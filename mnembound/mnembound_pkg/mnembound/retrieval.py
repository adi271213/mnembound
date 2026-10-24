"""
mnembound.retrieval — the four retrieval rungs.

1. ISOLATED                  — the probe's fragments of the querying tenant
2. BOUNDARY_PRESERVING_TOPK  — the probe's fragments in the querying tenant's
                               boundary; at tenant granularity the same set
                               as ISOLATED
3. SHARED_VECTOR_INDEX       — all of the probe's fragments, across tenants
4. SUMMARY_MERGE             — all of the probe's fragments, which the prompt
                               builder renders as a merged summary
                               (render_summary_text)

Every rung filters the evidence set attached to the probe at construction
time, so the construction-time labels stay exact; no rung queries an index.
retrieve(rung, probe, corpus) returns a `RetrievalResult` with the rung
(`rung`), the fragments the agent sees (`fragments`), the querying tenant
(`querying_tenant`) and the fragments the rung left out
(`excluded_fragments`). analysis/exposure_check.py measures what BM25 and
MiniLM retrieval over the whole fragment store returns.

The querying tenant is the tenant on whose behalf the agent runs; see
querying_tenant_for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from mnembound.schema import (
    ComponentRole,
    Corpus,
    Fragment,
    Probe,
    ProbeType,
)


class Rung(str, Enum):
    """The four sharing rungs."""

    ISOLATED = "isolated"
    BOUNDARY_PRESERVING_TOPK = "boundary_preserving_topk"
    SHARED_VECTOR_INDEX = "shared_vector_index"
    SUMMARY_MERGE = "summary_merge"


ALL_RUNGS: tuple[Rung, ...] = (
    Rung.ISOLATED,
    Rung.BOUNDARY_PRESERVING_TOPK,
    Rung.SHARED_VECTOR_INDEX,
    Rung.SUMMARY_MERGE,
)


@dataclass(frozen=True)
class RetrievalResult:
    """
    The output of one retrieval call.

    Fields:
        rung: which rung produced this result
        fragments: the fragments the agent sees (post-filtering)
        querying_tenant: the tenant on whose behalf the agent is querying
        context_text: empty for every rung; the summary-merge text is
            rendered when the prompt is built (render_summary_text).
        excluded_fragments: fragments of the probe's evidence set that the
            rung filtered out. For ISOLATED, these are the cross-tenant
            fragments. For BOUNDARY_PRESERVING_TOPK, the cross-boundary
            ones. SHARED_VECTOR_INDEX and SUMMARY_MERGE exclude nothing.
            summarize_rung reports their average number.
    """

    rung: Rung
    fragments: tuple[Fragment, ...]
    querying_tenant: str
    context_text: str = ""
    excluded_fragments: tuple[Fragment, ...] = field(default_factory=tuple)


# ----- Querying tenant resolution -----


def querying_tenant_for(probe: Probe) -> str:
    """
    Determine which tenant is querying for this probe.

    For SUPPORTED: the tenant of the probe's event, found through
    boundary_event_pairs. For PARTIAL (whose boundary_event_pairs is
    empty): the tenant of the first fragment, one of the event's own
    fragments, which precede the distractors.

    For ABSENT: the tenant of a fragment that carries the recall's subject
    in the subject role, if the evidence set holds one; otherwise the
    tenant with the most fragments in the evidence set (the first such
    tenant on a tie).

    For CROSS_BOUNDARY: the tenant of the fragment that supplies the
    subject component. The other fragments' tenants are the ones the trap
    pulls in.

    The resolution is deterministic given a probe.
    """
    if not probe.fragments:
        raise ValueError(
            f"probe {probe.probe_id} has no fragments; cannot determine "
            "querying tenant"
        )

    if probe.probe_type in (ProbeType.SUPPORTED, ProbeType.PARTIAL):
        # Distractors may come from other tenants, so the event's tenant is
        # found through its (boundary, event) pair when the probe has one.
        if probe.boundary_event_pairs:
            target_pair = next(iter(probe.boundary_event_pairs))
            for f in probe.fragments:
                if (f.boundary_id, f.event_id) == target_pair:
                    return f.tenant_id
        return probe.fragments[0].tenant_id

    # For CROSS_BOUNDARY, the querying tenant is the subject fragment's
    # tenant: the agent asks "what happened with subject X?" from the
    # tenant where X lives.
    if probe.probe_type == ProbeType.CROSS_BOUNDARY:
        for f in probe.fragments:
            if f.role == ComponentRole.SUBJECT:
                return f.tenant_id
        return probe.fragments[0].tenant_id

    # For ABSENT, the evidence set comes from other records than the
    # recall's: a fragment that carries the recall's subject decides if
    # there is one, otherwise the tenant with the most fragments.
    if probe.probe_type == ProbeType.ABSENT:
        for f in probe.fragments:
            if f.role == ComponentRole.SUBJECT and f.value == probe.recall.subject:
                return f.tenant_id
        tenant_counts: dict[str, int] = {}
        for f in probe.fragments:
            tenant_counts[f.tenant_id] = tenant_counts.get(f.tenant_id, 0) + 1
        return max(tenant_counts.items(), key=lambda kv: kv[1])[0]

    raise ValueError(f"unknown probe type: {probe.probe_type}")


# ----- Rung implementations -----


def retrieve_isolated(probe: Probe, corpus: Corpus) -> RetrievalResult:
    """
    ISOLATED rung: only fragments from the querying tenant are visible.

    Cross-tenant fragments are filtered out. For SUPPORTED probes, this
    is a no-op (all fragments are same-tenant). For CROSS_BOUNDARY
    traps, this strips the cross-tenant fragments, which means the
    composed recall is no longer fragment-supported — exactly the
    isolation guarantee.
    """
    tenant = querying_tenant_for(probe)
    keep: list[Fragment] = []
    drop: list[Fragment] = []
    for f in probe.fragments:
        if f.tenant_id == tenant:
            keep.append(f)
        else:
            drop.append(f)
    return RetrievalResult(
        rung=Rung.ISOLATED,
        fragments=tuple(keep),
        querying_tenant=tenant,
        excluded_fragments=tuple(drop),
    )


def retrieve_boundary_preserving_topk(
    probe: Probe, corpus: Corpus
) -> RetrievalResult:
    """
    BOUNDARY_PRESERVING_TOPK rung: the probe's fragments in the querying
    tenant's boundary.

    Under TENANT boundary granularity, this is equivalent to ISOLATED
    (one boundary per tenant). Under ENGAGEMENT or EVENT granularity,
    this is stricter: only fragments of the querying boundary are
    visible, which is the boundary of the querying tenant's subject
    fragment, else of the querying tenant's first fragment, else of the
    probe's first fragment.
    """
    tenant = querying_tenant_for(probe)
    # Determine the querying boundary: the boundary of the subject
    # fragment if any, else the first fragment.
    querying_boundary = None
    for f in probe.fragments:
        if f.role == ComponentRole.SUBJECT and f.tenant_id == tenant:
            querying_boundary = f.boundary_id
            break
    if querying_boundary is None:
        for f in probe.fragments:
            if f.tenant_id == tenant:
                querying_boundary = f.boundary_id
                break
    if querying_boundary is None and probe.fragments:
        querying_boundary = probe.fragments[0].boundary_id

    keep: list[Fragment] = []
    drop: list[Fragment] = []
    for f in probe.fragments:
        if f.boundary_id == querying_boundary:
            keep.append(f)
        else:
            drop.append(f)
    return RetrievalResult(
        rung=Rung.BOUNDARY_PRESERVING_TOPK,
        fragments=tuple(keep),
        querying_tenant=tenant,
        excluded_fragments=tuple(drop),
    )


def retrieve_shared_vector_index(
    probe: Probe, corpus: Corpus
) -> RetrievalResult:
    """
    SHARED_VECTOR_INDEX rung: the probe's evidence set unchanged,
    regardless of tenant or boundary. On a trap, the agent receives
    fragments of several tenants.
    """
    tenant = querying_tenant_for(probe)
    return RetrievalResult(
        rung=Rung.SHARED_VECTOR_INDEX,
        fragments=probe.fragments,
        querying_tenant=tenant,
    )


def retrieve_summary_merge(probe: Probe, corpus: Corpus) -> RetrievalResult:
    """
    SUMMARY_MERGE rung: every fragment of the probe's evidence set, which
    the agent receives as a merged summary.

    context_text is left empty, because rendering here would force a
    choice of tenant_visible vs tenant_hidden at retrieval time and could
    leak tenant-revealing text into downstream logs. The prompt builder
    calls `render_summary_text(probe, tenant_visible=...)` at the point of
    use, which makes the visibility choice explicit and local. The summary
    shows no provenance handles, so the agent cannot point at a source
    fragment.

    The `fragments` field holds every fragment, so scoring predicates can
    run.
    """
    tenant = querying_tenant_for(probe)

    return RetrievalResult(
        rung=Rung.SUMMARY_MERGE,
        fragments=probe.fragments,
        querying_tenant=tenant,
        context_text="",  # rendered on demand by render_summary_text()
    )


class SummaryMergeMode(str, Enum):
    """
    How the summary_merge rung renders its evidence text.

    PER_CLIENT_BLOCKS: one block per tenant, in tenant order, headed
        "Summary of recent activity for this client:" and listing the
        tenant's fragments sorted by role name. The prompt labels the
        evidence "Evidence summary by client".

    COLLAPSED: one bullet list of every fragment's text, sorted by tenant
        and role name, without tenant headers. The prompt labels the
        evidence "Merged memory summary". The paper's headline conditions
        use it.
    """

    PER_CLIENT_BLOCKS = "per_client_blocks"
    COLLAPSED = "collapsed"


def render_summary_text(
    probe: Probe,
    *,
    tenant_visible: bool,
    mode: SummaryMergeMode = SummaryMergeMode.COLLAPSED,
) -> str:
    """
    Render the summary_merge context text body (without an outer label
    header; the prompt builder adds that).

    Args:
        probe: The probe whose retrieved fragments are summarized.
        tenant_visible: REQUIRED keyword. If False, uses `f.sanitized_text`;
            otherwise `f.text`.
        mode: PER_CLIENT_BLOCKS or COLLAPSED. See SummaryMergeMode.
    """
    def fragment_text(f: Fragment) -> str:
        return f.text if tenant_visible else f.sanitized_text

    if mode == SummaryMergeMode.COLLAPSED:
        # One bullet list, sorted by tenant and role name, without tenant
        # headers.
        sorted_frags = sorted(
            probe.fragments,
            key=lambda f: (f.tenant_id, f.role.value),
        )
        return "\n".join(f"- {fragment_text(f)}" for f in sorted_frags)

    # PER_CLIENT_BLOCKS
    by_tenant: dict[str, list[Fragment]] = {}
    for f in probe.fragments:
        by_tenant.setdefault(f.tenant_id, []).append(f)
    summary_parts: list[str] = []
    for tid in sorted(by_tenant.keys()):
        frags = sorted(by_tenant[tid], key=lambda f: f.role.value)
        bullet_lines = [f"  - {fragment_text(f)}" for f in frags]
        summary_parts.append(
            "Summary of recent activity for this client:\n" + "\n".join(bullet_lines)
        )
    return "\n\n".join(summary_parts)


# ----- Dispatch -----


_RETRIEVAL_FUNCTIONS: dict[Rung, Callable[[Probe, Corpus], RetrievalResult]] = {
    Rung.ISOLATED: retrieve_isolated,
    Rung.BOUNDARY_PRESERVING_TOPK: retrieve_boundary_preserving_topk,
    Rung.SHARED_VECTOR_INDEX: retrieve_shared_vector_index,
    Rung.SUMMARY_MERGE: retrieve_summary_merge,
}


def retrieve(rung: Rung, probe: Probe, corpus: Corpus) -> RetrievalResult:
    """Dispatch to the appropriate rung's retrieval function."""
    fn = _RETRIEVAL_FUNCTIONS.get(rung)
    if fn is None:
        raise ValueError(f"unknown rung: {rung}")
    return fn(probe, corpus)


# ----- Per-rung statistics -----


@dataclass(frozen=True)
class RungSummary:
    """Aggregate stats over one rung's retrievals across many probes."""

    rung: Rung
    n_probes: int
    avg_fragments_retrieved: float
    avg_fragments_excluded: float
    # Per probe type, the fraction of fragments retained vs probe.fragments.
    retention_by_probe_type: dict[str, float]


def _sum_left_to_right(values: list[float]) -> float:
    # sum() of floats is compensated from Python 3.12 on; plain addition keeps
    # the released rung_stats.jsonl reproducible on every Python version.
    total = 0.0
    for v in values:
        total += v
    return total


def summarize_rung(
    rung: Rung, probes: list[Probe], corpus: Corpus
) -> RungSummary:
    """Run a rung over a list of probes and report aggregate statistics."""
    if not probes:
        return RungSummary(
            rung=rung,
            n_probes=0,
            avg_fragments_retrieved=0.0,
            avg_fragments_excluded=0.0,
            retention_by_probe_type={},
        )

    total_retrieved = 0
    total_excluded = 0
    retained_by_type: dict[str, list[float]] = {}

    for p in probes:
        result = retrieve(rung, p, corpus)
        total_retrieved += len(result.fragments)
        total_excluded += len(result.excluded_fragments)
        total_in = len(p.fragments)
        retention = len(result.fragments) / total_in if total_in > 0 else 0.0
        retained_by_type.setdefault(p.probe_type.value, []).append(retention)

    return RungSummary(
        rung=rung,
        n_probes=len(probes),
        avg_fragments_retrieved=total_retrieved / len(probes),
        avg_fragments_excluded=total_excluded / len(probes),
        retention_by_probe_type={
            t: _sum_left_to_right(v) / len(v) for t, v in retained_by_type.items()
        },
    )
