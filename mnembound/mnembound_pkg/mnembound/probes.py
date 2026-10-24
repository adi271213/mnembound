"""
mnembound.probes — Probe generation for the four MnemBound probe types.

A `Probe` is what an agent (or a deterministic verifier) is presented with.
Each probe carries:

- a `Recall`: the event description we are asking about
- a tuple of `Fragment`s: the probe's evidence set, which the rungs filter
  before the agent sees it
- exact ground-truth labels: is_fragment_supported, is_event_supported,
  and the set of distinct (boundary_id, event_id) pairs the supporting
  fragments resolve to (for CROSS_BOUNDARY probes)

The four probe types of the paper:

SUPPORTED      — the recall is an event of the corpus, and the evidence set
                 holds that event's five fragments.
ABSENT         — the recall is an event that no record holds, and the
                 evidence set comes from other events; it does not supply
                 all of the recall's components. The agent should abstain.
PARTIAL        — the recall is an event of the corpus, but the evidence set
                 holds only some of its fragments, so it is incomplete for a
                 single event.
CROSS_BOUNDARY — the cross-boundary trap probe: every component is supplied
                 by a real fragment, but the fragments come from different
                 events in two or more boundaries, and no record holds all
                 five components. A recall of it is fragment-faithful but
                 event-false.

Ground truth is computed at construction time using the store-level view
of the corpus (records and fragments with hidden metadata). The probe's
ground-truth fields are then frozen onto the Probe dataclass, so any
downstream scoring is exact and does not require LLM judgment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from mnembound.schema import (
    ALL_ROLES,
    ComponentRole,
    Corpus,
    Fragment,
    Probe,
    ProbeType,
    Recall,
    Record,
)


# ---------- Helpers ----------


@dataclass(frozen=True)
class _ComponentIndex:
    """
    Precomputed indices over a corpus's fragments and records.

    Used by probe generators to:
    - Find fragments that supply a given component value in a given role.
    - Check whether a tuple of component values matches any single record
      in the corpus (the constructive event-supported check).
    """

    fragments_by_record: dict[str, tuple[Fragment, ...]]
    fragments_by_role: dict[ComponentRole, tuple[Fragment, ...]]
    record_by_id: dict[str, Record]
    # For the event-support check: a set of frozensets, one per record,
    # where the frozenset contains the record's five (role, value) pairs.
    # A recall is event-supported iff its five (role, value) pairs are a
    # subset of one of these frozensets (which is the same as equality,
    # because all five roles are always present in our records).
    record_signatures: frozenset[frozenset[tuple[ComponentRole, str]]]
    # Index by signature -> record_id, so we can recover which event
    # supports a given recall if any does.
    signature_to_record_id: dict[frozenset[tuple[ComponentRole, str]], str]


def _build_component_index(corpus: Corpus) -> _ComponentIndex:
    """One-time precomputation over a corpus."""
    by_record: dict[str, list[Fragment]] = {}
    by_role: dict[ComponentRole, list[Fragment]] = {role: [] for role in ALL_ROLES}
    for f in corpus.fragments:
        by_record.setdefault(f.record_id, []).append(f)
        by_role[f.role].append(f)

    fragments_by_record = {rid: tuple(fs) for rid, fs in by_record.items()}
    fragments_by_role = {role: tuple(fs) for role, fs in by_role.items()}

    record_by_id = {r.record_id: r for r in corpus.records}

    sig_to_rid: dict[frozenset[tuple[ComponentRole, str]], str] = {}
    for r in corpus.records:
        sig = frozenset(
            (role, r.component(role)) for role in ALL_ROLES
        )
        # If two records have identical recall components, the signature
        # collides. This is unlikely at the corpus's default scale but we
        # don't crash on it; we just keep the first observed record_id.
        # Either record would satisfy event-supported equally.
        sig_to_rid.setdefault(sig, r.record_id)
    record_signatures = frozenset(sig_to_rid.keys())

    return _ComponentIndex(
        fragments_by_record=fragments_by_record,
        fragments_by_role=fragments_by_role,
        record_by_id=record_by_id,
        record_signatures=record_signatures,
        signature_to_record_id=sig_to_rid,
    )


def _recall_from_record(record: Record) -> Recall:
    """Construct the exact Recall corresponding to a record's components."""
    return Recall(
        subject=record.subject,
        action=record.action,
        object_=record.object_,
        outcome=record.outcome,
        time=record.time,
    )


def _recall_from_fragments(
    fragments: dict[ComponentRole, Fragment],
) -> Recall:
    """Construct a Recall whose components are the values of these fragments."""
    return Recall(
        subject=fragments[ComponentRole.SUBJECT].value,
        action=fragments[ComponentRole.ACTION].value,
        object_=fragments[ComponentRole.OBJECT].value,
        outcome=fragments[ComponentRole.OUTCOME].value,
        time=fragments[ComponentRole.TIME].value,
    )


def _recall_signature(recall: Recall) -> frozenset[tuple[ComponentRole, str]]:
    """The signature used for event-supported lookup."""
    return frozenset(
        (role, recall.component(role)) for role in ALL_ROLES
    )


def _is_event_supported_by_construction(
    recall: Recall, index: _ComponentIndex
) -> bool:
    """
    Construction-time event-supported check.

    Returns True iff some record in the corpus has the exact same five
    (role, value) pairs as `recall`.
    """
    return _recall_signature(recall) in index.record_signatures


def _is_fragment_supported_by_set(
    recall: Recall, fragments: Iterable[Fragment]
) -> bool:
    """
    Construction-time fragment-supported check.

    Returns True iff every component value of `recall` appears in some
    fragment of `fragments` with matching role.
    """
    # Build a {role: set of values} index over the fragment set.
    by_role: dict[ComponentRole, set[str]] = {role: set() for role in ALL_ROLES}
    for f in fragments:
        by_role[f.role].add(f.value)
    for role in ALL_ROLES:
        if recall.component(role) not in by_role[role]:
            return False
    return True


# ---------- Supported recall probes ----------


def generate_supported_probes(
    corpus: Corpus,
    n_probes: int,
    seed: int,
    distractor_fragments: int = 0,
) -> tuple[Probe, ...]:
    """
    Generate SUPPORTED probes: the recall is a real event in the corpus,
    and the evidence set contains the five fragments of that event (plus
    optional distractors from OTHER events).

    Args:
        corpus: The corpus to draw from.
        n_probes: How many probes to generate.
        seed: For deterministic sampling.
        distractor_fragments: Extra unrelated fragments to add to the
            evidence set. These come from other events and do NOT supply
            any component of the recall. Defaults to 0 (clean evidence).

    Returns:
        Tuple of n_probes SUPPORTED probes. Each has is_fragment_supported=
        True, is_event_supported=True, and boundary_event_pairs containing
        the single (boundary_id, event_id) of the underlying event.
    """
    if n_probes < 1:
        return ()
    if n_probes > len(corpus.records):
        raise ValueError(
            f"requested {n_probes} supported probes but corpus has only "
            f"{len(corpus.records)} records"
        )

    rng = random.Random(seed)
    index = _build_component_index(corpus)

    selected_records = rng.sample(corpus.records, n_probes)
    probes: list[Probe] = []

    for i, r in enumerate(selected_records):
        primary_fragments = index.fragments_by_record[r.record_id]
        assert len(primary_fragments) == 5

        retrieved = list(primary_fragments)

        if distractor_fragments > 0:
            # Pull distractor fragments from other records, deterministically.
            all_other = [
                f for f in corpus.fragments if f.record_id != r.record_id
            ]
            n_dist = min(distractor_fragments, len(all_other))
            retrieved.extend(rng.sample(all_other, n_dist))

        recall = _recall_from_record(r)

        probes.append(Probe(
            probe_id=f"supported-{i:06d}",
            probe_type=ProbeType.SUPPORTED,
            recall=recall,
            fragments=tuple(retrieved),
            is_fragment_supported=True,
            is_event_supported=True,
            boundary_event_pairs=frozenset({r.boundary_event_key()}),
        ))

    return tuple(probes)


# ---------- Absent recall probes ----------


def generate_absent_probes(
    corpus: Corpus,
    n_probes: int,
    seed: int,
    retrieval_size: int = 5,
) -> tuple[Probe, ...]:
    """
    Generate ABSENT probes: synthetic non-events constructed by component
    substitution and validated as neither fragment-supported nor
    event-supported.

    Construction details: we sample a base record, then replace ONE of its
    five components (chosen at random) with the corresponding component
    value from a different record. The resulting recall is therefore not a
    pure-random recall; it is a near-miss: four components from one real
    event plus one component from another real event. We validate that the
    composed recall does NOT appear as any single record in the corpus
    (rejection sampling), and we build the evidence set from fragments of
    yet other records so the recall is also not fragment-supported by the
    evidence set.

    Because the evidence set comes from other records, it supplies few of
    the recall's components (at most two of the five in the released
    corpora; run_paper.py check counts them), so the agent should abstain;
    a recall of it is unsupported, not a cross-boundary composition.

    Args:
        corpus: The corpus to draw from.
        n_probes: How many probes to generate.
        seed: For deterministic sampling.
        retrieval_size: How many fragments are in the evidence set
            (drawn from records other than the two that the recall is
            built from).

    Returns:
        Tuple of ABSENT probes. Each has is_fragment_supported=False and
        is_event_supported=False: no record holds the recall, and the
        evidence set does not supply all of its components.
    """
    if n_probes < 1:
        return ()

    rng = random.Random(seed)
    index = _build_component_index(corpus)

    if len(corpus.records) < 2:
        raise ValueError(
            "absent probes require at least 2 records in the corpus"
        )

    probes: list[Probe] = []
    attempts = 0
    max_attempts = n_probes * 20  # bound the search for valid probes

    while len(probes) < n_probes and attempts < max_attempts:
        attempts += 1

        # r_recall supplies the recall. r_retrieval only constrains
        # third_record below; the evidence set is drawn from all fragments
        # outside r_recall and third_record.
        r_recall, r_retrieval = rng.sample(corpus.records, 2)

        # Build the candidate recall from r_recall's components.
        # This recall IS event-supported (by r_recall). So we need to
        # MODIFY one component to make the recall novel.
        #
        # Approach: take r_recall's components but replace ONE component
        # with a value from a third record. Then check the resulting
        # recall is not event-supported.

        if len(corpus.records) < 3:
            return tuple(probes)

        third_record = rng.choice(corpus.records)
        if third_record.record_id in (r_recall.record_id, r_retrieval.record_id):
            continue

        # Pick which role to swap.
        swap_role = rng.choice(ALL_ROLES)
        novel_value = third_record.component(swap_role)
        if novel_value == r_recall.component(swap_role):
            continue  # no actual swap, recall would be event-supported

        # Construct the swapped recall.
        component_values = {role: r_recall.component(role) for role in ALL_ROLES}
        component_values[swap_role] = novel_value

        swapped_recall = Recall(
            subject=component_values[ComponentRole.SUBJECT],
            action=component_values[ComponentRole.ACTION],
            object_=component_values[ComponentRole.OBJECT],
            outcome=component_values[ComponentRole.OUTCOME],
            time=component_values[ComponentRole.TIME],
        )

        # Reject if any record happens to have these exact components.
        if _is_event_supported_by_construction(swapped_recall, index):
            continue

        # Build the evidence set from fragments of the other records.
        retrieval_pool = [
            f for f in corpus.fragments
            if f.record_id not in (r_recall.record_id, third_record.record_id)
        ]
        if len(retrieval_pool) < retrieval_size:
            continue
        retrieved = rng.sample(retrieval_pool, retrieval_size)

        # Final check: the evidence set must NOT fragment-support the
        # recall (otherwise it would be PARTIAL or worse).
        if _is_fragment_supported_by_set(swapped_recall, retrieved):
            continue

        probes.append(Probe(
            probe_id=f"absent-{len(probes):06d}",
            probe_type=ProbeType.ABSENT,
            recall=swapped_recall,
            fragments=tuple(retrieved),
            is_fragment_supported=False,
            is_event_supported=False,
            boundary_event_pairs=frozenset(),
        ))

    return tuple(probes)


# ---------- Partial-evidence probes ----------


def generate_partial_probes(
    corpus: Corpus,
    n_probes: int,
    seed: int,
    n_supported_components: int = 3,
    distractor_fragments: int = 0,
) -> tuple[Probe, ...]:
    """
    Generate PARTIAL probes: SOME components of the recall are
    fragment-supported by the evidence set; OTHERS are not.

    Construction: start with a real event. The evidence set contains
    `n_supported_components` of that event's five fragments, plus
    optionally `distractor_fragments` unrelated fragments from OTHER events.
    The recall asks about the full event. Therefore exactly
    (5 - n_supported_components) recall components are not fragment-
    supported by the evidence set, and the distractors do not change
    that (they come from other records and cannot fill the missing roles
    of the recall's underlying event).

    The recall is a real event of the corpus, but the evidence set does
    not support it: is_fragment_supported and is_event_supported are judged
    on the evidence set, which lacks the missing roles.

    Distractors and realism: with `distractor_fragments=0`, missing roles
    are conspicuously absent and the partial probe is easy to abstain on.
    With a moderate number of distractors (e.g., 5), the evidence looks like
    a realistic incomplete retrieval — the agent has to identify which roles
    are missing rather than noticing that the evidence set is tiny.
    The released corpora use `distractor_fragments=5`.

    Args:
        n_supported_components: How many of the 5 components are
            fragment-supported by the primary fragments. Must be 1..4
            (0 would be ABSENT, 5 would be SUPPORTED).
        distractor_fragments: How many extra unrelated fragments to add
            to the evidence set. These come from OTHER events than the
            probe's underlying event and never fill missing roles.
    """
    if n_probes < 1:
        return ()
    if not (1 <= n_supported_components <= 4):
        raise ValueError(
            f"n_supported_components must be 1..4, got {n_supported_components}"
        )
    if distractor_fragments < 0:
        raise ValueError(
            f"distractor_fragments must be non-negative, got {distractor_fragments}"
        )

    rng = random.Random(seed)
    index = _build_component_index(corpus)

    selected_records = rng.sample(
        corpus.records, min(n_probes, len(corpus.records))
    )
    probes: list[Probe] = []

    for i, r in enumerate(selected_records):
        if len(probes) >= n_probes:
            break

        primary_fragments = list(index.fragments_by_record[r.record_id])
        # Pick which subset of the event's fragments the evidence set holds.
        retained = rng.sample(primary_fragments, n_supported_components)

        # Add distractor fragments from OTHER records. We exclude any
        # fragment whose value matches a missing role's value in `recall`;
        # otherwise a distractor could accidentally fill a missing role
        # and invalidate the construction-time `is_fragment_supported=False`
        # label.
        recall = _recall_from_record(r)
        retained_roles = {f.role for f in retained}
        missing_roles = set(ALL_ROLES) - retained_roles
        missing_values_by_role = {
            role: recall.component(role) for role in missing_roles
        }

        if distractor_fragments > 0:
            candidate_distractors = [
                f for f in corpus.fragments
                if f.record_id != r.record_id
                and not (
                    f.role in missing_values_by_role
                    and f.value == missing_values_by_role[f.role]
                )
            ]
            n_dist = min(distractor_fragments, len(candidate_distractors))
            distractors = rng.sample(candidate_distractors, n_dist)
            retained = retained + distractors

        # Sanity check: the distractors never supply a missing component, so
        # the evidence set does not support the whole recall.
        assert not _is_fragment_supported_by_set(recall, retained), (
            f"partial probe {i} accidentally became fragment-supported "
            "after distractor injection"
        )

        probes.append(Probe(
            probe_id=f"partial-{i:06d}",
            probe_type=ProbeType.PARTIAL,
            recall=recall,
            fragments=tuple(retained),
            is_fragment_supported=False,
            is_event_supported=False,
            boundary_event_pairs=frozenset(),
        ))

    return tuple(probes)


# ---------- Cross-boundary trap probes ----------


def generate_cross_boundary_traps(
    corpus: Corpus,
    n_probes: int,
    seed: int,
    n_source_events: int = 5,
    max_retries_per_probe: int = 50,
    require_distinct_boundaries: bool = True,
    prefer_bridge_linked: bool = False,
) -> tuple[Probe, ...]:
    """
    Generate CROSS_BOUNDARY trap probes.

    Construction: select five fragments — one per role (subject, action,
    object, outcome, time) — from up to `n_source_events` DIFFERENT events,
    such that:

    1. Each fragment is a real authorized per-component fragment.
    2. The five supporting fragments resolve to different (boundary_id,
       event_id) pairs (at minimum, more than one distinct pair; with
       require_distinct_boundaries=True, more than one distinct
       boundary_id).
    3. The composed recall is NOT corpus-supported, i.e., no record
       contains all five (role, value) pairs together.

    The resulting probe has is_fragment_supported=True and
    is_event_supported=False: a recall of it is fragment-faithful but
    event-false.

    Bridge preference: when prefer_bridge_linked=True, the generator first
    tries to sample source records that share a bridge entity (in their
    `extra['bridge']` field or in their slot values), so that the bridge
    links the trap's records in the store. The agent sees the bridge only
    in a fragment whose text names it (see mnembound.bridges).

    When prefer_bridge_linked=True, every produced trap has a non-empty
    bridge_context if such traps are findable. If the corpus has no
    bridges, the generator falls back to pure-random sampling.

    Args:
        n_probes: How many traps to generate.
        seed: For deterministic sampling.
        n_source_events: How many distinct events the trap's fragments come
            from. 5 means every fragment comes from a different event;
            smaller values allow some role-collisions across events.
        max_retries_per_probe: Bound on rejection sampling.
        require_distinct_boundaries: If True, the n_source_events must
            also span >= 2 distinct boundary_ids. With TENANT granularity
            this means >= 2 tenants.
        prefer_bridge_linked: If True, preferentially sample source events
            that share a bridge entity. Falls back to pure-random if no
            bridge-linked candidates can be found within the retry budget.

    Returns:
        Tuple of CROSS_BOUNDARY probes, each fragment-supported and not
        event-supported by construction. Probes generated with
        prefer_bridge_linked=True may also carry non-empty bridge_context.
    """
    if n_probes < 1:
        return ()
    if not (2 <= n_source_events <= 5):
        raise ValueError(
            f"n_source_events must be 2..5, got {n_source_events}"
        )

    rng = random.Random(seed)
    index = _build_component_index(corpus)

    if len(corpus.records) < n_source_events:
        raise ValueError(
            f"need at least {n_source_events} records, have "
            f"{len(corpus.records)}"
        )

    # Pre-compute the bridge-keyed record index if needed.
    bridge_to_records: dict[str, list] = {}
    if prefer_bridge_linked:
        for r in corpus.records:
            # Extra-bridge
            bridge_val = r.extra.get("bridge") if r.extra else None
            if bridge_val:
                bridge_to_records.setdefault(bridge_val, []).append(r)
            # Slot-bridge (subject or object that is a bridge-type entity)
            for slot_val in (r.subject, r.object_):
                if isinstance(slot_val, str) and ":" in slot_val:
                    prefix = slot_val.split(":", 1)[0]
                    if prefix in ("ip", "hash", "cve", "vendor", "actor"):
                        bridge_to_records.setdefault(slot_val, []).append(r)
        # Drop bridges with only one record (cannot form cross-tenant trap).
        bridge_to_records = {
            b: rs for b, rs in bridge_to_records.items() if len(rs) >= 2
        }

    probes: list[Probe] = []
    total_attempts = 0
    max_total_attempts = n_probes * max_retries_per_probe
    # Track whether we're in bridge-mode or have fallen back.
    in_bridge_mode = prefer_bridge_linked and len(bridge_to_records) > 0

    while len(probes) < n_probes and total_attempts < max_total_attempts:
        total_attempts += 1

        # Sample n_source_events distinct records to draw fragments from.
        if in_bridge_mode:
            # Pick a bridge with at least n_source_events available records.
            eligible_bridges = [
                b for b, rs in bridge_to_records.items()
                if len(rs) >= n_source_events
            ]
            if not eligible_bridges:
                # Fall back to two-source bridge: pick a bridge with at
                # least 2 records, then fill remaining sources at random.
                eligible_bridges = list(bridge_to_records.keys())
                if not eligible_bridges:
                    in_bridge_mode = False
                    continue
            chosen_bridge = rng.choice(eligible_bridges)
            bridge_records = bridge_to_records[chosen_bridge]
            n_from_bridge = min(len(bridge_records), n_source_events)
            source_records = rng.sample(bridge_records, n_from_bridge)
            # Fill remaining sources from the rest of the corpus.
            if n_from_bridge < n_source_events:
                others = [r for r in corpus.records if r not in source_records]
                source_records += rng.sample(
                    others, n_source_events - n_from_bridge
                )
        else:
            source_records = rng.sample(corpus.records, n_source_events)

        # Assign each role to a fragment from one of the source records.
        roles = list(ALL_ROLES)
        rng.shuffle(roles)
        role_to_source = {
            role: source_records[i % n_source_events]
            for i, role in enumerate(roles)
        }

        role_to_fragment: dict[ComponentRole, Fragment] = {}
        for role, src in role_to_source.items():
            src_frags = index.fragments_by_record[src.record_id]
            chosen = next((f for f in src_frags if f.role == role), None)
            assert chosen is not None, (
                f"record {src.record_id} missing role {role}"
            )
            role_to_fragment[role] = chosen

        recall = _recall_from_fragments(role_to_fragment)

        # Check 1: not corpus-supported.
        if _is_event_supported_by_construction(recall, index):
            continue

        # Check 2: spans >= 2 distinct (boundary, event) pairs.
        be_pairs = {
            (f.boundary_id, f.event_id) for f in role_to_fragment.values()
        }
        if len(be_pairs) < 2:
            continue

        # Check 3: optional distinct boundaries.
        if require_distinct_boundaries:
            boundaries = {f.boundary_id for f in role_to_fragment.values()}
            if len(boundaries) < 2:
                continue

        # Compute bridge_context: the set of bridge entities shared by the
        # source records of the supporting fragments.
        contributing_record_ids = {f.record_id for f in role_to_fragment.values()}
        contributing_records = [
            index.record_by_id[rid] for rid in contributing_record_ids
        ]
        bridge_context = _compute_shared_bridges(contributing_records)

        # All five role-fragments form the evidence set.
        fragments_tuple = tuple(role_to_fragment[role] for role in ALL_ROLES)

        probes.append(Probe(
            probe_id=f"crossboundary-{len(probes):06d}",
            probe_type=ProbeType.CROSS_BOUNDARY,
            recall=recall,
            fragments=fragments_tuple,
            is_fragment_supported=True,
            is_event_supported=False,
            boundary_event_pairs=frozenset(be_pairs),
            bridge_context=bridge_context,
        ))

    return tuple(probes)


def _compute_shared_bridges(records: list) -> frozenset[str]:
    """
    Return the set of bridge entities that appear in MULTIPLE of the given
    records.

    A bridge entity is one of:
    - The value of Record.extra['bridge'], if present.
    - A subject or object_ value with prefix 'ip:' / 'hash:' / 'cve:' /
      'vendor:' / 'actor:'.

    We return the intersection of "bridges that appear in this record"
    across pairs of records — i.e., a bridge is in the context if at least
    two records share it.
    """
    per_record_bridges: list[set[str]] = []
    for r in records:
        bridges_here: set[str] = set()
        bridge_val = r.extra.get("bridge") if r.extra else None
        if bridge_val:
            bridges_here.add(bridge_val)
        for slot_val in (r.subject, r.object_):
            if isinstance(slot_val, str) and ":" in slot_val:
                prefix = slot_val.split(":", 1)[0]
                if prefix in ("ip", "hash", "cve", "vendor", "actor"):
                    bridges_here.add(slot_val)
        per_record_bridges.append(bridges_here)

    # A bridge is "shared" if it appears in >= 2 records.
    bridge_counts: dict[str, int] = {}
    for s in per_record_bridges:
        for b in s:
            bridge_counts[b] = bridge_counts.get(b, 0) + 1
    return frozenset(b for b, c in bridge_counts.items() if c >= 2)


# ---------- Convenience: build all four probe types at once ----------


@dataclass(frozen=True)
class ProbeMix:
    """A balanced mix of probes covering all four types."""

    supported: tuple[Probe, ...]
    absent: tuple[Probe, ...]
    partial: tuple[Probe, ...]
    cross_boundary: tuple[Probe, ...]

    def all_probes(self) -> tuple[Probe, ...]:
        return self.supported + self.absent + self.partial + self.cross_boundary


def generate_probe_mix(
    corpus: Corpus,
    seed: int,
    n_supported: int = 100,
    n_absent: int = 50,
    n_partial: int = 50,
    n_cross_boundary: int = 100,
    n_source_events_per_trap: int = 5,
    partial_distractor_fragments: int = 0,
    prefer_bridge_linked_traps: bool = False,
) -> ProbeMix:
    """
    Generate a balanced probe mix.

    The defaults give the 300 probes per corpus of the released runs: 100
    supported, 50 absent, 50 partial and 100 cross-boundary.

    Each probe type uses a sub-seed derived from the master seed so that
    changing one probe count does not perturb the others' sampling.

    Args:
        partial_distractor_fragments: How many unrelated fragments to add
            to PARTIAL probes' evidence sets (default 0; the released
            corpora use 5).
        prefer_bridge_linked_traps: When True, CROSS_BOUNDARY traps prefer
            source events that share a bridge entity. Requires the corpus
            to have been generated with bridges enabled.
    """
    rng = random.Random(seed)
    sub_seeds = {
        "supported": rng.randint(0, 2**31 - 1),
        "absent": rng.randint(0, 2**31 - 1),
        "partial": rng.randint(0, 2**31 - 1),
        "cross_boundary": rng.randint(0, 2**31 - 1),
    }

    return ProbeMix(
        supported=generate_supported_probes(
            corpus, n_supported, seed=sub_seeds["supported"]
        ),
        absent=generate_absent_probes(
            corpus, n_absent, seed=sub_seeds["absent"]
        ),
        partial=generate_partial_probes(
            corpus, n_partial, seed=sub_seeds["partial"],
            distractor_fragments=partial_distractor_fragments,
        ),
        cross_boundary=generate_cross_boundary_traps(
            corpus,
            n_cross_boundary,
            seed=sub_seeds["cross_boundary"],
            n_source_events=n_source_events_per_trap,
            prefer_bridge_linked=prefer_bridge_linked_traps,
        ),
    )
