"""
bcesv.py: the Boundary-Constrained Event-Source Verifier (BCESV) of §8.

Every variant applies the decision procedure of Algorithm 1 to a recall and
the provenance handles retrieved with it:

  1. For each recall component c, S_c is the set of retrieved handles whose
     fragment supplies c (the matching rule of the HARDENED scorer).
  2. Every handle in the union of the S_c is resolved to a group; a handle
     the resolver cannot resolve is unresolved.
  3. INVALID if the resolved handles fall into two or more groups.
  4. Otherwise ABSTAIN if some S_c is empty, some handle is unresolved, or no
     handle resolved.
  5. Otherwise SAFE.

The variants differ only in what a group is:

  FragmentOnly       no resolver: every handle is its own group.
  BcesvExact         trusted resolver: the boundary (tenant) of the handle.
  BcesvNoisy(p)      as BcesvExact, but each handle is unresolved with
                     probability p, decided deterministically from the seed
                     and the handle.
  BcesvReconstruct   no resolver: the connected components of the cohesion
                     rule, which links two fragments that share a tenant,
                     have timestamps at most RECON_TIME_WINDOW_SEC apart,
                     have an entity-set Jaccard of at least
                     RECON_ENTITY_JACCARD, and share a log source or an
                     incident template. It never reads boundary_id or
                     event_id.

Ground truth (ground_truth_boundary_invalid) is the HARDENED scorer of
analysis/rescore_final_ci.py, the scorer behind Table 1: the fragments that
supply the recall span two or more boundaries.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROLES = ("subject", "action", "object", "outcome", "time")
RECON_TIME_WINDOW_SEC = 600
RECON_ENTITY_JACCARD = 0.5


@dataclass
class StoreEntry:
    """Store-side metadata of one fragment, keyed by its public provenance handle."""
    role: str
    value: str
    sanitized_value: str
    text: str
    sanitized_text: str
    public_provenance_id: str
    record_id: str
    boundary_id: str
    event_id: str
    tenant_id: str
    timestamp: str
    log_source: str
    incident_template: str
    entity_set: tuple

    @classmethod
    def from_dict(cls, d: dict) -> "StoreEntry":
        return cls(
            role=d["role"],
            value=d.get("value", d["sanitized_value"]),
            sanitized_value=d["sanitized_value"],
            text=d.get("text", d["sanitized_text"]),
            sanitized_text=d["sanitized_text"],
            public_provenance_id=d["public_provenance_id"],
            record_id=d.get("record_id", d["event_id"]),
            boundary_id=d["boundary_id"],
            event_id=d["event_id"],
            tenant_id=d["tenant_id"],
            timestamp=d["timestamp"],
            log_source=d["log_source"],
            incident_template=d["incident_template"],
            entity_set=tuple(d.get("entity_set", [])),
        )


def load_store(condition_dir: Path) -> dict[str, StoreEntry]:
    """Fragment store of one condition, keyed by public provenance handle."""
    store: dict[str, StoreEntry] = {}
    with open(condition_dir / "fragments.store.jsonl") as f:
        for line in f:
            d = json.loads(line)
            store[d["public_provenance_id"]] = StoreEntry.from_dict(d)
    return store


def load_responses(condition_dir: Path) -> list[dict]:
    """Responses of one condition as dictionaries."""
    with open(condition_dir / "responses.jsonl") as f:
        return [json.loads(line) for line in f]


def supplying_provenance_ids(val: str, store: dict[str, StoreEntry], prov_ids: list[str],
                             view: str, casefold: bool = False) -> set[str]:
    """Handles in prov_ids whose fragment supplies the recall value val.

    A fragment supplies val if val equals its value, contains its value, or
    occurs in its text. With casefold, the comparison ignores letter case.
    """
    if not isinstance(val, str) or not val.strip():
        return set()
    val = val.strip()
    if casefold:
        val = val.lower()
    value_field = "sanitized_value" if view == "tenant_hidden" else "value"
    text_field = "sanitized_text" if view == "tenant_hidden" else "text"
    out = set()
    for pid in prov_ids:
        if pid not in store:
            continue
        f = store[pid]
        fv = str(getattr(f, value_field, "")).strip()
        ft = str(getattr(f, text_field, "")).strip()
        if casefold:
            fv, ft = fv.lower(), ft.lower()
        if not fv and not ft:
            continue
        if (fv and val == fv) or (fv and fv in val) or (ft and val in ft):
            out.add(pid)
    return out


def all_supplying(resp: dict, store: dict[str, StoreEntry], casefold: bool = False) -> dict[str, set[str]]:
    """Supplying handles for each of the five recall components."""
    recall = resp.get("recall") or {}
    view = resp.get("value_view", "tenant_hidden")
    prov_ids = resp.get("fragment_provenance_ids", [])
    return {c: supplying_provenance_ids(recall.get(c, ""), store, prov_ids, view, casefold) for c in ROLES}


def ground_truth_boundary_invalid(resp: dict, store: dict[str, StoreEntry]) -> bool:
    """HARDENED scorer: the supplying fragments span two or more boundaries."""
    if resp.get("abstained") or resp.get("recall") is None:
        return False
    return len({store[p].boundary_id for pids in all_supplying(resp, store).values() for p in pids}) > 1


def ground_truth_event_invalid(resp: dict, store: dict[str, StoreEntry]) -> bool:
    """Event-level counterpart: the supplying fragments span two or more events."""
    if resp.get("abstained") or resp.get("recall") is None:
        return False
    return len({store[p].event_id for pids in all_supplying(resp, store).values() for p in pids}) > 1


class VerifierBase:
    """Algorithm 1. Subclasses define how resolved handles form groups."""
    name = "base"

    def __init__(self, drop_prob: float = 0.0, seed: int = 42, casefold: bool = False):
        self.drop_prob = drop_prob
        self.seed = seed
        self.casefold = casefold

    def is_unresolved(self, pid: str) -> bool:
        """Whether the resolver fails on this handle: a fixed function of (seed, handle)."""
        if self.drop_prob <= 0.0:
            return False
        h = hashlib.sha256(f"{self.seed}:{pid}".encode()).digest()
        return int.from_bytes(h[:8], "big") / (1 << 64) < self.drop_prob

    def n_groups(self, pids: set[str], store: dict[str, StoreEntry]) -> int:
        raise NotImplementedError

    def predict(self, resp: dict, store: dict[str, StoreEntry]) -> str:
        """INVALID, SAFE or ABSTAIN for one response; a response without a recall is ABSTAIN."""
        if resp.get("abstained") or resp.get("recall") is None:
            return "ABSTAIN"
        resolved, unsupported, unresolved = set(), False, False
        for pids in all_supplying(resp, store, self.casefold).values():
            if not pids:
                unsupported = True
            for pid in pids:
                if self.is_unresolved(pid):
                    unresolved = True
                else:
                    resolved.add(pid)
        n = self.n_groups(resolved, store)
        if n > 1:
            return "INVALID"
        if unsupported or unresolved or n == 0:
            return "ABSTAIN"
        return "SAFE"


class FragmentOnly(VerifierBase):
    """No resolver: every handle is its own group, so two supplying handles refute."""
    name = "FragmentOnly"

    def n_groups(self, pids, store):
        return len(pids)


class BcesvExact(VerifierBase):
    """Trusted resolver: a handle's group is the boundary of its fragment."""

    def __init__(self, casefold: bool = False):
        super().__init__(casefold=casefold)
        self.name = "BcesvExact[casefold]" if casefold else "BcesvExact"

    def n_groups(self, pids, store):
        return len({store[p].boundary_id for p in pids})


class BcesvNoisy(BcesvExact):
    """BcesvExact with a resolver that fails on each handle with probability drop_prob."""

    def __init__(self, drop_prob: float, seed: int = 42):
        super().__init__()
        self.drop_prob = drop_prob
        self.seed = seed
        self.name = f"BcesvNoisy({round(drop_prob * 100)}%)"


class BcesvReconstruct(VerifierBase):
    """No resolver: groups are clusters of fragments inferred from store-side metadata.

    The cohesion rule links two fragments when every enabled feature agrees:
    same tenant, timestamps at most RECON_TIME_WINDOW_SEC apart, entity-set
    Jaccard at least RECON_ENTITY_JACCARD, and same log source or same incident
    template (when one of the two is disabled, the other must agree). Groups
    are the connected components. Disabling a feature gives the ablations of
    Appendix D.
    """

    def __init__(self, use_tenant: bool = True, use_time: bool = True, use_entity_set: bool = True,
                 use_log_source: bool = True, use_template: bool = True):
        super().__init__()
        self.use_tenant = use_tenant
        self.use_time = use_time
        self.use_entity_set = use_entity_set
        self.use_log_source = use_log_source
        self.use_template = use_template
        dropped = [tag for tag, on in (("no_tenant", use_tenant), ("no_time", use_time),
                                       ("no_entity", use_entity_set), ("no_logsrc", use_log_source),
                                       ("no_template", use_template)) if not on]
        self.name = f"BcesvRecon[{','.join(dropped)}]" if dropped else "BcesvReconstruct"

    def same_event(self, a: StoreEntry, b: StoreEntry) -> bool:
        if self.use_tenant and a.tenant_id != b.tenant_id:
            return False
        if self.use_time:
            try:
                gap = abs((datetime.fromisoformat(a.timestamp) - datetime.fromisoformat(b.timestamp)).total_seconds())
            except (ValueError, TypeError):
                return False
            if gap > RECON_TIME_WINDOW_SEC:
                return False
        if self.use_entity_set:
            ea, eb = set(a.entity_set), set(b.entity_set)
            if not ea or not eb or len(ea & eb) / len(ea | eb) < RECON_ENTITY_JACCARD:
                return False
        source_checks = []
        if self.use_log_source:
            source_checks.append(a.log_source == b.log_source)
        if self.use_template:
            source_checks.append(a.incident_template == b.incident_template)
        return not source_checks or any(source_checks)

    def n_groups(self, pids, store):
        frags = [store[p] for p in sorted(pids)]
        parent = list(range(len(frags)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                if self.same_event(frags[i], frags[j]):
                    parent[find(i)] = find(j)
        return len({find(i) for i in range(len(frags))})


def wilson_ci(k: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)
