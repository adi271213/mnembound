"""
mnembound.bridges — Cross-tenant bridge entities.

A "bridge" is an entity that legitimately appears in records belonging to
more than one tenant. SOC examples:

- A shared IP that participated in events at multiple MSSP clients
  (e.g., a phishing C2 server that hit clientA on Monday and clientB on
  Tuesday).
- A malware sample (hash) observed at multiple sites.
- A CVE referenced by alerts at multiple sites.
- A vendor product (Vendor-X EDR) mentioned in alerts at multiple sites.
- A named threat actor attributed to incidents at multiple sites.

With prefer_bridge_linked_traps, each cross-boundary trap is built from
source records that share a bridge entity, so the bridge links the trap's
records in the store. A bridge in Record.extra is never rendered into
fragment text; a slot bridge replaces the slot value in the record's text.
The agent therefore sees a trap's bridge only in a fragment whose text
names a slot bridge (analysis/corpus_stats.py counts these fragments for
the released traps).

Design:

1. Two densities (BridgeDensitySpec):
   - `extra` (default 5%): fraction of records that carry a bridge entity
     in their `Record.extra` field.
   - `slot` (default 0%): fraction of records that have a bridge entity in
     their subject or object slot. Off by default because it changes
     record semantics; the released corpora use 1%.

2. Five bridge categories:
   - shared_ip
   - malware_hash
   - cve
   - vendor
   - threat_actor

3. Each bridge category has a configurable pool size. Bridges are
   sampled deterministically from these pools.

4. Validation: `validation.assert_record_entities_in_pool_or_bridges`
   requires every subject and object outside the tenant's pool, and so
   every slot bridge, to be in the BridgePools whitelist. Extra bridges
   are drawn from the same pools.

5. Whitelist generation: `build_bridge_pools(seed, sizes)` returns a
   BridgePools whose entities are deterministic given the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from mnembound.validation import BridgePools


# Default bridge pool sizes: 125 entities. At 5% extra density on a
# 10-tenant × 500-record corpus, 250 records carry an extra bridge, two per
# entity on average.
DEFAULT_BRIDGE_POOL_SIZES: dict[str, int] = {
    "shared_ip": 50,
    "malware_hash": 30,
    "cve": 20,
    "vendor": 15,
    "threat_actor": 10,
}


# Default density knobs (fractions of records carrying a bridge).
DEFAULT_BRIDGE_DENSITY_EXTRA: float = 0.05
DEFAULT_BRIDGE_DENSITY_SLOT: float = 0.0

# When a record gets a slot-bridge, which slot does it go in? object_
# by default because subjects already carry tenant-specific user/process
# identifiers. shared_ip is the most natural object-slot bridge.
_BRIDGE_TO_SLOT: dict[str, str] = {
    "shared_ip": "object_",
    "malware_hash": "object_",
    "cve": "object_",
    "vendor": "object_",
    "threat_actor": "subject",
}


# Categories of bridges that may appear in a subject or object slot: all
# five.
SLOT_ELIGIBLE_BRIDGE_TYPES: frozenset[str] = frozenset({
    "shared_ip", "malware_hash", "cve", "vendor", "threat_actor",
})


# ----- Bridge entity factories -----


def _shared_ip(i: int) -> str:
    """An address of the RFC 5737 documentation range 198.51.100.0/24."""
    return f"ip:198.51.100.{i % 256}"


def _malware_hash(i: int, rng: random.Random) -> str:
    """A 64-char hex string that looks like SHA-256."""
    return "hash:" + "".join(rng.choice("0123456789abcdef") for _ in range(64))


def _cve(year: int, i: int) -> str:
    return f"cve:CVE-{year}-{10000 + i:05d}"


def _vendor(i: int) -> str:
    names = [
        "Sentinel", "Talos", "Lighthouse", "Aegis", "Nimbus", "Falcon",
        "Beacon", "Cipher", "Cobalt", "Drake", "Echelon", "Forge",
        "Glacier", "Harbor", "Ironwood",
    ]
    return f"vendor:{names[i % len(names)]}-{i // len(names):02d}"


def _threat_actor(i: int) -> str:
    actors = [
        "Lynx", "Hydra", "Specter", "Ghost", "Vulture", "Crow", "Serpent",
        "Wolf", "Cobra", "Mantis",
    ]
    return f"actor:APT-{actors[i % len(actors)]}-{i // len(actors):02d}"


# ----- Bridge pool construction -----


@dataclass(frozen=True)
class BridgeDensitySpec:
    """
    Bridge density specification.

    Two independent knobs:
    - extra:  fraction of records that carry a bridge in Record.extra
              (default 0.05).
    - slot:   fraction of records that carry a bridge in subject or
              object_ (default 0.0).

    A record may have both an extra-bridge AND a slot-bridge; the two
    knobs are independent. With both at 5%, ~0.25% of records have both
    (uncorrelated).
    """

    extra: float = DEFAULT_BRIDGE_DENSITY_EXTRA
    slot: float = DEFAULT_BRIDGE_DENSITY_SLOT

    def __post_init__(self) -> None:
        if not (0.0 <= self.extra <= 1.0):
            raise ValueError(f"extra density must be 0..1, got {self.extra}")
        if not (0.0 <= self.slot <= 1.0):
            raise ValueError(f"slot density must be 0..1, got {self.slot}")
        # Densities above 0.15 are rejected: such corpora are dominated by
        # bridge entities.
        if self.extra > 0.15:
            raise ValueError(
                f"extra density {self.extra} exceeds the realism cap 0.15; "
                "raise the cap in bridges.py to allow it"
            )
        if self.slot > 0.15:
            raise ValueError(
                f"slot density {self.slot} exceeds the realism cap 0.15"
            )

    @staticmethod
    def disabled() -> "BridgeDensitySpec":
        return BridgeDensitySpec(extra=0.0, slot=0.0)


def build_bridge_pools(
    seed: int, sizes: dict[str, int] | None = None
) -> BridgePools:
    """
    Build deterministic bridge pools.

    Pools are independent of the corpus's per-tenant pools — bridges live
    in a separate identifier namespace (ip:, hash:, cve:, vendor:, actor:)
    and never collide with the per-tenant entity prefixes (host:, user:,
    file:, proc:, malware:).

    Returns a BridgePools that the validation module will treat as the
    cross-tenant whitelist.
    """
    sizes = sizes or DEFAULT_BRIDGE_POOL_SIZES
    rng = random.Random(seed)

    n_ip = sizes.get("shared_ip", DEFAULT_BRIDGE_POOL_SIZES["shared_ip"])
    n_hash = sizes.get("malware_hash", DEFAULT_BRIDGE_POOL_SIZES["malware_hash"])
    n_cve = sizes.get("cve", DEFAULT_BRIDGE_POOL_SIZES["cve"])
    n_vendor = sizes.get("vendor", DEFAULT_BRIDGE_POOL_SIZES["vendor"])
    n_actor = sizes.get("threat_actor", DEFAULT_BRIDGE_POOL_SIZES["threat_actor"])

    ips = frozenset(_shared_ip(i) for i in range(n_ip))
    hashes = frozenset(_malware_hash(i, rng) for i in range(n_hash))
    # CVE years span 2022-2025 for a mix of realism.
    cves = frozenset(_cve(2022 + (i % 4), i) for i in range(n_cve))
    vendors = frozenset(_vendor(i) for i in range(n_vendor))
    actors = frozenset(_threat_actor(i) for i in range(n_actor))

    return BridgePools(
        shared_ips=ips,
        shared_hashes=hashes,
        shared_cves=cves,
        shared_vendors=vendors,
        shared_threat_actors=actors,
    )


# ----- Bridge sampling for record annotation -----


@dataclass(frozen=True)
class BridgeAssignment:
    """
    Per-record bridge assignment, produced by `assign_bridges_to_records`.

    Fields:
        extra_bridges_by_record_id: { record_id -> bridge_entity }.
            Records present in this dict carry that bridge entity in
            Record.extra under the key 'bridge'. Records absent from the
            dict carry no extra bridge.
        slot_bridges_by_record_id: { record_id -> (slot_name, bridge_entity) }.
            Records present in this dict have their `slot_name` field
            (one of 'subject' / 'object_') replaced with the bridge entity
            during generation. Records absent from the dict have their
            slot filled from the per-tenant pool as usual.
        bridges: the BridgePools used. Carried here so the generator can
            pass it into validation without recomputing.
    """

    extra_bridges_by_record_id: dict[str, str]
    slot_bridges_by_record_id: dict[str, tuple[str, str]]
    bridges: BridgePools


def assign_bridges_to_records(
    record_ids: list[str],
    bridges: BridgePools,
    density: BridgeDensitySpec,
    seed: int,
) -> BridgeAssignment:
    """
    Deterministically assign bridge entities to a subset of records.

    Args:
        record_ids: All record IDs in the corpus, in generator order.
        bridges: The BridgePools to draw from.
        density: How many records receive each type of bridge annotation.
        seed: For deterministic sampling.

    Returns:
        BridgeAssignment with two dicts keyed by record_id.

    Sampling:
    - We compute how many records get an extra-bridge:
        n_extra = round(density.extra * len(record_ids))
      and select those records uniformly at random. Each selected record
      gets a bridge entity sampled from a category, with category weights
      proportional to pool sizes (so larger pools get more usage).
    - Same procedure for slot-bridges, with the additional constraint
      that the bridge category must be in SLOT_ELIGIBLE_BRIDGE_TYPES.
    - The extra-bridge and slot-bridge samples are INDEPENDENT — a record
      can have both, neither, or one.
    """
    if not record_ids:
        return BridgeAssignment({}, {}, bridges)

    rng = random.Random(seed)
    record_ids_sorted = sorted(record_ids)  # deterministic input order

    # ---- Extra-bridges ----
    n_extra = round(density.extra * len(record_ids_sorted))
    extra_assignments: dict[str, str] = {}
    if n_extra > 0:
        selected_extra = rng.sample(record_ids_sorted, n_extra)
        for rid in selected_extra:
            category = _sample_category(rng, bridges)
            entity = _sample_bridge_entity(rng, bridges, category)
            if entity is not None:
                extra_assignments[rid] = entity

    # ---- Slot-bridges ----
    n_slot = round(density.slot * len(record_ids_sorted))
    slot_assignments: dict[str, tuple[str, str]] = {}
    if n_slot > 0:
        selected_slot = rng.sample(record_ids_sorted, n_slot)
        for rid in selected_slot:
            # Restrict to slot-eligible categories.
            category = _sample_category(rng, bridges, restrict_to_slot=True)
            entity = _sample_bridge_entity(rng, bridges, category)
            if entity is not None:
                slot_name = _BRIDGE_TO_SLOT[category]
                slot_assignments[rid] = (slot_name, entity)

    return BridgeAssignment(
        extra_bridges_by_record_id=extra_assignments,
        slot_bridges_by_record_id=slot_assignments,
        bridges=bridges,
    )


def _sample_category(
    rng: random.Random, bridges: BridgePools, restrict_to_slot: bool = False
) -> str:
    """Sample a bridge category weighted by pool size."""
    candidates: list[tuple[str, int]] = []
    if bridges.shared_ips:
        candidates.append(("shared_ip", len(bridges.shared_ips)))
    if bridges.shared_hashes:
        candidates.append(("malware_hash", len(bridges.shared_hashes)))
    if bridges.shared_cves:
        candidates.append(("cve", len(bridges.shared_cves)))
    if bridges.shared_vendors:
        candidates.append(("vendor", len(bridges.shared_vendors)))
    if bridges.shared_threat_actors:
        candidates.append(("threat_actor", len(bridges.shared_threat_actors)))

    if restrict_to_slot:
        candidates = [
            (cat, w) for cat, w in candidates
            if cat in SLOT_ELIGIBLE_BRIDGE_TYPES
        ]

    if not candidates:
        return "shared_ip"  # fallback: the empty pool gives no entity, and the record is skipped
    names = [c for c, _ in candidates]
    weights = [w for _, w in candidates]
    return rng.choices(names, weights=weights, k=1)[0]


def _sample_bridge_entity(
    rng: random.Random, bridges: BridgePools, category: str
) -> str | None:
    """Sample one entity from the specified bridge category."""
    pool: frozenset[str]
    if category == "shared_ip":
        pool = bridges.shared_ips
    elif category == "malware_hash":
        pool = bridges.shared_hashes
    elif category == "cve":
        pool = bridges.shared_cves
    elif category == "vendor":
        pool = bridges.shared_vendors
    elif category == "threat_actor":
        pool = bridges.shared_threat_actors
    else:
        return None
    if not pool:
        return None
    return rng.choice(sorted(pool))
