"""
mnembound.generator — Top-level corpus generator.

BoundaryGranularity is a top-level configuration parameter. The default,
TENANT, gives one confidentiality boundary per tenant; EVENT gives one
boundary per record. An entity sanitizer is built per corpus, and each
Fragment carries both tenant-visible and sanitized forms of `value` and
`text`; Fragment.view_for_v_frag(tenant_visible=...) selects the view.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from types import MappingProxyType

from mnembound.bridges import (
    BridgeDensitySpec,
    assign_bridges_to_records,
    build_bridge_pools,
)
from mnembound.identifiers import (
    BoundaryGranularity,
    boundary_id_for,
    build_entity_sanitizer,
    build_tenant_entity_pools,
    make_event_id,
    make_record_id,
)
from mnembound.probes import generate_probe_mix
from mnembound.schema import (
    ALL_ROLES,
    Corpus,
    Fragment,
    ProvenanceHandle,
    Record,
    make_resolver,
)
from mnembound.templates import (
    pick_template,
    render_template,
    sample_timestamp,
)
from mnembound.validation import (
    assert_disjoint_entity_pools,
    validate_corpus,
)


DEFAULT_BASE_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def derive_salt(seed: int) -> str:
    """
    Derive a corpus-level provenance salt deterministically from the seed.

    Stored on the Corpus (and in corpus.json); the same seed gives the same
    salt and so the same public handles.
    """
    rng = random.Random(seed)
    return f"mnembound:{rng.randint(0, 2**128 - 1):032x}"


def generate_corpus(
    seed: int,
    n_tenants: int = 10,
    records_per_tenant: int = 100,
    timestamp_window_days: int = 90,
    base_timestamp: datetime = DEFAULT_BASE_TIMESTAMP,
    boundary_granularity: BoundaryGranularity = BoundaryGranularity.TENANT,
    n_engagements_per_tenant: int = 5,
    n_supported_probes: int = 0,
    n_absent_probes: int = 0,
    n_partial_probes: int = 0,
    n_cross_boundary_probes: int = 0,
    bridge_density: BridgeDensitySpec | None = None,
    bridge_pool_sizes: dict[str, int] | None = None,
    partial_distractor_fragments: int = 0,
    prefer_bridge_linked_traps: bool = False,
) -> Corpus:
    """
    Generate a complete MnemBound corpus.

    Args:
        seed: Master seed. All randomness derives from this.
        n_tenants: Number of tenants (default 10).
        records_per_tenant: Records per tenant (default 100; the released
            corpora use 500).
        timestamp_window_days: Time window for record timestamps.
        base_timestamp: Start of the timestamp window.
        boundary_granularity: How boundaries are scoped within a tenant.
            Default TENANT: one boundary per tenant (the paper's MSSP
            framing). See identifiers.BoundaryGranularity for alternatives.
        n_engagements_per_tenant: Only used when granularity is ENGAGEMENT.
        n_supported_probes: How many SUPPORTED probes to attach.
        n_absent_probes: How many ABSENT probes to attach.
        n_partial_probes: How many PARTIAL probes to attach.
        n_cross_boundary_probes: How many CROSS_BOUNDARY trap probes to
            attach.
        bridge_density: Per-record bridge density specification. Default
            None means BridgeDensitySpec.disabled() (no bridges). Pass
            BridgeDensitySpec() for 5% extra and 0% slot bridges; the
            released corpora use BridgeDensitySpec(extra=0.05, slot=0.01).
            See mnembound.bridges for the design.
        bridge_pool_sizes: Override the default bridge pool sizes. Useful
            for ablations.
        partial_distractor_fragments: Number of unrelated fragments to
            inject into PARTIAL probes' retrieval sets (default 0).
            The released corpora use 5.
        prefer_bridge_linked_traps: When True, CROSS_BOUNDARY traps prefer
            source records that share a bridge entity. Defaults to False;
            the released corpora set it to True.

    Returns:
        A validated Corpus carrying records, fragments, provenance salt,
        entity sanitizer, a ProvenanceResolver, and (if requested) probes.
    """
    master_rng = random.Random(seed)

    # 1. Corpus-level salt for opaque provenance IDs.
    salt = derive_salt(master_rng.randint(0, 2**31 - 1))

    # 2. Per-tenant entity pools.
    pools = build_tenant_entity_pools(
        n_tenants=n_tenants, seed=master_rng.randint(0, 2**31 - 1)
    )
    assert_disjoint_entity_pools(pools)

    # 2b. Bridge pools and per-record bridge assignment.
    if bridge_density is None:
        bridge_density = BridgeDensitySpec.disabled()
    bridges = build_bridge_pools(
        seed=master_rng.randint(0, 2**31 - 1),
        sizes=bridge_pool_sizes,
    )
    # Pre-compute all record IDs so we can assign bridges before record
    # materialization. The IDs are deterministic functions of (tenant_index,
    # record_local), independent of template choices.
    all_record_ids: list[str] = []
    for ti in range(n_tenants):
        for rl in range(records_per_tenant):
            all_record_ids.append(make_record_id(ti, rl))
    bridge_assignment = assign_bridges_to_records(
        record_ids=all_record_ids,
        bridges=bridges,
        density=bridge_density,
        seed=master_rng.randint(0, 2**31 - 1),
    )

    # 3. Build the entity sanitizer (tenant-visible -> tenant-hidden token).
    # Note: bridge entities live in their own namespace (ip:, hash:, cve:,
    # vendor:, actor:) and are intentionally NOT sanitized — they are
    # tenant-neutral by construction and should remain visible in both
    # view modes.
    sanitizer = build_entity_sanitizer(pools, salt)

    # 4. Materialize records, per tenant, in numeric tenant_index order.
    records: list[Record] = []
    template_rng = random.Random(master_rng.randint(0, 2**31 - 1))
    timestamp_rng = random.Random(master_rng.randint(0, 2**31 - 1))
    entity_rng = random.Random(master_rng.randint(0, 2**31 - 1))

    pools_by_index = sorted(pools.values(), key=lambda p: p.tenant_index)

    for pool in pools_by_index:
        for record_local in range(records_per_tenant):
            tenant_index = pool.tenant_index

            boundary_id = boundary_id_for(
                tenant_index=tenant_index,
                record_local=record_local,
                granularity=boundary_granularity,
                n_engagements_per_tenant=n_engagements_per_tenant,
            )
            event_id = make_event_id(tenant_index, record_local)
            record_id = make_record_id(tenant_index, record_local)

            template_name = pick_template(template_rng)
            timestamp = sample_timestamp(
                timestamp_rng, base_timestamp, window_days=timestamp_window_days
            )
            output = render_template(template_name, entity_rng, pool, timestamp)

            # Default record fields from the template.
            subject = output.subject
            object_ = output.object_
            component_text = dict(output.component_text)
            extra: dict[str, object] = dict(output.extra)

            # Apply slot-bridge if this record was selected.
            slot_assignment = bridge_assignment.slot_bridges_by_record_id.get(
                record_id
            )
            if slot_assignment is not None:
                slot_name, bridge_entity = slot_assignment
                if slot_name == "subject":
                    original_value = subject
                    subject = bridge_entity
                elif slot_name == "object_":
                    original_value = object_
                    object_ = bridge_entity
                else:
                    original_value = None
                # Update component_text: substitute the original slot value
                # with the bridge entity in every component's text.
                if original_value is not None and original_value != bridge_entity:
                    for role, txt in component_text.items():
                        if original_value in txt:
                            component_text[role] = txt.replace(
                                original_value, bridge_entity
                            )

            # Apply extra-bridge if this record was selected.
            extra_bridge_entity = bridge_assignment.extra_bridges_by_record_id.get(
                record_id
            )
            if extra_bridge_entity is not None:
                extra["bridge"] = extra_bridge_entity

            record = Record(
                record_id=record_id,
                client_id=pool.tenant_id,
                boundary_id=boundary_id,
                event_id=event_id,
                subject=subject,
                action=output.action,
                object_=object_,
                outcome=output.outcome,
                time=output.time,
                log_source=output.log_source,
                incident_template=output.template.value,
                component_text=MappingProxyType(component_text),
                extra=MappingProxyType(extra),
            )
            records.append(record)

    records_tuple = tuple(records)

    # 5. Extract per-component fragments. Each record contributes five
    # fragments. Public provenance IDs are derived via salted SHA-256, and
    # each fragment carries both tenant-visible and sanitized forms.
    fragments: list[Fragment] = []
    public_to_internal: dict[str, tuple[str, int]] = {}

    for r in records_tuple:
        entity_set = frozenset({r.subject, r.object_})

        # Per-record sanitization: only the record's own subject and object_
        # can appear in its fragment text. We pre-compute the (visible, hidden)
        # replacement pairs for THIS record once, then reuse across its five
        # fragments, instead of scanning the whole sanitizer for every
        # fragment.
        per_record_pairs: list[tuple[str, str]] = []
        for visible in (r.subject, r.object_):
            hidden = sanitizer.get(visible)
            if hidden is not None and hidden != visible:
                per_record_pairs.append((visible, hidden))
        # Longest first, so that no entity is replaced inside a longer one.
        per_record_pairs.sort(key=lambda p: len(p[0]), reverse=True)

        for fragment_index, role in enumerate(ALL_ROLES):
            value = r.component(role)
            text = r.text_for(role)

            sanitized_value = sanitizer.get(value, value)
            sanitized_text = text
            for visible, hidden in per_record_pairs:
                if visible in sanitized_text:
                    sanitized_text = sanitized_text.replace(visible, hidden)

            internal = ProvenanceHandle(
                record_id=r.record_id, fragment_index=fragment_index
            )
            public_id = internal.public_id(salt)

            # Collision check.
            if public_id in public_to_internal:
                existing_rid, existing_fidx = public_to_internal[public_id]
                raise RuntimeError(
                    f"opaque provenance collision: public_id {public_id} maps "
                    f"to both ({existing_rid}, {existing_fidx}) and "
                    f"({r.record_id}, {fragment_index}). Increase "
                    "PUBLIC_HANDLE_HEX_LEN in schema.py."
                )
            public_to_internal[public_id] = (r.record_id, fragment_index)

            fragments.append(
                Fragment(
                    role=role,
                    value=value,
                    sanitized_value=sanitized_value,
                    text=text,
                    sanitized_text=sanitized_text,
                    public_provenance_id=public_id,
                    provenance=internal,
                    record_id=r.record_id,
                    boundary_id=r.boundary_id,
                    event_id=r.event_id,
                    tenant_id=r.client_id,
                    timestamp=r.time,
                    log_source=r.log_source,
                    incident_template=r.incident_template,
                    entity_set=entity_set,
                )
            )

    fragments_tuple = tuple(fragments)

    # 6. Store-level resolver.
    resolver = make_resolver(salt=salt, public_to_internal=public_to_internal)

    # 7. Entity pool unions and sanitizer for corpus storage (read-only).
    tenant_entity_unions = MappingProxyType(
        {tid: pool.all_entities() for tid, pool in pools.items()}
    )
    sanitizer_view = MappingProxyType(dict(sanitizer))

    # 8. Assemble base corpus (without probes) and validate. Probes are
    # generated against a fully-validated corpus to keep ground truth clean.
    base_corpus = Corpus(
        seed=seed,
        salt=salt,
        records=records_tuple,
        fragments=fragments_tuple,
        probes=(),
        tenant_entity_pools=tenant_entity_unions,
        entity_sanitizer=sanitizer_view,
        resolver=resolver,
        bridge_pools=bridges,
    )
    validate_corpus(base_corpus, bridges=bridges)

    # 9. Probe generation (if requested).
    probes: tuple = ()
    if (
        n_supported_probes > 0
        or n_absent_probes > 0
        or n_partial_probes > 0
        or n_cross_boundary_probes > 0
    ):
        mix = generate_probe_mix(
            base_corpus,
            seed=master_rng.randint(0, 2**31 - 1),
            n_supported=n_supported_probes,
            n_absent=n_absent_probes,
            n_partial=n_partial_probes,
            n_cross_boundary=n_cross_boundary_probes,
            partial_distractor_fragments=partial_distractor_fragments,
            prefer_bridge_linked_traps=prefer_bridge_linked_traps,
        )
        probes = mix.all_probes()

    # If no probes requested, return the validated base corpus directly.
    if not probes:
        return base_corpus

    # Otherwise rebuild the corpus with probes attached. Corpus is frozen,
    # so we construct a fresh instance with everything else preserved.
    from dataclasses import replace
    return replace(base_corpus, probes=probes)
