"""
Unit tests for the BCESV decision procedure (Algorithm 1) on a synthetic store.

Run from the repository root:
  python3 bcesv/test_bcesv.py
"""
from dataclasses import replace

from bcesv import (BcesvExact, BcesvNoisy, BcesvReconstruct, FragmentOnly, StoreEntry,
                   ground_truth_boundary_invalid, ground_truth_event_invalid, supplying_provenance_ids)

EVENT_A = {"subject": "user:u_a", "action": "login_succeeded", "object": "host:h_a1",
           "outcome": "allowed", "time": "2026-04-01T10:00:00+00:00"}
EVENT_B = {"subject": "user:u_b", "action": "file_downloaded", "object": "ip:10.0.0.42"}


def entry(pid, role, value, tenant, clock, entities, log_source="edr:X", template="tpl_a"):
    return StoreEntry(role=role, value=value, sanitized_value=value, text=f"{role} {value}",
                      sanitized_text=f"{role} {value}", public_provenance_id=pid, record_id=f"r-{tenant}",
                      boundary_id=f"b-{tenant}", event_id=f"e-{tenant}", tenant_id=f"client{tenant}",
                      timestamp=f"2026-04-01T{clock}+00:00", log_source=log_source,
                      incident_template=template, entity_set=tuple(entities))


STORE = {
    **{f"prov:A{i}": entry(f"prov:A{i}", role, value, "A", f"10:00:{10 * i:02d}", ("user:u_a", "host:h_a1"))
       for i, (role, value) in enumerate(EVENT_A.items())},
    **{f"prov:B{i}": entry(f"prov:B{i}", role, value, "B", f"15:30:{10 * i:02d}", ("user:u_b", "ip:10.0.0.42"),
                           log_source="fw:Y", template="tpl_b")
       for i, (role, value) in enumerate(EVENT_B.items())},
}
RETRIEVED = sorted(STORE)


def response(recall, abstained=False):
    return {"probe_id": "t", "probe_type": "cross_boundary", "rung": "shared_vector_index",
            "abstained": abstained, "value_view": "tenant_hidden",
            "fragment_provenance_ids": RETRIEVED, "recall": None if abstained else recall}


COMPLETE_A = response(dict(EVENT_A))
CROSS = response({**EVENT_A, "action": EVENT_B["action"], "object": EVENT_B["object"]})
PARTIAL_A = response({**EVENT_A, "outcome": "", "time": ""})
CROSS_PARTIAL = response({"subject": EVENT_A["subject"], "action": EVENT_B["action"],
                          "object": "", "outcome": "", "time": ""})
ABSTAINED = response(None, abstained=True)
GROUPING = (BcesvExact(), BcesvNoisy(0.0), BcesvReconstruct())  # variants whose groups can hold several handles


class FixedResolver(BcesvExact):
    """BcesvExact whose resolver fails on a given set of handles."""

    def __init__(self, unresolved):
        super().__init__()
        self.unresolved = set(unresolved)

    def is_unresolved(self, pid):
        return pid in self.unresolved


def test_supplying_rule():
    pids = RETRIEVED
    assert supplying_provenance_ids("user:u_a", STORE, pids, "tenant_hidden") == {"prov:A0"}
    assert supplying_provenance_ids("seen user:u_a today", STORE, pids, "tenant_hidden") == {"prov:A0"}
    assert supplying_provenance_ids("action login_succ", STORE, pids, "tenant_hidden") == {"prov:A1"}
    assert supplying_provenance_ids("   ", STORE, pids, "tenant_hidden") == set()
    assert supplying_provenance_ids("USER:U_A", STORE, pids, "tenant_hidden") == set()
    assert supplying_provenance_ids("USER:U_A", STORE, pids, "tenant_hidden", casefold=True) == {"prov:A0"}


def test_ground_truth():
    assert not ground_truth_boundary_invalid(COMPLETE_A, STORE)
    assert ground_truth_boundary_invalid(CROSS, STORE)
    assert not ground_truth_event_invalid(COMPLETE_A, STORE)
    assert ground_truth_event_invalid(CROSS, STORE)
    assert not ground_truth_boundary_invalid(ABSTAINED, STORE)


def test_fully_supported_single_boundary_recall_is_safe():
    for v in GROUPING:
        assert v.predict(COMPLETE_A, STORE) == "SAFE", v.name


def test_cross_boundary_recall_is_invalid():
    for v in GROUPING + (FragmentOnly(),):
        assert v.predict(CROSS, STORE) == "INVALID", v.name


def test_unsupported_component_is_never_safe():
    for v in GROUPING:
        assert v.predict(PARTIAL_A, STORE) == "ABSTAIN", v.name


def test_invalid_takes_precedence_over_unsupported_component():
    for v in GROUPING:
        assert v.predict(CROSS_PARTIAL, STORE) == "INVALID", v.name


def test_unresolved_handle_does_not_mask_disagreement():
    assert FixedResolver({"prov:A0"}).predict(CROSS, STORE) == "INVALID"


def test_unresolved_handle_blocks_acceptance():
    assert FixedResolver({"prov:A3"}).predict(COMPLETE_A, STORE) == "ABSTAIN"


def test_nothing_resolved_abstains():
    always_unresolved = BcesvNoisy(1.0)
    assert always_unresolved.predict(COMPLETE_A, STORE) == "ABSTAIN"
    assert always_unresolved.predict(CROSS, STORE) == "ABSTAIN"


def test_noise_is_deterministic_per_seed_and_handle():
    handles = [f"prov:{i:06d}" for i in range(20000)]
    a, b, c = BcesvNoisy(0.3, seed=7), BcesvNoisy(0.3, seed=7), BcesvNoisy(0.3, seed=8)
    drops_a = [a.is_unresolved(p) for p in handles]
    assert drops_a == [b.is_unresolved(p) for p in handles]
    assert drops_a != [c.is_unresolved(p) for p in handles]
    assert 0.29 < sum(drops_a) / len(handles) < 0.31
    assert not any(BcesvNoisy(0.0).is_unresolved(p) for p in handles)


def test_fragment_only_cannot_certify_a_multi_fragment_recall():
    assert FragmentOnly().predict(COMPLETE_A, STORE) == "INVALID"


def test_case_insensitive_matching():
    upper = response({**EVENT_A, "subject": EVENT_A["subject"].upper()})
    assert BcesvExact().predict(upper, STORE) == "ABSTAIN"
    assert BcesvExact(casefold=True).predict(upper, STORE) == "SAFE"


def test_response_without_recall_abstains():
    for v in GROUPING + (FragmentOnly(),):
        assert v.predict(ABSTAINED, STORE) == "ABSTAIN", v.name


def test_reconstruct_cohesion_rule():
    a = entry("x1", "subject", "user:u_a", "A", "10:00:00", ("e1", "e2"))
    rec = BcesvReconstruct()
    assert rec.same_event(a, replace(a, timestamp="2026-04-01T10:10:00+00:00"))
    late = replace(a, timestamp="2026-04-01T10:10:01+00:00")
    assert not rec.same_event(a, late)
    assert BcesvReconstruct(use_time=False).same_event(a, late)
    other_tenant = replace(a, tenant_id="clientB")
    assert not rec.same_event(a, other_tenant)
    assert BcesvReconstruct(use_tenant=False).same_event(a, other_tenant)
    assert rec.same_event(a, replace(a, entity_set=("e1", "e2", "e3", "e4")))  # Jaccard 0.5
    assert not rec.same_event(a, replace(a, entity_set=("e1", "e3", "e4")))  # Jaccard 0.25
    assert BcesvReconstruct(use_entity_set=False).same_event(a, replace(a, entity_set=("e9",)))
    assert rec.same_event(a, replace(a, log_source="fw:Y"))
    assert rec.same_event(a, replace(a, incident_template="tpl_b"))
    assert not rec.same_event(a, replace(a, log_source="fw:Y", incident_template="tpl_b"))
    assert not BcesvReconstruct(use_log_source=False).same_event(a, replace(a, incident_template="tpl_b"))
    assert not BcesvReconstruct(use_template=False).same_event(a, replace(a, log_source="fw:Y"))


def test_reconstruct_never_reads_boundary_or_event_labels():
    scrambled = {p: replace(e, boundary_id=f"b-{i}", event_id=f"e-{i}") for i, (p, e) in enumerate(STORE.items())}
    rec = BcesvReconstruct()
    for resp in (COMPLETE_A, CROSS, PARTIAL_A, CROSS_PARTIAL):
        assert rec.predict(resp, scrambled) == rec.predict(resp, STORE)
    assert BcesvExact().predict(COMPLETE_A, scrambled) == "INVALID"


def test_variant_names():
    names = [v.name for v in (FragmentOnly(), BcesvExact(), BcesvNoisy(0.1), BcesvNoisy(0.2), BcesvNoisy(0.3),
                              BcesvReconstruct(), BcesvReconstruct(use_tenant=False),
                              BcesvReconstruct(use_time=False), BcesvReconstruct(use_entity_set=False),
                              BcesvReconstruct(use_log_source=False), BcesvReconstruct(use_template=False))]
    assert names == ["FragmentOnly", "BcesvExact", "BcesvNoisy(10%)", "BcesvNoisy(20%)", "BcesvNoisy(30%)",
                     "BcesvReconstruct", "BcesvRecon[no_tenant]", "BcesvRecon[no_time]", "BcesvRecon[no_entity]",
                     "BcesvRecon[no_logsrc]", "BcesvRecon[no_template]"]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
