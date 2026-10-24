"""
paper_numbers.py: the counts and rates that the paper derives from the released
data, grouped by the part of the paper that reports them, including the values
plotted in Figures 3 and 4, the cells of Table 4 and the ranges of Table 6.

Every number is computed from data/<condition>/{responses,probes,
fragments.store,corpus,rung_stats} and analysis/audit_sample.csv with the
HARDENED scorer of rescore_final_ci.py and the EXACT and LOOSE scorers of
scorer_sensitivity.py. The last section checks that the scorer and the BCESV
ground truth (bcesv/bcesv.py) agree on every response.

Usage, from the repository root:
  python3 analysis/paper_numbers.py > analysis/paper_numbers.txt
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "bcesv"))
import bcesv
from rescore_final_ci import (CONDITIONS, ROLES, SAFE_RUNGS, boundary_invalid, load_store,
                              supplying_fragments, wilson_ci)
from scorer_sensitivity import boundary_invalid_exact, boundary_invalid_loose

UNSAFE_RUNGS = ("shared_vector_index", "summary_merge")
TYPES = ("supported", "cross_boundary", "absent", "partial")
SHORT = {"isolated": "Iso", "boundary_preserving_topk": "BP", "shared_vector_index": "SV", "summary_merge": "SM"}


def pct(k, n):
    return f"{k}/{n} = {100 * k / n:.1f}%"


def produced(r):
    """The response contains a recall (no abstention, no parse failure)."""
    return not r.get("abstained") and r.get("recall") is not None


class Condition:
    def __init__(self, data_dir, name, label, stores):
        d = data_dir / name
        self.name, self.label = name, label.strip()
        self.model, self.seed, self.setting = label.split()
        self.corpus = json.loads((d / "corpus.json").read_text())
        if self.corpus["seed"] not in stores:
            stores[self.corpus["seed"]] = load_store(d)
        self.store = stores[self.corpus["seed"]]
        with open(d / "responses.jsonl") as f:
            self.responses = [json.loads(line) for line in f]
        with open(d / "probes.jsonl") as f:
            self.probes = {p["probe_id"]: p for p in map(json.loads, f)}
        with open(d / "rung_stats.jsonl") as f:
            self.rung_stats = {s["rung"]: s for s in map(json.loads, f)}

    def select(self, probe_type=None, rungs=None):
        return [r for r in self.responses
                if (probe_type is None or r["probe_type"] == probe_type) and (rungs is None or r["rung"] in rungs)]

    def rate(self, rung, probe_type="cross_boundary"):
        rs = self.select(probe_type, (rung,))
        return sum(boundary_invalid(r, self.store) for r in rs), len(rs)

    def absent_abstention(self, rungs):
        rs = self.select("absent", rungs)
        return sum(1 for r in rs if r.get("abstained") and not r.get("parse_error")), len(rs)


def section_data(conds):
    print("[Data] §5 and §6, Appendix I")
    print(f"  conditions {len(conds)}; responses {sum(len(c.responses) for c in conds)}; "
          f"per condition {', '.join(str(n) for n in sorted({len(c.responses) for c in conds}))} (300 probes x 4 rungs)")
    mixes = sorted({tuple(Counter(p["probe_type"] for p in c.probes.values())[t] for t in TYPES) for c in conds})
    print("  probes per condition by type: "
          + " | ".join(", ".join(f"{t} {n}" for t, n in zip(TYPES, mix)) for mix in mixes))
    for seed in sorted({c.corpus["seed"] for c in conds}):
        c = next(c for c in conds if c.corpus["seed"] == seed)
        records_per_tenant = Counter(f["tenant_id"] for f in c.store.values() if f["role"] == "subject")
        print(f"  corpus seed {seed}: conditions {sum(x.corpus['seed'] == seed for x in conds)}; "
              f"tenants {len(records_per_tenant)}; records {len({f['event_id'] for f in c.store.values()})} "
              f"({', '.join(str(n) for n in sorted(set(records_per_tenant.values())))} per tenant); "
              f"fragments {len(c.store)}")
    views = Counter(r.get("value_view") for c in conds for r in c.responses)
    print("  responses by value view: " + ", ".join(f"{v} {n}" for v, n in sorted(views.items())))


def section_rungs(conds):
    print("\n[§5.3] Safe-rung equivalence and retention")
    same = total = 0
    for c in conds:
        by_probe = defaultdict(dict)
        for r in c.responses:
            by_probe[r["probe_id"]][r["rung"]] = r["fragment_provenance_ids"]
        for ctx in by_probe.values():
            total += 1
            same += ctx["isolated"] == ctx["boundary_preserving_topk"]
    print(f"  isolated and boundary_preserving_topk receive identical contexts: {same}/{total} probe pairs")
    for seed in sorted({c.corpus["seed"] for c in conds}):
        c = next(c for c in conds if c.corpus["seed"] == seed)
        kept = [len(r["fragment_provenance_ids"]) / len(c.probes[r["probe_id"]]["fragment_provenance_ids"])
                for r in c.select("cross_boundary", ("isolated",))]
        released = c.rung_stats["isolated"]["retention_by_probe_type"]["cross_boundary"]
        print(f"  seed {seed}: safe rungs keep {100 * sum(kept) / len(kept):.1f}% of a trap's source fragments "
              f"on average (rung_stats.jsonl: {100 * released:.1f}%)")


def section_results(conds):
    print("\n[§7.1] Trap responses, supported probes, event structure")
    no_recall = Counter()
    for c in conds:
        for r in c.select("cross_boundary"):
            group = "safe" if r["rung"] in SAFE_RUNGS else "unsafe"
            no_recall[group, "n"] += 1
            if not produced(r):
                no_recall[group, "parse failure" if r.get("parse_error") else "abstention"] += 1
    for g in ("safe", "unsafe"):
        k = no_recall[g, "parse failure"] + no_recall[g, "abstention"]
        print(f"  {g} rungs, trap responses without a recall: {k}/{no_recall[g, 'n']} "
              f"(parse failures {no_recall[g, 'parse failure']}, abstentions {no_recall[g, 'abstention']})")

    cells = [sum(produced(r) for r in c.select("supported", (rung,))) for c in conds for rung in SHORT]
    print(f"  supported probes with a recall: {sum(cells)}/{100 * len(cells)} "
          f"({sum(x == 100 for x in cells)} of {len(cells)} cells at 100/100)")
    total = Counter()
    for rung in ("isolated", "shared_vector_index", "summary_merge"):
        k = Counter()
        for c in conds:
            for r in c.select("supported", (rung,)):
                own = c.probes[r["probe_id"]]["fragment_provenance_ids"]
                assert len({c.store[p]["event_id"] for p in own}) == 1
                k["n"] += 1
                for casefold in (False, True):
                    k[casefold] += produced(r) and all(supplying_fragments(r, c.store, own, casefold).values())
        total.update(k)
        print(f"  {rung:20s} supported recalls whose five components all come from their own event's fragments: "
              f"ignoring case {pct(k[True], k['n'])}; case-sensitive {pct(k[False], k['n'])}")
    print(f"  {'together':20s} ignoring case {pct(total[True], total['n'])}")

    print("  both models on the same trap probe and unsafe rung, seed 42:")
    for setting in ("N/C", "C/C", "N/P", "C/P"):
        q, m = (next(c for c in conds if c.model == model and c.seed == "s42" and c.setting == setting)
                for model in ("Qwen", "Mistral"))
        mr = {(r["probe_id"], r["rung"]): produced(r) for r in m.select("cross_boundary", UNSAFE_RUNGS)}
        both = one = neither = 0
        for r in q.select("cross_boundary", UNSAFE_RUNGS):
            a, b = produced(r), mr[r["probe_id"], r["rung"]]
            both += a and b
            one += a != b
            neither += not a and not b
        print(f"    {setting}: {both + one + neither} probe-rung pairs; recall from both {both}; "
              f"exactly one abstains {one}; both abstain {neither}")

    lengths = defaultdict(list)
    for c in conds:
        for r in c.responses:
            if not produced(r):
                continue
            if r["probe_type"] == "cross_boundary":
                lengths["trap-probe recalls, all rungs"].append(len(r["raw_text"]))
                if boundary_invalid(r, c.store):
                    lengths["boundary-invalid recalls"].append(len(r["raw_text"]))
            elif r["probe_type"] == "supported":
                lengths["supported-probe recalls, all rungs"].append(len(r["raw_text"]))
    print("  median raw-response length in characters: " + "; ".join(
        f"{name} {statistics.median(v):g} (n={len(v)})" for name, v in lengths.items()))

    multi = recalls = combined = 0
    for c in conds:
        for r in c.select("cross_boundary", SAFE_RUNGS):
            multi += len({c.store[p]["event_id"] for p in r["fragment_provenance_ids"]}) > 1
            if produced(r):
                recalls += 1
                combined += boundary_invalid(r, c.store, key="event_id")
    n_safe = sum(len(c.select("cross_boundary", SAFE_RUNGS)) for c in conds)
    print(f"  safe rungs: trap contexts holding two or more events of the querying tenant {multi}/{n_safe}; "
          f"recalls {recalls}, of which {combined}/{recalls} combine two or more events")
    inv_e = sum(boundary_invalid(r, c.store, key="event_id") for c in conds for r in c.select("cross_boundary", UNSAFE_RUNGS))
    n_unsafe = sum(len(c.select("cross_boundary", UNSAFE_RUNGS)) for c in conds)
    print(f"  unsafe rungs: trap responses combining two or more events {pct(inv_e, n_unsafe)}")


def section_no_match(conds):
    print("\n[Appendix A.1] Components that match no source-fragment value (case-insensitive substring, either direction)")
    misses, k, any_miss = Counter(), 0, 0
    time_source = Counter()
    action_in_text = from_fragments = 0
    for c in conds:
        for r in c.select("cross_boundary", UNSAFE_RUNGS):
            if not produced(r):
                continue
            k += 1
            frags = [c.store[p] for p in r["fragment_provenance_ids"]]
            values = [f["sanitized_value"].strip().lower() for f in frags]
            missed = False
            all_in_text = True
            for comp in ROLES:
                v = r["recall"].get(comp, "")
                v = v.strip().lower() if isinstance(v, str) else ""
                if not (v and any(x and (v in x or x in v) for x in values)):
                    misses[comp] += 1
                    missed = True
                    roles = sorted({f["role"] for f in frags if v and v in f["sanitized_text"].lower()})
                    all_in_text = all_in_text and bool(roles)
                    if comp == "time":
                        time_source["/".join(roles) or "none"] += 1
                    elif comp == "action":
                        action_in_text += bool(roles)
            any_miss += missed
            from_fragments += all_in_text
    print(f"  unsafe-rung trap recalls {k}; with at least one such component {pct(any_miss, k)}")
    print("  per component: " + ", ".join(f"{comp} {misses[comp]}" for comp in sorted(ROLES, key=lambda x: -misses[x])))
    print("  such time components, by the source fragment whose text contains them: "
          + ", ".join(f"{role} {n}" for role, n in time_source.most_common()))
    print(f"  such action components that a source fragment's text contains: {pct(action_in_text, misses['action'])}")
    print("  recalls whose every component matches a source-fragment value or lies in a source fragment's text: "
          + pct(from_fragments, k))


def section_figures(conds):
    print("\n[Figure 3] Boundary-invalid rate per model, pooled over its 6 conditions")
    for model in ("Qwen", "Mistral"):
        parts = []
        for rung in SHORT:
            k = n = 0
            for c in conds:
                if c.model == model:
                    a, b = c.rate(rung)
                    k, n = k + a, n + b
            parts.append(f"{SHORT[rung]} {pct(k, n)}")
        print(f"  {model:8s}" + "; ".join(parts))

    print("\n[Figure 4, §7.2] Cautious prompt and per-client-blocks aggregator")
    s42 = [c for c in conds if c.seed == "s42"]
    for prompt, name in (("N", "neutral"), ("C", "cautious")):
        cs = [c for c in s42 if c.setting.startswith(prompt)]
        abst = [k / n for c in cs for k, n in (c.absent_abstention((rung,)) for rung in SHORT)]
        inv = [k / n for c in cs for k, n in (c.rate(rung) for rung in UNSAFE_RUNGS)]
        print(f"  Figure 4, {name}: median absent-probe abstention {100 * statistics.median(abst):.1f}% "
              f"({len(abst)} cells); median boundary-invalid rate {100 * statistics.median(inv):.1f}% ({len(inv)} unsafe-rung cells)")
    for prompt, name, pool in (("N", "neutral", conds), ("N", "neutral", s42), ("C", "cautious", conds)):
        per = sorted((k / n, c.label) for c in pool if c.setting.startswith(prompt)
                     for k, n in [c.absent_abstention(tuple(SHORT))])
        where = " on seed 42" if pool is s42 else ""
        print(f"  absent-probe abstention per {name} condition{where}, four rungs pooled: {100 * per[0][0]:.1f}% "
              f"({per[0][1]}) to {100 * per[-1][0]:.1f}% ({per[-1][1]}) over {len(per)} conditions")
    lowest = min((k / n, c.label, rung) for c in conds if c.setting.startswith("C")
                 for rung in SHORT for k, n in [c.absent_abstention((rung,))])
    print(f"  lowest absent-probe abstention in a cautious cell: {100 * lowest[0]:.1f}% ({lowest[1]}, {lowest[2]})")

    changed = n = 0
    for model in ("Qwen", "Mistral"):
        neutral, cautious = (next(c for c in conds if c.model == model and c.seed == "s42" and c.setting == s)
                             for s in ("N/C", "C/C"))
        for rung in UNSAFE_RUNGS:
            flags = [{r["probe_id"]: boundary_invalid(r, x.store) for r in x.select("cross_boundary", (rung,))}
                     for x in (neutral, cautious)]
            changed += sum(flags[0][p] != flags[1][p] for p in flags[0])
            n += len(flags[0])
            print(f"  {model:7s} {SHORT[rung]} neutral -> cautious (collapsed): "
                  f"{sum(flags[0].values())}/100 -> {sum(flags[1].values())}/100")
    print(f"  trap outcomes (boundary-invalid or not) that change from neutral to cautious: {changed} of {n}")
    for model in ("Qwen", "Mistral"):
        collapsed, blocks = (next(c for c in conds if c.model == model and c.seed == "s42" and c.setting == s)
                             for s in ("N/C", "N/P"))
        (a, _), (b, _) = collapsed.rate("summary_merge"), blocks.rate("summary_merge")
        print(f"  {model:7s} SM collapsed -> per-client-blocks (neutral): {a}/100 -> {b}/100")
    for name, keep in (("the cautious prompt", lambda c: c.setting.startswith("C")),
                       ("the per-client-blocks aggregator", lambda c: c.setting.endswith("P"))):
        rates = [k / n for c in conds if keep(c) for k, n in (c.rate(rung) for rung in UNSAFE_RUNGS)]
        print(f"  Table 6, unsafe-rung cells with {name}: {100 * min(rates):.1f}% to {100 * max(rates):.1f}% "
              f"({len(rates)} cells)")


def section_robustness(conds):
    print("\n[§7.4] Seeds and boundary span")
    for model in ("Qwen", "Mistral"):
        for seed in ("s42", "s43", "s44"):
            c = next(c for c in conds if c.model == model and c.seed == seed and c.setting == "N/C")
            unsafe = [c.rate(r) for r in UNSAFE_RUNGS]
            safe = [c.rate(r) for r in SAFE_RUNGS]
            print(f"  {model:7s} {seed} N/C: unsafe rungs {', '.join(pct(*x) for x in unsafe)}; "
                  f"safe rungs {', '.join(pct(*x) for x in safe)}")
    spans, unique = defaultdict(Counter), defaultdict(set)
    for c in conds:
        for r in c.select("cross_boundary", UNSAFE_RUNGS):
            span = len({c.store[p]["boundary_id"] for p in c.probes[r["probe_id"]]["fragment_provenance_ids"]})
            unique[span].add((c.seed, r["probe_id"]))
            spans[span]["n"] += 1
            spans[span]["InvB"] += boundary_invalid(r, c.store)
            spans[span]["InvE"] += boundary_invalid(r, c.store, key="event_id")
    for s in sorted(spans):
        print(f"  span {s}: distinct trap probes {len(unique[s])}; boundary-invalid {pct(spans[s]['InvB'], spans[s]['n'])}; "
              f"event-invalid {pct(spans[s]['InvE'], spans[s]['n'])}")


def section_traps(conds):
    print("\n[§8] How the source fragments of a distinct trap probe differ")
    tenants, days, templates, overlap = Counter(), [], Counter(), Counter()
    seen = set()
    for c in conds:
        for p in c.probes.values():
            if p["probe_type"] != "cross_boundary" or (c.seed, p["probe_id"]) in seen:
                continue
            seen.add((c.seed, p["probe_id"]))
            frags = [c.store[h] for h in p["fragment_provenance_ids"]]
            tenants[len({f["tenant_id"] for f in frags})] += 1
            times = [datetime.fromisoformat(f["timestamp"]) for f in frags]
            days.append((max(times) - min(times)).total_seconds() / 86400)
            templates[len({f["incident_template"] for f in frags})] += 1
            pairs = [(a, b) for i, a in enumerate(frags) for b in frags[i + 1:] if a["event_id"] != b["event_id"]]
            overlap["pairs with a shared entity"] += any(set(a["entity_set"]) & set(b["entity_set"]) for a, b in pairs)
    n = len(seen)
    print(f"  distinct trap probes {n}; by number of tenants: "
          + ", ".join(f"{k} trap{'s' * (k != 1)} with {t}" for t, k in sorted(tenants.items()))
          + f" (median {statistics.median(t for t, k in tenants.items() for _ in range(k)):g})")
    print(f"  days between the earliest and latest source fragment: median {statistics.median(days):.1f}, "
          f"minimum {min(days):.1f}, maximum {max(days):.1f}")
    print("  by number of incident templates: "
          + ", ".join(f"{k} trap{'s' * (k != 1)} with {t}" for t, k in sorted(templates.items())))
    print(f"  traps in which two fragments of different events share an entity: {overlap['pairs with a shared entity']}/{n}")


def section_table4(conds):
    print("\n[Table 4, Appendix A] Unsafe-rung cells: HARDENED with Wilson 95% CI, EXACT, LOOSE")
    rows = {"HARDENED": [], "EXACT": [], "LOOSE": []}
    for c in conds:
        for rung in UNSAFE_RUNGS:
            rs = c.select("cross_boundary", (rung,))
            k = sum(boundary_invalid(r, c.store) for r in rs)
            e = sum(boundary_invalid_exact(r, c.store) for r in rs if not r.get("parse_error"))
            l = sum(boundary_invalid_loose(r, c.store) for r in rs if not r.get("parse_error"))
            lo, hi = wilson_ci(k, len(rs))
            for name, x in (("HARDENED", k), ("EXACT", e), ("LOOSE", l)):
                rows[name].append(x / len(rs))
            print(f"  {c.model:7s} {c.seed} {c.setting} {SHORT[rung]}  {100 * k / len(rs):5.1f} [{100 * lo:4.1f}, {100 * hi:5.1f}]"
                  f"  {100 * e / len(rs):5.1f}  {100 * l / len(rs):5.1f}")
    print("  median of the 24 cells: " + "; ".join(f"{name} {100 * statistics.median(v):.1f}" for name, v in rows.items()))
    safe = [c.rate(r) for c in conds for r in SAFE_RUNGS]
    lo, hi = wilson_ci(0, 100)
    print(f"  safe-rung cells: {len(safe)}, of which at 0/100: {sum(k == 0 for k, _ in safe)} "
          f"(Wilson interval of 0/100: {100 * lo:.1f}% to {100 * hi:.1f}%)")


def section_audit(audit_csv):
    print("\n[Appendix F] Audit sample")
    with open(audit_csv) as f:
        rows = list(csv.DictReader(f))
    by_model = Counter(r["model"] for r in rows)
    by_type = Counter((r["probe_type"], r["scorer_decision"]) for r in rows)
    print(f"  rows {len(rows)}; by model: " + ", ".join(f"{m} {n}" for m, n in by_model.items())
          + "; by probe type and scorer decision: " + ", ".join(f"{t} {d} {n}" for (t, d), n in sorted(by_type.items())))


def section_consistency(conds, data_dir):
    print("\n[Check] HARDENED scorer (rescore_final_ci.py) and BCESV ground truth (bcesv.py)")
    agree = n = 0
    for c in conds:
        store = bcesv.load_store(data_dir / c.name)
        for r in c.responses:
            n += 1
            agree += boundary_invalid(r, c.store) == bcesv.ground_truth_boundary_invalid(r, store)
    print(f"  agree on {agree}/{n} responses")


def main():
    ap = argparse.ArgumentParser(description="Numbers that the paper derives from the released data.")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    stores = {}
    conds = [Condition(args.data, name, label, stores) for name, label in CONDITIONS]
    print("MnemBound: numbers that the paper derives from the released data (analysis/paper_numbers.py)\n")
    section_data(conds)
    section_rungs(conds)
    section_results(conds)
    section_no_match(conds)
    section_figures(conds)
    section_robustness(conds)
    section_traps(conds)
    section_table4(conds)
    section_audit(ROOT / "analysis" / "audit_sample.csv")
    section_consistency(conds, args.data)


if __name__ == "__main__":
    main()
