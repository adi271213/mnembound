"""
bcesv_run_all.py: Table 2, Figure 6 and Appendix D.

Runs every BCESV variant over the cross-boundary and supported responses of the
twelve conditions and writes, next to this script,

  bcesv_results.csv   counts per variant, condition, rung and probe type
  bcesv_summary.csv   per variant, on the shared-vector (SV) and summary-merge
                      (SM) rungs and pooled over both ("both"): recall and
                      precision on cross-boundary probes, the share of
                      supported recalls flagged INVALID (fp_sup) and accepted
                      as SAFE (acc_sup), abstention on cross-boundary probes,
                      the underlying counts, and Wilson 95% intervals

and prints the summary, the values plotted in Figure 6, how many trap
responses FragmentOnly flags INVALID, and the thresholds of the cohesion rule
of BcesvReconstruct. Ground truth is
ground_truth_boundary_invalid. Parse failures are counted in bcesv_results.csv
and excluded from the metrics.

Usage, from the repository root:
  python3 bcesv/bcesv_run_all.py > bcesv/bcesv_run_all.log
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from bcesv import (RECON_ENTITY_JACCARD, RECON_TIME_WINDOW_SEC, BcesvExact, BcesvNoisy, BcesvReconstruct,
                   FragmentOnly, ground_truth_boundary_invalid, load_responses, load_store, wilson_ci)

HERE = Path(__file__).resolve().parent
CONDITIONS = ["qwen_full_neutral", "qwen_full_cautious", "qwen_full_pcb", "qwen_full_cautious_pcb",
              "qwen_seed43_neutral", "qwen_seed44_neutral", "mistral_full_neutral", "mistral_full_cautious",
              "mistral_full_pcb", "mistral_full_cautious_pcb", "mistral_seed43_neutral", "mistral_seed44_neutral"]
RUNGS = ("isolated", "boundary_preserving_topk", "shared_vector_index", "summary_merge")
UNSAFE = {"shared_vector_index": "SV", "summary_merge": "SM"}


def variants():
    """The eleven variants of Appendix D, then BcesvExact with case-insensitive matching (§8)."""
    return [FragmentOnly(), BcesvExact(), BcesvNoisy(0.1), BcesvNoisy(0.2), BcesvNoisy(0.3),
            BcesvReconstruct(),
            BcesvReconstruct(use_tenant=False), BcesvReconstruct(use_time=False),
            BcesvReconstruct(use_entity_set=False), BcesvReconstruct(use_log_source=False),
            BcesvReconstruct(use_template=False),
            BcesvExact(casefold=True)]


def run(data_dir):
    vs = variants()
    cells = Counter()
    for cond in CONDITIONS:
        store, responses = load_store(data_dir / cond), load_responses(data_dir / cond)
        for r in responses:
            if r["probe_type"] not in ("cross_boundary", "supported"):
                continue
            if r.get("parse_error"):
                for v in vs:
                    cells[(v.name, cond, r["rung"], r["probe_type"], "parse_fail")] += 1
                continue
            gt = ground_truth_boundary_invalid(r, store)
            for v in vs:
                pred = v.predict(r, store)
                outcome = ("tp" if gt else "fp") if pred == "INVALID" else ("fn" if gt else "tn")
                key = (v.name, cond, r["rung"], r["probe_type"])
                cells[key + (pred,)] += 1
                cells[key + (outcome,)] += 1
    return vs, cells


def write_results(vs, cells):
    names = [v.name for v in vs]
    path = HERE / "bcesv_results.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "condition", "rung", "probe_type", "tp", "fp", "fn", "tn",
                    "invalid", "safe", "abstain", "parse_fail"])
        for name in names:
            for cond in CONDITIONS:
                for rung in RUNGS:
                    for pt in ("cross_boundary", "supported"):
                        k = (name, cond, rung, pt)
                        w.writerow(list(k) + [cells[k + (x,)] for x in
                                              ("tp", "fp", "fn", "tn", "INVALID", "SAFE", "ABSTAIN", "parse_fail")])
    return path, len(names) * len(CONDITIONS) * len(RUNGS) * 2


def summarize(vs, cells):
    rows = []
    for v in vs:
        for scope in ("SV", "SM", "both"):
            rungs = [r for r, s in UNSAFE.items() if scope in (s, "both")]

            def count(pt, x):
                return sum(cells[(v.name, cond, rung, pt, x)] for cond in CONDITIONS for rung in rungs)

            tp, fn, fp = count("cross_boundary", "tp"), count("cross_boundary", "fn"), count("cross_boundary", "fp")
            n_trap = sum(count("cross_boundary", x) for x in ("INVALID", "SAFE", "ABSTAIN"))
            n_sup = sum(count("supported", x) for x in ("INVALID", "SAFE", "ABSTAIN"))
            sup_flagged, sup_accepted = count("supported", "INVALID"), count("supported", "SAFE")
            trap_abstain = count("cross_boundary", "ABSTAIN")
            row = {"variant": v.name, "rung": scope}
            for metric, k, n in (("recall", tp, tp + fn), ("precision", tp, tp + fp),
                                 ("fp_sup", sup_flagged, n_sup), ("acc_sup", sup_accepted, n_sup)):
                lo, hi = wilson_ci(k, n)
                row.update({metric: k / n if n else 0.0, f"{metric}_lo": lo, f"{metric}_hi": hi})
            row.update({"trap_abstain": trap_abstain / n_trap, "tp": tp, "fn": fn, "fp": fp, "n_trap": n_trap,
                        "n_sup": n_sup, "sup_flagged": sup_flagged, "sup_accepted": sup_accepted})
            rows.append(row)
    path = HERE / "bcesv_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    return path, rows


def main():
    ap = argparse.ArgumentParser(description="Table 2, Figure 6 and Appendix D.")
    ap.add_argument("data_dir", nargs="?", type=Path, default=HERE.parent / "data")
    args = ap.parse_args()
    vs, cells = run(args.data_dir)
    path, n = write_results(vs, cells)
    print(f"wrote {path.name} ({n} rows)")
    path, rows = summarize(vs, cells)
    print(f"wrote {path.name} ({len(rows)} rows)")

    print("\nTable 2 and Appendix D: unsafe rungs, pooled over the 12 conditions")
    print(f"{'variant':24s} {'rung':4s} {'recall':>7s} {'prec':>7s} {'fp_sup':>7s} {'acc_sup':>8s} {'trap_abst':>9s}")
    for r in rows:
        print(f"{r['variant']:24s} {r['rung']:4s} {100 * r['recall']:6.1f}% {100 * r['precision']:6.1f}% "
              f"{100 * r['fp_sup']:6.1f}% {100 * r['acc_sup']:7.1f}% {100 * r['trap_abstain']:8.1f}%")

    print("\nFigure 6: shared-vector rung, BcesvExact (p = 0) and BcesvNoisy(p)")
    by_name = {(r["variant"], r["rung"]): r for r in rows}
    for p, name in ((0, "BcesvExact"), (10, "BcesvNoisy(10%)"), (20, "BcesvNoisy(20%)"), (30, "BcesvNoisy(30%)")):
        r = by_name[(name, "SV")]
        print(f"  p = {p:2d}%: recall {100 * r['recall']:5.1f}%, precision {100 * r['precision']:5.1f}%, "
              f"supported accepted {100 * r['acc_sup']:5.1f}%")

    print()
    for rung in ("SV", "SM"):
        r = by_name[("FragmentOnly", rung)]
        print(f"FragmentOnly, {rung}: {r['tp'] + r['fp']} of the {r['n_trap']} trap responses flagged INVALID")
    print(f"The cohesion rule of BcesvReconstruct links two fragments with the same tenant, timestamps at most {RECON_TIME_WINDOW_SEC} s "
          f"apart, an entity-set Jaccard of at least {RECON_ENTITY_JACCARD}, and the same log source or incident "
          "template; its groups are the connected components of these links")


if __name__ == "__main__":
    main()
