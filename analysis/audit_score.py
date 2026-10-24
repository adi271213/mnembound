"""
audit_score.py: the manual scorer audit (§7.4, R3; Appendix F).

Reads the labeled audit sample (analysis/audit_sample.csv by default) and
prints the agreement between the HARDENED scorer's decision and the manual
label: the overall rate, the rate per stratum, the confusion matrix, Cohen's
kappa with the Landis and Koch interpretation, and any disagreements. The
released labels were assigned with the scorer's decision visible, so the
agreement measures consistency with the scorer, not independent validation.

Usage, from the repository root:
  python3 analysis/audit_score.py > analysis/audit_score.log
  python3 analysis/audit_score.py --csv "$tmp/relabel.csv"     (see analysis/audit_labeler.py)
"""
import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser(description="Score the manual scorer audit (Appendix F).")
    ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent / "audit_sample.csv",
                    help="labeled audit sample")
    args = ap.parse_args()
    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    labeled = [r for r in rows if r.get("human_decision", "").strip()]
    n_total = len(rows)
    n_labeled = len(labeled)
    print(f"Total sampled rows: {n_total}")
    print(f"Labeled rows:       {n_labeled}")
    print()

    if n_labeled == 0:
        print(f"No labeled rows in {args.csv}. Label them with analysis/audit_labeler.py, "
              "then re-run this script.")
        return

    n_agree = sum(1 for r in labeled if r.get("agree", "").strip() == "1")
    print(f"Agreement rate:     {n_agree}/{n_labeled} = "
          f"{n_agree/n_labeled*100:.1f}%")
    print()

    def stratum(r):
        if r["probe_type"] == "cross_boundary":
            return ("cross_boundary_INVALID"
                    if r["scorer_decision"] == "INVALID"
                    else "cross_boundary_SAFE")
        if r["probe_type"] == "supported":
            return "supported"
        return r["probe_type"]

    by_stratum = defaultdict(lambda: [0, 0])  # [agree, total]
    for r in labeled:
        s = stratum(r)
        by_stratum[s][1] += 1
        if r.get("agree", "").strip() == "1":
            by_stratum[s][0] += 1

    print("Per-stratum agreement:")
    for s, (a, t) in sorted(by_stratum.items()):
        print(f"  {s:30s} {a}/{t} = {a/t*100:.1f}%" if t else
              f"  {s:30s} 0/0")
    print()

    print("Confusion matrix (scorer rows, human columns):")
    decisions = sorted({r["scorer_decision"] for r in labeled} |
                       {r["human_decision"] for r in labeled})
    conf = {(s, h): 0 for s in decisions for h in decisions}
    for r in labeled:
        s, h = r["scorer_decision"], r["human_decision"]
        if h in decisions and s in decisions:
            conf[(s, h)] += 1
    header = " " * 13 + "  ".join(f"{d:>10s}" for d in decisions)
    print(header)
    for s in decisions:
        row = f"{s:13s} " + "  ".join(
            f"{conf[(s, h)]:>10d}" for h in decisions)
        print(row)
    print()

    po = n_agree / n_labeled if n_labeled else 0
    scorer_marg = Counter(r["scorer_decision"] for r in labeled)
    human_marg = Counter(r["human_decision"] for r in labeled)
    pe = sum(
        (scorer_marg[d] / n_labeled) * (human_marg[d] / n_labeled)
        for d in decisions
    )
    if pe < 1.0:
        kappa = (po - pe) / (1 - pe)
        print(f"Cohen's kappa:     {kappa:.3f}")
        if kappa >= 0.81:
            interp = "near-perfect agreement"
        elif kappa >= 0.61:
            interp = "substantial agreement"
        elif kappa >= 0.41:
            interp = "moderate agreement"
        elif kappa >= 0.21:
            interp = "fair agreement"
        elif kappa >= 0.0:
            interp = "slight agreement"
        else:
            interp = "no agreement / disagreement"
        print(f"Interpretation:    {interp} (Landis & Koch, 1977)")
    else:
        print("Cohen's kappa:     undefined (pe = 1.0)")
    print()

    disagreements = [r for r in labeled
                     if r.get("agree", "").strip() != "1"]
    if disagreements:
        print(f"Disagreements ({len(disagreements)} rows):")
        for r in disagreements:
            print(f"  audit_id={r['audit_id']:>3s}  "
                  f"scorer={r['scorer_decision']:<10s}  "
                  f"human={r['human_decision']:<10s}  "
                  f"({r['model']} {r['probe_type']} on {r['rung']})")
            if r.get("human_notes", "").strip():
                print(f"    notes: {r['human_notes']}")


if __name__ == "__main__":
    main()
