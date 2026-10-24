"""
parse_fail_sensitivity.py: parse-failure handling (§7.4, R2; Appendix C).

A parse failure is a response that the parser of
mnembound/mnembound_pkg/mnembound/agent_io.py rejected (parse_error set): the
output was neither an abstention nor a recall whose five components are
strings. On the cross-boundary probes of each cell this script reports

  MAIN   boundary-invalid rate with parse failures counted as not invalid (the
         paper's scoring; counting them as abstentions gives the same rate)
  WORST  boundary-invalid rate with every parse failure counted as
         boundary-invalid (the hostile bound)
  PF     number of parse failures

followed by the parse failures over all probe types, how many of them are
JSON recalls with a missing or null component, the largest hostile bound on a
safe rung, and the gap to the unsafe rungs of the same condition.

Usage, from the repository root:
  python3 analysis/parse_fail_sensitivity.py > analysis/parse_fail_sensitivity.log
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from rescore_final_ci import CONDITIONS, RUNGS, SAFE_RUNGS, boundary_invalid, load_store

HERE = Path(__file__).resolve().parent
ROLES = ("subject", "action", "object", "outcome", "time")


def missing_or_null(raw_text):
    """Components that a JSON recall leaves missing or null, or None when the text is no JSON recall."""
    try:
        obj = json.loads(raw_text)
    except ValueError:
        return None
    recall = obj.get("recall") if isinstance(obj, dict) else None
    return [k for k in ROLES if recall.get(k) is None] if isinstance(recall, dict) else None


def count(condition_dir):
    """Counters per rung: cross-boundary probes, their parse failures and MAIN invalids, and all parse failures."""
    store = load_store(condition_dir)
    c = Counter()
    with open(condition_dir / "responses.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rung = r["rung"]
            c[rung, "n_all"] += 1
            c[rung, "pf_all"] += bool(r.get("parse_error"))
            if r.get("parse_error"):
                missing = missing_or_null(r.get("raw_text", ""))
                if missing:
                    c["null_recall"] += 1
                    for k in missing:
                        c["null", k] += 1
            if r["probe_type"] != "cross_boundary":
                continue
            c[rung, "n"] += 1
            if r.get("parse_error"):
                c[rung, "pf"] += 1
            elif boundary_invalid(r, store):
                c[rung, "inv"] += 1
    return c


def pct(k, n):
    return f"{k}/{n}={100 * k / n:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description="Parse-failure handling (§7.4, Appendix C).")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    args = ap.parse_args()

    print("Parse-failure sensitivity on cross-boundary trap probes.")
    print("MAIN  = parse failures counted as not boundary-invalid (paper scoring)")
    print("WORST = parse failures counted as boundary-invalid (hostile bound)")
    print("PF    = parse failures among the cell's cross-boundary probes")
    print()
    print(f"{'condition':16s} {'rung':28s} {'MAIN':>14s} {'WORST':>14s} {'PF':>4s}")
    print("-" * 80)
    counts = {}
    for path_name, label in CONDITIONS:
        c = counts[label] = count(args.data / path_name)
        for rung in RUNGS:
            n, inv, pf = c[rung, "n"], c[rung, "inv"], c[rung, "pf"]
            print(f"{label:16s} {rung:28s} {pct(inv, n):>14s} {pct(inv + pf, n):>14s} {pf:4d}")
        print()

    total = sum(c[r, "pf_all"] for c in counts.values() for r in RUNGS)
    n_all = sum(c[r, "n_all"] for c in counts.values() for r in RUNGS)
    trap_pf = sum(c[r, "pf"] for c in counts.values() for r in RUNGS)
    print(f"Parse failures: {total} of {n_all} responses ({100 * total / n_all:.2f}%), {trap_pf} of them on cross-boundary probes")
    null_recall = sum(c["null_recall"] for c in counts.values())
    per_role = {k: sum(c["null", k] for c in counts.values()) for k in ROLES}
    print(f"Parse failures that are JSON recalls with a missing or null component: {null_recall} of {total}; "
          "missing or null, per component: " + ", ".join(f"{k} {n}" for k, n in per_role.items()))
    print("Cells with parse failures, over all 300 probes of the rung:")
    for label, c in counts.items():
        for rung in RUNGS:
            if c[rung, "pf_all"]:
                print(f"  {label:16s} {rung:28s} {c[rung, 'pf_all']:3d}/{c[rung, 'n_all']}")

    hostile = {(label, r): (c[r, "inv"] + c[r, "pf"]) / c[r, "n"] for label, c in counts.items() for r in SAFE_RUNGS}
    rate = max(hostile.values())
    worst = [key for key, v in hostile.items() if v == rate]
    unsafe = [counts[label][r, "inv"] / counts[label][r, "n"]
              for label in {label for label, _ in worst} for r in RUNGS if r not in SAFE_RUNGS]
    print()
    print(f"Largest hostile bound on a safe rung: {100 * rate:.1f}% "
          f"({'; '.join(f'{label.strip()} {r}' for label, r in worst)})")
    print(f"Same conditions, unsafe rungs (MAIN): {100 * min(unsafe):.1f}% to {100 * max(unsafe):.1f}%")
    gaps = []
    for label, c in counts.items():
        safe_worst = max((c[r, "inv"] + c[r, "pf"]) / c[r, "n"] for r in SAFE_RUNGS)
        unsafe_min = min(c[r, "inv"] / c[r, "n"] for r in RUNGS if r not in SAFE_RUNGS)
        gaps.append((unsafe_min - safe_worst, label))
    gap, label = min(gaps)
    print(f"Smallest gap, within a condition, between the lowest unsafe-rung rate and the highest hostile "
          f"safe-rung bound: {100 * gap:.1f} percentage points ({label.strip()})")


if __name__ == "__main__":
    main()
