"""
audit_labeler.py: interactive labeling of an audit sample (Appendix F).

Shows each unlabeled row of the CSV (the recall and a summary of the retrieved
fragments), asks for a label, records whether it agrees with the scorer's
decision, and saves after every row, so labeling can stop and resume.

By default the scorer's decision is hidden, so the labels are independent of
the scorer. The released labels in analysis/audit_sample.csv were assigned
with the decision visible; --show-scorer reproduces that procedure. The tool
never writes analysis/audit_sample.csv itself: label a copy.

Usage, from the repository root, with the copy in a temporary folder:
  tmp=$(mktemp -d)
  cp analysis/audit_sample.csv "$tmp/relabel.csv"
  python3 analysis/audit_labeler.py "$tmp/relabel.csv" --clear
  python3 analysis/audit_score.py --csv "$tmp/relabel.csv"

Keys:
  i  INVALID      cross-boundary probe, the recall combines boundaries
  s  SAFE         cross-boundary probe, the recall stays within one boundary
  a  ABSTAIN      the model declined
  p  PARSE_FAIL   the parser rejected the output (a recall with a missing or null component)
  r  RECALL       supported, absent or partial probe, the model produced a recall
  n  NO_MATCH     no recall, no abstention and no parse error
  ?  show the row again    b  relabel the previous row    q  save and quit
"""
import argparse
import csv
import sys
from pathlib import Path

DECISIONS = {"i": "INVALID", "s": "SAFE", "a": "ABSTAIN", "p": "PARSE_FAIL", "r": "RECALL", "n": "NO_MATCH"}
LABEL_FIELDS = ("human_decision", "human_notes", "agree")
RELEASED = Path(__file__).resolve().parent / "audit_sample.csv"


def show_row(row, idx, total, n_labeled, show_scorer):
    print("\n" + "=" * 78)
    print(f"  Row {idx + 1} / {total}    (labeled so far: {n_labeled})    audit_id: {row['audit_id']}")
    print("=" * 78)
    for key in ("model", "condition", "probe_type", "rung", "probe_id"):
        print(f"  {key + ':':12s} {row[key]}")
    print("\n  Recall:")
    for c in ("subject", "action", "object", "outcome", "time"):
        print(f"    {c + ':':8s} {row.get('recall_' + c, '')}")
    print(f"\n  Retrieved fragments:\n    {row.get('fragment_summary', '')}\n")
    if show_scorer:
        print(f"  Scorer decided: {row['scorer_decision']}\n")


def save(rows, fieldnames, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fieldnames} for r in rows)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Label an audit sample.")
    ap.add_argument("csv", type=Path, help="copy of the audit sample to label")
    ap.add_argument("--clear", action="store_true", help="remove existing labels first")
    ap.add_argument("--show-scorer", action="store_true", help="show the scorer's decision, as in the released audit")
    args = ap.parse_args()
    if args.csv.resolve() == RELEASED:
        sys.exit("label a copy: this tool does not write analysis/audit_sample.csv")
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    fieldnames += [k for k in LABEL_FIELDS if k not in fieldnames]
    if args.clear:
        for r in rows:
            for k in LABEL_FIELDS:
                r[k] = ""
        save(rows, fieldnames, args.csv)

    print(f"Loaded {len(rows)} rows from {args.csv}; "
          f"already labeled: {sum(1 for r in rows if r.get('human_decision', '').strip())}")
    print("Labels: " + "  ".join(f"[{k}] {v}" for k, v in DECISIONS.items()) + "  [?] again  [b] back  [q] quit")
    i = 0
    try:
        while i < len(rows):
            row = rows[i]
            if row.get("human_decision", "").strip():
                i += 1
                continue
            n_labeled = sum(1 for r in rows if r.get("human_decision", "").strip())
            show_row(row, i, len(rows), n_labeled, args.show_scorer)
            while True:
                key = input("label (i/s/a/p/r/n/?/b/q) > ").strip().lower()
                if key == "q":
                    raise KeyboardInterrupt
                if key == "?":
                    show_row(row, i, len(rows), n_labeled, args.show_scorer)
                    continue
                if key == "b":
                    previous = [j for j in range(i - 1, -1, -1) if rows[j].get("human_decision", "").strip()]
                    if not previous:
                        print("  no earlier labeled row")
                        continue
                    for k in LABEL_FIELDS:
                        rows[previous[0]][k] = ""
                    save(rows, fieldnames, args.csv)
                    i = previous[0]
                    break
                if key in DECISIONS:
                    row["human_decision"] = DECISIONS[key]
                    row["agree"] = "1" if DECISIONS[key] == row.get("scorer_decision") else "0"
                    row["human_notes"] = input("notes (ENTER to skip) > ").strip()
                    save(rows, fieldnames, args.csv)
                    i += 1
                    break
                print(f"  unknown key {key!r}")
    except KeyboardInterrupt:
        print("\nsaving and exiting")
    save(rows, fieldnames, args.csv)
    done = sum(1 for r in rows if r.get("human_decision", "").strip())
    print(f"labeled {done} of {len(rows)} rows in {args.csv}")


if __name__ == "__main__":
    main()
