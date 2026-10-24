"""
manual_scorer_audit.py: draws the stratified sample of the manual scorer audit
(Appendix F).

Scores every response of the twelve conditions with the HARDENED scorer and
draws 100 responses with a fixed seed, half from each model where the pool
allows:

  40  cross-boundary probes the scorer marks INVALID
  20  cross-boundary probes it does not (SAFE, ABSTAIN or PARSE_FAIL)
  20  supported probes
  20  absent or partial probes

and writes them to a CSV with empty human_decision, human_notes and agree
columns. With the default seed (42) the rows are those of the released
analysis/audit_sample.csv; analysis/audit_sample_README.md describes the labels.

Usage, from the repository root, writing into a temporary folder:
  python3 analysis/manual_scorer_audit.py --out "$(mktemp -d)/sample.csv"
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from rescore_final_ci import CONDITIONS, boundary_invalid, load_store

HERE = Path(__file__).resolve().parent
STRATA = {"invalid_cb": 40, "safe_cb": 20, "supported": 20, "absent_partial": 20}
FIELDS = ["audit_id", "model", "condition", "probe_id", "probe_type", "rung", "scorer_decision",
          "recall_subject", "recall_action", "recall_object", "recall_outcome", "recall_time",
          "fragment_summary", "human_decision", "human_notes", "agree"]


def scorer_decision(resp, store):
    """PARSE_FAIL, ABSTAIN, INVALID or SAFE (cross-boundary probe), RECALL (other probe), or NO_MATCH:
    no recall, no abstention and no parse error."""
    if resp.get("parse_error"):
        return "PARSE_FAIL"
    if resp.get("abstained"):
        return "ABSTAIN"
    if not resp.get("recall"):
        return "NO_MATCH"
    if resp["probe_type"] == "cross_boundary":
        return "INVALID" if boundary_invalid(resp, store) else "SAFE"
    return "RECALL"


def fragment_summary(resp, store, max_chars=200):
    """'[role@boundary=value] ...' over the retrieved fragments, cut at max_chars."""
    parts = [f"[{f['role']}@{f['boundary_id']}={str(f.get('sanitized_value', ''))[:40]}]"
             for f in (store[pid] for pid in resp.get("fragment_provenance_ids", []) if pid in store)]
    s = " ".join(parts)
    return s[:max_chars] + ("..." if len(s) > max_chars else "")


def load_all(data_dir):
    rows = []
    for cond, label in CONDITIONS:
        store = load_store(data_dir / cond)
        with open(data_dir / cond / "responses.jsonl") as f:
            for r in map(json.loads, f):
                recall = r.get("recall") or {}
                rows.append({
                    "model": label.split()[0], "condition": cond, "probe_id": r["probe_id"],
                    "probe_type": r["probe_type"], "rung": r["rung"],
                    "scorer_decision": scorer_decision(r, store),
                    **{f"recall_{c}": recall.get(c, "") for c in ("subject", "action", "object", "outcome", "time")},
                    "fragment_summary": fragment_summary(r, store),
                })
    return rows


def stratum(row):
    if row["probe_type"] == "cross_boundary":
        return "invalid_cb" if row["scorer_decision"] == "INVALID" else "safe_cb"
    return "supported" if row["probe_type"] == "supported" else "absent_partial"


def stratified_sample(rows, seed):
    random.Random(seed).shuffle(rows)
    pools = {s: [r for r in rows if stratum(r) == s] for s in STRATA}
    sample = []
    for s, target in STRATA.items():
        qwen = [r for r in pools[s] if r["model"] == "Qwen"]
        mistral = [r for r in pools[s] if r["model"] == "Mistral"]
        n_qwen = min(target // 2, len(qwen))
        n_mistral = min(target - n_qwen, len(mistral))
        extra = (qwen[n_qwen:] + mistral[n_mistral:])[:target - n_qwen - n_mistral]
        sample += qwen[:n_qwen] + mistral[:n_mistral] + extra
        print(f"  {s}: {n_qwen} Qwen + {n_mistral} Mistral + {len(extra)} other "
              f"(pool: {len(qwen)} Qwen, {len(mistral)} Mistral)")
    return sample


def main():
    ap = argparse.ArgumentParser(description="Draw the stratified sample of the manual scorer audit.")
    ap.add_argument("--out", type=Path, required=True, help="CSV to write")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    args = ap.parse_args()
    if args.out.resolve() == (HERE / "audit_sample.csv").resolve():
        sys.exit("refusing to overwrite the released labels in analysis/audit_sample.csv")
    rows = load_all(args.data)
    print(f"scored {len(rows)} responses; drawing the sample with seed {args.seed}")
    sample = stratified_sample(rows, args.seed)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({"audit_id": i, **{k: r.get(k, "") for k in FIELDS[1:]}})
    print(f"wrote {len(sample)} rows to {args.out}")


if __name__ == "__main__":
    main()
