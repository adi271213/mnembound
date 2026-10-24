"""
qualitative_cases.py: the case studies of §7.3 and Appendix E.

For each model, under the headline condition (seed 42, neutral prompt,
collapsed aggregator), selects the cross-boundary trap response on the
shared-vector rung that

  (a) produced a recall (no abstention, no parse failure),
  (b) is boundary-invalid under the HARDENED scorer, and
  (c) spans the most distinct boundaries, ties broken by the lowest probe_id,

and prints it as Markdown: the retrieved fragments, the recall, and for each
component the fragments that supply it and how (exact value, fragment value
inside the component, or component inside the fragment's text).

Usage, from the repository root:
  python3 analysis/qualitative_cases.py > analysis/qualitative_cases.md
"""
import argparse
import json
from pathlib import Path

from rescore_final_ci import ROLES, boundary_invalid, load_store, match_kind

HERE = Path(__file__).resolve().parent
CASES = {
    "Qwen3-30B-A3B-Instruct-2507-FP8 (seed 42, neutral / collapsed)": "qwen_full_neutral",
    "Mistral-Small-3.2-24B-Instruct-2506-FP8 (seed 42, neutral / collapsed)": "mistral_full_neutral",
}


def matches(resp, store):
    """component -> [(fragment number, fragment, match kind)] over the retrieved fragments."""
    view = resp.get("value_view", "tenant_hidden")
    frags = [store[pid] for pid in resp["fragment_provenance_ids"] if pid in store]
    out = {}
    for c in ROLES:
        val = resp["recall"].get(c, "")
        val = val.strip() if isinstance(val, str) else ""
        out[c] = [(i, f, k) for i, f in enumerate(frags, 1) if val and (k := match_kind(val, f, view))]
    return frags, out


def find_case(condition_dir):
    store = load_store(condition_dir)
    best, best_span = None, 0
    with open(condition_dir / "responses.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r["probe_type"] != "cross_boundary" or r["rung"] != "shared_vector_index"
                    or r.get("abstained") or r.get("parse_error") or not r.get("recall")
                    or not boundary_invalid(r, store)):
                continue
            span = len({f["boundary_id"] for ms in matches(r, store)[1].values() for _, f, _ in ms})
            if span > best_span or (span == best_span and r["probe_id"] < best["probe_id"]):
                best, best_span = r, span
    return best, store


def render(label, resp, store):
    frags, comp = matches(resp, store)
    spanned = sorted({f["boundary_id"] for ms in comp.values() for _, f, _ in ms})
    lines = [f"## {label}: {resp['probe_id']}", "",
             f"Rung: shared_vector_index. Value view: {resp['value_view']}. "
             f"Boundaries spanned by the recall: {', '.join(spanned)} ({len(spanned)}).", "",
             "### Retrieved evidence", "",
             "| # | role | boundary | sanitized value | sanitized text |", "|---|---|---|---|---|"]
    for i, f in enumerate(frags, 1):
        lines.append(f"| {i} | {f['role']} | {f['boundary_id']} | `{f['sanitized_value']}` | `{f['sanitized_text']}` |")
    lines += ["", "### Recall", "", "```json", json.dumps({"recall": resp["recall"]}, indent=2), "```", "",
              "### Supplying fragments per component", "",
              "| component | value | fragment (boundary) | match |", "|---|---|---|---|"]
    for c in ROLES:
        val = resp["recall"].get(c, "")
        if not comp[c]:
            lines.append(f"| {c} | `{val}` | none | none |")
        for i, f, kind in comp[c]:
            lines.append(f"| {c} | `{val}` | #{i} ({f['boundary_id']}) | {kind} |")
    kinds = [kind for c in ROLES for _, _, kind in comp[c]]
    single = sum(len(comp[c]) == 1 for c in ROLES)
    lines += ["",
              f"Match kinds: {', '.join(f'{k} {kinds.count(k)}' for k in sorted(set(kinds)))}. "
              f"Components supplied by exactly one fragment: {single} of {len(ROLES)}.",
              ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Case studies of §7.3 and Appendix E.")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    args = ap.parse_args()
    out = ["# Case studies: fragment-faithful, event-false recalls (§7.3, Appendix E)", "",
           "One cross-boundary trap per model under the headline condition on the shared-vector rung: the "
           "boundary-invalid recall that spans the most boundaries, ties broken by the lowest probe_id.", ""]
    for label, condition in CASES.items():
        resp, store = find_case(args.data / condition)
        out.append(render(label, resp, store) if resp else f"## {label}: no qualifying response\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
