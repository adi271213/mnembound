"""
corpus_stats.py: corpus and trap-probe statistics (Appendix I).

For each condition, writes data/<condition>/corpus_stats.json with

  n_fragments, n_records, n_boundaries   size of the fragment store
  probes_by_type                         probes per type
  cross_boundary_probes                  trap probes: how many carry a bridge
                                         context, their bridge categories
                                         (shared_ip, malware_hash, cve, vendor,
                                         threat_actor), and the number of
                                         distinct boundaries spanned by each
                                         probe's five source fragments
  bridge entities                        how many distinct bridge entities the
                                         traps use and how many boundaries each
                                         reaches

and prints one row per condition followed by the same statistics over the
distinct trap probes (the conditions that share a corpus seed share their
probes, so 12 condition folders hold 300 distinct trap probes) and, for each
distinct trap probe, how many of its five fragments show its bridge entity in
their text.

Usage, from the repository root:
  python3 analysis/corpus_stats.py > analysis/corpus_stats.log
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIDGE_PREFIX = {"ip": "shared_ip", "hash": "malware_hash", "cve": "cve", "vendor": "vendor", "actor": "threat_actor"}


def bridge_category(entity: str) -> str:
    """Bridge category of an entity such as 'ip:198.51.100.7'."""
    prefix = entity.split(":", 1)[0]
    return BRIDGE_PREFIX.get(prefix, prefix)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def stats_for_condition(condition_dir: Path) -> dict:
    fragments = load_jsonl(condition_dir / "fragments.store.jsonl")
    probes = load_jsonl(condition_dir / "probes.jsonl")
    boundary_of = {f["public_provenance_id"]: f["boundary_id"] for f in fragments}
    traps = [p for p in probes if p["probe_type"] == "cross_boundary"]
    categories, spans = Counter(), Counter()
    boundaries_of_bridge = defaultdict(set)
    for p in traps:
        bds = {boundary_of[pid] for pid in p["fragment_provenance_ids"]}
        spans[len(bds)] += 1
        for b in p.get("bridge_context", []):
            categories[bridge_category(b)] += 1
            boundaries_of_bridge[b] |= bds
    with_bridge = sum(1 for p in traps if p.get("bridge_context"))
    coverage = Counter(len(b) for b in boundaries_of_bridge.values())
    return {
        "condition": condition_dir.name,
        "n_fragments": len(fragments),
        "n_records": len({f["event_id"] for f in fragments}),
        "n_boundaries": len({f["boundary_id"] for f in fragments}),
        "probes_by_type": dict(Counter(p["probe_type"] for p in probes)),
        "cross_boundary_probes": {
            "total": len(traps),
            "with_bridge_context": with_bridge,
            "fraction_with_bridge_context": with_bridge / len(traps),
            "bridge_category_distribution": dict(categories),
            "boundary_span_distribution": dict(sorted(spans.items())),
            "mean_boundary_span": sum(k * v for k, v in spans.items()) / len(traps),
        },
        "distinct_bridges_in_cross_boundary_probes": len(boundaries_of_bridge),
        "bridge_boundary_coverage_distribution": dict(sorted(coverage.items())),
        "mean_boundaries_per_bridge": sum(len(b) for b in boundaries_of_bridge.values()) / len(boundaries_of_bridge),
    }


def bridge_shown(condition_dir: Path) -> Counter:
    """Trap probes by the number of their five fragments whose sanitized text names one of the trap's bridges."""
    text = {f["public_provenance_id"]: f["sanitized_text"] for f in load_jsonl(condition_dir / "fragments.store.jsonl")}
    shown = Counter()
    for p in load_jsonl(condition_dir / "probes.jsonl"):
        if p["probe_type"] == "cross_boundary":
            shown[sum(any(b in text[pid] for b in p["bridge_context"]) for pid in p["fragment_provenance_ids"])] += 1
    return shown


def main():
    ap = argparse.ArgumentParser(description="Corpus and trap-probe statistics (Appendix I).")
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    args = ap.parse_args()
    conditions = sorted(d for d in args.data.iterdir() if d.is_dir())
    header = (f"{'condition':28s} {'seed':>4s} {'trap probes':>11s} {'with bridge context':>19s} "
              f"{'mean boundaries per trap':>24s} {'distinct bridges':>16s} {'mean boundaries per bridge':>26s}")
    print(header)
    print("-" * len(header))
    distinct, first_folder, folder_totals = {}, {}, []
    for cond in conditions:
        s = stats_for_condition(cond)
        with open(cond / "corpus_stats.json", "w") as f:
            json.dump(s, f, indent=2)
        cb = s["cross_boundary_probes"]
        folder_totals.append((cb["total"], cb["with_bridge_context"]))
        seed = json.loads((cond / "corpus.json").read_text())["seed"]
        distinct.setdefault(seed, cb)
        first_folder.setdefault(seed, cond)
        print(f"{cond.name:28s} {seed:4d} {cb['total']:11d} {cb['with_bridge_context']:19d} "
              f"{cb['mean_boundary_span']:24.2f} {s['distinct_bridges_in_cross_boundary_probes']:16d} "
              f"{s['mean_boundaries_per_bridge']:26.2f}")
    print()
    print(f"Wrote corpus_stats.json to {len(conditions)} condition folders.")
    print(f"Trap probes over the {len(conditions)} condition folders: {sum(t for t, _ in folder_totals)}; "
          f"with a bridge context: {sum(b for _, b in folder_totals)}")
    traps = sum(cb["total"] for cb in distinct.values())
    with_bridge = sum(cb["with_bridge_context"] for cb in distinct.values())
    spans = Counter()
    for cb in distinct.values():
        spans.update({int(k): v for k, v in cb["boundary_span_distribution"].items()})
    print(f"Distinct corpora (seeds): {len(distinct)} ({', '.join(map(str, sorted(distinct)))})")
    print(f"Distinct trap probes: {traps}; with a bridge context: {with_bridge}")
    print(f"Boundaries spanned per distinct trap probe: {dict(sorted(spans.items()))}; "
          f"range {min(spans)} to {max(spans)}; mean {sum(k * v for k, v in spans.items()) / traps:.2f}")
    shown = {seed: bridge_shown(d) for seed, d in sorted(first_folder.items())}
    total = sum(shown.values(), Counter())
    keys = list(range(max(3, max(total) + 1)))
    print(f"Distinct trap probes whose bridge entity is named in the text of {', '.join(map(str, keys[:-1]))} or "
          f"{keys[-1]} of their five fragments (the evidence of the unsafe rungs): "
          + "; ".join(f"seed {s} " + ", ".join(str(c[k]) for k in keys) for s, c in shown.items())
          + "; all " + ", ".join(str(total[k]) for k in keys))


if __name__ == "__main__":
    main()
