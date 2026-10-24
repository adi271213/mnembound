"""
run_paper.py: the generation procedure behind data/<condition>/.

Each of the twelve conditions of the paper is one model, one corpus seed, one
system prompt and one summary-merge aggregator. For a condition, `run`
generates the corpus of its seed with the configuration of the paper
(PAPER_CORPUS below), sends every probe on every rung to the model behind an
OpenAI-compatible endpoint with the decoding settings of the paper (temperature
0, top_p 1, at most 400 output tokens; needs the openai package), and writes
corpus.json, probes.jsonl, fragments.store.jsonl, rung_stats.jsonl and
responses.jsonl in the format of data/<condition>/. With --stub a
deterministic stub that recalls each probe's event from the evidence replaces
the model.

`check` prints the corpus configuration and the decoding settings of `run`,
regenerates the three corpora (seeds 42, 43 and 44) and compares them with
the released data: corpus.json, probes.jsonl, fragments.store.jsonl and
rung_stats.jsonl byte for byte, and the context of every released response
with the retrieval of its rung. For each corpus it also counts the records
that carry a bridge entity, the trap probes whose five source records carry
the trap's bridge, how many of an absent recall's components its evidence
supplies, and the probes and responses whose question names a raw,
tenant-specific subject (see agent_io.build_prompt). Its output is released
as run_paper_check.log.

Usage, from mnembound/mnembound_pkg:
  python3 run_paper.py check ../../data > run_paper_check.log
  python3 run_paper.py run qwen_full_neutral --out-dir "$(mktemp -d)" --base-url http://localhost:8000/v1
  python3 run_paper.py run qwen_full_neutral --out-dir "$(mktemp -d)" --stub
"""
import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from mnembound.agent_io import PromptMode, build_prompt, run_probe, stub_agent_always_recall
from mnembound.bridges import BridgeDensitySpec
from mnembound.generator import generate_corpus
from mnembound.retrieval import ALL_RUNGS, SummaryMergeMode, retrieve, summarize_rung
from mnembound.schema import ALL_ROLES, ComponentRole, ProbeType
from mnembound.scoring import ValueView
from mnembound.serialization import (fragment_to_store_dict, write_corpus_metadata, write_probes_jsonl,
                                     write_responses_jsonl, write_rung_stats_jsonl)

QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MISTRAL = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8"
CONDITIONS = {  # name: (model, seed, system prompt, summary-merge aggregator)
    "qwen_full_neutral": (QWEN, 42, "neutral", "collapsed"),
    "qwen_full_cautious": (QWEN, 42, "cautious", "collapsed"),
    "qwen_full_pcb": (QWEN, 42, "neutral", "per_client_blocks"),
    "qwen_full_cautious_pcb": (QWEN, 42, "cautious", "per_client_blocks"),
    "qwen_seed43_neutral": (QWEN, 43, "neutral", "collapsed"),
    "qwen_seed44_neutral": (QWEN, 44, "neutral", "collapsed"),
    "mistral_full_neutral": (MISTRAL, 42, "neutral", "collapsed"),
    "mistral_full_cautious": (MISTRAL, 42, "cautious", "collapsed"),
    "mistral_full_pcb": (MISTRAL, 42, "neutral", "per_client_blocks"),
    "mistral_full_cautious_pcb": (MISTRAL, 42, "cautious", "per_client_blocks"),
    "mistral_seed43_neutral": (MISTRAL, 43, "neutral", "collapsed"),
    "mistral_seed44_neutral": (MISTRAL, 44, "neutral", "collapsed"),
}
PAPER_CORPUS = dict(n_tenants=10, records_per_tenant=500, bridge_density=BridgeDensitySpec(extra=0.05, slot=0.01),
                    n_supported_probes=100, n_absent_probes=50, n_partial_probes=50, n_cross_boundary_probes=100,
                    partial_distractor_fragments=5, prefer_bridge_linked_traps=True)
DECODING = dict(temperature=0.0, top_p=1.0, max_tokens=400)
# Fields of fragments.store.jsonl in the release: the raw value and text of
# each fragment and its record id are left out.
STORE_FIELDS = ["public_provenance_id", "boundary_id", "role", "sanitized_value", "sanitized_text",
                "tenant_id", "event_id", "timestamp", "entity_set", "log_source", "incident_template"]


def build_corpus(seed):
    return generate_corpus(seed=seed, **PAPER_CORPUS)


def write_corpus_files(corpus, out_dir):
    """corpus.json, probes.jsonl, fragments.store.jsonl and rung_stats.jsonl as released."""
    write_corpus_metadata(corpus, out_dir / "corpus.json")
    write_probes_jsonl(corpus.probes, out_dir / "probes.jsonl")
    write_rung_stats_jsonl([summarize_rung(r, list(corpus.probes), corpus) for r in ALL_RUNGS],
                           out_dir / "rung_stats.jsonl")
    with open(out_dir / "fragments.store.jsonl", "w") as f:
        for frag in corpus.fragments:
            record = fragment_to_store_dict(frag)
            f.write(json.dumps({k: record[k] for k in STORE_FIELDS}) + "\n")


def make_openai_agent(base_url, model):
    import openai
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")

    def agent(system, user):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                timeout=120, **DECODING)
            return resp.choices[0].message.content or ""
        except Exception as e:
            return json.dumps({"abstain": True, "reason": f"agent_error: {e}"})

    return agent


def run(args):
    model, seed, prompt, aggregator = CONDITIONS[args.condition]
    corpus = build_corpus(seed)
    agent = None if args.stub else make_openai_agent(args.base_url, model)
    responses, t0 = [], time.perf_counter()
    for rung in ALL_RUNGS:
        for probe in corpus.probes:
            result = retrieve(rung, probe, corpus)
            if args.stub:
                def agent(system, user, _p=probe, _r=result):
                    return stub_agent_always_recall(_p, _r, value_view=ValueView.TENANT_HIDDEN)
            responses.append(run_probe(probe, result, agent, tenant_visible=False, model=model, seed=seed,
                                       prompt_mode=PromptMode(prompt), summary_merge_mode=SummaryMergeMode(aggregator)))
        print(f"{rung.value:26s} done after {time.perf_counter() - t0:.1f} s", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_corpus_files(corpus, args.out_dir)
    write_responses_jsonl(responses, args.out_dir / "responses.jsonl")
    print(f"wrote {len(responses)} responses to {args.out_dir / 'responses.jsonl'}")


def raw_subject(probe, corpus):
    """Whether the question of the probe names its raw, tenant-specific subject (see agent_io.build_prompt)."""
    subject = probe.recall.subject
    return subject in corpus.entity_sanitizer and not any(
        f.role == ComponentRole.SUBJECT and f.value == subject for f in probe.fragments)


def check(args):
    names = sorted(CONDITIONS)
    types = [t.value for t in (ProbeType.SUPPORTED, ProbeType.ABSENT, ProbeType.PARTIAL, ProbeType.CROSS_BOUNDARY)]
    c = PAPER_CORPUS
    print(f"corpus configuration (PAPER_CORPUS): {c['n_tenants']} tenants of {c['records_per_tenant']} records; "
          f"bridge density {c['bridge_density'].extra:.0%} in Record.extra and {c['bridge_density'].slot:.0%} in the "
          f"subject or object slot; probes {c['n_supported_probes']} supported, {c['n_absent_probes']} absent, "
          f"{c['n_partial_probes']} partial, {c['n_cross_boundary_probes']} cross_boundary; "
          f"{c['partial_distractor_fragments']} distractor fragments in partial probes; "
          f"bridge-linked traps {c['prefer_bridge_linked_traps']}")
    print("decoding of run (DECODING): " + ", ".join(f"{k} {v}" for k, v in DECODING.items()))
    print()
    print(f"{'condition':26s} {'corpus.json':>12s} {'probes':>7s} {'store':>6s} {'rung_stats':>10s} {'contexts':>9s}")
    all_same = True
    n_responses = 0
    raw_responses, raw_probes, all_probes = Counter(), Counter(), Counter()  # over the twelve conditions
    for seed in (42, 43, 44):
        corpus = build_corpus(seed)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_corpus_files(corpus, tmp)
            regenerated = {f: (tmp / f).read_bytes() for f in
                           ("corpus.json", "probes.jsonl", "fragments.store.jsonl", "rung_stats.jsonl")}
        probes = {p.probe_id: p for p in corpus.probes}
        raw = {p.probe_id for p in corpus.probes if raw_subject(p, corpus)}
        seed_names = [n for n in names if CONDITIONS[n][1] == seed]
        seed_raw_responses = 0
        for name in seed_names:
            d = args.data / name
            same = [regenerated[f] == (d / f).read_bytes() for f in regenerated]
            mismatched = 0
            with open(d / "responses.jsonl") as f:
                for r in map(json.loads, f):
                    n_responses += 1
                    probe = probes[r["probe_id"]]
                    result = retrieve(next(x for x in ALL_RUNGS if x.value == r["rung"]), probe, corpus)
                    mismatched += [x.public_provenance_id for x in result.fragments] != r["fragment_provenance_ids"]
                    if probe.probe_id in raw:
                        assert f"involving {probe.recall.subject}?" in build_prompt(probe, result, tenant_visible=False).user
                        raw_responses[probe.probe_type.value] += 1
                        seed_raw_responses += 1
            all_same &= all(same) and not mismatched
            print(f"{name:26s} " + " ".join(f"{'same' if s else 'DIFFERENT':>{w}s}" for s, w in zip(same, (12, 7, 6, 10)))
                  + f" {'all' if not mismatched else str(mismatched) + ' differ':>9s}")
        extra = {r.record_id for r in corpus.records if r.extra.get("bridge")}
        bridges = corpus.bridge_pools.all_bridges()
        slot = {r.record_id for r in corpus.records if r.subject in bridges or r.object_ in bridges}
        pools = corpus.bridge_pools
        print(f"  seed {seed}: bridge pools: shared_ip {len(pools.shared_ips)}, malware_hash {len(pools.shared_hashes)}, "
              f"cve {len(pools.shared_cves)}, vendor {len(pools.shared_vendors)}, "
              f"threat_actor {len(pools.shared_threat_actors)}")
        print(f"  seed {seed}: records {len(corpus.records)}; with a bridge in Record.extra {len(extra)}, "
              f"in the subject or object slot {len(slot)}, either {len(extra | slot)}")
        records = {r.record_id: r for r in corpus.records}
        traps = [p for p in corpus.probes if p.probe_type == ProbeType.CROSS_BOUNDARY]
        one_bridge = [p for p in traps if len(p.bridge_context) == 1]
        carried = in_extra = in_slot = 0
        for p in one_bridge:
            (b,) = p.bridge_context
            sources = [records[f.record_id] for f in p.fragments]
            in_extra += sum(r.extra.get("bridge") == b for r in sources)
            in_slot += sum(b in (r.subject, r.object_) for r in sources)
            carried += all(r.extra.get("bridge") == b or b in (r.subject, r.object_) for r in sources)
        print(f"  seed {seed}: trap probes {len(traps)}; with one bridge entity {len(one_bridge)}; whose five source "
              f"records all carry it {carried} (in Record.extra {in_extra} of {5 * len(one_bridge)} source records, "
              f"in the subject or object slot {in_slot})")
        absent = [p for p in corpus.probes if p.probe_type == ProbeType.ABSENT]
        supplied = Counter(sum(any(f.role == role and f.value == p.recall.component(role) for f in p.fragments)
                               for role in ALL_ROLES) for p in absent)
        print(f"  seed {seed}: absent probes {len(absent)}, by how many of the recall's five components their "
              "evidence supplies: " + ", ".join(f"{k}: {supplied[k]}" for k in range(max(supplied) + 1)))
        by_type = Counter(p.probe_type.value for p in corpus.probes)
        raw_by_type = Counter(probes[pid].probe_type.value for pid in raw)
        print(f"  seed {seed}: probes whose question names a raw, tenant-specific subject: "
              + ", ".join(f"{t} {raw_by_type[t]} of {by_type[t]}" for t in types)
              + f"; their responses in the {len(seed_names)} conditions: {seed_raw_responses}")
        for t in types:
            raw_probes[t] += len(seed_names) * raw_by_type[t]
            all_probes[t] += len(seed_names) * by_type[t]
    print(f"responses whose question names a raw, tenant-specific subject: {sum(raw_responses.values())} of "
          f"{n_responses} (" + ", ".join(f"{t} {raw_responses[t]}" for t in types if raw_responses[t]) + ")")
    print(f"probes of the {len(names)} conditions whose question names a raw, tenant-specific subject: "
          + ", ".join(f"{t} {raw_probes[t]} of {all_probes[t]}" for t in types))
    print("regeneration matches the released data" if all_same else "regeneration DIFFERS from the released data")
    return 0 if all_same else 1


def main():
    ap = argparse.ArgumentParser(description="Generation procedure of the paper's conditions.")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one condition")
    r.add_argument("condition", choices=sorted(CONDITIONS))
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--stub", action="store_true", help="use the deterministic stub instead of a model")
    c = sub.add_parser("check", help="regenerate the corpora and compare them with the released data")
    c.add_argument("data", type=Path)
    args = ap.parse_args()
    return run(args) if args.command == "run" else check(args)


if __name__ == "__main__":
    sys.exit(main())
