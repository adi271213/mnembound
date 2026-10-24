# mnembound

Benchmark generator and harness of MnemBound (§5 and §6 of the paper). It generates a synthetic multi-tenant SOC corpus with construction-time tenant, boundary and event labels, builds the four probe types, applies the four retrieval rungs, renders the agent prompts, parses the responses, and scores them with exact predicates. It has no third-party dependencies; running a model needs the `openai` client package.

## Modules

| Module | Purpose |
|---|---|
| `generator.py` | corpus generation: tenants, records, fragments, sanitized (tenant-hidden) views |
| `bridges.py` | bridge entities shared across tenants: IPs, malware hashes, CVEs, vendors, threat actors |
| `probes.py` | supported, cross-boundary, absent and partial probes |
| `retrieval.py` | the four retrieval rungs and the summary-merge renderings |
| `agent_io.py` | prompt construction, response parsing, deterministic stub agents |
| `scoring.py` | exact predicates and rates: fragment support, event support, and the event-level violation (fragment-supported, not event-supported) |
| `schema.py` | records, fragments, probes, provenance handles |
| `identifiers.py` | disjoint per-tenant identifier pools and the entity sanitizer |
| `templates.py` | the five SOC incident templates |
| `validation.py` | corpus invariants |
| `serialization.py`, `export.py` | JSON and JSONL writers and a reader, view-restricted exports |

## Tests

```bash
python3 -m pytest -q mnembound/mnembound_pkg/tests    # from the repository root
```

137 tests cover generation and its determinism, identifier and bridge invariants, probes, rungs, prompt rendering, parsing, scoring and serialization.

## The released corpora and runs

`run_paper.py` holds the configuration of the paper (`PAPER_CORPUS`: 10 tenants of 500 records, bridge density 5% in `Record.extra` and 1% in the subject or object slot, 100 supported, 100 cross-boundary, 50 absent and 50 partial probes, 5 distractor fragments in partial probes, bridge-linked traps) and the twelve conditions of `data/`.

```bash
cd mnembound/mnembound_pkg
python3 run_paper.py check ../../data > run_paper_check.log
```

records the corpus configuration and the decoding settings of `run`, regenerates the corpora of seeds 42, 43 and 44 and compares them with the released data: `corpus.json`, `probes.jsonl`, `fragments.store.jsonl` and `rung_stats.jsonl` are byte-identical for all twelve conditions, and the context recorded with every one of the 14,400 responses is the one the rung's retrieval returns. It also counts, per corpus, the records that carry a bridge entity, the trap probes whose five source records carry the trap's bridge, and the probes and responses whose question names a raw subject. Its output is released as `run_paper_check.log`.

```bash
python3 run_paper.py run qwen_full_neutral --out-dir "$(mktemp -d)" --base-url http://localhost:8000/v1
```

runs one condition the way its responses were produced: every probe on every rung, sent to the model behind an OpenAI-compatible endpoint with temperature 0, top_p 1 and at most 400 output tokens, and writes the condition's files in the format of `data/`. With `--stub`, a deterministic stub that recalls each probe's event from the evidence replaces the model. The responses in `data/` were produced with these settings by run scripts on an earlier version of this package, which generates the same corpora and probes and renders the same prompts; the model, serving and hardware settings are listed in the README at the repository root.

`run_smoke.py` runs the same pipeline on a small corpus (10 tenants of 200 records, 100 probes) and prints the package's per-rung rates, a check of the expected rung pattern, parse failures and abstentions; with `--stub` it needs no model. It writes into a new temporary folder unless `--out-dir` is given.

## Scoring

The predicates of `scoring.py` compare whole recall components with fragment values of the same role, so a paraphrased component counts as unsupported. Their `is_boundary_invalid` is event-level: a recall is invalid when every component is supplied but no single event (`boundary_id`, `event_id`) supplies all five, so a recall that combines two events of one tenant counts too. The tests and `run_smoke.py` use them. The rates reported in the paper are computed from the released responses by `analysis/rescore_final_ci.py`, whose HARDENED rule is boundary-level (the supplying fragments span two or more boundaries, the paper's Inv_B) and also accepts a fragment value inside a component and a component inside a fragment's text.

## Prompts

`agent_io.build_prompt` renders the two system prompts and the user prompt (Appendix H). With `tenant_visible=False`, the evidence is the sanitized fragment text, and the question names the sanitized subject of the probe's recall when the probe holds that subject's fragment. A probe without that fragment keeps the raw, tenant-specific subject value in its question. In the released runs this holds for 598 of the 600 absent probes and for the 232 of the 600 partial probes that lack the subject role, that is for 3,320 of the 14,400 prompts; `run_paper_check.log` reports the counts.
