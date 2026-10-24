# MnemBound: Fragment-Faithful, Event-False Recall in Multi-Tenant Agent Memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23617448.svg)](https://doi.org/10.5281/zenodo.23617448)

Artifact of the ACSAC 2026 paper by Aditya Jain, Anmol Pandey (University of Warwick) and Liang Chen (University of Hertfordshire).

A multi-tenant agent can combine authorized fragments from different tenants into one recalled event that never happened. MnemBound measures this failure on a synthetic SOC corpus with construction-time ground truth. The repository holds the benchmark generator, the 14,400 responses of the two models analysed in the paper together with their corpora and probes, the scripts that compute every number, table and figure of the paper from these data, the BCESV post-hoc verifier, and the paper itself. Reproducing the results needs no GPU.

## Quick start

Python 3.10 or newer.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./verify.sh       # every file against MANIFEST.sha256, and the size of every condition
./reproduce.sh    # every step of REPRODUCE.md, then verify.sh again
```

`reproduce.sh` recomputes the released outputs from the data, overwriting them, and the closing `verify.sh` confirms that they come out byte-identical. Two outputs depend on library versions and are written to temporary files instead: the exposure check, whose rerun is compared with the released rows, and the PDF of Figure 5. The exposure check downloads the all-MiniLM-L6-v2 model on its first run. [REPRODUCE.md](REPRODUCE.md) lists every step with its expected output.

## From the paper to the artifact

Every command runs from the repository root. The bracketed names are sections of `analysis/paper_numbers.txt`.

| Paper | Command | Output |
|---|---|---|
| Table 1; the 0 of 2,400 and 2,378 of 2,400 of the Abstract, §1, §7.1 and §11; the per-cell rates of §7.1, §9, §10 and §11 (0% in every safe-rung cell, 97% to 100% in the unsafe-rung cells); the 14 of 24 unsafe-rung cells at 100/100 | `python3 analysis/rescore_final_ci.py` | `analysis/table1.txt`; per-cell rates in `analysis/rescore_final_ci.log` and `.csv` |
| §5 and §6: tenants, records, fragments, probes per type, responses | `python3 analysis/paper_numbers.py` | [Data] |
| §5.3: safe-rung equivalence, share of trap fragments kept | `python3 analysis/paper_numbers.py` | [§5.3]; the unsafe rungs keep all five, as `data/<condition>/rung_stats.jsonl` records |
| §7.1: responses without a recall, supported recalls, the two models on the same probe, response lengths, event structure | `python3 analysis/paper_numbers.py` | [§7.1] |
| Figure 3 | `python3 analysis/paper_numbers.py` | [Figure 3] |
| Figure 4 and §7.2; §9's absent-probe abstention of at least 80% in every cautious cell; the two 97% to 100% ranges of Table 6 | `python3 analysis/paper_numbers.py` | [Figure 4, §7.2] |
| §7.3 and Appendix E | `python3 analysis/qualitative_cases.py` | `analysis/qualitative_cases.md` |
| §7.4, R1, and Appendix B, including how often each model copies a fragment's text instead of its value (R1 and §8) | `python3 analysis/scorer_sensitivity.py` | `analysis/scorer_sensitivity.log` |
| §7.4, R2, and Appendix C (Table 5), including the 92 parse failures, all of them recalls with a missing or null component | `python3 analysis/parse_fail_sensitivity.py` | `analysis/parse_fail_sensitivity.log` |
| §7.4, bridge categories and boundary spans; Figure 5 | `python3 analysis/make_figure_robustness.py` | `analysis/make_figure_robustness.log`; the PDF goes into a temporary folder, and the released figure is `paper/fig_robustness.pdf` |
| §7.4, span-2 probes and R4 (seeds) | `python3 analysis/paper_numbers.py` | [§7.4] |
| §7.4, R3, and Appendix F | `python3 analysis/audit_score.py` | `analysis/audit_score.log`; the sample by model and stratum in [Appendix F] |
| §7.5 | `python3 analysis/exposure_check.py`, which writes into a temporary folder (REPRODUCE.md, step 10) | the released run: `analysis/exposure_check.csv`, `.log` |
| §4, the FragmentOnly sentence; §8, Table 2, Figure 6 and Appendix D, the trap responses that FragmentOnly flags, and the thresholds of the cohesion rule | `python3 bcesv/bcesv_run_all.py` | `bcesv/bcesv_summary.csv`, `bcesv_results.csv`, `bcesv_run_all.log` |
| §7.1, the 300 unique trap probes; §8, how the source fragments of a trap differ | `python3 analysis/paper_numbers.py` | [§8] |
| Appendix A: Table 4, the supported-probe rates and the absent-probe abstention ranges | `python3 analysis/paper_numbers.py` | [Table 4, Appendix A], [§7.1], [Figure 4, §7.2] |
| §7.1 and Appendix A.1: components that match no source-fragment value, and how many of them a source fragment's text contains | `python3 analysis/paper_numbers.py` | [Appendix A.1] |
| Appendix H, the system and user prompts | none: rendered by `mnembound/mnembound_pkg/mnembound/agent_io.py` and `retrieval.py` | |
| Appendix H, the 3,320 prompts whose question names a raw subject (598 of the 600 absent and 232 of the 600 partial probes); Appendix I, the corpus configuration and bridge pools, the 296 to 298 records of a corpus that carry a bridge, and the traps whose five source records all carry their bridge | `cd mnembound/mnembound_pkg && python3 run_paper.py check ../../data > run_paper_check.log` | `mnembound/mnembound_pkg/run_paper_check.log` |
| Appendix I, trap probes: bridge contexts, boundary spans, and in how many of its five fragments the text names the trap's bridge | `python3 analysis/corpus_stats.py` | `data/<condition>/corpus_stats.json`, `analysis/corpus_stats.log` |
| Table 6 | restates numbers of the rows above | `analysis/table1.txt`; [Figure 4, §7.2]; `bcesv/bcesv_run_all.log` |
| §6, twelve modules and 137 unit tests | `python3 -m pytest -q mnembound/mnembound_pkg/tests` | 137 passed; the modules are listed in `mnembound/mnembound_pkg/README.md` |
| Figures 1 and 2, Algorithm 1, Table 3 | none: drawn in `paper/main.tex`; the seeds and corpus sizes in Figure 2 are those of [Data] | |
| The paper | `./reproduce.sh --paper`, or `latexmk -cd -pdf -outdir="$(mktemp -d)" paper/main.tex` | a 16-page PDF; the released one is `paper/main.pdf` |

## Layout

```
analysis/                 scoring, sensitivity, audit, exposure and figure scripts with their outputs
bcesv/                    the BCESV verifier, its driver, unit tests and outputs (BCESV_README.md)
data/                     the twelve conditions: corpus, probes, fragment store and responses
mnembound/mnembound_pkg/  generator, probes, rungs, prompts, exact scoring, runners, run_paper_check.log (README.md)
paper/                    LaTeX source, bibliography, Figure 5 and the compiled paper
README.md                 this file
REPRODUCE.md              every step of the reproduction, with its expected output
CHANGELOG.md              changes since the reviewed artifact
CITATION.cff              citation metadata
LICENSE                   Apache License 2.0
requirements.txt          Python packages for Figure 5, the exposure check and the tests
reproduce.sh              reruns every step of REPRODUCE.md
verify.sh                 checks the release against MANIFEST.sha256 and the size of every condition
generate_manifest.sh      writes MANIFEST.sha256
MANIFEST.sha256           SHA-256 of every released file
.gitignore                files that git leaves out
```

## Data

Each condition is one model, one corpus seed, one system prompt and one summary-merge aggregator:

| Folder | Model | Seed | Prompt | Aggregator |
|---|---|---|---|---|
| `qwen_full_neutral` | Qwen3-30B-A3B-Instruct-2507-FP8 | 42 | neutral | collapsed |
| `qwen_full_cautious` | Qwen3-30B-A3B-Instruct-2507-FP8 | 42 | cautious | collapsed |
| `qwen_full_pcb` | Qwen3-30B-A3B-Instruct-2507-FP8 | 42 | neutral | per-client-blocks |
| `qwen_full_cautious_pcb` | Qwen3-30B-A3B-Instruct-2507-FP8 | 42 | cautious | per-client-blocks |
| `qwen_seed43_neutral`, `qwen_seed44_neutral` | Qwen3-30B-A3B-Instruct-2507-FP8 | 43, 44 | neutral | collapsed |
| `mistral_*` (the same six) | Mistral-Small-3.2-24B-Instruct-2506-FP8 | as above | as above | as above |

Conditions with the same seed share one corpus, so the twelve folders hold three distinct corpora. Each folder contains:

| File | Lines | Content |
|---|---|---|
| `responses.jsonl` | 1,200 | one response per probe and rung (300 x 4): `probe_id`, `probe_type`, `rung`, `model`, `seed`, `value_view` (`tenant_hidden`), `tenant_visible` (false), `fragment_provenance_ids` (the handles of the fragments in the context, in retrieval order), `raw_text` (the model output), `recall` (the five parsed components, or null), `abstained`, and `abstain_reason` or `parse_error` when they apply; a response that the parser rejects (in these runs, a recall with a missing or null component) has `parse_error` set, `abstained` true and `recall` null |
| `probes.jsonl` | 300 | `probe_id`, `probe_type` (100 supported, 100 cross_boundary, 50 absent, 50 partial), `recall` (the probe's target event, tenant-visible values), `fragment_provenance_ids` (the probe's evidence), `is_fragment_supported`, `is_event_supported`, `boundary_event_pairs`, `bridge_context` (the bridge entities that the trap's source records share) |
| `fragments.store.jsonl` | 25,000 | the store's record of each fragment: `public_provenance_id`, `role`, `sanitized_value` and `sanitized_text` (what the agent saw), `tenant_id`, `boundary_id`, `event_id`, `timestamp`, `log_source`, `incident_template`, `entity_set` (the subject and object of the fragment's record, tenant-visible) |
| `corpus.json` | 21 | `seed`, `salt` (the provenance salt), `n_records`, `n_fragments`, `n_probes`, `tenants`, `n_tenant_entity_pools`, `n_sanitizer_entries` (entities the sanitizer maps), `has_bridges` |
| `rung_stats.jsonl` | 4 | per rung: fragments retrieved and excluded per probe, and the share of each probe type's fragments kept |
| `corpus_stats.json` | 37 (38 for seed 44) | trap-probe statistics written by `analysis/corpus_stats.py` |

The agent saw sanitized fragment text, in which tenant-specific entities are replaced by tokens such as `user:u_6cb60e35`. On the isolated, boundary-preserving and shared-vector rungs the prompt lists the fragments in the order of `fragment_provenance_ids`, each with the first 20 characters of its handle; on the summary-merge rung it shows no handles and sorts the fragments by tenant and role, as one list (collapsed) or one block per tenant (per-client-blocks). The question at the end of each prompt names the subject of the probe's recall. For 598 of the 600 absent probes and 232 of the 600 partial probes, the probe holds no fragment with that subject and the question names the raw, tenant-specific value instead: 3,320 of the 14,400 prompts, none of them on supported or cross-boundary probes (`mnembound/mnembound_pkg/run_paper_check.log`). The tenant, boundary, event and entity fields of the fragment store are what a deployer's trusted resolver holds; the scorer uses them as ground truth, `BcesvExact` takes the boundary from them as its resolver, and the cohesion rule of `BcesvReconstruct` uses the tenant, timestamp, entity set, log source and template.

## Runs that produced the responses

| Item | Value |
|---|---|
| Models | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` and `unsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8`, at the Hugging Face revisions current on 24 May 2026, `5a5a776300a41aaa681dd7ff0106608ef2bc90db` and `2a8994ec06d4b4353764dedd6ca492377f1d3996`; neither repository has changed since |
| Serving | vLLM 0.21 through its OpenAI-compatible API |
| Decoding | temperature 0, top_p 1, at most 400 output tokens |
| Hardware | one NVIDIA H100 SXM 80 GB |
| FP8 | CUTLASS block-scaled kernels, with vLLM's DeepGEMM kernel disabled to work around an FP8 numerical bug |
| Date | 24 May 2026 |

The generation procedure and its configuration are in `mnembound/mnembound_pkg/run_paper.py`. Its `check` command regenerates the three corpora and shows that the package reproduces the released `corpus.json`, `probes.jsonl`, `fragments.store.jsonl` and `rung_stats.jsonl` byte for byte, and the context of every released response; its `run` command sends the same prompts, with the same decoding settings, to a model behind an OpenAI-compatible server. A rerun need not reproduce the released responses exactly; every result in the paper is computed from the released responses.

## Reproducing and checking

`verify.sh` exits non-zero when a file is missing, unlisted or different from `MANIFEST.sha256`, or when a condition does not hold 1,200 responses, 300 probes (100, 100, 50, 50 by type), 25,000 fragments and 4 rung summaries. `make_figure_robustness.py` writes Figure 5 into a new temporary folder unless `--out` names a file, so that the released `paper/fig_robustness.pdf` is kept: the PDF's bytes depend on the matplotlib version and the fonts installed, while the plotted counts do not. The MiniLM rows of `analysis/exposure_check.csv` were computed on the CPU with sentence-transformers 6.2.0, transformers 5.20.2, torch 2.14.1 and numpy 2.5.4, as `analysis/exposure_check.log` records. MiniLM scores depend on these versions, so `reproduce.sh` writes its rerun of the exposure check to a temporary file and runs `python3 analysis/exposure_check.py --compare` on it, which requires the BM25 rows to be identical to the released ones and the MiniLM rows to give the counts reported in §7.5.

## Citation and license

[CITATION.cff](CITATION.cff) gives the citation; the archived copy of this repository has the DOI [10.5281/zenodo.23617448](https://doi.org/10.5281/zenodo.23617448). Code and data are released under the Apache License 2.0 ([LICENSE](LICENSE)). [CHANGELOG.md](CHANGELOG.md) lists the changes since the version reviewed for the paper.
