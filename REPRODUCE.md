# Reproducing the results

Every step runs from the repository root on the released data and needs no GPU; `README.md` describes the environment in which the models produced the responses. Each step lists the values to compare with the paper.

- Steps 3 to 9, 11 and 13 rewrite released outputs, and rerunning them leaves those outputs byte-identical.
- Step 10 writes a rerun of the exposure check into a temporary folder and compares it with the released rows, whose MiniLM part depends on the library versions listed in `README.md`.
- Step 12 writes Figure 5 into a temporary folder and rewrites its log, which comes out byte-identical; the PDF's bytes depend on the matplotlib version and the fonts installed.
- Steps 1 and 2 write nothing, and step 14 builds the paper in a temporary directory.

`./reproduce.sh` runs steps 1 to 13 in order, with the outputs of step 10 and the PDF of step 12 going to a temporary directory, then runs `./verify.sh`. On a MacBook Pro with an Apple M4 Pro and 24 GB of memory, with the all-MiniLM-L6-v2 model already downloaded, four runs took 57 to 75 seconds.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./verify.sh
```

`verify.sh` prints `verify.sh: all checks passed`.

## 1. Package tests (§6)

```bash
python3 -m pytest -q mnembound/mnembound_pkg/tests
```

`137 passed`.

## 2. Verifier tests (§8)

```bash
python3 bcesv/test_bcesv.py
```

`16 tests passed`.

## 3. Table 1 and the per-cell rates of Appendix A

```bash
python3 analysis/rescore_final_ci.py > analysis/rescore_final_ci.log
```

The printed table is byte-identical to the log of the original analysis, `analysis/rescore_final_ci.log`. The script also writes `analysis/rescore_final_ci.csv` (the 48 cells) and `analysis/table1.txt`:

| | Safe rungs (24 cells) | Unsafe rungs (24 cells) |
|---|---|---|
| Cell-min | 0/100 = 0.0% [0.0, 3.7] | 97/100 = 97.0% [91.5, 99.0] |
| Cell-max | 0/100 = 0.0% [0.0, 3.7] | 100/100 = 100.0% [96.3, 100.0] |
| Pooled | 0/2400 = 0.0% [0.0, 0.2] | 2378/2400 = 99.1% [98.6, 99.4] |

with Wilson 95% intervals: every safe-rung cell is at 0% and the unsafe-rung cells are at 97% to 100% (§7.1, §9, §10 and §11). `table1.txt` adds that 14 of the 24 unsafe-rung cells are at 100/100.

## 4. Scorer sensitivity (§7.4, R1; Appendix B)

```bash
python3 analysis/scorer_sensitivity.py > analysis/scorer_sensitivity.log
```

Median unsafe-rung rate over the 24 cells: EXACT 91.5%, HARDENED 100.0%, LOOSE 100.0%; safe-rung median 0.0% under all three, a gap of at least 91.5 percentage points. Mistral shared-vector cells: 39% to 68% under EXACT, 97% to 100% under HARDENED. Components of the unsafe-rung trap recalls that equal no fragment's value but occur in a fragment's text, which HARDENED matches and EXACT does not: Qwen 1,129 of 2,985 (37.8%) on shared-vector and 463 of 2,980 (15.5%) on summary-merge, Mistral 1,761 of 3,000 (58.7%) and 555 of 2,985 (18.6%) (R1 and §8).

## 5. Parse failures (§7.4, R2; Appendix C)

```bash
python3 analysis/parse_fail_sensitivity.py > analysis/parse_fail_sensitivity.log
```

92 of the 14,400 responses (0.64%) are parse failures, 86 of them on cross-boundary probes. All 92 are JSON recalls with a missing or null component: the object in all 92, the outcome in 73 and the time in 72. Counting every parse failure as boundary-invalid, no safe-rung cell exceeds 19.0% (Mistral, seed 42, cautious prompt, isolated rung, with either aggregator); the unsafe rungs of the same conditions are at 97.0% to 100.0%, a gap of at least 78.0 percentage points.

## 6. Case studies (§7.3, Appendix E)

```bash
python3 analysis/qualitative_cases.py > analysis/qualitative_cases.md
```

Qwen `crossboundary-000005` (boundaries b-B, b-D, b-E, b-H, b-I) and Mistral `crossboundary-000002` (b-A, b-B, b-D, b-G, b-H), five boundaries each, every component supplied by exactly one fragment.

## 7. Scorer audit (§7.4, R3; Appendix F)

```bash
python3 analysis/audit_score.py > analysis/audit_score.log
```

Agreement 100/100, Cohen's kappa 1.000; the output is byte-identical to the log of the original analysis. The labels were assigned with the scorer's decision visible, so the audit measures consistency with the scorer, not independent validation. `analysis/audit_sample_README.md` describes the sample and the labels, and how to label a copy of it again, in a temporary folder, without seeing the scorer's decision.

## 8. Corpus statistics (Appendix I)

```bash
python3 analysis/corpus_stats.py > analysis/corpus_stats.log
```

Rewrites `data/<condition>/corpus_stats.json`. All 1,200 cross-boundary probes of the twelve folders carry a bridge context; the 300 distinct trap probes (100 per seed) span 2 to 5 boundaries, 4.17 on average (13, 41, 129 and 117 probes span 2, 3, 4 and 5). In 246 of the 300 traps (84, 72 and 90 on seeds 42, 43 and 44) no fragment's text names the trap's bridge entity, and in the other 54 exactly one fragment's text does.

## 9. Counts derived in the text, Figures 3 and 4, Tables 4 and 6

```bash
python3 analysis/paper_numbers.py > analysis/paper_numbers.txt
```

Among others:

- §5.3: the isolated and boundary-preserving rungs receive identical contexts for 3,600 of 3,600 probe pairs; the safe rungs keep 23.4%, 33.4% and 30.6% of a trap's five source fragments on seeds 42, 43 and 44.
- §7.1: trap responses without a recall, 98 of 2,400 on the safe rungs and 10 of 2,400 on the unsafe rungs; supported probes with a recall, 4,800 of 4,800; supported recalls whose five components come from their own event's fragments when letter case is ignored, 3,401 of 3,600 (94.5%: isolated 96.6%, shared-vector 96.8%, summary-merge 90.0%, which is 33.8% when letter case counts); on the safe rungs, 616 of the 2,400 trap contexts hold two or more events of the querying tenant and 604 of the 2,302 recalls combine such events; on the unsafe rungs, 2,388 of 2,400 trap responses combine two or more events.
- Figure 3: Qwen shared-vector 597/600 (99.5%), summary-merge 591/600 (98.5%); Mistral 597/600 (99.5%), 593/600 (98.8%); safe rungs 0/600.
- Figure 4: median absent-probe abstention 30.0% (neutral) and 96.0% (cautious) over 16 seed-42 cells; median boundary-invalid rate 100.0% and 98.5% over 8 unsafe-rung cells. §7.2: absent-probe abstention per condition 8.0% to 53.5% over the 8 neutral conditions (8.5% to 49.5% over the 4 on seed 42) and 90.5% to 98.0% over the 4 cautious ones, with no cautious cell below 80.0% (§9); the cautious prompt changes 3 of 400 outcomes.
- Table 6: the unsafe-rung cells with the cautious prompt and those with the per-client-blocks aggregator are both at 97.0% to 100.0% (8 cells each).
- §7.4: the 13 span-2 probes give 43/52 boundary-invalid and 52/52 event-invalid recalls.
- §7.1 and Appendix A.1: 2,346 of the 2,390 unsafe-rung trap recalls (98.2%) have a component that matches no source-fragment value; per component, action 2,269, time 1,127, object 112, outcome 77, subject 33; 1,125 of these 1,127 times occur in the text of the trap's action fragment, and 2,079 of the 2,269 actions (91.6%) in the text of a source fragment; 2,159 of the 2,390 recalls (90.3%) consist only of components that match a source-fragment value or lie in a source fragment's text.
- Table 4: the 24 unsafe-rung cells under HARDENED (with Wilson 95% intervals), EXACT and LOOSE, with medians of 100.0%, 91.5% and 100.0%.
- The HARDENED scorer and the BCESV ground truth agree on 14,400 of 14,400 responses.

## 10. Exposure check (§7.5)

```bash
tmp=$(mktemp -d)
python3 analysis/exposure_check.py --out "$tmp/exposure_check.csv" > "$tmp/exposure_check.log"
python3 analysis/exposure_check.py --compare "$tmp/exposure_check.csv"
```

Writes the rows and the log of a rerun into a temporary folder, as `./reproduce.sh` does; on its first run the check downloads all-MiniLM-L6-v2. `--compare` prints `BM25 rows: identical to the released rows` and that the MiniLM counts of §7.5 (queries, exposed contexts, source fragments retrieved and queries returning all five, under both tie-breaking orders) are identical to the released rows, and fails otherwise. With numpy 2.2, the rerun prints RuntimeWarnings from the matrix product of the MiniLM scores (divide by zero, overflow, invalid value); they are harmless, and the outputs are unchanged.

The released rows, `analysis/exposure_check.csv`, hold one row per seed, retriever, tie-breaking order and query family, and their log, `analysis/exposure_check.log`, lists the library versions, the pooled counts, and the counts that differ between the two tie-breaking orders. BM25 over the sanitized text of each of the three corpora, top ten: claim queries give an exposed context for 208 of 300 traps (69.3%) and generic queries for 0 of 120, the same under both tie-breaking orders for every seed; no claim query returns all five source fragments of its trap, and 889 of the 1,500 source fragments are retrieved under either order (seeds 42 and 44 differ by one fragment between the orders, in opposite directions); 57 of the 120 generic queries return fewer than ten fragments. MiniLM: 0 of 300 and 0 of 120 exposed contexts under both orders, and no top ten covers all five roles; 148 (hash order) or 147 (file order) of the 1,500 source fragments are retrieved, and no query returns all five.

## 11. Post-hoc verifier (§8, Table 2, Figure 6, Appendix D)

```bash
python3 bcesv/bcesv_run_all.py > bcesv/bcesv_run_all.log
```

Writes `bcesv/bcesv_results.csv` (1,152 rows) and `bcesv/bcesv_summary.csv` (36 rows). On the shared-vector (SV) and summary-merge (SM) rungs:

| Variant | Recall | Precision | Supported flagged | Supported accepted |
|---|---|---|---|---|
| FragmentOnly | 100.0% | 99.7% / 99.4% | 100.0% | 0.0% |
| BcesvExact | 100.0% | 100.0% | 0.0% | 96.5% / 33.8% |
| BcesvNoisy(10%) | 99.4% / 98.6% | 100.0% | 0.0% | 55.8% / 19.2% |
| BcesvNoisy(20%) | 96.6% / 94.3% | 100.0% | 0.0% | 29.8% / 12.8% |
| BcesvNoisy(30%) | 94.6% / 89.2% | 100.0% | 0.0% | 16.9% / 7.5% |
| BcesvReconstruct | 100.0% | 99.7% / 99.4% | 0.0% | 96.5% / 33.8% |

`BcesvExact[casefold]` accepts 93.4% of the supported recalls of both rungs. The five single-feature ablations of BcesvReconstruct give the same rates as BcesvReconstruct. `FragmentOnly` flags 1,197 of the 1,200 shared-vector and 1,191 of the 1,200 summary-merge trap responses INVALID. The log ends with the thresholds of the cohesion rule of BcesvReconstruct: the same tenant, timestamps at most 600 s apart, an entity-set Jaccard of at least 0.5, and the same log source or incident template.

## 12. Figure 5 (§7.4)

```bash
python3 analysis/make_figure_robustness.py --out "$(mktemp -d)/fig_robustness.pdf" > analysis/make_figure_robustness.log
```

Writes Figure 5 into a temporary folder and prints the plotted counts. By bridge category: Shared IP 798/804, Malware hash 693/704, CVE 332/332, Vendor 348/352, Threat actor 207/208. By boundary span: 2: 43/52, 3: 279/284, 4: 742/744, 5: 1314/1320. Without `--out`, the script also writes into a new temporary folder; the released figure is `paper/fig_robustness.pdf`.

## 13. Corpus regeneration and prompts (§5, Appendices H and I)

```bash
cd mnembound/mnembound_pkg && python3 run_paper.py check ../../data > run_paper_check.log && cd ../..
```

Rewrites `mnembound/mnembound_pkg/run_paper_check.log`. It records the corpus configuration (10 tenants of 500 records; bridge density 5% in `Record.extra` and 1% in the subject or object slot; 100 supported, 50 absent, 50 partial and 100 cross-boundary probes; 5 distractor fragments in partial probes) and the decoding settings of `run` (temperature 0, top_p 1, at most 400 output tokens). It regenerates the corpora of seeds 42, 43 and 44: `corpus.json`, `probes.jsonl`, `fragments.store.jsonl` and `rung_stats.jsonl` are the same as the released files for all twelve conditions, and so is the context of every response. Per corpus, the bridge pools hold 50 shared IPs, 30 malware hashes, 20 CVEs, 15 vendors and 10 threat actors; 250 records carry a bridge entity in `Record.extra` and 50 in their subject or object slot, and 298, 298 and 296 records carry one or the other; each of the 100 trap probes has one bridge entity, which all five of its source records carry. The evidence of an absent probe supplies at most two of the five components of its recall (none for 35, 30 and 25 of the 50 absent probes of seeds 42, 43 and 44). The question of 3,320 of the 14,400 prompts names the raw, tenant-specific subject of an absent or partial probe (2,392 absent and 928 partial prompts; 598 of the 600 absent and 232 of the 600 partial probes of the twelve conditions); see `mnembound/mnembound_pkg/README.md`. The log ends with `regeneration matches the released data`.

## 14. The paper

```bash
latexmk -cd -pdf -outdir="$(mktemp -d)" paper/main.tex
```

Builds the 16-page paper (needs a TeX distribution with IEEEtran, pgfplots and TikZ) in a temporary directory, so that no build files are written into `paper/`. Its text is the same as that of the released `paper/main.pdf`; the PDF's bytes depend on the TeX installation. `./reproduce.sh --paper` includes this step.

## Checking a full run

`./reproduce.sh` ends with `./verify.sh`, which fails if any released file has changed. In a git checkout, `git status` also shows no changes and no new files. The library versions with which the MiniLM rows of step 10 were computed are listed in `README.md` and in `analysis/exposure_check.log`.
