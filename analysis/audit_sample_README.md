# Manual scorer audit

`audit_sample.csv` holds the 100 responses of the manual audit of the HARDENED scorer (§7.4, R3; Appendix F), with the manual labels.

## Sample

`manual_scorer_audit.py` drew the sample with seed 42 from the 14,400 responses, half from each model:

| Stratum | Rows |
|---|---|
| cross-boundary probes the scorer marks INVALID | 40 |
| cross-boundary probes it does not (SAFE, ABSTAIN or PARSE_FAIL) | 20 |
| supported probes | 20 |
| absent or partial probes | 20 (11 absent, 9 partial) |

Running `python3 analysis/manual_scorer_audit.py --out "$(mktemp -d)/sample.csv"` writes the same 100 rows without labels, into a temporary folder outside the repository.

## Columns

`audit_id`, `model`, `condition`, `probe_id`, `probe_type`, `rung` identify the response. `scorer_decision` is the scorer's verdict: INVALID or SAFE for a cross-boundary probe with a recall, RECALL for another probe with a recall, ABSTAIN, PARSE_FAIL or NO_MATCH (see Labels). `recall_subject` to `recall_time` are the recall, and `fragment_summary` lists the retrieved fragments as `[role@boundary=sanitized value]`, with each value cut at 40 characters and the list cut at 200 characters (marked `...`). `human_decision` is the manual label, `agree` is 1 when it equals `scorer_decision` and 0 otherwise, and `human_notes` holds optional notes.

## Labels

| Label | Meaning |
|---|---|
| INVALID | cross-boundary probe: the recall combines fragments of two or more boundaries |
| SAFE | cross-boundary probe: the recall stays within one boundary |
| ABSTAIN | the model declined |
| PARSE_FAIL | the parser rejected the output; every parse failure in the released responses is a recall with a missing or null component |
| RECALL | supported, absent or partial probe: the model produced a recall |
| NO_MATCH | no recall, no abstention and no parse error (the scorer's fallback) |

The released labels were assigned with the scorer's decision visible, so the audit measures consistency with the scorer, not independent validation. They agree with the scorer on 100 of 100 rows (Cohen's kappa 1.000; `python3 analysis/audit_score.py`, output in `audit_score.log`).

## Labeling the sample again

`audit_labeler.py` labels a copy of the sample and hides the scorer's decision unless `--show-scorer` is given. The copy goes into a temporary folder outside the repository:

```bash
tmp=$(mktemp -d)
cp analysis/audit_sample.csv "$tmp/relabel.csv"
python3 analysis/audit_labeler.py "$tmp/relabel.csv" --clear
python3 analysis/audit_score.py --csv "$tmp/relabel.csv"
```
