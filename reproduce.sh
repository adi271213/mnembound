#!/usr/bin/env bash
# Reruns every step of REPRODUCE.md from the released data, rewriting the
# released outputs in place, then checks the repository with verify.sh.
# The exposure check and the Figure 5 PDF depend on library versions and go
# to a temporary directory; the exposure rerun is compared with the released
# rows. With --paper it also builds the paper (needs latexmk) into the
# temporary directory and reports its page count.
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

step() { echo; echo "== $*"; }

step "1. package tests";         python3 -m pytest -q mnembound/mnembound_pkg/tests
step "2. verifier tests";        python3 bcesv/test_bcesv.py
step "3. Table 1, Appendix A";   python3 analysis/rescore_final_ci.py > analysis/rescore_final_ci.log
step "4. scorer sensitivity";    python3 analysis/scorer_sensitivity.py > analysis/scorer_sensitivity.log
step "5. parse failures";        python3 analysis/parse_fail_sensitivity.py > analysis/parse_fail_sensitivity.log
step "6. case studies";          python3 analysis/qualitative_cases.py > analysis/qualitative_cases.md
step "7. scorer audit";          python3 analysis/audit_score.py > analysis/audit_score.log
step "8. corpus statistics";     python3 analysis/corpus_stats.py > analysis/corpus_stats.log
step "9. derived counts";        python3 analysis/paper_numbers.py > analysis/paper_numbers.txt
step "10. exposure check (rerun to a temporary file, compared with the released rows)"
python3 analysis/exposure_check.py --out "$tmp/exposure_check.csv" > "$tmp/exposure_check.log"
python3 analysis/exposure_check.py --compare "$tmp/exposure_check.csv"
step "11. BCESV";                python3 bcesv/bcesv_run_all.py > bcesv/bcesv_run_all.log
step "12. Figure 5 (counts; the PDF goes to a temporary file)"
python3 analysis/make_figure_robustness.py --out "$tmp/fig_robustness.pdf" > analysis/make_figure_robustness.log
step "13. corpus regeneration and prompts"
(cd mnembound/mnembound_pkg && python3 run_paper.py check ../../data > run_paper_check.log)

if [[ "${1:-}" == "--paper" ]]; then
    step "14. paper"
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir="$tmp/paper" paper/main.tex > "$tmp/latexmk.out"
    # The log is not UTF-8 (pdfTeX writes some characters as single bytes), so it is read byte-wise.
    LC_ALL=C tr -d '\n' < "$tmp/paper/main.log" | LC_ALL=C grep -o "Output written on [^)]*)" \
        | LC_ALL=C sed -E 's|on .*/main\.pdf|on main.pdf|'
fi

step "verify"
./verify.sh
