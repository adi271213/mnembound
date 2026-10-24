#!/usr/bin/env bash
# Checks the release: the set of files and every file's SHA-256 against
# MANIFEST.sha256, and the size of every condition in data/. Exits non-zero
# on any mismatch.
set -euo pipefail
cd "$(dirname "$0")"

status=0
fail() { echo "FAIL: $*"; status=1; }

expected=$(cut -c67- MANIFEST.sha256)
actual=$(./generate_manifest.sh --list)
missing=$(LC_ALL=C comm -23 <(echo "$expected") <(echo "$actual"))
extra=$(LC_ALL=C comm -13 <(echo "$expected") <(echo "$actual"))
[[ -z "$missing" ]] || fail "listed in MANIFEST.sha256 but missing: $(echo $missing)"
[[ -z "$extra" ]] || fail "present but not listed in MANIFEST.sha256: $(echo $extra)"

if command -v sha256sum >/dev/null 2>&1; then sha=(sha256sum); else sha=(shasum -a 256); fi
if ! "${sha[@]}" -c --quiet MANIFEST.sha256; then
    fail "files whose SHA-256 differs from MANIFEST.sha256 are listed above"
fi
echo "manifest: $(wc -l < MANIFEST.sha256 | tr -d ' ') files checked"

lines() { wc -l < "$1" | tr -d ' '; }
conditions=(data/*/)
[[ ${#conditions[@]} -eq 12 ]] || fail "expected 12 conditions in data/, found ${#conditions[@]}"
for c in "${conditions[@]}"; do
    c=${c%/}
    for f in corpus.json corpus_stats.json fragments.store.jsonl probes.jsonl responses.jsonl rung_stats.jsonl; do
        [[ -f "$c/$f" ]] || fail "$c/$f is missing"
    done
    [[ -f "$c/responses.jsonl" && $(lines "$c/responses.jsonl") -eq 1200 ]] || fail "$c: expected 1200 responses"
    [[ -f "$c/probes.jsonl" && $(lines "$c/probes.jsonl") -eq 300 ]] || fail "$c: expected 300 probes"
    [[ -f "$c/fragments.store.jsonl" && $(lines "$c/fragments.store.jsonl") -eq 25000 ]] || fail "$c: expected 25000 fragments"
    [[ -f "$c/rung_stats.jsonl" && $(lines "$c/rung_stats.jsonl") -eq 4 ]] || fail "$c: expected 4 rung summaries"
    for t in supported:100 cross_boundary:100 absent:50 partial:50; do
        n=$(grep -c "\"probe_type\": \"${t%%:*}\"" "$c/probes.jsonl" || true)
        [[ "$n" -eq "${t##*:}" ]] || fail "$c: expected ${t##*:} ${t%%:*} probes, found $n"
    done
done
echo "data: ${#conditions[@]} conditions checked"

if [[ $status -ne 0 ]]; then
    echo "verify.sh: FAILED"
    exit 1
fi
echo "verify.sh: all checks passed"
