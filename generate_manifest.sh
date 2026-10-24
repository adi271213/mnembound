#!/usr/bin/env bash
# Writes MANIFEST.sha256, the SHA-256 of every released file.
# With --list, prints the released files instead; verify.sh uses that list.
set -euo pipefail
cd "$(dirname "$0")"

# Released files: everything except version control, local environments,
# caches, LaTeX build files and the manifest itself.
list_files() {
    find . \( -name .git -o -name .venv -o -name venv -o -name env -o -name __pycache__ \
              -o -name .pytest_cache -o -name .vscode -o -name .idea -o -name '*.egg-info' \) -prune -o -type f \
         ! -name MANIFEST.sha256 ! -name .DS_Store ! -name '._*' ! -name '*.pyc' ! -name '*.pyo' ! -name '*.swp' \
         ! -path './paper/*.aux' ! -path './paper/*.bbl' ! -path './paper/*.blg' ! -path './paper/*.log' \
         ! -path './paper/*.out' ! -path './paper/*.fls' ! -path './paper/*.fdb_latexmk' \
         ! -path './paper/*.synctex.gz' -print | sed 's|^\./||' | LC_ALL=C sort
}

if [[ "${1:-}" == "--list" ]]; then
    list_files
    exit 0
fi

if command -v sha256sum >/dev/null 2>&1; then sha=(sha256sum); else sha=(shasum -a 256); fi
list_files | tr '\n' '\0' | xargs -0 "${sha[@]}" > MANIFEST.sha256
echo "MANIFEST.sha256: $(wc -l < MANIFEST.sha256 | tr -d ' ') files"
