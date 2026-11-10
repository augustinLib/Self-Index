#!/usr/bin/env bash
set -euo pipefail

cpu=0
for option in "$@"; do
    case "$option" in
        --cpu) cpu=1 ;;
        -h|--help)
            printf '%s\n' 'Usage: bash setup.sh [--cpu]' \
                'Default: the core dependencies (BRIGHT and table runs, BM25 and local encoders).' \
                '--cpu: install the core Torch version from the CPU wheel index.'
            exit 0 ;;
        *) printf 'Unknown setup option: %s\n' "$option" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"
if [ -e .git ]; then
    git submodule update --init externals/BrowseComp-Plus externals/LongMemEval-V2
fi
[ -d venv ] || python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
if [ "$cpu" = 1 ]; then
    torch_requirement=$(sed -n '/^torch==/p' requirements.txt)
    venv/bin/python -m pip install "$torch_requirement" \
        --index-url https://download.pytorch.org/whl/cpu
fi
venv/bin/python -m pip install -e .
venv/bin/python run.py exec repo.check
if [ -z "${JAVA_HOME:-}" ] && ! command -v java >/dev/null 2>&1; then
    printf '\nBM25 needs a Java 21 runtime (JAVA_HOME or java on PATH); none was found.\n'
fi
printf '\nSetup complete. Export the service URLs listed in .env.example.\n'
printf 'List experiments: venv/bin/python run.py list\n'
