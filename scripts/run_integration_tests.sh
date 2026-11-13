#!/usr/bin/env bash
# Builds and installs charmtyles + charmnumeric into the active env, then runs the integration tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHARMNUMERIC_ROOT="${ROOT}/example/charmnumeric"

python -m pip install -e "${ROOT}"
python -m pip install -e "${CHARMNUMERIC_ROOT}[tests]" -C build-dir=build/tests
python -m charmnumeric.selftest

eval "$(python -c 'import charmnumeric._build_info as b; print(f"export CHARMNUMERIC_SERVER={b.SERVER}"); print(f"export CHARMNUMERIC_CHARMRUN={b.CHARMRUN}" if b.CHARMRUN else "")')"
python -m pytest "${CHARMNUMERIC_ROOT}/tests" -m integration -v "$@"
