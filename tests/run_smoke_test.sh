#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
W="${1:-$(mktemp -d "${TMPDIR:-/tmp}/gbm_chst11_smoke.XXXXXX")}"
mkdir -p "$W"
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"
echo "building synthetic data in $W ..."
"$PY" tests/make_mock_data.py --out "$W" > "$W/make_mock.log" 2>&1 || {
  echo "mock generation failed:"; tail -20 "$W/make_mock.log"; exit 1; }
export GBM_DATA="$W/data" GBM_RESULTS="$W/results" GBM_LOGS="$W/logs" GBM_AXIS_MIN_SAMPLES=5
bash run_all.sh --quick --jobs 2 --python "$PY" --only "18 19 20 21 22 25 26 27 28 30"
rc=$?
echo; echo "smoke-test outputs: $W/results/robustness   (exit $rc)"
exit $rc
