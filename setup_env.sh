#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 required"' || exit 1
if ! "$PY" -m venv --help >/dev/null 2>&1; then
  echo "python venv module missing: sudo apt install python3-venv"; exit 1
fi
[[ -d .venv ]] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt
fc-list 2>/dev/null | grep -qiE "arial|liberation sans" || \
  echo "tip: sudo apt install fonts-liberation   (Arial-metric font for figures)"
.venv/bin/python - <<'P'
import importlib
for m in ["numpy","pandas","scipy","anndata","scanpy","infercnvpy","decoupler","statsmodels",
          "matplotlib","gseapy","lifelines","igraph","leidenalg"]:
    v = importlib.import_module(m)
    print(f"  {m:<12} {getattr(v,'__version__','?')}")
print("environment OK")
P
