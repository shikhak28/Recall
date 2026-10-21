#!/usr/bin/env bash
# One-shot setup.  ./setup.sh        -> auto (GPU build if nvidia-smi works)
#                  ./setup.sh cpu    -> force CPU build
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
if [[ "${1:-auto}" != "cpu" ]] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo ">> NVIDIA GPU found: installing GPU build"
  .venv/bin/pip install -q -r requirements-gpu.txt
  # fastembed/rapidocr may pull the CPU onnxruntime too; two onnxruntimes in one env break CUDA.
  .venv/bin/pip uninstall -y -q onnxruntime || true
  .venv/bin/pip install -q --force-reinstall --no-deps "onnxruntime-gpu>=1.21"
else
  echo ">> CPU build"
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python - <<'PY'
from recall import config
print("CUDA available to Recall:", config.use_cuda(), "| OCR workers:", config.default_workers())
PY
[[ -f .env ]] || cp .env.example .env
echo ">> Done. Put your NEBIUS_API_KEY in .env, then:  .venv/bin/streamlit run app.py"
