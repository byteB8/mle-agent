#!/usr/bin/env bash
# One-time server setup for the LLM server, entirely under this directory (no root, nothing on /).
#   ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
BASE=$(pwd)
PY=${PY:-$HOME/wrf/wrfEnv/bin/python}   # any Python 3.11 interpreter; only used to create the venv
mkdir -p tmp logs compat

# 1. vLLM in its own venv (it pins its own torch; keep it out of shared envs)
[ -x venv/bin/python ] || "$PY" -m venv venv
TMPDIR=$BASE/tmp PIP_CACHE_DIR=$BASE/tmp/pipcache \
  venv/bin/pip install --timeout 180 --retries 10 --progress-bar off vllm==0.30.0

# 2. CUDA 13 forward-compat driver libs (host driver is older than the CUDA torch was built for)
if [ ! -d compat/x/usr/local/cuda-13.0/compat ]; then
  DEB=cuda-compat-13-0_580.95.05-0ubuntu1_amd64.deb
  curl -sfL -o "compat/$DEB" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/$DEB"
  dpkg-deb -x "compat/$DEB" compat/x && rm "compat/$DEB"
fi

# 3. model weights
HF_HOME=$BASE/hf venv/bin/python -c \
  "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-Coder-30B-A3B-Instruct')"

LD_LIBRARY_PATH=$BASE/compat/x/usr/local/cuda-13.0/compat venv/bin/python -c \
  "import torch; assert torch.cuda.is_available(); print('ok', torch.__version__, torch.cuda.get_device_name(0))"
