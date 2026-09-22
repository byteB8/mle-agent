#!/usr/bin/env bash
# Start the local LLM server (vLLM, OpenAI-compatible) on one GPU, bound to localhost.
#   ./serve.sh [gpu|auto] [port]      auto = highest free GPU index first (3 > 2 > 1 > 0)
set -euo pipefail
PORT=${2:-8011}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
HF=${HF:-$HOME/work/exp/hf}

docker rm -f srv >/dev/null 2>&1 || true
GPU=${1:-auto}
if [ "$GPU" = auto ]; then
  sleep 3  # let a replaced server release its memory
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 1000 {print $1}' | sort -rn | head -1)
  [ -n "$GPU" ] || { echo "no free GPU"; exit 1; }
fi
docker run -d --name srv --gpus "device=${GPU}" --ipc=host \
  -p 127.0.0.1:${PORT}:8000 -v "${HF}:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:v0.30.0 \
  --model "${MODEL}" --served-model-name coder \
  --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching
echo "started srv on GPU ${GPU}, port ${PORT}; logs: docker logs -f srv"
