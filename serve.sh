#!/usr/bin/env bash
# Start the local LLM server (vLLM, OpenAI-compatible) on one GPU, bound to localhost.
#   ./serve.sh [gpu] [port]
set -euo pipefail
GPU=${1:-1}
PORT=${2:-8011}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
HF=${HF:-$HOME/work/exp/hf}

docker rm -f srv >/dev/null 2>&1 || true
docker run -d --name srv --gpus "device=${GPU}" --ipc=host \
  -p 127.0.0.1:${PORT}:8000 -v "${HF}:/root/.cache/huggingface" -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:latest \
  --model "${MODEL}" --served-model-name coder \
  --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching
echo "started srv on GPU ${GPU}, port ${PORT}; logs: docker logs -f srv"
