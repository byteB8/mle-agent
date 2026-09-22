#!/usr/bin/env bash
# Start/stop the local LLM server (vLLM, OpenAI-compatible) on one GPU, bound to localhost.
#   ./serve.sh [gpu|auto] [port]      auto = highest free GPU index first (3 > 2 > 1 > 0)
#   ./serve.sh stop
# Runs natively from its own venv (not Docker) so the GPU process belongs to the invoking user,
# and all caches/temp files stay under $BASE instead of the root filesystem.
set -euo pipefail
cd "$(dirname "$0")"
BASE=$(pwd)
PIDFILE=$BASE/logs/srv.pid
mkdir -p "$BASE/logs" "$BASE/cache" "$BASE/tmp"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"; sleep 5
fi
rm -f "$PIDFILE"
[ "${1:-}" = stop ] && { echo stopped; exit 0; }

PORT=${2:-8011}
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
GPU=${1:-auto}
if [ "$GPU" = auto ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 1000 {print $1}' | sort -rn | head -1)
  [ -n "$GPU" ] || { echo "no free GPU"; exit 1; }
fi

# vLLM's torch is built for CUDA 13; the host driver (565) only speaks CUDA 12.7. NVIDIA's forward-compat
# libcuda (datacenter GPUs only) bridges that without root -- see setup.sh.
COMPAT=$BASE/compat/x/usr/local/cuda-13.0/compat
[ -d "$COMPAT" ] && export LD_LIBRARY_PATH=$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$BASE/venv/bin:$PATH   # JIT kernel builds call ninja from the venv
# the host nvcc is CUDA 10.1, too old for FlashInfer JIT; use the (prebuilt-free) PyTorch sampler instead
export VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_WORKSPACE_BASE=$BASE/cache
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$BASE/hf HF_HUB_OFFLINE=1 TMPDIR=$BASE/tmp \
       VLLM_CACHE_ROOT=$BASE/cache/vllm TORCHINDUCTOR_CACHE_DIR=$BASE/cache/inductor \
       TRITON_CACHE_DIR=$BASE/cache/triton XDG_CACHE_HOME=$BASE/cache
setsid nohup venv/bin/vllm serve "$MODEL" --served-model-name coder \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 65536 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching > "$BASE/logs/srv.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started vLLM (pid $(cat "$PIDFILE")) on GPU ${GPU}, port ${PORT}; log: logs/srv.log"
