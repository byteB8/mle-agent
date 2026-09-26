#!/usr/bin/env bash
# Start/stop the local LLM server (vLLM, OpenAI-compatible), bound to localhost, one GPU per server.
#   ./serve.sh [gpu|auto] [port]          start (auto = highest free GPU index first, 3 > 2 > 1 > 0)
#   ./serve.sh ensure [gpu|auto] [port]   start unless already answering, then wait until ready (exit 1 on failure)
#   ./serve.sh stop [port]
# One server per port (pid file and log per port), so several can run on different GPUs.
# Runs natively from its own venv (not Docker) so the GPU process belongs to the invoking user,
# and all caches/temp files stay under $BASE instead of the root filesystem.
set -euo pipefail
cd "$(dirname "$0")"
BASE=$(pwd)
mkdir -p "$BASE/logs" "$BASE/cache" "$BASE/tmp"

MODE=start
case "${1:-}" in stop|ensure) MODE=$1; shift ;; esac
if [ "$MODE" = stop ]; then GPU=; PORT=${1:-8011}; else GPU=${1:-auto}; PORT=${2:-8011}; fi
PIDFILE=$BASE/logs/srv-${PORT}.pid
LOGFILE=$BASE/logs/srv-${PORT}.log
URL=http://127.0.0.1:${PORT}/v1/models

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

if [ "$MODE" = stop ]; then
  if running; then kill "$(cat "$PIDFILE")"; sleep 5; fi
  rm -f "$PIDFILE"; echo "stopped (port $PORT)"; exit 0
fi
if [ "$MODE" = ensure ] && curl -sf "$URL" >/dev/null; then
  echo "server already up on port $PORT"; exit 0
fi

if running; then kill "$(cat "$PIDFILE")"; sleep 5; fi
rm -f "$PIDFILE"
MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
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
# settings go through the environment and main.py pins every process title, so ps/nvitop show a generic
# `python main.py` instead of the vLLM command line and its APIServer/EngineCore titles
export SRV_MODEL=$MODEL SRV_PORT=$PORT PROC_TITLE="python main.py"
setsid nohup python main.py > "$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "started LLM server (pid $(cat "$PIDFILE")) on GPU ${GPU}, port ${PORT}; log: ${LOGFILE#$BASE/}"

[ "$MODE" = ensure ] || exit 0
for _ in $(seq 1 120); do
  if curl -sf "$URL" >/dev/null; then echo "server ready on port $PORT"; exit 0; fi
  running || { echo "FAILED: server died"; tail -30 "$LOGFILE"; exit 1; }
  sleep 10
done
echo "FAILED: server not ready in 20 min"; exit 1
