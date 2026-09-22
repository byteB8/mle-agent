#!/usr/bin/env bash
# Unattended batch: make sure the LLM server is up, run several seeds of one task in parallel,
# summarise, then stop the server so the GPU is released on the shared box.
#   ./batch.sh <task_dir> <tag> "<seeds>" [extra run.py args...]
#   e.g. ./batch.sh data/dev-adult baseline "1 2 3 4" --time-limit-min 30 --max-steps 40
set -uo pipefail
cd "$(dirname "$0")"
TASK=$1; TAG=$2; SEEDS=$3; shift 3
PY=${PY:-$HOME/wrf/wrfEnv/bin/python}
PORT=${PORT:-8011}
LOG=logs/batch-${TAG}.log
exec >>"$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

# 1. wait for any in-flight venv install, retry once if vllm is still missing
while pgrep -u "$(id -un)" -f "venv/bin/pip install" >/dev/null; do sleep 30; done
if ! venv/bin/python -c "import vllm" 2>/dev/null; then
  say "vllm missing, retrying install"
  TMPDIR=$PWD/tmp PIP_CACHE_DIR=$PWD/tmp/pipcache \
    venv/bin/pip install --timeout 180 --retries 10 --progress-bar off vllm==0.30.0 >>logs/venv.log 2>&1
  venv/bin/python -c "import vllm" || { say "FAILED: vllm not installable"; exit 1; }
fi
say "vllm $(venv/bin/python -c 'import vllm;print(vllm.__version__)') ready"

# 2. start the server unless it is already answering
if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  ./serve.sh auto "$PORT"
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
    kill -0 "$(cat logs/srv.pid)" 2>/dev/null || { say "FAILED: server died"; tail -30 logs/srv.log; exit 1; }
    sleep 10
  done
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null || { say "FAILED: server not ready in 20 min"; exit 1; }
fi
say "server up; GPU processes and owners:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r pid mem; do
  echo "  pid $pid user $(ps -o user= -p "$pid") mem$mem"
done

# 3. run seeds in parallel, each in its own sandbox
pids=()
for s in $SEEDS; do
  "$PY" run.py --task "$TASK" --tag "$TAG" --seed "$s" --base-url "http://127.0.0.1:${PORT}/v1" "$@" \
    > "logs/${TAG}-s${s}.log" 2>&1 &
  pids+=($!)
  sleep 2   # distinct run-id timestamps
done
say "launched seeds: $SEEDS"
for p in "${pids[@]}"; do wait "$p"; done

# 4. summary
say "summary:"
"$PY" - "$TAG" <<'EOF'
import glob, json, sys, statistics as st
tag = sys.argv[1]
rows = [json.load(open(f)) for f in sorted(glob.glob(f"runs/{tag}-*/result.json"))]
for r in rows:
    print(f"  {r['run_id']}: score={r['score']} stop={r['stop_reason']} steps={r['steps']} "
          f"t={r['elapsed_s']}s tokens={r['total_prompt_tokens'] + r['completion_tokens']}")
sc = [r["score"] for r in rows if r["score"] is not None]
if sc:
    print(f"  n={len(sc)}/{len(rows)} valid  mean={st.mean(sc):.4f}  sd={st.pstdev(sc):.4f}  "
          f"min={min(sc):.4f}  max={max(sc):.4f}")
EOF

# 5. release the GPU
./serve.sh stop
say "done"
