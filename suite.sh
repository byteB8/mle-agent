#!/usr/bin/env bash
# Arms over a *suite* of tasks: within an arm every (task, seed) runs in parallel; arms run one after another
# on one LLM server, which is released at the end.
#   GPU=2 PORT=8012 [KEEP_SERVER=1] ./suite.sh "<task dirs>" "<seeds>" tag:"run.py args" [tag:"run.py args" ...]
#   e.g. GPU=2 PORT=8012 ./suite.sh "data/leaf-classification data/random-acts-of-pizza" "0" \
#          pilot_tree:"--agent tree --time-limit-min 30 --cpus 4" pilot_react:"--time-limit-min 30 --cpus 4"
set -uo pipefail
cd "$(dirname "$0")"
TASKS=$1; SEEDS=$2; shift 2
PY=${PY:-$HOME/wrf/wrfEnv/bin/python}
PORT=${PORT:-8011}
LOG=logs/suite-${PORT}.log
exec >>"$LOG" 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

say "suite start: tasks=[$TASKS] seeds=[$SEEDS] arms: $*"
./serve.sh ensure "${GPU:-auto}" "$PORT" || { say "FAILED: server did not come up"; exit 1; }
for arm in "$@"; do
  tag=${arm%%:*}; args=${arm#*:}
  say "arm $tag: $args"
  pids=()
  for t in $TASKS; do
    for s in $SEEDS; do
      # shellcheck disable=SC2086  # args are intentionally word-split
      "$PY" run.py --task "$t" --tag "$tag" --seed "$s" --base-url "http://127.0.0.1:${PORT}/v1" $args \
        > "logs/${tag}-$(basename "$t")-s${s}.log" 2>&1 &
      pids+=($!)
      sleep 2   # distinct run-id timestamps
    done
  done
  for p in "${pids[@]}"; do wait "$p"; done
  "$PY" - "$tag" <<'PYEOF'
import glob, json, sys
for f in sorted(glob.glob(f"runs/{sys.argv[1]}-*/result.json")):
    r = json.load(open(f))
    extra = f" nodes={r['nodes']} buggy={r['buggy_nodes']} best_val={r['best_val']}" if "nodes" in r else ""
    print(f"  {r['run_id']}: {r['metric']}={r['score']} stop={r['stop_reason']} t={r['elapsed_s']}s{extra}")
PYEOF
done
[ "${KEEP_SERVER:-0}" = 1 ] || ./serve.sh stop "$PORT"   # KEEP_SERVER=1: caller runs more rounds on it
say "suite done"
