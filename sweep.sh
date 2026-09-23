#!/usr/bin/env bash
# Run several experiment arms back to back on one LLM server, then release the GPU.
# Arms run sequentially so wall-clock numbers stay comparable; seeds within an arm run in parallel.
#   ./sweep.sh <task_dir> "<seeds>" <tag>:"<run.py args>" [<tag>:"<run.py args>" ...]
#   e.g. ./sweep.sh data/dev-adult "0 1 2 3 4" base120:"--max-steps 120" gate:"--max-steps 120 --min-submit-frac 0.5"
set -uo pipefail
cd "$(dirname "$0")"
TASK=$1; SEEDS=$2; shift 2
LOG=logs/sweep.log
say() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }
say "sweep start: $*"
for arm in "$@"; do
  tag=${arm%%:*}; args=${arm#*:}
  say "arm $tag: $args"
  # shellcheck disable=SC2086  # args are intentionally word-split
  KEEP_SERVER=1 ./batch.sh "$TASK" "$tag" "$SEEDS" $args
  grep -A8 "summary:" "logs/batch-${tag}.log" | tail -1 >>"$LOG"
done
./serve.sh stop >>"$LOG" 2>&1
say "sweep done"
