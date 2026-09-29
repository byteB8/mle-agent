#!/usr/bin/env bash
# Push code (local = source of truth) to the compute server, or pull results back.
#   ./sync.sh push | data | pull
# Remote is read from .remote (gitignored), e.g.:  REMOTE=user@host  PORT=22  DIR=~/work/exp
#
# `push` sends an explicit whitelist of code paths and only deletes *inside* the code directories, so
# server-side state (venvs, model weights, caches, data, runs, logs, ...) can never be removed by a sync,
# whatever it is called. (A mirror-with-excludes version once wiped a new server-only venv.)
set -euo pipefail
cd "$(dirname "$0")"
source .remote
SSH="ssh -p ${PORT:-22}"
case "${1:-push}" in
  push) rsync -az -e "$SSH" --delete --exclude __pycache__ core env "${REMOTE}:${DIR}/"   # code dirs: exact mirror
        rsync -az -e "$SSH" ./*.py ./*.sh "${REMOTE}:${DIR}/" ;;                          # top-level scripts: no delete
  data) rsync -az -e "$SSH" --info=stats1 data/ "${REMOTE}:${DIR}/data/" ;;   # task dirs (built locally by prep.py)
  pull) rsync -az -e "$SSH" --exclude 'work/' "${REMOTE}:${DIR}/runs/" results/runs/ ;;
  *) echo "usage: $0 push|data|pull"; exit 1 ;;
esac
