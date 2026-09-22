#!/usr/bin/env bash
# Push code (local = source of truth) to the compute server, or pull results back.
#   ./sync.sh push | pull
# Remote is read from .remote (gitignored), e.g.:  REMOTE=user@host  PORT=22  DIR=~/work/exp
set -euo pipefail
cd "$(dirname "$0")"
source .remote
SSH="ssh -p ${PORT:-22}"
case "${1:-push}" in
  push) rsync -az -e "$SSH" --delete \
          --exclude .git --exclude '*.md' --exclude .remote --exclude sync.sh --exclude __pycache__ \
          --exclude runs/ --exclude data/ --exclude hf/ --exclude cache/ --exclude venv/ --exclude tmp/ --exclude compat/ --exclude logs/ --exclude results/ \
          ./ "${REMOTE}:${DIR}/" ;;
  pull) rsync -az -e "$SSH" --exclude 'work/' "${REMOTE}:${DIR}/runs/" results/runs/ ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
