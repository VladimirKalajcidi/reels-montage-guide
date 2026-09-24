#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REELS_SERVER_HOST:-}" ]]; then
  echo "Set REELS_SERVER_HOST, for example: export REELS_SERVER_HOST=user@server"
  exit 1
fi

if [[ -z "${REELS_SERVER_DIR:-}" ]]; then
  echo "Set REELS_SERVER_DIR, for example: export REELS_SERVER_DIR=/srv/reels_good"
  exit 1
fi

LOCAL_ROOT="${REELS_LOCAL_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCAL_VIDEOS_DIR="${REELS_LOCAL_VIDEOS_DIR:-$LOCAL_ROOT/videos/}"
LOCAL_META_FILE="${REELS_LOCAL_META_FILE:-$LOCAL_ROOT/webapp/data/video-meta.json}"
SSH_PORT_ARGS=()

if [[ -n "${REELS_SERVER_PORT:-}" ]]; then
  SSH_PORT_ARGS=(-e "ssh -p ${REELS_SERVER_PORT}")
fi

echo "Syncing videos from ${LOCAL_VIDEOS_DIR} to ${REELS_SERVER_HOST}:${REELS_SERVER_DIR}/videos/"
rsync -az --delete "${SSH_PORT_ARGS[@]}" \
  --include='*/' \
  --include='*_edit.mp4' \
  --include='*_hook*.mp4' \
  --include='source.txt' \
  --include='montage-plan*.md' \
  --exclude='*' \
  "${LOCAL_VIDEOS_DIR}" \
  "${REELS_SERVER_HOST}:${REELS_SERVER_DIR}/videos/"

if [[ -f "${LOCAL_META_FILE}" ]]; then
  echo "Syncing metadata ${LOCAL_META_FILE}"
  rsync -az "${SSH_PORT_ARGS[@]}" \
    "${LOCAL_META_FILE}" \
    "${REELS_SERVER_HOST}:${REELS_SERVER_DIR}/webapp/data/video-meta.json"
fi

echo "Done."
