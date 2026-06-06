#!/usr/bin/env bash
# Sparklers Wakeword Studio — container entrypoint.
#
# On every container start:
#   1. Symlink the baked-in Piper voices into the runtime data dir so
#      the user's volume mount doesn't shadow them.
#   2. Make sure the writable dirs exist on the mounted volumes.
#   3. exec the CMD (uvicorn by default).

set -euo pipefail

log() { echo "[entrypoint] $*"; }

DATA_DIR=${SPARKLERS_DATA_DIR:-/app/data}
MODELS_DIR=${SPARKLERS_MODELS_DIR:-/app/models}
BAKED_VOICES=${SPARKLERS_PIPER_VOICES:-/opt/sparklers-ww/piper_voices}

mkdir -p "${DATA_DIR}/train/positive" "${DATA_DIR}/train/negative" \
         "${DATA_DIR}/piper_voices" "${MODELS_DIR}"

# Seed the volume's piper_voices/ from the baked-in copy on first start
# (idempotent: only copies missing files).
if [ -d "${BAKED_VOICES}" ]; then
    log "syncing baked Piper voices → ${DATA_DIR}/piper_voices"
    for f in "${BAKED_VOICES}"/*; do
        name=$(basename "$f")
        if [ ! -e "${DATA_DIR}/piper_voices/${name}" ]; then
            cp "$f" "${DATA_DIR}/piper_voices/${name}"
        fi
    done
fi

# Confirm tegrastats is available — the dashboard's system monitor needs it
if command -v tegrastats >/dev/null 2>&1; then
    log "tegrastats: available"
else
    log "WARN: tegrastats not found — system monitor panel will be empty"
fi

log "starting: $*"
cd /app
exec "$@"
