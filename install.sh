#!/usr/bin/env bash
# Sparklers Wakeword Studio — one-shot installer.
#
# Run this on a fresh NVIDIA Jetson with JetPack 6.x and you'll have
# the wake-word studio running at http://<jetson-ip>:8082 in ~3 minutes
# (after the docker image pull).
#
#   curl -fsSL https://raw.githubusercontent.com/Arijit1080/sparklers-wakeword-studio/main/install.sh | bash

set -euo pipefail

GH_USER="Arijit1080"
REPO="sparklers-wakeword-studio"
COMPOSE_URL="https://raw.githubusercontent.com/${GH_USER}/${REPO}/main/docker-compose.yml"
IMAGE="ghcr.io/${GH_USER}/${REPO}:latest"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/${REPO}}"

log()   { printf "\n\033[1;34m[install]\033[0m %s\n" "$*"; }
warn()  { printf "\n\033[1;33m[install]\033[0m %s\n" "$*"; }
fail()  { printf "\n\033[1;31m[install]\033[0m %s\n" "$*"; exit 1; }

docker_run() {
    if groups | grep -q docker; then docker "$@"
    elif command -v sg >/dev/null 2>&1; then sg docker -c "docker $*"
    else sudo docker "$@"
    fi
}

# 1. sanity
log "Step 1/5 — checking host"
[ "$(uname -m)" = "aarch64" ] || fail "Only aarch64 (Jetson) is supported."
[ -f /etc/nv_tegra_release ] || fail "Not a Jetson? /etc/nv_tegra_release missing."

# 2. docker
log "Step 2/5 — Docker"
if ! command -v docker >/dev/null 2>&1; then
    log "  installing Docker…"
    curl -fsSL https://get.docker.com | sudo sh
fi

if ! groups | grep -q docker; then
    sudo usermod -aG docker "$USER"
    warn "Added '$USER' to the docker group. This session uses 'sg docker' so you don't have to log out."
fi

# 3. port
log "Step 3/5 — checking port 8082"
if ss -tln 2>/dev/null | grep -q ":8082 "; then
    fail "Port 8082 is in use. Free it before running install.sh."
fi

# 4. compose
log "Step 4/5 — fetching docker-compose.yml"
mkdir -p "${INSTALL_DIR}"
curl -fsSL "${COMPOSE_URL}" -o "${INSTALL_DIR}/docker-compose.yml"

# 5. pull + start
log "Step 5/5 — pulling image and starting Sparklers Wakeword Studio"
cd "${INSTALL_DIR}"
docker_run compose pull
docker_run compose up -d

IP=$(hostname -I | awk '{print $1}')
cat <<EOF

==============================================================
 Sparklers Wakeword Studio is starting up.

 Open in your browser:

   http://${IP:-<jetson-ip>}:8082

 First start unpacks the bundled Piper voices to the data
 volume (~5 s).  Then the home page is ready.

 Stop / start:

   docker compose -f ${INSTALL_DIR}/docker-compose.yml down
   docker compose -f ${INSTALL_DIR}/docker-compose.yml up -d

 Upgrade to a newer image:

   docker compose -f ${INSTALL_DIR}/docker-compose.yml pull
   docker compose -f ${INSTALL_DIR}/docker-compose.yml up -d

==============================================================
EOF
