# syntax=docker/dockerfile:1.7
#
# Sparklers Wakeword Studio — Dockerfile
# --------------------------------------
# Targets NVIDIA Jetson Orin / Xavier on JetPack 6.x (L4T R36.x).
# Layers are ordered slowest-to-change → fastest-to-change so editing
# Python code rebuilds in seconds, not minutes.

ARG L4T_TAG=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ----- 1. System packages -----
# We only need ALSA + numpy linkers + tzdata.  No GStreamer (audio-only),
# no opencv, no torchvision deps.
# tegrastats is shipped with the L4T BSP at /usr/bin/tegrastats — not an
# apt package on its own.  The base l4t-jetpack image already has it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-setuptools python3-wheel \
        libopenblas0-pthread libsndfile1 \
        alsa-utils libasound2-plugins libportaudio2 \
        tzdata ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Sanity: confirm tegrastats is on PATH so the dashboard's sys monitor works.
RUN command -v tegrastats >/dev/null 2>&1 \
    || echo "WARN: tegrastats not in image — sys monitor will be empty"

# ----- 2. Python deps (split for layer cache) -----
# numpy first so pandas/scipy etc compile against a known version
RUN pip3 install --upgrade pip && \
    pip3 install 'numpy<2'

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# ----- 3. Pre-download OpenWakeWord pretrained models + Piper voices -----
# Bakes ~400 MB into the image so the container is fully offline-capable
# from the first run.
COPY tools/download_voices.py /tmp/download_voices.py
RUN mkdir -p /opt/sparklers-ww/piper_voices && \
    cd /opt/sparklers-ww && \
    python3 -c "import openwakeword; openwakeword.utils.download_models()" && \
    python3 -c "\
import sys; sys.path.insert(0, '/tmp'); \
import download_voices; \
download_voices.VOICES_DIR = __import__('pathlib').Path('/opt/sparklers-ww/piper_voices'); \
download_voices.main()"

# ----- 4. App code (LAST, edits don't bust expensive layers above) -----
WORKDIR /app
COPY audio/ /app/audio/
COPY apps/ /app/apps/
COPY tools/ /app/tools/
COPY web/   /app/web/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ----- 5. Volumes for persistent state -----
# /app/data    — training WAVs (regenerated on each train, not strictly needed but useful for debugging)
# /app/models  — trained .joblib classifiers (the only thing you really want to persist)
VOLUME ["/app/data", "/app/models"]

ENV PYTHONPATH=/app \
    SPARKLERS_DATA_DIR=/app/data \
    SPARKLERS_MODELS_DIR=/app/models \
    SPARKLERS_PIPER_VOICES=/opt/sparklers-ww/piper_voices

EXPOSE 8082

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8082"]
