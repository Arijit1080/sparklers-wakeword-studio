"""Runtime loader for the pre-baked keyword-independent negatives.

The bake step (tools/bake_negatives.py) runs at Docker build time and
writes an .npz file with voice-distractor embeddings + clean noisy-
distractor base WAVs.  This module is the runtime counterpart: the
training pipeline calls `load(voice_list)` and either gets back a
NegativesCache it can use directly, or None — in which case it falls
back to live synth.

Mismatch handling is intentionally strict.  If the bake hash and the
runtime hash don't match (someone changed DISTRACTOR_TEXTS, added a
voice, swapped the feature extractor's mel params, etc.), we'd rather
fall back to live synth than ship a model trained on stale features.
The warning is logged loudly so the cache miss isn't silent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from tools.generate_samples import (
    DISTRACTOR_TEXTS, TARGET_LEN, SAMPLE_RATE,
)
from audio.oww_features import (
    EMBED_DIM, WINDOW_FRAMES, CHUNK_SAMPLES,
)

# These must mirror the constants in tools/bake_negatives.py exactly.
# Bumping any of them invalidates the cache (which is the correct
# behavior — old image won't match new code's expectations).
_N_PER_VOICE = 80
_N_NOISY = 200
_SEED = 4242
_VERSION = 1

_LOG = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get(
    "SPARKLERS_CACHE_DIR",
    "/opt/sparklers-ww/cache"))


def _cache_signature(voice_list: list[str]) -> str:
    payload = {
        "version": _VERSION,
        "voices": sorted(voice_list),
        "distractor_texts": list(DISTRACTOR_TEXTS),
        "embed_dim": EMBED_DIM,
        "window_frames": WINDOW_FRAMES,
        "chunk_samples": CHUNK_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "target_len": TARGET_LEN,
        "n_per_voice": _N_PER_VOICE,
        "n_noisy": _N_NOISY,
        "seed": _SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


class NegativesCache:
    """In-memory view of the baked negatives.

    Attribute layout matches the .npz keys, but typed for the runtime.
    """
    def __init__(self,
                 voice_distractor_features: np.ndarray,
                 voice_distractor_voices: np.ndarray,
                 noisy_base_wavs: np.ndarray,
                 noisy_base_voices: np.ndarray,
                 n_per_voice: int,
                 voices: list[str]) -> None:
        self.voice_distractor_features = voice_distractor_features
        self.voice_distractor_voices = voice_distractor_voices
        self.noisy_base_wavs = noisy_base_wavs
        self.noisy_base_voices = noisy_base_voices
        self.n_per_voice_baked = n_per_voice
        self.voices = list(voices)

    @property
    def n_voice_distractors(self) -> int:
        return int(self.voice_distractor_features.shape[0])

    @property
    def n_noisy(self) -> int:
        return int(self.noisy_base_wavs.shape[0])

    def features_for(self, voice: str, n_wanted: int) -> np.ndarray:
        """Return up to `n_wanted` cached feature rows for `voice`.

        If the cache has fewer than `n_wanted` for that voice, returns
        all of them — caller is responsible for supplementing with
        live synth if more are required.
        """
        mask = self.voice_distractor_voices == voice
        rows = self.voice_distractor_features[mask]
        if rows.shape[0] >= n_wanted:
            return rows[:n_wanted]
        return rows


def load(voice_list: list[str]) -> NegativesCache | None:
    """Try to load the pre-baked negatives cache.

    Returns None on any failure (missing file, hash mismatch, malformed
    npz, etc.) and the caller is expected to fall back to live synth.
    """
    sig = _cache_signature(voice_list)
    path = CACHE_DIR / f"negatives_{sig}.npz"
    if not path.exists():
        _LOG.warning(
            "negatives cache MISS: %s not present (expected hash %s)",
            path, sig,
        )
        return None
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as exc:
        _LOG.warning("negatives cache: failed to load %s (%s)", path, exc)
        return None
    try:
        cached_sig = str(z["cache_hash"])
        if cached_sig != sig:
            _LOG.warning(
                "negatives cache: hash mismatch  file=%s expected=%s",
                cached_sig, sig,
            )
            return None
        cache = NegativesCache(
            voice_distractor_features=z["voice_distractor_features"]
                .astype(np.float32),
            voice_distractor_voices=z["voice_distractor_voices"],
            noisy_base_wavs=z["noisy_base_wavs"].astype(np.int16),
            noisy_base_voices=z["noisy_base_voices"],
            n_per_voice=int(z["n_per_voice"]),
            voices=[str(v) for v in z["voices"]],
        )
        _LOG.info(
            "negatives cache HIT: %d voice-distractor features + "
            "%d noisy base WAVs from %s",
            cache.n_voice_distractors, cache.n_noisy, path.name,
        )
        return cache
    finally:
        z.close()
