"""Feature extraction for the sklearn classifier path.

We piggy-back on OpenWakeWord's `AudioFeatures` helper class — it already
implements the correct rolling-buffer mel + embedding pipeline that OWW's
own classifiers consume.

For each utterance we compute the full sequence of 96-D embeddings,
take the LAST `WINDOW_FRAMES` of them (the most recent ~1.28 s of voice
context), and flatten to a 1536-D vector that sklearn can train on.

Used by:
    tools/train_keyword_sklearn.py    (offline, batch)
    apps/recognize_custom.py          (live streaming)
"""

from __future__ import annotations

import numpy as np

CHUNK_SAMPLES = 1280              # 80 ms @ 16 kHz
EMBED_DIM = 96
WINDOW_FRAMES = 16                # 1.28 s of context — matches OWW classifier IO


class OWWFeatures:
    """Thin wrapper over openwakeword.utils.AudioFeatures.

    The OWW class is stateful (it maintains internal mel and embedding
    buffers), so use one instance per stream/thread.  We expose two paths:
      • `embed_clip(audio_int16)` — offline: full clip → fixed 1536-D vector
      • `embed_chunk(chunk_int16)` — live: 80 ms chunk → push to buffer →
                                     return either None (not enough context
                                     yet) or the current 16-frame embedding
                                     window flattened to (1536,)
    """

    def __init__(self) -> None:
        from openwakeword.utils import AudioFeatures
        # Force ONNX backend (we don't have tflite-runtime configured)
        self._af = AudioFeatures(inference_framework="onnx")

    # ------- offline -------

    def embed_clip(self, audio_int16: np.ndarray) -> np.ndarray:
        """Compute embeddings for the whole clip, return last
        WINDOW_FRAMES of them flattened.  Pads with zeros if the clip is
        too short to produce that many embeddings."""
        # Reset internal state for a clean per-clip run
        self._af.reset()
        n = audio_int16.size
        # Drip the audio in 1280-sample chunks (what AudioFeatures expects)
        cursor = 0
        while cursor + CHUNK_SAMPLES <= n:
            self._af(audio_int16[cursor:cursor + CHUNK_SAMPLES])
            cursor += CHUNK_SAMPLES
        # Read the embedding buffer
        embs = self._af.get_features(WINDOW_FRAMES)   # shape (1, WINDOW_FRAMES, EMBED_DIM)
        if embs.size == 0:
            return np.zeros(WINDOW_FRAMES * EMBED_DIM, dtype=np.float32)
        flat = embs.reshape(-1).astype(np.float32)
        # pad/truncate to exactly WINDOW_FRAMES * EMBED_DIM
        target = WINDOW_FRAMES * EMBED_DIM
        if flat.size < target:
            out = np.zeros(target, dtype=np.float32)
            out[-flat.size:] = flat
            return out
        return flat[-target:]

    # ------- live -------

    def feed_chunk(self, chunk_int16: np.ndarray) -> np.ndarray | None:
        """Push 80 ms of audio into the rolling buffer.  Returns the
        current 16-frame embedding window flattened (1536,) if available,
        else None (still warming up)."""
        self._af(chunk_int16)
        embs = self._af.get_features(WINDOW_FRAMES)
        target = WINDOW_FRAMES * EMBED_DIM
        if embs.size < target:
            return None
        return embs.reshape(-1)[-target:].astype(np.float32)

    def reset(self) -> None:
        self._af.reset()
