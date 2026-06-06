"""Simple energy-based voice-activity detector.

For enrollment we only need to **trim leading/trailing silence** off short
recordings — full-blown VAD (silero, webrtcvad) is overkill and adds deps.
The streaming recognize loop will get a proper VAD later if needed.

Algorithm: split into 20 ms frames, mark frames whose RMS exceeds the
noise-floor estimate (lower 30%-percentile of all frames) by a margin,
then crop to the [first_active, last_active] range with small padding.
"""

from __future__ import annotations

import numpy as np

from .mic import Capture, SAMPLE_RATE


FRAME_MS = 20
MARGIN_DB = 9.0      # frame is "active" if RMS > noise + 9 dB
PAD_MS = 100         # keep this much silence on each side


def _frame_rms_db(samples: np.ndarray) -> np.ndarray:
    """Return one dBFS value per FRAME_MS frame."""
    n = int(SAMPLE_RATE * FRAME_MS / 1000)
    trim = (len(samples) // n) * n
    frames = samples[:trim].reshape(-1, n).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def trim_silence(cap: Capture) -> Capture:
    """Return a Capture with leading + trailing silence removed.

    If the recording is essentially silent the original is returned (with a
    note via `.peak_dbfs` showing why downstream code can skip it).
    """
    db = _frame_rms_db(cap.samples)
    if len(db) < 5:
        return cap

    noise = float(np.percentile(db, 30))
    active = db > (noise + MARGIN_DB)
    if not active.any():
        return cap   # all noise — give up, caller can re-prompt

    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))

    frame_n = int(SAMPLE_RATE * FRAME_MS / 1000)
    pad_n = int(SAMPLE_RATE * PAD_MS / 1000)
    start = max(0, first * frame_n - pad_n)
    end = min(len(cap.samples), (last + 1) * frame_n + pad_n)

    return Capture(samples=cap.samples[start:end], sample_rate=cap.sample_rate)


def is_speech(cap: Capture, threshold_db: float = -45.0) -> bool:
    """Quick gate: does the recording contain ANY audible speech?"""
    return cap.peak_dbfs > threshold_db
