"""Audio augmentation for wakeword training data.

The wakeword classifier learns its decision boundary from whatever
embeddings we hand it.  Without augmentation, both classes ride in the
"clean TTS in a vacuum" region of feature space — so the classifier
under-confidence-fires on real-mic-in-a-real-room speech (typically
0.4-0.5 instead of 0.85+).

This module provides cheap, pure-numpy/scipy effects we use to:

  1. Roughen the TTS positives so they look like real-mic audio
     (reverb + light SNR mix + small pitch jitter).
  2. Pitch-shift user-recorded samples ±2 / ±4 semitones so 10
     recordings turn into 50 augmented variants covering a wider
     fundamental-frequency range (synthetic speaker diversity).

All functions operate on int16 mono numpy arrays at SAMPLE_RATE.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, resample_poly

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------- pitch shift

def pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Cheap pitch-shift via rational resample.

    Note: this also TIME-stretches the clip — +4 semitones makes the
    audio ~21% shorter (and higher pitched), -4 semitones makes it
    ~26% longer (and lower).  Caller is expected to pad/crop back to
    its target length afterwards.  For wakeword training where the
    feature extractor only consumes the last 1.28 s of context, that
    side effect is fine; the keyword stays intelligible up to ±5
    semitones.
    """
    if abs(semitones) < 0.01:
        return audio
    ratio = 2.0 ** (semitones / 12.0)
    # resample_poly(x, up, down) produces output of length len(x)*up/down.
    # We want frequencies × ratio (pitch up = ratio>1), which means
    # compressing time → output length = N / ratio.  So up/down = 1/ratio.
    if ratio >= 1.0:
        up, down = 1000, int(round(1000 * ratio))
    else:
        up, down = int(round(1000 / ratio)), 1000
    resampled = resample_poly(audio.astype(np.float32), up, down)
    return np.clip(resampled, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------- reverb

def _make_small_room_ir(rng: np.random.Generator,
                         sr: int = SAMPLE_RATE,
                         max_delay_s: float = 0.25,
                         decay_s: float = 0.12) -> np.ndarray:
    """Synthetic small-room impulse response.

    Models direct path + 5-7 discrete early reflections + an
    exponentially-decaying diffuse noise tail.  Randomized per-call via
    `rng` so each augmented sample sees a slightly different "room"
    geometry — gives the classifier exposure to many acoustics rather
    than memorizing one fake room.
    """
    n = int(sr * max_delay_s)
    ir = np.zeros(n, dtype=np.float32)
    ir[0] = 1.0  # direct sound

    n_refl = int(rng.integers(5, 8))
    for _ in range(n_refl):
        delay_ms = float(rng.uniform(5, 60))
        idx = min(n - 1, int(delay_ms / 1000 * sr))
        amp = float(rng.uniform(0.15, 0.45))
        sign = 1.0 if rng.random() > 0.5 else -1.0
        ir[idx] += amp * sign

    tail = rng.standard_normal(n).astype(np.float32)
    envelope = np.exp(-np.arange(n) / (decay_s * sr))
    ir += tail * envelope * 0.08

    peak = float(np.abs(ir).max())
    if peak > 1e-6:
        ir /= peak
    return ir


def apply_reverb(audio: np.ndarray, rng: np.random.Generator,
                  wet: float = 0.35) -> np.ndarray:
    """Convolve `audio` with a synthetic small-room IR.

    `wet` is the dry/wet mix ratio: 0 = pass-through, 1 = fully
    reverberated.  Default 0.35 sounds like a small office or kitchen.
    """
    if wet <= 0.001:
        return audio
    ir = _make_small_room_ir(rng)
    dry = audio.astype(np.float32)
    wet_sig = fftconvolve(dry, ir, mode="same")
    out = (1.0 - wet) * dry + wet * wet_sig
    return np.clip(out, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------- noise

def mix_noise(audio: np.ndarray, rng: np.random.Generator,
               snr_db: float) -> np.ndarray:
    """Mix Gaussian white noise into `audio` at the given SNR.

    For "light" augmentation use SNR 20-30 dB (barely audible noise).
    For "realistic room" use 10-20 dB.  Below 10 dB starts to swamp
    the keyword.
    """
    f = audio.astype(np.float32) / 32768.0
    sig_p = float(np.mean(f * f)) + 1e-12
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_p = float(np.mean(noise * noise)) + 1e-12
    target_n_p = sig_p / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_n_p / noise_p)
    out = f + noise
    return np.clip(out * 32768.0, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------- combined

def augment_positive(audio: np.ndarray, rng: np.random.Generator,
                      *,
                      pitch_semitones: float | None = None,
                      reverb_prob: float = 0.7,
                      noise_prob: float = 0.6,
                      pitch_jitter_prob: float = 0.3,
                      ) -> np.ndarray:
    """Apply a random combination of effects to a positive sample.

    The defaults are tuned for "TTS-to-real-mic" domain crossing —
    every augmented copy gets at least one effect on average.
    """
    out = audio
    # 1. pitch — either a forced value (e.g. ±4 for user-sample
    # variants) or a small random jitter for TTS positives
    if pitch_semitones is not None:
        out = pitch_shift(out, pitch_semitones)
    elif rng.random() < pitch_jitter_prob:
        out = pitch_shift(out, float(rng.uniform(-1.5, 1.5)))
    # 2. reverb
    if rng.random() < reverb_prob:
        wet = float(rng.uniform(0.20, 0.45))
        out = apply_reverb(out, rng, wet=wet)
    # 3. noise
    if rng.random() < noise_prob:
        snr = float(rng.uniform(15.0, 28.0))
        out = mix_noise(out, rng, snr_db=snr)
    return out
