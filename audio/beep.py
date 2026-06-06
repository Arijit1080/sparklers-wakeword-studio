"""Short sine-wave beep generator + speaker playback.

Used by the headless enrollment flow to cue the speaker:
    READY beep  (high)  → "say it now"
    DONE  beep  (low)   → "stop, sample captured"
    RETRY beep  (chirp) → "that one was silent, try again"
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

BEEP_SAMPLE_RATE = 16_000


def make_tone(freq_hz: float, duration_ms: int = 200,
              amplitude: float = 0.35, fade_ms: int = 10) -> np.ndarray:
    """Generate a 16-bit mono sine tone with attack/release fade to avoid
    clicks at the speaker."""
    sr = BEEP_SAMPLE_RATE
    n = int(sr * duration_ms / 1000)
    t = np.arange(n) / sr
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    fade = int(sr * fade_ms / 1000)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * 32767).astype(np.int16)


def make_chirp(f_start: float, f_end: float, duration_ms: int = 300,
               amplitude: float = 0.35) -> np.ndarray:
    """Linear-frequency sweep — used as the 'retry' cue so it's
    distinguishable from the steady READY/DONE tones."""
    sr = BEEP_SAMPLE_RATE
    n = int(sr * duration_ms / 1000)
    t = np.arange(n) / sr
    freq = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    wave = amplitude * np.sin(phase)
    fade = int(sr * 0.01)
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * 32767).astype(np.int16)


READY_BEEP = make_tone(880.0, duration_ms=180)      # A5
DONE_BEEP = make_tone(523.0, duration_ms=120)       # C5
RETRY_BEEP = make_chirp(440.0, 220.0, duration_ms=350)
COMPLETE_BEEP = np.concatenate([
    make_tone(523.0, 120), make_tone(659.0, 120), make_tone(784.0, 200)
])  # C-E-G arpeggio at the end


def play(samples: np.ndarray, device=None) -> None:
    """Play int16 mono samples through the USB codec by default.

    sounddevice's notion of "default" is whatever ALSA reports for index 37
    ("default"), which on this Jetson can drift to HDMI when no monitor's
    audio sink is attached.  We auto-pin to the USB codec instead so the
    enrollment cues always come out the right port.
    """
    if device is None:
        # local import to avoid circular module loads at audio/__init__.py time
        from .mic import find_usb_codec_output_index
        device = find_usb_codec_output_index()
    sd.play(samples, samplerate=BEEP_SAMPLE_RATE, blocking=True, device=device)
