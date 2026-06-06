"""Thin wrapper around sounddevice for fixed-sample-rate mono capture.

The hotword pipeline only ever wants 16 kHz mono int16.  Centralising that
here keeps the apps/* scripts from sprinkling `sounddevice` kwargs around.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16_000   # TitaNet ingests 16 kHz; matches arecord defaults
DTYPE = "int16"


@dataclass
class Capture:
    """One recording.  `samples` is int16 mono at `sample_rate`."""

    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def peak_dbfs(self) -> float:
        peak = float(np.abs(self.samples).max()) / 32768.0
        return 20 * np.log10(peak + 1e-9)

    @property
    def rms_dbfs(self) -> float:
        s = self.samples.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(s * s)))
        return 20 * np.log10(rms + 1e-9)


def list_input_devices() -> list[dict]:
    """Return all input-capable devices as a list of summaries."""
    return [
        {
            "index": i,
            "name": d["name"],
            "channels": d["max_input_channels"],
            "default_sr": d["default_samplerate"],
        }
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


def find_usb_codec_index() -> Optional[int]:
    """Heuristic: pick the first USB-Audio-class input device."""
    for d in list_input_devices():
        n = d["name"].lower()
        if any(t in n for t in ("usb pnp", "usb audio", "sss1629", "jmtek")):
            return d["index"]
    return None


def find_usb_codec_output_index() -> Optional[int]:
    """Same heuristic, for OUTPUT side.  The USB codec usually presents the
    same device index for both directions but the lookup is symmetric for
    safety against weird driver layouts."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] <= 0:
            continue
        n = d["name"].lower()
        if any(t in n for t in ("usb pnp", "usb audio", "sss1629", "jmtek")):
            return i
    return None


def record_blocking(duration_s: float, device: Optional[int] = None) -> Capture:
    """Block for `duration_s` seconds, return the recording as a Capture.

    Uses sounddevice's `rec` which is a one-shot capture — fine for short
    enrollment clips.  For continuous streaming the recognize loop will use
    `InputStream` directly.
    """
    nframes = int(duration_s * SAMPLE_RATE)
    buf = sd.rec(
        nframes,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=DTYPE,
        device=device,
        blocking=True,
    )
    # sd.rec returns shape (frames, channels); flatten to (frames,)
    return Capture(samples=buf.reshape(-1), sample_rate=SAMPLE_RATE)


def record_with_meter(
    duration_s: float,
    device: Optional[int] = None,
    on_chunk=None,
    chunk_ms: int = 50,
) -> Capture:
    """Like record_blocking but invokes `on_chunk(rms_dbfs)` every chunk_ms
    so the caller can render a live VU meter.
    """
    nframes_total = int(duration_s * SAMPLE_RATE)
    chunk_frames = int(SAMPLE_RATE * chunk_ms / 1000)
    out = np.zeros(nframes_total, dtype=np.int16)
    written = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=DTYPE,
        device=device,
        blocksize=chunk_frames,
    ) as stream:
        t_end = time.monotonic() + duration_s
        while written < nframes_total and time.monotonic() < t_end:
            need = min(chunk_frames, nframes_total - written)
            block, _ = stream.read(need)
            block = block.reshape(-1)
            out[written : written + len(block)] = block
            written += len(block)
            if on_chunk is not None:
                s = block.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(s * s) + 1e-12))
                on_chunk(20 * np.log10(rms + 1e-9))

    return Capture(samples=out[:written], sample_rate=SAMPLE_RATE)


def save_wav(cap: Capture, path: str) -> None:
    """Write the capture to a 16-bit PCM WAV at the recorded sample rate."""
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(cap.sample_rate)
        w.writeframes(cap.samples.astype(np.int16).tobytes())


def load_wav(path: str) -> Capture:
    """Read a 16-bit PCM mono WAV.  Resamples if not 16 kHz."""
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert sw == 2, f"only 16-bit WAV supported, got {sw*8}-bit"
    samples = np.frombuffer(raw, dtype=np.int16)
    if nch == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        # cheap linear resample; good enough for offline files
        from scipy.signal import resample_poly
        samples = resample_poly(samples.astype(np.float32), SAMPLE_RATE, sr)
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return Capture(samples=samples, sample_rate=SAMPLE_RATE)
