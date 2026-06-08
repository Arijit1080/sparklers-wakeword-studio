"""Multiprocessing worker for parallel Piper TTS synthesis.

Wakeword training runs ~1080 Piper synth calls in a single train.  Each
call takes ~1 second on the Jetson Orin CPU, so a serial pass burns
~18 minutes just on TTS.  Piper itself is single-threaded (each voice is
its own ONNX session), but each voice is independent of the others —
so the natural parallelism unit is "one voice per worker process".  We
load the voice once per worker and feed it every task assigned to that
voice; with 6 voices on a 6-core Orin we get a ~5× speedup on the TTS
phase.

Used by web/state.py's _train_worker for both keyword positives and
multi-word hard negatives, and by tools/bake_negatives.py at Docker
build time for the keyword-independent negatives.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path


def synth_voice_tasks(voice_name: str, voices_dir: str,
                       tasks: list, seed: int) -> int:
    """Synthesize a batch of (out_path, text) tasks for a single voice.

    Args:
        voice_name: stem of the Piper model, e.g. "en_US-amy-medium"
        voices_dir: directory containing {voice_name}.onnx and .json
        tasks: list of (out_path_str, text) tuples
        seed: random seed for Piper's per-utterance tempo/noise jitter

    Returns:
        Number of files written.
    """
    # In spawn-mode the child has a clean sys.path, so make sure tools/
    # is importable before we pull in the synth helpers.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from tools.generate_samples import (
        _load_voice, _synthesize, _pad_or_crop_centered, _save_wav,
    )

    random.seed(seed)
    voice = _load_voice(Path(voices_dir), voice_name)
    for out_path, text in tasks:
        audio = _synthesize(voice, text)
        audio = _pad_or_crop_centered(audio)
        _save_wav(audio, Path(out_path))
    return len(tasks)


def synth_voice_tasks_to_array(voice_name: str, voices_dir: str,
                                texts: list, seed: int):
    """Same as synth_voice_tasks but returns an in-memory (N, TARGET_LEN)
    int16 array instead of writing files.  Used by bake_negatives.py so
    we can collect everything in the parent process for an .npz dump
    without touching the filesystem from each worker.
    """
    import numpy as np
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from tools.generate_samples import (
        _load_voice, _synthesize, _pad_or_crop_centered, TARGET_LEN,
    )

    random.seed(seed)
    voice = _load_voice(Path(voices_dir), voice_name)
    out = np.zeros((len(texts), TARGET_LEN), dtype=np.int16)
    for i, text in enumerate(texts):
        audio = _synthesize(voice, text)
        audio = _pad_or_crop_centered(audio)
        out[i] = audio
    return voice_name, out


def embed_wav_batch(wav_paths: list):
    """Embed a batch of WAVs into (N, WINDOW_FRAMES * EMBED_DIM) float32.

    OWW's AudioFeatures defaults to single-threaded ONNX (ncpu=1), so
    running N of these in separate processes gives ~Nx throughput on the
    Jetson's 6-core Orin CPU.  Each worker pays a ~2-3s startup cost
    loading the mel + embedding ONNX sessions; amortized over hundreds
    of clips per batch that's fine.

    Pin per-worker thread counts to 1 before importing onnxruntime so
    six workers don't each try to use six cores at once.
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import numpy as np
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from audio.oww_features import OWWFeatures, WINDOW_FRAMES, EMBED_DIM
    from audio.mic import load_wav

    extractor = OWWFeatures()
    out = np.zeros((len(wav_paths), WINDOW_FRAMES * EMBED_DIM),
                    dtype=np.float32)
    for i, p in enumerate(wav_paths):
        out[i] = extractor.embed_clip(load_wav(str(p)).samples)
    return out
