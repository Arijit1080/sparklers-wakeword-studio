"""Pre-bake the keyword-INDEPENDENT negative training data at Docker
build time.

Wakeword training spends most of its wall-clock on two things that
don't depend on the keyword the user is training:

  1. Synthesizing ~480 voice-distractor utterances with Piper TTS
     (6 voices × 80 distractors each) — ~8 minutes.
  2. Computing the OWW ONNX feature vector for each of those 480
     wavs plus another 200 "noisy distractor" wavs — ~3 minutes.

We do all of that once at image-build time and ship the result in the
container.  The runtime trainer then only synthesizes the keyword
positives + multi-word hard-negatives + per-run noise-mixed variants,
and only embeds those plus the few procedural silence frames.  End
result: training drops from ~30 min to ~5 min.

Output: $SPARKLERS_CACHE_DIR/negatives_<hash>.npz containing
    cache_hash                  scalar str
    voice_distractor_features   (n_voices * N_PER_VOICE, 1536) float32
                                  fully embedded — train uses as-is
    voice_distractor_voices     (n_voices * N_PER_VOICE,) <U64
                                  which voice each row came from (for
                                  slice-by-voice when neg_per_voice is
                                  smaller than what we baked)
    noisy_base_wavs             (N_NOISY, TARGET_LEN) int16
                                  clean TTS — train re-mixes per-run
                                  noise then embeds (noise mix is fast)
    noisy_base_voices           (N_NOISY,) <U64
    n_per_voice                 scalar int
    n_noisy                     scalar int
    voices                      (n_voices,) <U64

Hash covers everything that affects sample CONTENT: voice list, the
distractor text corpus, mel/embed window params, the random seed.  If
any of those change at runtime, the loader rejects the cache and falls
back to live synth.

Run:
    PYTHONPATH=. \\
    SPARKLERS_PIPER_VOICES=/path/to/voices \\
    SPARKLERS_CACHE_DIR=/path/to/cache \\
        python3 tools/bake_negatives.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Make tools/ + audio/ importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.generate_samples import (   # noqa: E402
    DISTRACTOR_TEXTS, TARGET_LEN, SAMPLE_RATE,
)
from tools.parallel_synth import synth_voice_tasks_to_array   # noqa: E402
from audio.oww_features import (   # noqa: E402
    OWWFeatures, EMBED_DIM, WINDOW_FRAMES, CHUNK_SAMPLES,
)

VOICES_DIR = Path(os.environ.get(
    "SPARKLERS_PIPER_VOICES",
    str(ROOT / "data" / "piper_voices")))
CACHE_DIR = Path(os.environ.get(
    "SPARKLERS_CACHE_DIR",
    str(ROOT / "data" / "cache")))

# Bake-time counts.  Runtime can request fewer (we slice) but if a user
# bumps neg_per_voice above N_PER_VOICE we top up with live synth.
N_PER_VOICE = 80
N_NOISY = 200
SEED = 4242   # different from the train-time seed so we don't get the
              # same text-jitter combos as live synth would produce
VERSION = 1


def cache_signature(voice_list: list[str]) -> str:
    """Stable short hash over every input that affects the cached bytes.

    Anything that would make a baked sample look different on the
    bytes-level must be in here.  If the runtime computes a different
    signature, the loader falls back to live synth.
    """
    payload = {
        "version": VERSION,
        "voices": sorted(voice_list),
        "distractor_texts": list(DISTRACTOR_TEXTS),
        "embed_dim": EMBED_DIM,
        "window_frames": WINDOW_FRAMES,
        "chunk_samples": CHUNK_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "target_len": TARGET_LEN,
        "n_per_voice": N_PER_VOICE,
        "n_noisy": N_NOISY,
        "seed": SEED,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _voice_distractor_texts(seed: int) -> list[str]:
    """Pick N_PER_VOICE distractor texts (with replacement) for one voice.

    We seed per voice so each voice's text mix is reproducible but the
    voices don't all hit the exact same sentences in the same order.
    """
    import random
    r = random.Random(seed)
    return [r.choice(DISTRACTOR_TEXTS) for _ in range(N_PER_VOICE)]


def _noisy_base_texts(seed: int, count: int) -> list[str]:
    import random
    r = random.Random(seed)
    return [r.choice(DISTRACTOR_TEXTS) for _ in range(count)]


def main() -> int:
    voices = sorted([p.stem for p in VOICES_DIR.glob("*.onnx")])
    if not voices:
        print(f"FAIL: no Piper voices in {VOICES_DIR}/  "
              f"(did download_voices.py run yet?)", file=sys.stderr)
        return 1
    print(f"baking negatives for {len(voices)} voices: {voices}")
    print(f"  cache dir: {CACHE_DIR}")
    print(f"  voices dir: {VOICES_DIR}")

    sig = cache_signature(voices)
    out_path = CACHE_DIR / f"negatives_{sig}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"already baked: {out_path} — skipping")
        return 0

    n_workers = min(len(voices), max(1, (os.cpu_count() or 6) - 1))
    print(f"  parallel workers: {n_workers}")
    voice_dir_str = str(VOICES_DIR)

    # --- 1. voice-distractor TTS, one process per voice -----------
    t0 = time.perf_counter()
    vd_wavs: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(
                synth_voice_tasks_to_array,
                v, voice_dir_str,
                _voice_distractor_texts(SEED + i),
                SEED + i,
            ): v
            for i, v in enumerate(voices)
        }
        for f in as_completed(futs):
            vname, arr = f.result()
            vd_wavs[vname] = arr
            print(f"  TTS voice-distractor done: {vname}  "
                  f"({arr.shape[0]} utts)")
    vd_secs = time.perf_counter() - t0
    print(f"voice-distractor TTS: {vd_secs:.1f}s")

    # --- 2. noisy-distractor base TTS (clean, no noise yet) -------
    # Round-robin assign to voices so the noisy set has the same voice
    # distribution that the live train code would produce.
    t0 = time.perf_counter()
    noisy_assignments = [voices[i % len(voices)] for i in range(N_NOISY)]
    per_voice_texts: dict[str, list[str]] = {v: [] for v in voices}
    per_voice_global_idx: dict[str, list[int]] = {v: [] for v in voices}
    noisy_text_seed = SEED + 10_000
    noisy_texts = _noisy_base_texts(noisy_text_seed, N_NOISY)
    for gi, v in enumerate(noisy_assignments):
        per_voice_texts[v].append(noisy_texts[gi])
        per_voice_global_idx[v].append(gi)

    noisy_wavs = np.zeros((N_NOISY, TARGET_LEN), dtype=np.int16)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(
                synth_voice_tasks_to_array,
                v, voice_dir_str,
                per_voice_texts[v],
                SEED + 20_000 + i,
            ): v
            for i, v in enumerate(voices) if per_voice_texts[v]
        }
        for f in as_completed(futs):
            vname, arr = f.result()
            for local_k, gi in enumerate(per_voice_global_idx[vname]):
                noisy_wavs[gi] = arr[local_k]
            print(f"  TTS noisy-base done: {vname}  ({arr.shape[0]} utts)")
    print(f"noisy-base TTS:        {time.perf_counter() - t0:.1f}s")

    # --- 3. embed the voice-distractors ---------------------------
    # OWW's AudioFeatures is single-threaded ONNX, and at this scale
    # the synth wins dominate by 6×.  No parallelism gain worth the
    # extra ONNX session memory per worker — embed serially in this
    # process.
    t0 = time.perf_counter()
    extractor = OWWFeatures()
    vd_rows = []
    vd_voice_labels = []
    for v in voices:
        arr = vd_wavs[v]
        for i in range(arr.shape[0]):
            vd_rows.append(extractor.embed_clip(arr[i]))
            vd_voice_labels.append(v)
    vd_features = np.stack(vd_rows).astype(np.float32)
    print(f"voice-distractor embed: {time.perf_counter() - t0:.1f}s  "
          f"shape={vd_features.shape}")

    # --- 4. write .npz --------------------------------------------
    np.savez_compressed(
        out_path,
        cache_hash=np.array(sig),
        voice_distractor_features=vd_features,
        voice_distractor_voices=np.array(vd_voice_labels),
        noisy_base_wavs=noisy_wavs,
        noisy_base_voices=np.array(noisy_assignments),
        n_per_voice=np.array(N_PER_VOICE),
        n_noisy=np.array(N_NOISY),
        voices=np.array(voices),
    )
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"\n[OK] wrote {out_path}  ({sz:.1f} MB)")
    print(f"     hash={sig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
