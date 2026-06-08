# DEVLOG — Sparklers Wakeword Studio

A running diary of what we changed, why, and what the numbers looked
like.  Reads top-down, newest entry last.

---

## 2026-06-08  Cut training from 30 min → 3 min (10× speedup)

### The starting point

A "train wakeword" run on the Jetson Orin Nano Super (8 GB, JetPack 6.2)
took ~30 minutes from `POST /api/train/start` to `models/<kw>.joblib`
landing on disk.  Way too slow for the studio's iterate-and-listen loop.

Profiling the existing `_train_worker` in `web/state.py`, the breakdown
on the default training config (`n_per_voice=50`, `neg_per_voice=80`,
6 baked Piper voices, 200 silence + 200 noisy + 300 hard-negative
augmentation) looked roughly like:

| Phase                                 | Count | Time     |
|---------------------------------------|------:|---------:|
| Piper synth — positives (keyword)     |   300 |  ~5 min  |
| Piper synth — voice distractors       |   480 |  ~8 min  |
| Piper synth — hard negatives          |   300 |  ~5 min  |
| Piper synth — noisy distractors (TTS) |   200 |  ~3 min  |
| Silence (procedural numpy)            |   200 |   ~0.5 s |
| ONNX feature extraction (mel + embed) |  1480 |  ~5 min  |
| sklearn `LogisticRegression.fit()`    |     1 |   <1 s   |
| **Total**                             |       | ~30 min  |

The two dominant costs were both unfortunate accidents:

1. **Serial Piper TTS** — every utterance was generated in a single
   for-loop on one core, even though the Orin has six.  Each Piper voice
   is independent (its own ONNX session) so this is embarrassingly
   parallel by voice.
2. **All work redone every train** — most of the negatives don't depend
   on the user's chosen keyword.  Voice distractors and noisy distractors
   are read from a static `DISTRACTOR_TEXTS` list and synthesized by the
   same six baked-in Piper voices.  Same inputs every time → same outputs
   every time → no need to recompute on every train.

### What we changed

**(1) Pre-baked negatives at Docker build time** — `tools/bake_negatives.py`

A new script that runs as a Docker build step (Step 3.5 in the
`Dockerfile`).  It synthesizes the 480 voice-distractor WAVs + 200
clean noisy-distractor TTS base WAVs, and pre-computes the OWW
mel+embedding feature vectors for the 480 voice distractors.  Output is
a single `negatives_<hash>.npz` at `/opt/sparklers-ww/cache/`.

The hash is a SHA-256 over the inputs that affect cache content (voice
list, the distractor text corpus, mel/embed window params, the seed).
At runtime, `tools/negatives_cache.py` recomputes the hash and either
loads the npz or falls back to live synth.  No silent staleness.

Build-time cost: ~6.5 min, once, cached by Docker as long as the four
files copied in Step 3.5 don't change.  Image bloat: 11.3 MB.

**(2) Parallel Piper synth** — `tools/parallel_synth.py`

`ProcessPoolExecutor` with spawn context, one process per voice.  Each
worker loads its Piper voice once and runs every task assigned to it
(positives + hard negatives).  Goes from ~600 serial Piper calls to 6
parallel streams of ~100 each.

We use spawn (not fork) so the workers don't inherit the live-listen
thread's ONNX state.  Costs ~1s/worker once at start; insignificant
against the ~100s of TTS each one then runs.

**(3) Parallel feature extraction** — `embed_wav_batch()`

Same `ProcessPoolExecutor` pattern, but for the embed phase.  Each
worker creates its own `OWWFeatures` instance and chews through a chunk
of WAV paths.  OWW's `AudioFeatures` already defaults to single-threaded
ONNX (`ncpu=1`), so 5-6 workers on a 6-core Orin scale near linearly.

We pin `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1`
in each worker before `import numpy`, so numpy operations inside
`load_wav` and `embed_clip` don't try to use 6 cores per process and
thrash with the other 5 workers.

### Iteration story (the bugs we hit)

**Bug 1 — embed worker count clamped to 1.**

First attempt at the embed dispatch:
```python
n_embed_workers = min(
    max(1, len(pos_wavs) + len(neg_wavs) > 0),   # <— bug
    max(1, (os.cpu_count() or 6) - 1),
    6,
)
```
The first argument computes `len(...) > 0` (a boolean) before the
`max(1, ...)`, so `True` becomes `1`.  `min(1, 5, 6)` = 1, single
worker, embed phase ran serial.  Symptom: the test run took 406 s, but
`[train] embed: ...` reported only `1 workers`.

Fix: drop the count check entirely — fewer WAVs than workers just
produces fewer batches, the worker pool handles that fine.

**Bug 2 — environment vars set after numpy was imported.**

Workers do `os.environ.setdefault("OMP_NUM_THREADS", "1")` before
`import numpy`.  `setdefault` is a no-op if the env var was inherited
from the parent.  In our case the parent didn't set them, so setdefault
worked, but it's a footgun for future changes — switched to direct
assignment to make the intent explicit.

### Numbers after the change

Measured on Jetson Orin Nano Super 8 GB, JetPack 6.2.  Keyword `"hey
aurora"` (two words → exercises the hard-negative path).

```
[train] negatives cache HIT — 480 VD feats, 200 noisy base WAVs
[train] parallel synth: 600 samples in 107.6s (~6/s)
[train] embed: 71.8s  (pos=300 live_neg=700 cached_VD=480, 5 workers)
=== TOTAL TRAIN TIME: 184s (~3 min 4 sec) ===
```

| Phase                | Before  | After   | Speedup |
|----------------------|--------:|--------:|--------:|
| Piper synth          | ~21 min |  108 s  |  ~12×   |
| ONNX embed           | ~5 min  |   72 s  |  ~4×    |
| Other (silence/fit)  |   ~1 s  |   ~3 s  |   ~1×   |
| **Total**            | ~30 min |  184 s  | **10×** |

A single-word keyword (no hard-negative pass) drops further to ~2 min
because the synth phase only needs the 300 positives (~50 s on 6
workers) rather than the 600 positives + hard-negs.

### What we did NOT do (and why)

- **Bake the hard-negatives.**  They depend on the keyword (`prefix`/
  `suffix` variants), so they can't be precomputed at image-build time.
- **Bake the noise-mixed noisy distractors.**  We pre-bake the *clean*
  TTS base WAVs and re-do the noise mix per-train so the noise
  realization varies run-to-run.  The mix itself is sub-millisecond.
- **GPU acceleration for Piper or OWW.**  Both are small ONNX models
  where CPU latency is dominated by per-call overhead, not compute.
  CUDA didn't help in earlier microbenchmarks and adds a memory cost
  we'd rather spend elsewhere on the 8 GB device.

### Files touched

- new: `tools/bake_negatives.py`
- new: `tools/parallel_synth.py`
- new: `tools/negatives_cache.py`
- edit: `web/state.py` (the `_train_worker` body)
- edit: `docker/Dockerfile` (added build step 3.5 + `SPARKLERS_CACHE_DIR` env)
