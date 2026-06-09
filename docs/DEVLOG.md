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

---

## 2026-06-08  Fix the "0.4 - 0.5 misses" — TTS aug + optional user voice

### The symptom

After shipping the 10× speedup the user reported real-mic misses: even
when they clearly said the keyword the classifier scored 0.4-0.5, often
sitting just below threshold and never firing.  Lowering the threshold
to 0.3 didn't help.  Bumping `suggested_threshold` on the model didn't
help either.

### The cause

Pure TTS-trained classifier domain gap.  AUC on the held-out 10% TTS
eval slice was **1.000** — the model has learned to discriminate the
keyword perfectly *for studio-clean Piper voices in a vacuum*.  Real
human speech through a USB codec into a small room lives outside that
distribution: the embedding lands in a region the classifier has no
training data for, so it falls back to ~0.5 (the LR prior).

This is the same domain-mismatch failure mode that every TTS-trained
wakeword has to fix one way or another.  Two complementary moves:

  1. **Roughen the TTS positives** so the positive class covers
     "reverb-stained, lightly noisy, slightly mistuned audio" rather
     than only "studio-clean".  Pure data augmentation, no user input.
  2. **Add a few real recordings of the user**, pitch-shifted to
     synthesize additional speaker fundamentals.  Optional but huge.

### What we built

**`tools/audio_augment.py`** — three pure-numpy/scipy effects:

  - `pitch_shift(audio, semitones)` — rational `resample_poly` based.
    Also time-stretches; caller re-pads/crops to TARGET_LEN.  Works
    cleanly up to about ±5 semitones.
  - `apply_reverb(audio, rng, wet)` — convolves with a synthetic
    small-room IR (direct + 5-7 randomized early reflections +
    exponentially-decaying noise tail).  IR is `rng`-derived so each
    augmented sample sees a slightly different "room".
  - `mix_noise(audio, rng, snr_db)` — Gaussian white noise at a given
    SNR.  Reused for both noisy distractors (10-15 dB) and light
    positive jitter (20-28 dB).

Plus `augment_positive(audio, rng)` — random combination of the three
with probability-weighted defaults tuned for "TTS-to-real-mic".

**`web/state.py` train flow** now runs *after* the parallel Piper synth:

  - Each TTS positive gets one augmented copy (300 → 600 positives).
    Pass is ~0.9 s of pure-numpy work — embed picks both up.
  - If `data/train/user_pos/<kw_safe>/sample_NN.wav` exists, each user
    sample is pitch-shifted to {-4, -2, 0, +2, +4} semitones and each
    variant gets a small reverb/noise pass on top.  10 recordings →
    50 augmented positives in ~0.5 s.

**`audio/mic.py` `record_blocking`** — patched the same way as the
voiceclone studio's `record_clip`: query native input channels, open
at that count, mix to mono in software.  Avoids the PaErrorCode -9998
"Invalid number of channels" failure on USB codecs that refuse mono.

**Optional UI flow** on `/train`:

  - New "🎙 Record yourself" card above the train form
  - `POST /api/train/record_samples` (background thread, status updates
    via the existing SSE bus)
  - `POST /api/train/clear_user_samples`
  - `GET /api/train/user_samples?keyword=X` — count for the keyword

State machine added a new `state="recording"` between `idle` and
`training`.  Listening/training endpoints already refuse non-idle
states so no extra guards needed.

Sample sequence: READY beep (A5, 180 ms) → 1.6 s capture → DONE beep
(C5, 120 ms) → 1 s pause × N samples, COMPLETE beep at the end.
WAVs persist under `/app/data/train/user_pos/<kw_safe>/` across
trains until the user clicks "Clear my samples".

### Numbers after the change

| Train scenario | Time | Positives | Embed batches |
|---|---:|---:|---:|
| Pre-aug, no user samples (yesterday)    | 188 s | 300 | 1000 wavs |
| TTS-aug only, no user samples           | 206 s | 600 | 1300 wavs |
| TTS-aug + 3 user recs (test_rec, smoke) | 163 s | 375 | 1075 wavs |
| TTS-aug + 10 user recs (typical)        | ~220 s (est) | 650 | 1350 wavs |

The +18 s vs. pre-aug is the augmentation pass (~1 s) plus the doubled
positive count flowing through the parallel embed (~+17 s).  Still well
under the original 30 min, still under 4 min total.

Recognition impact won't show in the train log — only in real-mic
testing.  The expected shift on the user's own voice is from 0.4-0.5
to 0.85+ with 10 user samples; other speakers in the same room should
land 0.70+.

### Files touched

- new: `tools/audio_augment.py`
- new: API endpoints in `web/app.py` for record/clear/count
- new: "Record yourself" card + JS in `web/templates/train.html`
- edit: `web/state.py` — added `start_recording_samples` +
  `_record_worker` + augmentation pass in `_train_worker` + new
  `state="recording"` + `rec_*` fields on ServiceStatus
- edit: `audio/mic.py` — `record_blocking` opens at native channels
  and down-mixes to mono

---

## 2026-06-09  Tuning the precision/recall knob until it actually worked

Yesterday's "TTS aug + optional user voice" shipped but in a real
room with a real mic it failed in two opposite ways depending on the
day:

* prefix false-fire: trained "hey krishna", saying just "hey" lit it
* under-confident: real-mic "hey krishna" scored 0.3-0.5 → missed

Today was the messy debug arc that landed on settings that actually
hold up.  No silver bullet — a half-dozen small dials, each pulling
the boundary one way or the other.

### What we tried, in order

1. **Per-model threshold from the trained `suggested_threshold`.**
   The model already picks an F1-optimal threshold during training; the
   listener was ignoring it and using a single global.  Fixed the
   listener to use the model's own value.  Helped slightly — but
   `suggested_threshold` was landing at 0.10 because the eval set
   separated trivially.  Added a floor (0.30, later 0.50) so a too-
   low suggestion couldn't fire on phoneme noise.
2. **Per-model UI override.**  Added per-model threshold inputs on the
   home page with persistent storage in `_thresholds.json`.  User
   asked us to tear it back out — "one threshold is enough."  Ripped.
3. **Bumped hard-negative counts** — `N_HARD_PREFIX` 150 → 300, plus
   the same augmentation pass on hardnegs as we'd been doing on
   positives.  Stopped the prefix false-fire for a while, then it
   came back as soon as we re-enabled user-voice training.
4. **Threshold selection changed** from "lowest F1-tie" to "highest
   F1-tie with eval precision ≥ 0.98".  Less aggressive thresholds.
5. **Removed `class_weight="balanced"`.**  "Balanced" was upweighting
   the (fewer) positives ~3×, tipping the boundary into negative
   space and causing the "fires on random speech" failure.  Dropped to
   `class_weight=None` with `C=0.5` for slightly more regularization.
6. **Dropped the TTS-positive augmentation entirely.**  The aug copies
   were expanding positive variety without an equivalent negative
   expansion, widening the positive region.  Kept the negative
   augmentation in place — net effect: "reverb + noise" carries no
   class signal anymore.
7. **User-voice pitch shifts: from ±4 → ±2 → none.**  Each step
   helped a different failure mode and broke a different one.  ±4
   garbled "krishna" past the embedding window → phantom "hey alone is
   positive" examples.  ±2 was less destructive but still
   over-emphasized the user's voice.  Zero pitch shifts meant the
   model only fired on the trained speaker.
8. **The actual answer for user-voice: `sample_weight=0.4`.**  Use
   the recordings as-is (no pitch shifts, no aug copies) but weight
   them at 40% in `clf.fit`.  10 user samples × 0.4 = 4 effective
   positives against 300 TTS × 1.0 = 300.  User voice is ~1.3% of
   the positive signal — enough to anchor real-mic acoustics for
   hard-to-pronounce names, not enough to dominate the boundary.
9. **Multi-person append-mode recording.**  The original record flow
   wiped prior samples on every click.  Changed it to find the next
   unused filename index and append, so 3-sample batches can stack:
   person A → swap → person B → swap → person C → 9 samples covering
   3 voices.  Added a "How many" input (default 3) to the UI.
10. **Mic gain tuning.**  100% (+31 dB) on the Waveshare USB codec
    looked like clipping when recording for training, so we dropped
    to 50%.  Bad call — at 50% the live listening audio was so quiet
    the embeddings landed nowhere near the trained positive region.
    Bumped back to 100%; mild clipping at inference time is fine.

### Other small simplifications

- The home page threshold input default went 0.5 → 0.4 → 0.3 as we
  iterated.  Final: 0.3.
- Ripped per-model threshold storage, API endpoints, and UI controls.
  One global threshold, no per-model magic, no floors or caps in the
  listener — just `if score > threshold: fire`.
- Listener logic simplified from "max of (global, floor, capped
  suggestion)" to a single comparison.

### Numbers

Real-mic on a Jetson Orin Nano Super 8GB with the Waveshare USB
codec mic, threshold 0.3, mic at 100% gain:

| keyword type                   | real-mic score | "prefix alone" |
|--------------------------------|---------------:|---------------:|
| `hey krishna`, TTS-only        |       0.4-0.7  |       <0.10    |
| `hey krishna`, +10 user voice  |       0.6-0.85 |       <0.15    |
| `hey krishna`, +9 user (3 ppl) |       0.5-0.8  |       <0.10    |

### Files touched today

- edit: `web/state.py` — single-threshold listener, `class_weight=None`,
  C=0.5, dropped positive-aug pass, kept hardneg-aug pass,
  multi-person append recording, `sample_weight=0.4` for user-voice
  rows, highest-F1-tie + precision-floor threshold selection
- edit: `web/templates/home.html` — threshold default 0.3, removed
  per-model controls
- edit: `web/templates/train.html` — "How many" input, append mode
  copy, hint about multi-person batches
- removed: per-model threshold API + UI + persistence (was added and
  then yanked in the same day per user feedback)
