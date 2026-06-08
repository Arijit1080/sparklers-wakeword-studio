"""Process-wide state for the wakeword web UI.

Unifies two model backends:
    • Pretrained OWW ONNX classifiers (loaded via openwakeword.Model)
    • Custom sklearn .joblib classifiers (our own training output)

A single background thread services them all per audio chunk.  Training
runs in the same worker pool but as a separate task — the audio pipeline
yields to it until done.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.beep import DONE_BEEP, READY_BEEP, play   # noqa: E402
from audio.mic import (   # noqa: E402
    SAMPLE_RATE, find_usb_codec_index, find_usb_codec_output_index,
)
from audio.oww_features import OWWFeatures, EMBED_DIM, WINDOW_FRAMES   # noqa: E402
from web.sysmon import SystemMonitor   # noqa: E402

MODELS_DIR = ROOT / "models"
CHUNK_SAMPLES = 1280   # OWW expects 80 ms / 16 kHz


@dataclass
class ServiceStatus:
    state: str = "idle"                  # idle | listening | training | recording
    model_keys: list[str] = field(default_factory=list)
    threshold: float = 0.5
    patience: int = 2
    vad_threshold: float = 0.3
    cooldown_s: float = 2.0
    n_triggers: int = 0
    last_trigger_t: float = 0.0
    last_error: str = ""
    embed_ms_p50: float = 0.0
    # training-specific
    train_keyword: str = ""
    train_phase: str = ""
    train_progress: float = 0.0          # 0..1
    train_progress_text: str = ""
    # user-voice-recording-specific (optional flow before training)
    rec_keyword: str = ""
    rec_done: int = 0
    rec_total: int = 0
    rec_status: str = ""                 # "" | "ready" | "recording" | "pause" | "done"


class WakewordService:
    """Singleton.  Owns mic + warm models + the background worker."""

    def __init__(self) -> None:
        self.status = ServiceStatus()
        self._lock = threading.Lock()
        self._listen_thread: Optional[threading.Thread] = None
        self._train_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._oww_model = None           # openwakeword.Model for pretrained
        self._sklearn_models: dict[str, Any] = {}    # name → payload dict
        self._features: Optional[OWWFeatures] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: list[asyncio.Queue[dict]] = []
        self._triggers: deque[dict] = deque(maxlen=200)
        self._embed_times: deque[float] = deque(maxlen=50)
        self._in_device = find_usb_codec_index()
        self._out_device = find_usb_codec_output_index()
        self._ensure_oww_models()
        # System resource monitor — broadcasts cpu/ram/gpu every 1 s
        self._sysmon = SystemMonitor(
            on_update=lambda s: self.emit({"type": "sysstats", **s}),
        )
        if not self._sysmon.start():
            self.status.last_error = "tegrastats not available — no sys stats"

    # ---------- lifecycle ----------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def shutdown(self) -> None:
        self.stop_listening()
        try: self._sysmon.stop()
        except Exception: pass    # noqa: BLE001, E701

    def _ensure_oww_models(self) -> None:
        try:
            import openwakeword
            test = (Path(openwakeword.__file__).parent
                    / "resources/models/hey_jarvis_v0.1.onnx")
            if not test.exists():
                openwakeword.utils.download_models()
        except Exception as e:    # noqa: BLE001
            self.status.last_error = f"OWW model download failed: {e}"

    def _ensure_features(self) -> OWWFeatures:
        if self._features is None:
            self._features = OWWFeatures()
        return self._features

    # ---------- SSE bus ----------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=400)
        with self._lock:
            self._subscribers.append(q)
            try:
                q.put_nowait({"type": "status", **asdict(self.status)})
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def emit(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        with self._lock:
            subs = list(self._subscribers)
        if self._loop is None:
            return
        for q in subs:
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass

    def _push_status(self, **extra) -> None:
        with self._lock:
            for k, v in extra.items():
                setattr(self.status, k, v)
        self.emit({"type": "status", **asdict(self.status)})

    # ---------- model discovery ----------

    def list_models(self) -> list[dict]:
        out: list[dict] = []
        for b in ["hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy",
                  "timer", "weather"]:
            out.append({"key": b, "source": "pretrained",
                        "format": "onnx", "size_kb": None, "path": None})
        if MODELS_DIR.exists():
            for p in sorted(MODELS_DIR.glob("*.joblib")):
                meta = {}
                try:
                    payload = joblib.load(p)
                    meta = {
                        "auc": payload.get("eval", {}).get("auc"),
                        "suggested_threshold": payload.get("suggested_threshold"),
                        "n_clips": payload.get("train", {}).get("n_train_pos", 0)
                                   + payload.get("train", {}).get("n_train_neg", 0),
                    }
                except Exception:    # noqa: BLE001
                    pass
                out.append({"key": p.stem, "source": "custom",
                            "format": "joblib",
                            "size_kb": p.stat().st_size // 1024,
                            "path": str(p), **meta})
            for p in sorted(MODELS_DIR.glob("*.onnx")):
                out.append({"key": p.stem, "source": "custom",
                            "format": "onnx",
                            "size_kb": p.stat().st_size // 1024,
                            "path": str(p)})
        return out

    # ---------- live listening ----------

    def start_listening(self, models: list[str],
                          threshold: float = 0.5,
                          patience: int = 2,
                          vad_threshold: float = 0.3,
                          cooldown_s: float = 2.0) -> dict:
        if self.status.state == "listening":
            return {"ok": True, "msg": "already listening"}
        if self.status.state == "training":
            return {"ok": False, "error": "service is busy training"}

        # Classify model names: pretrained (OWW) vs sklearn .joblib
        pretrained: list[str] = []
        sklearn_models: dict[str, Any] = {}
        for name in models:
            jp = MODELS_DIR / f"{name}.joblib"
            op = MODELS_DIR / f"{name}.onnx"
            if jp.exists():
                sklearn_models[name] = joblib.load(jp)
            elif op.exists():
                pretrained.append(str(op))
            else:
                pretrained.append(name)
        try:
            self._sklearn_models = sklearn_models
            self._oww_model = None
            if pretrained:
                from openwakeword import Model
                kw = dict(inference_framework="onnx",
                          wakeword_models=pretrained)
                if vad_threshold > 0:
                    kw["vad_threshold"] = vad_threshold
                self._oww_model = Model(**kw)
            if sklearn_models:
                self._ensure_features()
        except Exception as e:    # noqa: BLE001
            return {"ok": False, "error": f"model load failed: {e}"}

        keys: list[str] = []
        if self._oww_model is not None:
            keys += list(self._oww_model.models.keys())
        keys += list(sklearn_models.keys())
        self._push_status(state="listening", model_keys=keys,
                          threshold=float(threshold), patience=int(patience),
                          vad_threshold=float(vad_threshold),
                          cooldown_s=float(cooldown_s),
                          n_triggers=0, last_error="")
        self._stop.clear()
        self._listen_thread = threading.Thread(
            target=self._listen_worker,
            args=(threshold, patience, cooldown_s),
            name="WW-Listen", daemon=True,
        )
        self._listen_thread.start()
        try: play(READY_BEEP, device=self._out_device)
        except Exception: pass    # noqa: BLE001, E701
        return {"ok": True, "models": keys}

    def stop_listening(self) -> dict:
        self._stop.set()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        self._push_status(state="idle")
        return {"ok": True}

    def _listen_worker(self, threshold: float, patience: int,
                        cooldown_s: float) -> None:
        try:
            streak: dict[str, int] = {}
            last_fire: dict[str, float] = {}
            # Reset feature buffer so we start fresh
            if self._features is not None:
                self._features.reset()
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                device=self._in_device, blocksize=CHUNK_SAMPLES,
            ) as stream:
                # Track recent RMS so the VAD gate looks at the WHOLE 1.28 s
                # window, not just the current 80 ms chunk.  Without this,
                # a single voiced chunk followed by 15 silent chunks would
                # still pass — yet the embedding is dominated by silence.
                rms_history: deque[float] = deque(maxlen=WINDOW_FRAMES)
                # dBFS below which we consider the window "no speech"
                VAD_GATE_DBFS = -42.0

                while not self._stop.is_set():
                    block, _ = stream.read(CHUNK_SAMPLES)
                    block = block.reshape(-1)
                    t0 = time.monotonic()

                    # Window-level VAD gate
                    f = block.astype(np.float32) / 32768.0
                    block_rms = float(np.sqrt(np.mean(f * f) + 1e-12))
                    block_dbfs = 20.0 * np.log10(block_rms + 1e-9)
                    rms_history.append(block_dbfs)
                    # window is "silent" if EVERY chunk is below the gate
                    window_silent = (
                        len(rms_history) == WINDOW_FRAMES
                        and max(rms_history) < VAD_GATE_DBFS
                    )

                    all_scores: dict[str, float] = {}

                    if self._oww_model is not None:
                        oww_scores = self._oww_model.predict(block)
                        for k, v in oww_scores.items():
                            all_scores[k] = 0.0 if window_silent else float(v)

                    if self._sklearn_models and self._features is not None:
                        feat = self._features.feed_chunk(block)
                        if feat is not None and not window_silent:
                            feat2 = feat.reshape(1, -1)
                            for name, payload in self._sklearn_models.items():
                                clf = payload["model"]
                                p = float(clf.predict_proba(feat2)[0, 1])
                                all_scores[name] = p
                        elif feat is None:
                            # warming up the feature buffer
                            pass
                        else:
                            # silent window: pin sklearn scores to 0
                            for name in self._sklearn_models:
                                all_scores[name] = 0.0

                    pred_ms = 1000.0 * (time.monotonic() - t0)
                    self._embed_times.append(pred_ms)
                    if self._embed_times:
                        self.status.embed_ms_p50 = float(np.median(self._embed_times))

                    now = time.monotonic()
                    top_key, top_val = "", -1.0
                    for k, v in all_scores.items():
                        if v > top_val: top_key, top_val = k, v
                        if v > threshold:
                            streak[k] = streak.get(k, 0) + 1
                        else:
                            streak[k] = 0
                        if (streak[k] >= patience
                                and now - last_fire.get(k, 0) > cooldown_s):
                            last_fire[k] = now
                            streak[k] = 0
                            self.status.n_triggers += 1
                            self.status.last_trigger_t = time.time()
                            evt = {"type": "trigger", "key": k, "score": v,
                                   "n": self.status.n_triggers,
                                   "ts": time.time()}
                            self._triggers.appendleft(evt)
                            self.emit(evt)
                            try: play(DONE_BEEP, device=self._out_device)
                            except Exception: pass    # noqa: BLE001, E701
                    self.emit({"type": "scores",
                               "top_key": top_key,
                               "top": top_val if top_val >= 0 else 0,
                               "all": all_scores,
                               "embed_ms": pred_ms})
        except Exception as e:    # noqa: BLE001
            self._push_status(state="idle",
                               last_error=f"{type(e).__name__}: {e}")
            self.emit({"type": "error", "msg": str(e)})
        finally:
            self._push_status(state="idle")

    def triggers(self, limit: int = 100) -> list[dict]:
        return list(self._triggers)[:limit]

    # ---------- model management ----------

    def delete_model(self, name: str) -> dict:
        """Delete a custom model (.joblib).  Refuses if the service is
        busy listening or training."""
        if self.status.state != "idle":
            return {"ok": False,
                    "error": f"can't delete while {self.status.state} — "
                             f"stop first"}
        safe = name.strip().lower().replace(" ", "_")
        if not safe.replace("_", "").isalnum():
            return {"ok": False, "error": "invalid name"}
        deleted = []
        for ext in ("joblib", "onnx"):
            p = MODELS_DIR / f"{safe}.{ext}"
            if p.exists():
                try:
                    p.unlink()
                    deleted.append(p.name)
                except OSError as e:
                    return {"ok": False, "error": str(e)}
        if not deleted:
            return {"ok": False, "error": "no such model"}
        return {"ok": True, "deleted": deleted}

    # ---------- optional: record-yourself for training ----------

    def _user_pos_dir(self, kw_safe: str) -> Path:
        return ROOT / "data" / "train" / "user_pos" / kw_safe

    def count_user_samples(self, keyword: str) -> int:
        kw_safe = keyword.strip().lower().replace(" ", "_")
        d = self._user_pos_dir(kw_safe)
        return len(list(d.glob("*.wav"))) if d.exists() else 0

    def clear_user_samples(self, keyword: str) -> dict:
        if self.status.state != "idle":
            return {"ok": False,
                    "error": f"service is busy: {self.status.state}"}
        kw_safe = keyword.strip().lower().replace(" ", "_")
        d = self._user_pos_dir(kw_safe)
        n = 0
        if d.exists():
            for p in d.glob("*.wav"):
                p.unlink()
                n += 1
            try:
                d.rmdir()
            except OSError:
                pass
        return {"ok": True, "deleted": n}

    def start_recording_samples(self, keyword: str,
                                  n_samples: int = 10,
                                  sample_seconds: float = 1.6) -> dict:
        if self.status.state != "idle":
            return {"ok": False,
                    "error": f"service is busy: {self.status.state}"}
        if not keyword.strip():
            return {"ok": False, "error": "keyword required"}
        kw_safe = keyword.strip().lower().replace(" ", "_")
        if not kw_safe.replace("_", "").isalnum():
            return {"ok": False, "error": "keyword must be alphanumeric"}
        self._stop.clear()
        t = threading.Thread(
            target=self._record_worker,
            args=(keyword, kw_safe, int(n_samples), float(sample_seconds)),
            name="WW-Record", daemon=True,
        )
        t.start()
        return {"ok": True}

    def _record_worker(self, keyword: str, kw_safe: str,
                        n_samples: int, sample_seconds: float) -> None:
        """Beep + record N short clips of the user saying the keyword.

        Each iteration: READY beep → 1.6 s capture → DONE beep → 1.0 s
        pause for the user to breathe.  WAVs land under
        /app/data/train/user_pos/<kw_safe>/sample_NN.wav and persist
        across trains (until the user clicks Clear or changes keyword).
        """
        from audio.mic import record_blocking, save_wav
        from audio.beep import READY_BEEP, DONE_BEEP, COMPLETE_BEEP, play
        from audio.mic import Capture
        try:
            self._push_status(
                state="recording", rec_keyword=keyword,
                rec_done=0, rec_total=n_samples, rec_status="ready",
            )
            self.emit({"type": "record_start", "keyword": keyword,
                       "n_samples": n_samples})
            out_dir = self._user_pos_dir(kw_safe)
            out_dir.mkdir(parents=True, exist_ok=True)
            # wipe any prior recordings for this keyword so the saved set
            # always matches what just got captured
            for p in list(out_dir.glob("*.wav")):
                p.unlink()

            for i in range(n_samples):
                if self._stop.is_set():
                    self._push_status(rec_status="cancelled")
                    self.emit({"type": "record_cancel"})
                    return
                # cue the user
                self._push_status(rec_done=i, rec_status="ready",
                                   rec_keyword=keyword)
                try: play(READY_BEEP, device=self._out_device)
                except Exception as e:    # noqa: BLE001
                    print(f"[record] beep failed: {e}", flush=True)
                # capture
                self._push_status(rec_done=i, rec_status="recording",
                                   rec_keyword=keyword)
                try:
                    cap = record_blocking(
                        sample_seconds, device=self._in_device,
                    )
                except Exception as e:    # noqa: BLE001
                    print(f"[record] capture failed: {e}", flush=True)
                    self._push_status(
                        state="idle", last_error=f"record failed: {e}",
                        rec_status="error",
                    )
                    self.emit({"type": "record_error", "msg": str(e)})
                    return
                save_wav(cap, str(out_dir / f"sample_{i:02d}.wav"))
                self._push_status(rec_done=i + 1, rec_status="pause",
                                   rec_keyword=keyword)
                try: play(DONE_BEEP, device=self._out_device)
                except Exception:
                    pass
                # pause for breath, but allow stop mid-pause
                for _ in range(10):
                    if self._stop.is_set():
                        break
                    time.sleep(0.1)

            try: play(COMPLETE_BEEP, device=self._out_device)
            except Exception:
                pass
            self._push_status(
                state="idle", rec_status="done", rec_done=n_samples,
            )
            self.emit({"type": "record_done", "keyword": keyword,
                       "count": n_samples})
        except Exception as e:    # noqa: BLE001
            print(f"[record] fatal: {e}", flush=True)
            self._push_status(state="idle",
                               last_error=f"record fatal: {e}",
                               rec_status="error")
            self.emit({"type": "record_error", "msg": str(e)})

    def stop_recording_samples(self) -> dict:
        if self.status.state != "recording":
            return {"ok": False, "error": "not recording"}
        self._stop.set()
        return {"ok": True}

    # ---------- training ----------

    def start_training(self, keyword: str, n_per_voice: int = 50,
                        neg_per_voice: int = 80) -> dict:
        if self.status.state != "idle":
            return {"ok": False, "error": f"service is busy: {self.status.state}"}
        if not keyword.strip():
            return {"ok": False, "error": "keyword required"}
        keyword = keyword.strip()
        kw_safe = keyword.lower().replace(" ", "_")
        if not kw_safe.replace("_", "").isalnum():
            return {"ok": False, "error": "keyword must be alphanumeric"}
        self._stop.clear()
        self._train_thread = threading.Thread(
            target=self._train_worker,
            args=(keyword, kw_safe, int(n_per_voice), int(neg_per_voice)),
            name="WW-Train", daemon=True,
        )
        self._train_thread.start()
        return {"ok": True}

    def _train_worker(self, keyword: str, kw_safe: str,
                       n_per_voice: int, neg_per_voice: int) -> None:
        try:
            self._push_status(state="training", train_keyword=keyword,
                               train_phase="starting", train_progress=0.0,
                               train_progress_text="initializing…")
            self.emit({"type": "train_start", "keyword": keyword,
                       "n_per_voice": n_per_voice,
                       "neg_per_voice": neg_per_voice})

            # ----- 1. ensure piper voices -----
            self._push_status(train_phase="voices",
                               train_progress_text="checking Piper voices…")
            from tools.download_voices import VOICES, VOICES_DIR, _fetch
            VOICES_DIR.mkdir(parents=True, exist_ok=True)
            missing = []
            for name, mu, cu in VOICES:
                if not (VOICES_DIR / f"{name}.onnx").exists():
                    missing.append((name, mu, cu))
            if missing:
                self.emit({"type": "train_progress", "phase": "voices",
                           "msg": f"downloading {len(missing)} voice(s)…"})
                for i, (name, mu, cu) in enumerate(missing):
                    self._push_status(
                        train_phase="voices",
                        train_progress=0.05 * (i / max(1, len(missing))),
                        train_progress_text=f"voice {i+1}/{len(missing)}: {name}",
                    )
                    _fetch(mu, VOICES_DIR / f"{name}.onnx")
                    _fetch(cu, VOICES_DIR / f"{name}.onnx.json")

            # ----- 2. plan + generate samples -----
            # We include three negative classes — voice distractors,
            # silence/low-noise (so the model knows ambient room audio is
            # NOT the keyword), and noisy distractors (real rooms aren't
            # studio-clean TTS).
            #
            # Speed strategy (vs. the original serial pipeline):
            #   • Pre-baked cache (tools/bake_negatives.py at Docker
            #     build time) supplies voice-distractor FEATURES and
            #     noisy-distractor base WAVs — saves ~480 Piper calls
            #     and ~480 ONNX feature passes.
            #   • Parallel Piper across one process per voice for the
            #     keyword-specific work that the cache can't cover
            #     (positives + multi-word hard negatives).
            N_SILENCE = 200
            N_NOISY = 200
            N_HARD_SUFFIX = 150
            N_HARD_PREFIX = 150
            voices = sorted([p.stem for p in VOICES_DIR.glob("*.onnx")])
            from tools.generate_samples import (
                _load_voice, _synthesize, _pad_or_crop_centered, _save_wav,
                _make_silence, _mix_with_noise,
                POS_DIR, NEG_DIR, DISTRACTOR_TEXTS,
            )
            from tools.parallel_synth import synth_voice_tasks
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            import random
            import time as _time
            random.seed(42)
            rng = np.random.default_rng(42)
            POS_DIR.mkdir(parents=True, exist_ok=True)
            NEG_DIR.mkdir(parents=True, exist_ok=True)
            # wipe any prior synthesis for this keyword to keep counts clean
            for p in list(POS_DIR.glob("*.wav")) + list(NEG_DIR.glob("*.wav")):
                p.unlink()

            # try the pre-baked cache; on miss we live-synth everything
            try:
                from tools.negatives_cache import load as _load_neg_cache
                neg_cache = _load_neg_cache(voices)
            except Exception as _exc:
                print(f"[train] negatives cache load error: {_exc}",
                      flush=True)
                neg_cache = None
            if neg_cache is not None:
                print(f"[train] negatives cache HIT — "
                      f"{neg_cache.n_voice_distractors} VD feats, "
                      f"{neg_cache.n_noisy} noisy base WAVs", flush=True)
            else:
                print("[train] negatives cache MISS — live-synth fallback",
                      flush=True)

            # --- build per-voice task lists for parallel synth ---
            keyword_variants = [keyword, keyword.lower(),
                                 keyword + ".", keyword + "!"]
            per_voice_tasks: dict[str, list[tuple[str, str]]] = {
                v: [] for v in voices
            }
            # positives — always live-synthed
            for v in voices:
                for i in range(n_per_voice):
                    per_voice_tasks[v].append((
                        str(POS_DIR / f"{v}_{i:03d}.wav"),
                        random.choice(keyword_variants),
                    ))
            # voice distractors — live-synth only on cache miss
            if neg_cache is None:
                for v in voices:
                    for i in range(neg_per_voice):
                        per_voice_tasks[v].append((
                            str(NEG_DIR / f"{v}_{i:03d}.wav"),
                            random.choice(DISTRACTOR_TEXTS),
                        ))

            # hard negatives for multi-word keywords.  Two failure modes:
            #   1. suffix alone:  "krishna" should NOT fire "hey krishna"
            #   2. prefix alone:  "hey there" should NOT fire any "hey X"
            # Without (2) the classifier latches onto "hey" as the
            # positive signal and fires on just "heyyy".
            words = keyword.strip().split()
            suffix_variants: list[str] = []
            prefix_variants: list[str] = []
            if len(words) >= 2:
                prefix = words[0]
                suffix = " ".join(words[1:])
                suffix_variants = [
                    suffix, suffix.lower(), suffix + ".", suffix + "!",
                    f"hi {suffix}", f"hello {suffix}",
                    f"yo {suffix}", f"oh {suffix}",
                    f"{suffix} please", f"{suffix} here",
                ]
                prefix_variants = [
                    prefix, prefix + ".", prefix + "!",
                    f"{prefix} there", f"{prefix} you",
                    f"{prefix} buddy", f"{prefix} guys",
                    f"{prefix} what's up", f"{prefix} how are you",
                    f"{prefix} listen", f"{prefix} stop",
                    f"oh {prefix}", f"say {prefix}",
                    f"{prefix} {prefix}",
                ]
                for i in range(N_HARD_SUFFIX):
                    v = voices[i % len(voices)]
                    per_voice_tasks[v].append((
                        str(NEG_DIR / f"_hardneg_sfx_{i:04d}.wav"),
                        random.choice(suffix_variants),
                    ))
                for i in range(N_HARD_PREFIX):
                    v = voices[i % len(voices)]
                    per_voice_tasks[v].append((
                        str(NEG_DIR / f"_hardneg_pfx_{i:04d}.wav"),
                        random.choice(prefix_variants),
                    ))

            # noisy distractor BASE WAVs — only live-synth on cache miss.
            # When we have the cache, we lift base WAVs directly out of
            # it and just mix per-run noise in below (mix is ~ms-cheap).
            if neg_cache is None:
                for i in range(N_NOISY):
                    v = voices[i % len(voices)]
                    per_voice_tasks[v].append((
                        str(NEG_DIR / f"_noisy_base_{i:04d}.wav"),
                        random.choice(DISTRACTOR_TEXTS),
                    ))

            total_tasks = sum(len(t) for t in per_voice_tasks.values())
            self.emit({"type": "train_progress", "phase": "synth",
                       "msg": f"parallel-synth {total_tasks} samples "
                              f"on {len(voices)} workers…"})
            self._push_status(
                train_phase="synth", train_progress=0.06,
                train_progress_text=
                f"spawning {len(voices)} Piper workers…",
            )

            # --- run parallel synth ---
            # spawn (not fork) so workers don't inherit ONNX/audio state
            # held by the live-listen thread.  Spawn costs ~1s/worker once.
            spawn_ctx = mp.get_context("spawn")
            done = 0
            t_synth0 = _time.perf_counter()
            with ProcessPoolExecutor(max_workers=len(voices),
                                      mp_context=spawn_ctx) as ex:
                futs = {
                    ex.submit(synth_voice_tasks,
                              v, str(VOICES_DIR),
                              per_voice_tasks[v], 42 + idx): v
                    for idx, v in enumerate(voices)
                    if per_voice_tasks[v]
                }
                for f in as_completed(futs):
                    v = futs[f]
                    n_done = f.result()
                    done += n_done
                    frac = 0.06 + 0.24 * (done / max(1, total_tasks))
                    self._push_status(
                        train_phase="synth", train_progress=frac,
                        train_progress_text=
                        f"synth {done}/{total_tasks} (voice {v} done)",
                    )
            synth_secs = _time.perf_counter() - t_synth0
            print(f"[train] parallel synth: {done} samples in "
                  f"{synth_secs:.1f}s "
                  f"(~{done / max(0.1, synth_secs):.0f}/s)", flush=True)

            # --- silence (procedural, fast — stays in main thread) ---
            for i in range(N_SILENCE):
                noise_dbfs = float(rng.uniform(-65.0, -35.0))
                audio = _make_silence(rng, noise_dbfs=noise_dbfs)
                _save_wav(audio, NEG_DIR / f"_silence_{i:04d}.wav")
            self._push_status(
                train_phase="synth", train_progress=0.32,
                train_progress_text=f"silence {N_SILENCE} done",
            )

            # --- per-run noise mix on noisy-distractor bases ---
            # We always do the noise mix per-train (so the noise
            # realization differs run-to-run) but the underlying clean
            # TTS came either from disk (cache miss) or from the cache.
            from audio.mic import load_wav
            if neg_cache is not None:
                noisy_base = neg_cache.noisy_base_wavs
            else:
                base_paths = sorted(NEG_DIR.glob("_noisy_base_*.wav"))
                noisy_base = np.stack([
                    load_wav(str(p)).samples for p in base_paths
                ]) if base_paths else np.zeros((0, 0), dtype=np.int16)
                # tidy up base WAVs so they aren't picked up by embed glob
                for p in base_paths:
                    p.unlink()

            noisy_mixed = np.zeros(
                (noisy_base.shape[0], noisy_base.shape[1] if noisy_base.ndim == 2 else 0),
                dtype=np.int16,
            ) if noisy_base.size else np.zeros((0, 0), dtype=np.int16)
            for i in range(noisy_base.shape[0]):
                mixed = _mix_with_noise(
                    noisy_base[i], rng,
                    snr_db=float(rng.uniform(5.0, 15.0)),
                )
                noisy_mixed[i] = mixed
                _save_wav(mixed, NEG_DIR / f"_noisy_{i:04d}.wav")

            # --- augment the TTS positives ---
            # For every clean Piper positive we produce one augmented
            # copy with random reverb + light noise + small pitch jitter.
            # This drags the positive-class decision boundary toward
            # "real-mic-in-a-real-room" so the classifier doesn't fire
            # only on studio-clean TTS.  Each aug is fast (~ms) so the
            # whole pass is dominated by disk I/O.
            from tools.audio_augment import augment_positive
            t_aug0 = _time.perf_counter()
            tts_pos_wavs = sorted(POS_DIR.glob("*.wav"))
            n_tts_aug = 0
            for p in tts_pos_wavs:
                # skip anything that's already an aug copy (shouldn't be
                # any at this point, but be defensive across retrains)
                if "_aug" in p.stem or "_user" in p.stem:
                    continue
                clip = load_wav(str(p)).samples
                aug = augment_positive(clip, rng)
                # the aug step may have changed clip length (pitch jitter)
                aug = _pad_or_crop_centered(aug)
                _save_wav(aug, POS_DIR / f"{p.stem}_aug.wav")
                n_tts_aug += 1

            # --- load user-recorded positives if any, pitch-shift ---
            # Persisted at /app/data/train/user_pos/<kw_safe>/sample_NN.wav
            # from the optional "Record yourself" flow.  Each sample
            # turns into 5 augmented variants ({-4,-2,0,+2,+4} semitones)
            # — synthetic speaker diversity from a single recording set.
            USER_POS_DIR = ROOT / "data" / "train" / "user_pos" / kw_safe
            user_paths = sorted(USER_POS_DIR.glob("*.wav")) \
                if USER_POS_DIR.exists() else []
            n_user_aug = 0
            if user_paths:
                from tools.audio_augment import pitch_shift
                pitch_variants = [-4.0, -2.0, 0.0, +2.0, +4.0]
                for ui, p in enumerate(user_paths):
                    clip = load_wav(str(p)).samples
                    for psh in pitch_variants:
                        shifted = pitch_shift(clip, psh) \
                            if abs(psh) > 0.01 else clip
                        # also apply mild reverb/noise so user samples
                        # cover position-in-room variation
                        shifted = augment_positive(
                            shifted, rng,
                            pitch_semitones=0.0,  # already pitch-shifted
                            pitch_jitter_prob=0.0,
                            reverb_prob=0.5, noise_prob=0.5,
                        )
                        shifted = _pad_or_crop_centered(shifted)
                        _save_wav(
                            shifted,
                            POS_DIR / f"_user_{ui:02d}_p{int(psh):+d}.wav",
                        )
                        n_user_aug += 1
            print(f"[train] augmented positives: "
                  f"+{n_tts_aug} TTS-aug, +{n_user_aug} user "
                  f"({len(user_paths)} recordings × 5 pitch variants) "
                  f"in {_time.perf_counter() - t_aug0:.1f}s",
                  flush=True)

            # ----- 3. embed (parallel; skips cached voice-distractor rows) -----
            # Strategy: fan WAV paths out to N worker processes, each
            # running its own OWWFeatures (single-threaded ONNX so they
            # don't fight for cores).  At Jetson Orin scale this turns a
            # ~5 min serial embed phase into ~1 min.
            #
            # The noisy mixed WAVs are already on disk at _noisy_*.wav
            # (we wrote them above for debug visibility), so we just
            # glob them up like everything else.
            self.emit({"type": "train_progress", "phase": "embed",
                       "msg": "extracting features for keyword-specific "
                              "+ procedural samples…"})
            self._push_status(
                train_phase="embed", train_progress=0.35,
                train_progress_text="spawning embed workers…",
            )

            pos_wavs = sorted(POS_DIR.glob("*.wav"))
            neg_wavs = sorted(NEG_DIR.glob("*.wav"))  # silence + hardneg
                                                       # + noisy + (VD if miss)
            t_embed0 = _time.perf_counter()

            import os as _os
            from tools.parallel_synth import embed_wav_batch
            # 6 workers on 6-core Orin gives the best wall-clock; more
            # workers just add ONNX session memory pressure for no gain.
            n_embed_workers = min(6, max(1, (_os.cpu_count() or 6) - 1))

            def _chunk(items: list, n: int) -> list[list]:
                if n <= 0 or not items:
                    return []
                k, r = divmod(len(items), n)
                out: list[list] = []
                idx = 0
                for w in range(n):
                    take = k + (1 if w < r else 0)
                    if take:
                        out.append(items[idx:idx + take])
                        idx += take
                return out

            pos_batches = _chunk([str(p) for p in pos_wavs], n_embed_workers)
            neg_batches = _chunk([str(p) for p in neg_wavs], n_embed_workers)

            # Dispatch both pos + neg batches into the same pool so all
            # workers stay busy.  We keep their results separated via the
            # (kind, idx) labels so the final stack preserves filename order.
            pos_results: list[np.ndarray | None] = [None] * len(pos_batches)
            neg_results: list[np.ndarray | None] = [None] * len(neg_batches)
            n_embed_total = len(pos_wavs) + len(neg_wavs)
            done_embed = 0
            with ProcessPoolExecutor(max_workers=n_embed_workers,
                                      mp_context=spawn_ctx) as ex:
                futs = {}
                for i, batch in enumerate(pos_batches):
                    futs[ex.submit(embed_wav_batch, batch)] = ("pos", i, len(batch))
                for i, batch in enumerate(neg_batches):
                    futs[ex.submit(embed_wav_batch, batch)] = ("neg", i, len(batch))
                for f in as_completed(futs):
                    kind, idx, n = futs[f]
                    arr = f.result()
                    if kind == "pos":
                        pos_results[idx] = arr
                    else:
                        neg_results[idx] = arr
                    done_embed += n
                    frac = 0.35 + 0.50 * (done_embed / max(1, n_embed_total))
                    self._push_status(
                        train_phase="embed", train_progress=frac,
                        train_progress_text=
                        f"embed {done_embed}/{n_embed_total}",
                    )

            empty = np.zeros((0, WINDOW_FRAMES * EMBED_DIM), dtype=np.float32)
            Xp = (np.vstack([a for a in pos_results if a is not None])
                  if any(a is not None for a in pos_results) else empty)
            Xn_live = (np.vstack([a for a in neg_results if a is not None])
                       if any(a is not None for a in neg_results) else empty)

            # cached voice-distractor features (no synth, no embed)
            if neg_cache is not None:
                vd_chunks = []
                shortfall_texts: list[tuple[str, str]] = []
                for v in voices:
                    rows = neg_cache.features_for(v, neg_per_voice)
                    vd_chunks.append(rows)
                    if rows.shape[0] < neg_per_voice:
                        shortfall = neg_per_voice - rows.shape[0]
                        print(f"[train] cache short by {shortfall} for "
                              f"voice {v} — will top up live", flush=True)
                        for _k in range(shortfall):
                            shortfall_texts.append(
                                (v, random.choice(DISTRACTOR_TEXTS))
                            )
                if shortfall_texts:
                    # rare path: live-synth + embed the shortfall in this
                    # process.  Keeps the codepath simple; if a user ever
                    # actually pushes neg_per_voice this high we can
                    # parallelize this path too.
                    extractor = self._ensure_features()
                    for v, text in shortfall_texts:
                        voice = _load_voice(VOICES_DIR, v)
                        audio = _synthesize(voice, text)
                        audio = _pad_or_crop_centered(audio)
                        vd_chunks.append(extractor.embed_clip(audio)[None, :])
                Xn_cached = np.vstack(vd_chunks) if vd_chunks else empty
            else:
                Xn_cached = empty

            Xn = np.vstack([Xn_live, Xn_cached])
            print(f"[train] embed: {_time.perf_counter() - t_embed0:.1f}s  "
                  f"(pos={Xp.shape[0]} live_neg={Xn_live.shape[0]} "
                  f"cached_VD={Xn_cached.shape[0]}, "
                  f"{n_embed_workers} workers)", flush=True)

            # ----- 4. fit -----
            self._push_status(train_phase="fit",
                               train_progress=0.88,
                               train_progress_text="fitting classifier…")
            X = np.vstack([Xp, Xn])
            y = np.concatenate([np.ones(len(Xp), dtype=np.int8),
                                 np.zeros(len(Xn), dtype=np.int8)])
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(X))
            X, y = X[idx], y[idx]
            n_eval = max(20, int(len(X) * 0.10))
            X_eval, y_eval = X[:n_eval], y[:n_eval]
            X_train, y_train = X[n_eval:], y[n_eval:]
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score, f1_score
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(
                    C=1.0, class_weight="balanced",
                    max_iter=2000, random_state=42,
                    solver="lbfgs",
                )),
            ])
            clf.fit(X_train, y_train)
            p_eval = clf.predict_proba(X_eval)[:, 1]
            auc = float(roc_auc_score(y_eval, p_eval))
            best_t, best_f1 = 0.5, 0.0
            for t in np.arange(0.1, 0.95, 0.05):
                f = f1_score(y_eval, (p_eval > t).astype(np.int8),
                              zero_division=0)
                if f > best_f1:
                    best_t, best_f1 = float(t), float(f)

            # ----- 5. save -----
            self._push_status(train_phase="save",
                               train_progress=0.96,
                               train_progress_text="saving model…")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            out = MODELS_DIR / f"{kw_safe}.joblib"
            joblib.dump({
                "model": clf, "keyword": keyword,
                "window_frames": WINDOW_FRAMES, "embed_dim": EMBED_DIM,
                "feature_dim": WINDOW_FRAMES * EMBED_DIM,
                "suggested_threshold": best_t,
                "eval": {"auc": auc, "f1_at_best": best_f1,
                          "n_eval_pos": int(y_eval.sum()),
                          "n_eval_neg": int(len(y_eval) - y_eval.sum())},
                "train": {"n_train_pos": int(y_train.sum()),
                           "n_train_neg": int(len(y_train) - y_train.sum())},
            }, out, compress=3)

            self._push_status(state="idle", train_phase="done",
                               train_progress=1.0,
                               train_progress_text=
                               f"saved {out.relative_to(ROOT)} "
                               f"(AUC {auc:.3f}, threshold {best_t:.2f})")
            self.emit({"type": "train_done", "keyword": kw_safe,
                       "path": str(out.relative_to(ROOT)),
                       "auc": auc, "threshold": best_t,
                       "f1": best_f1})
            try: play(DONE_BEEP, device=self._out_device)
            except Exception: pass    # noqa: BLE001, E701
        except Exception as e:    # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._push_status(state="idle",
                               last_error=f"{type(e).__name__}: {e}",
                               train_phase="error",
                               train_progress_text=str(e))
            self.emit({"type": "error", "msg": str(e)})


SERVICE: WakewordService = WakewordService()
