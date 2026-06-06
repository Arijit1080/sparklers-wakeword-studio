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
    state: str = "idle"                  # idle | listening | training
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

            # ----- 2. generate samples -----
            # We include three negative classes — voice distractors,
            # silence/low-noise (so the model knows ambient room audio is
            # NOT the keyword), and noisy distractors (real rooms aren't
            # studio-clean TTS).
            N_SILENCE = 200
            N_NOISY = 200
            voices = sorted([p.stem for p in VOICES_DIR.glob("*.onnx")])
            total_synth = len(voices) * (n_per_voice + neg_per_voice) + N_SILENCE + N_NOISY
            self.emit({"type": "train_progress", "phase": "synth",
                       "msg": f"generating {total_synth} samples…"})
            from tools.generate_samples import (
                _load_voice, _synthesize, _pad_or_crop_centered, _save_wav,
                _make_silence, _mix_with_noise,
                POS_DIR, NEG_DIR, DISTRACTOR_TEXTS,
            )
            import random
            random.seed(42)
            rng = np.random.default_rng(42)
            POS_DIR.mkdir(parents=True, exist_ok=True)
            NEG_DIR.mkdir(parents=True, exist_ok=True)
            # wipe any prior synthesis for this keyword to keep counts clean
            for p in list(POS_DIR.glob("*.wav")) + list(NEG_DIR.glob("*.wav")):
                p.unlink()
            keyword_variants = [keyword, keyword.lower(),
                                 keyword + ".", keyword + "!"]
            done = 0
            for v in voices:
                voice = _load_voice(VOICES_DIR, v)
                for i in range(n_per_voice):
                    text = random.choice(keyword_variants)
                    audio = _synthesize(voice, text)
                    audio = _pad_or_crop_centered(audio)
                    _save_wav(audio, POS_DIR / f"{v}_{i:03d}.wav")
                    done += 1
                    if done % 25 == 0 or done == total_synth:
                        frac = 0.05 + 0.30 * (done / total_synth)
                        self._push_status(
                            train_phase="synth",
                            train_progress=frac,
                            train_progress_text=
                            f"synth {done}/{total_synth}  ({v} pos)",
                        )
                for i in range(neg_per_voice):
                    text = random.choice(DISTRACTOR_TEXTS)
                    audio = _synthesize(voice, text)
                    audio = _pad_or_crop_centered(audio)
                    _save_wav(audio, NEG_DIR / f"{v}_{i:03d}.wav")
                    done += 1
                    if done % 25 == 0 or done == total_synth:
                        frac = 0.05 + 0.30 * (done / total_synth)
                        self._push_status(
                            train_phase="synth",
                            train_progress=frac,
                            train_progress_text=
                            f"synth {done}/{total_synth}  ({v} neg)",
                        )

            # silence + low-level noise negatives
            for i in range(N_SILENCE):
                noise_dbfs = float(rng.uniform(-65.0, -35.0))
                audio = _make_silence(rng, noise_dbfs=noise_dbfs)
                _save_wav(audio, NEG_DIR / f"_silence_{i:04d}.wav")
                done += 1
                if done % 50 == 0:
                    frac = 0.05 + 0.30 * (done / total_synth)
                    self._push_status(
                        train_phase="synth", train_progress=frac,
                        train_progress_text=f"synth silence {i+1}/{N_SILENCE}",
                    )

            # ---- hard negatives for multi-word keywords ----
            # Two failure modes we need to teach against:
            #   1. suffix alone:  "krishna" should NOT fire "hey krishna"
            #   2. prefix alone:  "hey" (or "hey there") should NOT fire any
            #                     "hey X" model
            # We generate ~150 of each.  Without (2) the classifier latches
            # onto "hey" as the positive signal and fires on just "heyyy".
            words = keyword.strip().split()
            hard_neg_count = 0
            if len(words) >= 2:
                prefix = words[0]
                suffix = " ".join(words[1:])

                # --- suffix-only hard negatives ---
                suffix_variants = [
                    suffix, suffix.lower(), suffix + ".", suffix + "!",
                    f"hi {suffix}", f"hello {suffix}",
                    f"yo {suffix}", f"oh {suffix}",
                    f"{suffix} please", f"{suffix} here",
                ]
                # --- prefix-only hard negatives ---
                # Real-world phrases that start with "hey" but aren't the
                # wake word.  Drawn-out "heyyy", common "hey + something",
                # "hey" embedded mid-sentence.
                prefix_variants = [
                    prefix, prefix + ".", prefix + "!",
                    f"{prefix} there", f"{prefix} you",
                    f"{prefix} buddy", f"{prefix} guys",
                    f"{prefix} what's up", f"{prefix} how are you",
                    f"{prefix} listen", f"{prefix} stop",
                    f"oh {prefix}", f"say {prefix}",
                    f"{prefix} {prefix}",
                ]

                N_HARD_SUFFIX = 150
                N_HARD_PREFIX = 150
                N_HARD = N_HARD_SUFFIX + N_HARD_PREFIX
                self.emit({"type": "train_progress", "phase": "synth",
                           "msg": f"hard-negatives: {N_HARD_SUFFIX} suffix-only "
                                  f"+ {N_HARD_PREFIX} prefix-only…"})
                for i in range(N_HARD_SUFFIX):
                    v = voices[i % len(voices)]
                    voice = _load_voice(VOICES_DIR, v)
                    text = random.choice(suffix_variants)
                    audio = _synthesize(voice, text)
                    audio = _pad_or_crop_centered(audio)
                    _save_wav(audio, NEG_DIR / f"_hardneg_sfx_{i:04d}.wav")
                    hard_neg_count += 1
                    done += 1
                    if done % 25 == 0:
                        frac = 0.05 + 0.30 * (done / (total_synth + N_HARD))
                        self._push_status(
                            train_phase="synth", train_progress=frac,
                            train_progress_text=
                            f"suffix-only neg {i+1}/{N_HARD_SUFFIX}  ('{text}')",
                        )
                for i in range(N_HARD_PREFIX):
                    v = voices[i % len(voices)]
                    voice = _load_voice(VOICES_DIR, v)
                    text = random.choice(prefix_variants)
                    audio = _synthesize(voice, text)
                    audio = _pad_or_crop_centered(audio)
                    _save_wav(audio, NEG_DIR / f"_hardneg_pfx_{i:04d}.wav")
                    hard_neg_count += 1
                    done += 1
                    if done % 25 == 0:
                        frac = 0.05 + 0.30 * (done / (total_synth + N_HARD))
                        self._push_status(
                            train_phase="synth", train_progress=frac,
                            train_progress_text=
                            f"prefix-only neg {i+1}/{N_HARD_PREFIX}  ('{text}')",
                        )

            # noisy distractor negatives (TTS distractors + noise mix)
            for i in range(N_NOISY):
                v = voices[i % len(voices)]
                voice = _load_voice(VOICES_DIR, v)
                text = random.choice(DISTRACTOR_TEXTS)
                audio = _synthesize(voice, text)
                audio = _pad_or_crop_centered(audio)
                audio = _mix_with_noise(audio, rng,
                                         snr_db=float(rng.uniform(5.0, 15.0)))
                _save_wav(audio, NEG_DIR / f"_noisy_{i:04d}.wav")
                done += 1
                if done % 25 == 0 or done == total_synth:
                    frac = 0.05 + 0.30 * (done / total_synth)
                    self._push_status(
                        train_phase="synth", train_progress=frac,
                        train_progress_text=f"synth noisy {i+1}/{N_NOISY}",
                    )

            # ----- 3. embed everything -----
            self.emit({"type": "train_progress", "phase": "embed",
                       "msg": "extracting features for all samples…"})
            extractor = self._ensure_features()
            from audio.mic import load_wav
            pos_wavs = sorted(POS_DIR.glob("*.wav"))
            neg_wavs = sorted(NEG_DIR.glob("*.wav"))
            n_total = len(pos_wavs) + len(neg_wavs)
            Xp = np.zeros((len(pos_wavs), WINDOW_FRAMES * EMBED_DIM),
                           dtype=np.float32)
            Xn = np.zeros((len(neg_wavs), WINDOW_FRAMES * EMBED_DIM),
                           dtype=np.float32)
            for i, p in enumerate(pos_wavs):
                Xp[i] = extractor.embed_clip(load_wav(str(p)).samples)
                if (i + 1) % 30 == 0 or i + 1 == len(pos_wavs):
                    frac = 0.35 + 0.50 * ((i + 1) / n_total)
                    self._push_status(
                        train_phase="embed",
                        train_progress=frac,
                        train_progress_text=
                        f"embed {i+1}/{len(pos_wavs)} pos",
                    )
            for j, p in enumerate(neg_wavs):
                Xn[j] = extractor.embed_clip(load_wav(str(p)).samples)
                if (j + 1) % 30 == 0 or j + 1 == len(neg_wavs):
                    frac = 0.35 + 0.50 * ((len(pos_wavs) + j + 1) / n_total)
                    self._push_status(
                        train_phase="embed",
                        train_progress=frac,
                        train_progress_text=
                        f"embed {j+1}/{len(neg_wavs)} neg",
                    )

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
