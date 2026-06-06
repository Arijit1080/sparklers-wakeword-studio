"""Generate positive ("Arijit") + negative samples for training.

For each Piper voice:
    - synthesize the keyword N times with slight tempo variation → positives
    - synthesize random distractor sentences → negatives

Each sample is saved as a 16 kHz mono WAV of fixed length (default 1.6 s
— enough to fit a 1.28 s training window with some padding on each end).

Run:
    python3 tools/generate_samples.py --keyword Arijit --n-per-voice 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "data" / "piper_voices"
POS_DIR = ROOT / "data" / "train" / "positive"
NEG_DIR = ROOT / "data" / "train" / "negative"

SAMPLE_RATE = 16_000
TARGET_LEN_S = 1.6
TARGET_LEN = int(SAMPLE_RATE * TARGET_LEN_S)


DISTRACTOR_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "good morning everyone how are you today",
    "I really enjoyed the book that you recommended",
    "let's grab a coffee this afternoon if you have time",
    "the weather forecast says rain tomorrow afternoon",
    "could you please send me the report by tomorrow",
    "I am thinking about going for a walk in the park",
    "what time does the meeting start tomorrow",
    "the train will arrive at platform nine in five minutes",
    "she finished her homework before dinner last night",
    "do you know where I left my keys",
    "the new restaurant downtown has great food",
    "please make sure to lock the door when you leave",
    "thanks for the help with the project last week",
    "I will be working from home on Friday",
    "the children are playing in the garden",
    "let me check my calendar for next Tuesday",
    "the new software update is available now",
    "remember to take out the trash on Wednesday",
    "we should plan a vacation for the summer",
    "the package should arrive by Thursday",
    "he is studying engineering at the university",
    "the museum exhibit opens at ten in the morning",
    "I forgot to bring my umbrella today",
    "we watched a great movie at the cinema last night",
    "the concert tickets sold out in less than an hour",
    "she has been practicing piano for over a decade",
    "the highway was closed due to construction work",
    "I hope you have a wonderful weekend",
    "thank you for coming to the celebration",
    # phonetically near-target distractors — these help discrimination
    "are you sure about this",
    "I really need to think about it",
    "her name is Anika",
    "are we ready to go",
    "Adit is a friend of mine",
    "the car is in the garage",
    "are these your books",
    "we should leave now",
]


def _load_voice(voice_dir: Path, name: str):
    """Load a Piper voice from on-disk .onnx + .json."""
    from piper import PiperVoice
    model = voice_dir / f"{name}.onnx"
    return PiperVoice.load(str(model))


def _synthesize(voice, text: str) -> np.ndarray:
    """Run Piper TTS, return int16 16 kHz mono numpy array."""
    from piper import SynthesisConfig
    cfg = SynthesisConfig(
        length_scale=random.uniform(0.85, 1.15),    # tempo jitter
        noise_scale=random.uniform(0.55, 0.75),
        noise_w_scale=random.uniform(0.6, 0.9),
    )
    # Piper exposes synthesize() returning AudioChunk(s) with .audio_int16_array
    out_chunks: list[np.ndarray] = []
    sr = SAMPLE_RATE
    for chunk in voice.synthesize(text, cfg):
        out_chunks.append(chunk.audio_int16_array)
        sr = chunk.sample_rate
    audio = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.int16)
    # Resample to 16 kHz if needed (most Piper voices are 22050 Hz)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio.astype(np.float32), SAMPLE_RATE, sr)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio


def _pad_or_crop_centered(audio: np.ndarray, target_len: int = TARGET_LEN
                           ) -> np.ndarray:
    """Place the audio centered in a `target_len` int16 buffer.  If the
    audio is too long, randomly crop a centered window of target_len."""
    out = np.zeros(target_len, dtype=np.int16)
    if len(audio) >= target_len:
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len].astype(np.int16)
    # leave a random offset so the keyword isn't always perfectly centered
    slack = target_len - len(audio)
    offset = random.randint(int(slack * 0.20), int(slack * 0.80))
    out[offset:offset + len(audio)] = audio
    return out


def _save_wav(samples: np.ndarray, path: Path) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.astype(np.int16).tobytes())


def _make_silence(rng: np.random.Generator,
                   noise_dbfs: float = -50.0) -> np.ndarray:
    """A TARGET_LEN sample of low-level Gaussian noise.  Used as a
    silence/no-speech negative — the model needs to see these or it'll
    confidently misfire on silence (no training signal otherwise)."""
    noise = rng.standard_normal(TARGET_LEN).astype(np.float32)
    target_amp = 10 ** (noise_dbfs / 20.0)
    noise = noise / (np.std(noise) + 1e-9) * target_amp
    return np.clip(noise * 32767, -32768, 32767).astype(np.int16)


def _mix_with_noise(samples: np.ndarray, rng: np.random.Generator,
                     snr_db: float) -> np.ndarray:
    """Mix Gaussian noise into a sample at the given SNR.  Returns int16."""
    f = samples.astype(np.float32) / 32768.0
    sig_p = np.mean(f * f) + 1e-12
    noise = rng.standard_normal(len(samples)).astype(np.float32)
    noise_p = np.mean(noise * noise) + 1e-12
    target_n_p = sig_p / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_n_p / noise_p)
    out = f + noise
    return np.clip(out * 32768, -32768, 32767).astype(np.int16)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyword", required=True,
                   help="the wake word to generate, e.g. 'Arijit'")
    p.add_argument("--n-per-voice", type=int, default=50,
                   help="positive samples per Piper voice (default 50)")
    p.add_argument("--neg-per-voice", type=int, default=80,
                   help="negative samples per voice (default 80)")
    p.add_argument("--n-silence", type=int, default=200,
                   help="silence/low-noise negative samples (default 200). "
                        "Critical for avoiding confident mispredictions on "
                        "real-room ambient audio.")
    p.add_argument("--n-noisy-neg", type=int, default=200,
                   help="noisy-distractor negatives (default 200).  Adds "
                        "TTS distractors mixed with white noise at 5-15 dB SNR.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    voices = sorted([p.stem for p in VOICES_DIR.glob("*.onnx")])
    if not voices:
        print(f"no voices in {VOICES_DIR}/ — run tools/download_voices.py first",
              file=sys.stderr)
        return 1
    print(f"found {len(voices)} voices: {voices}")
    POS_DIR.mkdir(parents=True, exist_ok=True)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    # phonetic variants help when Piper's English G2P mangles the name
    KEYWORD_VARIANTS = [args.keyword, args.keyword.lower(),
                        args.keyword + ".", args.keyword + "!"]

    n_pos = 0
    n_neg = 0
    for v in voices:
        print(f"\n• {v}")
        voice = _load_voice(VOICES_DIR, v)
        # positives
        for i in range(args.n_per_voice):
            text = random.choice(KEYWORD_VARIANTS)
            audio = _synthesize(voice, text)
            audio = _pad_or_crop_centered(audio)
            _save_wav(audio, POS_DIR / f"{v}_{i:03d}.wav")
            n_pos += 1
        print(f"  positives done: +{args.n_per_voice}")
        # negatives
        for i in range(args.neg_per_voice):
            text = random.choice(DISTRACTOR_TEXTS)
            audio = _synthesize(voice, text)
            audio = _pad_or_crop_centered(audio)
            _save_wav(audio, NEG_DIR / f"{v}_{i:03d}.wav")
            n_neg += 1
        print(f"  negatives done: +{args.neg_per_voice}")

    # ---- pure silence / ambient noise negatives ----
    if args.n_silence > 0:
        print(f"\n• silence/ambient negatives ({args.n_silence}) — "
              f"critical for filtering real-room background")
        for i in range(args.n_silence):
            # vary the noise floor from very quiet (-65) to slightly audible (-35)
            noise_dbfs = float(rng.uniform(-65.0, -35.0))
            audio = _make_silence(rng, noise_dbfs=noise_dbfs)
            _save_wav(audio, NEG_DIR / f"_silence_{i:04d}.wav")
            n_neg += 1
        print(f"  done")

    # ---- noisy distractor negatives (simulate real-room speech-with-noise) ----
    if args.n_noisy_neg > 0 and voices:
        print(f"\n• noisy distractor negatives ({args.n_noisy_neg}) — "
              f"distractors mixed with white noise at 5-15 dB SNR")
        for i in range(args.n_noisy_neg):
            v = voices[i % len(voices)]
            voice = _load_voice(VOICES_DIR, v)
            text = random.choice(DISTRACTOR_TEXTS)
            audio = _synthesize(voice, text)
            audio = _pad_or_crop_centered(audio)
            audio = _mix_with_noise(audio, rng,
                                     snr_db=float(rng.uniform(5.0, 15.0)))
            _save_wav(audio, NEG_DIR / f"_noisy_{i:04d}.wav")
            n_neg += 1
        print(f"  done")

    print(f"\n✓ generated {n_pos} positives → {POS_DIR.relative_to(ROOT)}/")
    print(f"  generated {n_neg} negatives → {NEG_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
