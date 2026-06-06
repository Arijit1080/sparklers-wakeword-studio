"""Live wake-word recognition using OpenWakeWord pre-trained models.

Reads audio from the USB codec in 80 ms chunks (1280 samples @ 16 kHz)
and runs OWW's predict() on each chunk.  Any model exceeding the
threshold fires a trigger.

Usage:
    cd ~/jetson-wakeword-studio && source venv/bin/activate
    python3 apps/recognize_loop.py                          # listen for all built-in keywords
    python3 apps/recognize_loop.py --models hey_jarvis      # listen for only one
    python3 apps/recognize_loop.py --threshold 0.5 -v       # verbose, custom threshold
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.beep import DONE_BEEP, READY_BEEP, play   # noqa: E402
from audio.mic import SAMPLE_RATE, find_usb_codec_index   # noqa: E402

CHUNK_SAMPLES = 1280   # OWW expects 80 ms chunks @ 16 kHz


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="",
                   help="comma-separated subset of wake-word models to listen for. "
                        "default: all pre-trained.  Examples: 'hey_jarvis', "
                        "'alexa,hey_mycroft'")
    p.add_argument("--custom", default="",
                   help="absolute path to a custom .onnx wake-word model")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="trigger when score > this (default 0.5)")
    p.add_argument("--patience", type=int, default=1,
                   help="require this many consecutive above-threshold frames "
                        "before firing (default 1).  Higher = fewer false fires "
                        "but slower trigger.")
    p.add_argument("--vad-threshold", type=float, default=0.0,
                   help="OWW's built-in VAD threshold (0 = off). Try 0.3 to "
                        "skip scoring on silence — improves recall.")
    p.add_argument("--cooldown-s", type=float, default=2.0,
                   help="seconds to suppress after a trigger (default 2)")
    p.add_argument("--device", type=int, default=None,
                   help="sounddevice input index. default: auto-detect USB codec")
    p.add_argument("--no-beep", action="store_true",
                   help="don't play a confirmation beep on trigger")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print every chunk's per-model scores")
    args = p.parse_args()

    from openwakeword import Model     # delayed import for fast --help
    from audio.mic import find_usb_codec_output_index   # for ready beep routing

    wakeword_models = None
    if args.custom:
        wakeword_models = [args.custom]
    elif args.models:
        # OWW's Model() takes a list of model file basenames or paths
        wakeword_models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("loading OpenWakeWord (ONNX)…")
    t0 = time.monotonic()
    model_kwargs = dict(inference_framework="onnx")
    if args.vad_threshold > 0:
        model_kwargs["vad_threshold"] = args.vad_threshold
    if wakeword_models:
        model_kwargs["wakeword_models"] = wakeword_models
    model = Model(**model_kwargs)
    keys = list(model.models.keys())
    print(f"  loaded in {time.monotonic()-t0:.1f}s — {len(keys)} keyword(s):")
    for k in keys:
        print(f"    • {k}")

    device = args.device if args.device is not None else find_usb_codec_index()
    if device is None:
        print("no USB codec detected", file=sys.stderr)
        return 1

    out_dev = find_usb_codec_output_index()
    play(READY_BEEP, device=out_dev)
    print("\nLISTENING.  speak a wake word.  Ctrl+C to stop.\n")

    last_fire_t: dict[str, float] = {}
    streak: dict[str, int] = {}     # consecutive above-threshold count per model
    n_fires = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            device=device, blocksize=CHUNK_SAMPLES,
        ) as stream:
            while True:
                block, _ = stream.read(CHUNK_SAMPLES)
                block = block.reshape(-1)
                t_e = time.monotonic()
                scores = model.predict(block)
                pred_ms = 1000 * (time.monotonic() - t_e)

                # Patience-filtered trigger logic.  We track per-model streak
                # of consecutive above-threshold predictions; only fire when
                # the streak reaches `--patience`, and only outside cooldown.
                fired_names: list[str] = []
                now = time.monotonic()
                for k, v in scores.items():
                    if v > args.threshold:
                        streak[k] = streak.get(k, 0) + 1
                    else:
                        streak[k] = 0
                    if (streak[k] >= args.patience
                            and now - last_fire_t.get(k, 0) > args.cooldown_s):
                        fired_names.append(k)
                        streak[k] = 0

                if fired_names:
                    n_fires += 1
                    for k in fired_names:
                        last_fire_t[k] = time.monotonic()
                        print(f"  🟢 TRIGGER #{n_fires}  '{k}'  score={scores[k]:.3f}  "
                              f"({pred_ms:.1f}ms)")
                    if not args.no_beep:
                        play(DONE_BEEP, device=out_dev)
                elif args.verbose:
                    top = max(scores.items(), key=lambda x: x[1])
                    if top[1] > 0.01:
                        bar = "█" * int(top[1] * 20)
                        print(f"  {bar:<20}  {top[0]}={top[1]:.3f}  "
                              f"({pred_ms:.1f}ms)")
    except KeyboardInterrupt:
        print(f"\n\nstopped.  total triggers: {n_fires}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
