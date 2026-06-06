"""Live wake-word recognition using OUR sklearn classifier.

Architecture:
    mic → 80 ms chunk → OWW melspec + embedding → push to 16-frame ring →
    flatten 16×96 = 1536-D → sklearn.predict_proba → above-threshold + patience → fire

Use this for models trained with tools/train_keyword_sklearn.py.

Run:
    python3 apps/recognize_custom.py --keyword arijit
    python3 apps/recognize_custom.py --keyword arijit --threshold 0.7 --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.beep import DONE_BEEP, READY_BEEP, play   # noqa: E402
from audio.mic import (   # noqa: E402
    SAMPLE_RATE, find_usb_codec_index, find_usb_codec_output_index,
)
from audio.oww_features import (   # noqa: E402
    CHUNK_SAMPLES, EMBED_DIM, OWWFeatures, WINDOW_FRAMES,
)

MODELS_DIR = ROOT / "models"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyword", required=True,
                   help="filename stem of the trained .joblib model")
    p.add_argument("--threshold", type=float, default=None,
                   help="trigger when P(keyword) > this. default: model's "
                        "suggested_threshold")
    p.add_argument("--patience", type=int, default=2,
                   help="require this many consecutive above-threshold "
                        "frames before firing (default 2)")
    p.add_argument("--cooldown-s", type=float, default=2.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    model_path = MODELS_DIR / f"{args.keyword}.joblib"
    if not model_path.exists():
        print(f"missing model: {model_path}", file=sys.stderr)
        print(f"  train it:   python3 tools/train_keyword_sklearn.py "
              f"--keyword {args.keyword}", file=sys.stderr)
        return 1
    print(f"loading {model_path.relative_to(ROOT)}…")
    payload = joblib.load(model_path)
    clf = payload["model"]
    threshold = args.threshold if args.threshold is not None \
        else payload.get("suggested_threshold", 0.5)
    print(f"  keyword:    {payload['keyword']}")
    print(f"  features:   {payload['feature_dim']}-D "
          f"({payload['window_frames']} × {payload['embed_dim']})")
    print(f"  threshold:  {threshold:.3f}")
    print(f"  eval AUC:   {payload.get('eval',{}).get('auc','?')}")

    print("\nloading OWW feature extractor…")
    extractor = OWWFeatures()

    in_dev = args.device if args.device is not None else find_usb_codec_index()
    out_dev = find_usb_codec_output_index()
    if in_dev is None:
        print("no USB codec detected", file=sys.stderr)
        return 1

    # Warmup: feed a few silent chunks so the buffer fills + onnxruntime JIT's
    print("warming up…")
    t0 = time.monotonic()
    for _ in range(WINDOW_FRAMES + 2):
        extractor.feed_chunk(np.zeros(CHUNK_SAMPLES, dtype=np.int16))
    extractor.reset()
    print(f"  warm: {1000*(time.monotonic()-t0):.1f}ms")

    streak = 0
    last_fire_t = 0.0
    n_fires = 0

    play(READY_BEEP, device=out_dev)
    print("\nLISTENING.  speak the keyword.  Ctrl+C to stop.\n")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            device=in_dev, blocksize=CHUNK_SAMPLES,
        ) as stream:
            while True:
                block, _ = stream.read(CHUNK_SAMPLES)
                block = block.reshape(-1)
                t_e = time.monotonic()
                feat = extractor.feed_chunk(block)
                if feat is None:
                    continue
                score = float(clf.predict_proba(feat.reshape(1, -1))[0, 1])
                ms = 1000.0 * (time.monotonic() - t_e)
                hit = score > threshold
                streak = streak + 1 if hit else 0
                now = time.monotonic()
                if (streak >= args.patience
                        and now - last_fire_t > args.cooldown_s):
                    last_fire_t = now
                    n_fires += 1
                    streak = 0
                    print(f"  🟢 TRIGGER #{n_fires}  score={score:.3f}  "
                          f"({ms:.1f}ms)")
                    if not args.no_beep:
                        play(DONE_BEEP, device=out_dev)
                elif args.verbose:
                    bar = "█" * int(score * 20)
                    marker = "*" if hit else " "
                    print(f"  {bar:<20}  score={score:.3f} {marker}  "
                          f"streak={streak}  ({ms:.1f}ms)")
    except KeyboardInterrupt:
        print(f"\n\nstopped.  total triggers: {n_fires}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
