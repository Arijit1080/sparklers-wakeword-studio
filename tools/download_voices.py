"""Download a small but diverse set of Piper TTS voice models.

These are used by tools/generate_samples.py to synthesize the keyword and
distractor samples for training.

We deliberately keep this list small (~5–8 voices) — for a
single-keyword classifier with augmentation, more voices give
diminishing returns vs. the disk + bandwidth cost.

Run:
    python3 tools/download_voices.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = ROOT / "data" / "piper_voices"

# Curated list of Piper voices — mix of US English (different speakers /
# qualities) plus GB English for diversity. Each is the .onnx weight file
# + the .json config.  Sources: https://github.com/rhasspy/piper
VOICES = [
    ("en_US-amy-medium",        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"),
    ("en_US-ryan-medium",       "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"),
    ("en_US-libritts_r-medium", "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json"),
    ("en_US-lessac-medium",     "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"),
    ("en_GB-alan-medium",       "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"),
    ("en_GB-jenny_dioco-medium","https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
                                "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json"),
]


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  download {url.split('/')[-1]} → {dest.name}…", flush=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0"))
        n_read = 0
        chunk = 1 << 16
        while True:
            data = resp.read(chunk)
            if not data:
                break
            out.write(data)
            n_read += len(data)
            if total:
                pct = 100.0 * n_read / total
                print(f"\r    {pct:5.1f}%  ({n_read/1024/1024:.1f} / "
                      f"{total/1024/1024:.1f} MB)", end="", flush=True)
        print()


def main() -> int:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(VOICES)} Piper voices to {VOICES_DIR}/")
    for name, model_url, cfg_url in VOICES:
        print(f"\n• {name}")
        _fetch(model_url, VOICES_DIR / f"{name}.onnx")
        _fetch(cfg_url, VOICES_DIR / f"{name}.onnx.json")
    total_bytes = sum(p.stat().st_size for p in VOICES_DIR.glob("*"))
    print(f"\n✓ {len(VOICES)} voices, total {total_bytes/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
