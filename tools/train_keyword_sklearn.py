"""Train a sklearn binary classifier on OWW-extracted features.

Pipeline:
    1. Walk data/train/positive/ + data/train/negative/
    2. For each WAV, run audio → OWW melspec → embedding → fixed 1536-D
       feature vector (16 frames × 96-D each, center-cropped)
    3. Train LogisticRegression with class-balanced loss
    4. Hold out ~10% for eval, report TPR/FPR/AUC
    5. Save the trained pipeline as models/<keyword>.joblib

Run:
    python3 tools/train_keyword_sklearn.py --keyword arijit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.mic import load_wav   # noqa: E402
from audio.oww_features import OWWFeatures, EMBED_DIM, WINDOW_FRAMES   # noqa: E402

POS_DIR = ROOT / "data" / "train" / "positive"
NEG_DIR = ROOT / "data" / "train" / "negative"
MODELS_DIR = ROOT / "models"


def _embed_dir(extractor: OWWFeatures, d: Path) -> np.ndarray:
    """Embed every WAV in `d`.  Returns (N, 1536) float32."""
    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        return np.zeros((0, WINDOW_FRAMES * EMBED_DIM), dtype=np.float32)
    out = np.zeros((len(wavs), WINDOW_FRAMES * EMBED_DIM), dtype=np.float32)
    print(f"  embedding {len(wavs)} clips from {d.relative_to(ROOT)}/…")
    t0 = time.monotonic()
    for i, p in enumerate(wavs):
        clip = load_wav(str(p))
        out[i] = extractor.embed_clip(clip.samples)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.monotonic() - t0)
            eta = (len(wavs) - i - 1) / rate
            print(f"    {i+1}/{len(wavs)}  ({rate:.1f}/s, ETA {eta:.0f}s)",
                  flush=True)
    dt = time.monotonic() - t0
    print(f"    done in {dt:.1f}s ({len(wavs)/dt:.1f}/s)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyword", required=True,
                   help="lowercase name for the output model file")
    p.add_argument("--eval-frac", type=float, default=0.1,
                   help="fraction of samples held out for evaluation (default 0.1)")
    p.add_argument("--C", type=float, default=1.0,
                   help="LogisticRegression regularization strength (default 1.0)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print("loading OWW feature extractor (melspec + embedding)…")
    extractor = OWWFeatures()

    print("\n=== embedding positives ===")
    Xp = _embed_dir(extractor, POS_DIR)
    print(f"  shape: {Xp.shape}")
    print("\n=== embedding negatives ===")
    Xn = _embed_dir(extractor, NEG_DIR)
    print(f"  shape: {Xn.shape}")

    if len(Xp) < 10 or len(Xn) < 10:
        print("not enough samples — generate more first", file=sys.stderr)
        return 1

    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp), dtype=np.int8),
                         np.zeros(len(Xn), dtype=np.int8)])
    # shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # train / eval split
    n_eval = max(20, int(len(X) * args.eval_frac))
    X_eval, y_eval = X[:n_eval], y[:n_eval]
    X_train, y_train = X[n_eval:], y[n_eval:]
    print(f"\ntrain: {len(X_train)}   eval: {len(X_eval)}")
    print(f"  train pos/neg: {int(y_train.sum())}/{len(y_train) - int(y_train.sum())}")
    print(f"  eval  pos/neg: {int(y_eval.sum())}/{len(y_eval) - int(y_eval.sum())}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n=== fitting LogisticRegression ===")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=args.C, class_weight="balanced",
            max_iter=2000, random_state=args.seed,
            solver="lbfgs",
        )),
    ])
    t0 = time.monotonic()
    clf.fit(X_train, y_train)
    print(f"  fit in {time.monotonic() - t0:.2f}s")

    # ----- eval -----
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    p_eval = clf.predict_proba(X_eval)[:, 1]
    yhat = (p_eval > 0.5).astype(np.int8)
    print("\n=== eval @ threshold 0.5 ===")
    print(classification_report(y_eval, yhat, target_names=["other", args.keyword],
                                digits=3))
    cm = confusion_matrix(y_eval, yhat)
    print(f"  confusion: TN={cm[0,0]} FP={cm[0,1]}  FN={cm[1,0]} TP={cm[1,1]}")
    auc = roc_auc_score(y_eval, p_eval)
    print(f"  ROC-AUC: {auc:.4f}")

    # threshold sweep — pick the one that maximizes F1
    from sklearn.metrics import f1_score
    print("\n=== threshold sweep (eval F1) ===")
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        f = f1_score(y_eval, (p_eval > t).astype(np.int8))
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    print(f"  best threshold: {best_t:.2f}   F1: {best_f1:.4f}")

    # ----- save -----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{args.keyword}.joblib"
    payload = {
        "model": clf,
        "keyword": args.keyword,
        "window_frames": WINDOW_FRAMES,
        "embed_dim": EMBED_DIM,
        "feature_dim": WINDOW_FRAMES * EMBED_DIM,
        "suggested_threshold": best_t,
        "eval": {
            "auc": auc,
            "f1_at_best": best_f1,
            "n_eval_pos": int(y_eval.sum()),
            "n_eval_neg": int(len(y_eval) - y_eval.sum()),
        },
        "train": {
            "n_train_pos": int(y_train.sum()),
            "n_train_neg": int(len(y_train) - y_train.sum()),
        },
    }
    joblib.dump(payload, out, compress=3)
    print(f"\n✓ saved → {out.relative_to(ROOT)}  ({out.stat().st_size//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
