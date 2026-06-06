# Train a custom wake word for this Jetson

You can run the entire training pipeline on a free Google Colab T4 GPU. Most of the work — generating ~1000 synthetic samples in 50+ voices, augmenting with noise + reverb, training, exporting to ONNX — is wrapped in **the official OpenWakeWord training notebook** so we don't reinvent it.

## End-to-end (~25 min wall clock)

### 1. Open the notebook

[**OpenWakeWord — Automatic Custom Model Training (official Colab)**](https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)

### 2. Sign in with any Google account (free tier is fine)

### 3. Change the target word

In the cell labeled **"Set the target word for the model"**, set:

```python
target_phrase = ["Arijit", "arijit"]   # both capitalizations
```

You can add more variations — even mis-spellings that catch how Piper TTS might phonetically render the name:

```python
target_phrase = ["Arijit", "arijit", "Aurijit", "Arrijit"]
```

### 4. (Optional but recommended) Set runtime to GPU

`Runtime → Change runtime type → Hardware accelerator: T4 GPU`

Without GPU, training takes ~60 min. With T4 GPU it's ~10–15 min.

### 5. Run all cells

`Runtime → Run all`. The notebook will:

- install dependencies (~3 min)
- generate ~1000 positive samples of "Arijit" using Piper TTS across ~50 voices
- download OpenWakeWord's pre-curated negative dataset (~3 GB)
- compute embeddings for everything (~3 min)
- train classifier for ~10000 steps (~6 min on T4)
- export to ONNX

When it's done, the last cell shows a download link for `arijit.onnx` (or whichever filename matches your target_phrase).

### 6. Copy to the Jetson

From your Mac:

```bash
scp ~/Downloads/arijit.onnx jetson@jetson.local:/home/jetson/jetson-wakeword-studio/models/
```

### 7. Refresh the web UI

Open `http://<jetson-ip>:8082`. Your new model appears in the "Custom" group at the top of the model picker, ahead of the pre-trained list. Tick it, hit "Start listening", say "Arijit".

## Expected accuracy

For a 3-syllable distinct word like "Arijit" with the standard pipeline:

- **Recall**: 90–95% on first try
- **False positives**: < 1 per hour in normal conversation

If accuracy isn't great after the first deployment, the next step is **the verifier model** (phase 2) which fine-tunes for your specific voice + mic:

1. Use the web UI to record 20–30 real samples of you saying the word
2. Run `apps/train_verifier.py --name arijit` (sklearn only, no GPU needed, ~30 s)
3. The verifier head sits on top of the base ONNX and rejects most false positives without hurting recall

(Phase 2 verifier UI coming after the Colab base model is working.)

## Training multiple keywords

Each keyword is its own ONNX file. Just re-run the Colab notebook with a different `target_phrase`. ~25 min per word. The web UI handles any number of custom models — they all show up in the list and can be enabled together.
