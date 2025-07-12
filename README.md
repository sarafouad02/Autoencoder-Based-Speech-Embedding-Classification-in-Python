# Autoencoder-Based-Speech-Embedding-Classification-in-Python

# ReducedMNIST Audio: Data Augmentation & Autoencoder Pipelines

## Overview

This project explores representation learning and classification on spoken-digit spectrograms using two strategies:

1. **Baseline Averaged-Frame Classifier**: Simple MLP on averaged spectrogram frames.
2. **Autoencoder-Based Embeddings**: Three AE pipelines to encode entire utterances into fixed-length vectors, followed by MLP classification.

Datasets:
  • `Train_spectrograms`: PNG spectrograms of digits 0–9 from multiple speakers.
  • `Test_spectrograms`: Held-out utterances for evaluation.

## Requirements

```bash
Python 3.8+
pip install numpy Pillow torch torchvision
```

## 1. Baseline Classifier

* **Feature Extraction**: Load spectrogram (128 × T); average over time → 128‑dim vector.
* **Model**: 128 → 64 (ReLU) → 10 (logits) MLP.
* **Training**: 20 epochs, Adam lr=1e‑3, batch size=32.

```bash
python baseline_classifier.py
```

Outputs training loss per epoch, final test accuracy.

## 2. Autoencoder Pipelines

Each AE learns latent embeddings of entire utterances for downstream classification.

### 2.1 Concatenation + Padding (`autoencoder_concat.py`)

1. **Dataset**: Pad each spectrogram to `max_frames` T; flatten → 128×T input vector.
2. **Autoencoder**: 〈input\_dim → 512 → latent\_dim (e.g. 256) → 512 → input\_dim〉.
3. **Latent Extraction**: Use encoder to obtain z for each utterance.
4. **Classifier**: 256 → 64 (ReLU) → 10 MLP; 20 epochs.

```bash
python autoencoder_concat.py
```

Logs AE MSE loss and classifier accuracy.

### 2.2 Sliding-Window Embeddings (`ae_sliding_window_embeddings.py`)

1. **Frame Pairs**: Train AE on concatenated frame‑pairs (2×128 → latent 128 → reconstruct 256).
2. **Utterance Embedding**: Recursively encode pair (prev\_z, next\_frame) → new z; average all z’s → 128‑dim vector.
3. **Classifier**: 128 → 64 (ReLU) → 10 MLP; 20 epochs.

```bash
python ae_sliding_window_embeddings.py
```

Prints AE training MSE and classifier test accuracy.

### 2.3 Minimum-Error Sliding-Window (`ae_min_error_embeddings.py`)

1. **Frame-Pair AE**: Same as sliding-window AE.
2. **Min-Error Embedding**: At each step, form multiple candidates (latent + frame, frame + frame, etc.), pick lowest reconstruction error to compute next z; average z’s.
3. **Classifier**: 128 → 64 (ReLU) → 10 MLP; 20 epochs.

