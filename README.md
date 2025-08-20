# BirdCLEF 2025 Sound Event Detection (SED) Inference Pipeline

This repository provides a pipeline for performing sound event detection on the BirdCLEF 2025 dataset using pre-trained SED models.

# Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Audio Preprocessing](#audio-preprocessing)
- [Model Architecture](#model-architecture)
- [Inference](#inference)
- [Utilities](#utilities)
- [License](#license)

## Overview
The pipeline processes audio soundscapes, extracts log-Mel spectrograms, and predicts the presence of bird species using ensemble pre-trained models. It also includes functionality to enhance low-ranked class predictions using a power transformation.

## Installation
Requires Python 3.9+ and the following libraries:

```bash
pip install torch torchaudio timm numpy pandas

# BirdCLEF 2025 Audio Classification Pipeline

This repository provides a pipeline for bird sound classification using the **BirdCLEF 2025** dataset.  
It includes audio preprocessing, feature extraction, a Timm-based CNN with attention, and ensemble inference.

---

## Dataset Structure

The pipeline expects the **BirdCLEF 2025 dataset** organized as follows:

../input/birdclef-2025/
├── test_soundscapes/
├── train_soundscapes/
└── train_audio/

markdown
Copy
Edit

- **Test soundscapes:** `../input/birdclef-2025/test_soundscapes/`  
- **Train soundscapes:** `../input/birdclef-2025/train_soundscapes/`  
- **Train audio per class:** `../input/birdclef-2025/train_audio/`  

**Note:** Audio files should be in `.ogg` format.  

---

## Audio Preprocessing

The pipeline performs the following preprocessing steps:

1. Convert stereo audio files to **mono**.  
2. Slice audio into **5-second non-overlapping chunks**.  
3. Compute **log-Mel spectrograms** with **128 Mel bands**.  
4. Apply **spectrogram normalization**.  

---

## Model Architecture

- **Backbone:** `eca_nfnet_l0` (from Timm)  
- **Input channels:** 3 (after repeating Mel spectrogram)  
- **Attention block:** `AttBlockV2` for clipwise and segmentwise predictions  
- **Ensemble:** 3 models (`sed0.pth`, `sed1.pth`, `sed2.pth`)  

### Attention Block

- Computes **normalized attention** over segments.  
- Aggregates segmentwise predictions into **clipwise logits**.  

---

## Inference

1. Load models from the `MODELS` list.  
2. Convert audio files to Mel spectrograms using `audio_to_mel()`.  
3. Pass through each model to obtain **clipwise logits**.  
4. Average predictions from all models.  
5. Optionally, apply `apply_power_to_low_ranked_cols()` to emphasize **lower-ranked classes**.  
6. Save predictions in the format required for BirdCLEF 2025 submission.

Example:

```python
predictions = prediction('sample_audio')
