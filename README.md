# Birdclef-2025
BirdCLEF 2025 Sound Event Detection (SED) Inference Pipeline

This repository provides a pipeline for performing sound event detection on the BirdCLEF 2025 dataset using pre-trained SED models.

Table of Contents

Overview

Installation

Dataset

Audio Preprocessing

Model Architecture

Inference

Utilities

License

Overview

The pipeline processes audio soundscapes, extracts log-mel spectrograms, and predicts the presence of bird species using ensemble pre-trained models. Special functionality is included to enhance low-ranked class predictions using a power transformation.

Installation

Requires Python 3.9+ and the following libraries:

pip install torch torchaudio timm numpy pandas

Dataset

Test soundscapes: ../input/birdclef-2025/test_soundscapes/

Train soundscapes: ../input/birdclef-2025/train_soundscapes/

Train audio per class: ../input/birdclef-2025/train_audio/

Audio files should be in .ogg format.

Audio Preprocessing

Mono conversion for stereo files.

Audio is sliced into 5-second non-overlapping chunks.

Log-Mel spectrograms are computed using 128 Mel bands.

Spectrogram normalization is applied.

Model Architecture

The pipeline uses a Timm-based CNN with Attention:

Backbone: eca_nfnet_l0

Input channels: 3 (after repeating Mel spectrogram)

Attention block: AttBlockV2 for clipwise and segmentwise predictions

Ensemble: 3 models (sed0.pth, sed1.pth, sed2.pth)

Attention Block

The attention block computes normalized attention over segments and aggregates predictions into clipwise logits.

Inference

Load models from MODELS list.

Convert audio files to Mel spectrograms using audio_to_mel().

Pass through each model to obtain clipwise logits.

Average predictions from all models.

Optionally apply apply_power_to_low_ranked_cols() to emphasize lower-ranked classes.

Save predictions in the format required for BirdCLEF 2025 submission.

Example:

predictions = prediction('sample_audio')

Utilities

apply_power_to_low_ranked_cols(p, top_k=30, exponent=2): Emphasize low-ranked classes in predictions.

interpolate(): Upsample framewise outputs.

pad_framewise_output(): Resize predictions to match audio frame count.

normalize_std(): Standardize Mel spectrograms.

License

This project is released under the MIT License.
