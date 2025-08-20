# BirdCLEF 2025 Sound Event Detection (SED) Inference Pipeline

This repository provides a pipeline for performing sound event detection on the BirdCLEF 2025 dataset using pre-trained SED models.

## Table of Contents
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
