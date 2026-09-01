# AI-Powered Robust Speech Enhancement for Defence Communication

An AI-based speech enhancement project focused on improving speech intelligibility in noisy defence and high-noise communication environments.

## What We Are Building

The system takes clean speech, adds realistic environmental and defence-related noise at multiple SNR levels, and processes the noisy speech using an AI-based speech enhancement model.

### Pipeline

Clean Speech → Noise + Speech Mixing → Multiple SNR Levels → Noisy Speech → DeepFilterNet3 → Enhanced Speech → Objective Evaluation

## Noise Conditions

The current dataset pipeline includes:

- Helicopter
- Airplane
- Engine / Vehicle
- Wind
- Train
- Siren
- Artillery
- Drone

Noisy speech is generated at multiple SNR levels to evaluate the system under different noise conditions.

## Current Model

We are currently using **DeepFilterNet3** as the baseline speech enhancement model.

The baseline is evaluated using:

- **SNR** – Noise reduction performance
- **STOI** – Speech intelligibility
- **PESQ** – Perceived speech quality
- **RTF / Latency** – Computational performance

## Current Progress

- Speech and noise preprocessing pipeline
- Noise preparation and conversion
- Noisy-clean speech pair generation
- Multiple SNR conditions
- DeepFilterNet3 baseline integration
- Batch evaluation pipeline
- Performance analysis in progress
- Model experimentation and fine-tuning in progress
- Real-time prototype in progress

## Project Structure

SIH-26052/
│
├── scripts/
│   ├── audio.py
│   ├── evaluate.py
│   ├── mix_dataset.py
│   ├── noisy.py
│   ├── prepare_noise.py
│   └── stft.py
│
├── evaluate_all.py
├── .gitignore
│
├── dataset/        # Local dataset (not uploaded)
├── output/         # Generated outputs (not uploaded)
├── models/         # Local models (not uploaded)
└── defence_noise/  # Local noise files (not uploaded)

Dataset, audio files, generated outputs, and model files are kept locally and excluded from the repository because of their size.

## Status

**Work in Progress**

The current repository contains the initial speech enhancement and evaluation pipeline. Further work will focus on improving enhancement quality, verifying evaluation metrics, reducing computational latency, testing defence-specific noise conditions, and developing the final real-time prototype.
