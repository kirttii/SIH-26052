AI-Powered Robust Speech Enhancement for Defence Communication

An AI-based speech enhancement project focused on improving speech intelligibility in noisy defence and high-noise communication environments.

What We Are Building

The system takes clean speech, adds different types of realistic environmental and defence-related noise at multiple SNR levels, and processes the noisy speech using an AI-based speech enhancement model.

Clean Speech  
     ↓  
Noise + Speech Mixing  
     ↓  
Different SNR Levels  
     ↓  
Noisy Speech  
     ↓  
DeepFilterNet3  
     ↓  
Enhanced Speech  
     ↓  
SNR / STOI / PESQ Evaluation  

Noise Conditions  
The current dataset pipeline includes noise categories such as:  
Helicopter  
Airplane  
Engine / Vehicle  
Wind  
Train  
Siren  
Artillery  
Drone  
The noisy speech is generated at multiple SNR levels to evaluate the system under different noise conditions.  
Current Model  
We are currently using DeepFilterNet3 as the baseline speech enhancement model.  
The baseline is evaluated using:  
SNR – Noise reduction performance  
STOI – Speech intelligibility  
PESQ – Perceived speech quality  
RTF / Latency – Computational performance  
Current Progress  
Speech and noise preprocessing pipeline  
Noise preparation and conversion  
Noisy-clean speech pair generation  
Multiple SNR conditions  
DeepFilterNet3 baseline integration  
Batch evaluation pipeline  
Performance analysis in progress  
Model experimentation and fine-tuning in progress  
Real-time prototype in progress  
Project Structure  
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
Status  
Work in Progress  
The current repository contains the initial speech enhancement and evaluation pipeline. Further work will focus on improving enhancement quality, reducing computational latency, testing defence-specific noise conditions, and developing the final real-time prototype.  
  
This is correct
