# 🎙️ AI-Powered Robust Speech Enhancement for Defence Communication

An AI-driven speech enhancement pipeline designed to improve speech intelligibility in noisy and dynamic environments such as defence, aerospace, transportation, and industrial communication.

The system generates realistic noisy-clean speech pairs, applies AI-based speech enhancement using DeepFilterNet, and evaluates the enhanced output using objective speech-quality metrics.

---

## 🎯 Problem Statement

Communication in defence and high-noise environments can be severely affected by:

- 🚁 Helicopter and aircraft noise
- 🚗 Vehicle and engine noise
- 💨 Wind noise
- 🚂 Train and transportation noise
- 🚨 Sirens and impulsive environmental sounds
- 🔊 Low-SNR communication conditions

Traditional noise reduction techniques may struggle with rapidly changing and non-stationary noise.

This project explores an AI-based approach for robust speech enhancement while measuring both speech quality and computational performance.

---

## 💡 Proposed Solution

The project implements an end-to-end speech enhancement pipeline:

```text
Clean Speech
     ↓
Noise Selection
     ↓
Noise + Speech Mixing
     ↓
Multiple SNR Conditions
     ↓
Noisy Speech
     ↓
DeepFilterNet
     ↓
Enhanced Speech
     ↓
Objective Evaluation
     ↓
SNR | STOI | PESQ | Latency | RTF
