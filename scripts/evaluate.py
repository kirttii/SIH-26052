import time
import numpy as np
import soundfile as sf
from pystoi import stoi
from pesq import pesq


SR = 16000

CLEAN = "dataset/mixed/clean_0000.wav"
NOISY = "dataset/mixed/noisy_0000_10dB_airplane.wav"

# Change this filename if DeepFilterNet produced a different name
ENHANCED = "output/baseline/noisy_0000_10dB_airplane_DeepFilterNet2.wav"


def load_audio(path):
    audio, sr = sf.read(path)

    if sr != SR:
        raise ValueError(
            f"{path} has sample rate {sr}, expected {SR}"
        )

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32)


def match_length(a, b):
    length = min(len(a), len(b))
    return a[:length], b[:length]


def calculate_snr(clean, test):
    clean, test = match_length(clean, test)

    noise = clean - test

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    return 10 * np.log10(
        (signal_power + 1e-10) /
        (noise_power + 1e-10)
    )


print("=" * 50)
print("AUDIO ENHANCEMENT EVALUATION")
print("=" * 50)

print("\nLoading audio...")

clean = load_audio(CLEAN)
noisy = load_audio(NOISY)

print("Clean :", len(clean) / SR, "seconds")
print("Noisy :", len(noisy) / SR, "seconds")


# -----------------------------
# BASELINE: NOISY AUDIO
# -----------------------------

noisy_clean, noisy_ref = match_length(noisy, clean)

snr_noisy = calculate_snr(noisy_ref, noisy_clean)

stoi_noisy = stoi(
    noisy_ref,
    noisy_clean,
    SR,
    extended=False
)

pesq_noisy = pesq(
    SR,
    noisy_ref,
    noisy_clean,
    "wb"
)


print("\n--- NOISY AUDIO ---")
print(f"SNR  : {snr_noisy:.2f} dB")
print(f"STOI : {stoi_noisy:.3f}")
print(f"PESQ : {pesq_noisy:.3f}")


# -----------------------------
# ENHANCED AUDIO
# -----------------------------

print("\nLoading enhanced audio...")

start = time.perf_counter()

enhanced = load_audio(ENHANCED)

processing_time = time.perf_counter() - start

enhanced_clean, enhanced_ref = match_length(
    enhanced,
    clean
)

snr_enhanced = calculate_snr(
    enhanced_ref,
    enhanced_clean
)

stoi_enhanced = stoi(
    enhanced_ref,
    enhanced_clean,
    SR,
    extended=False
)

pesq_enhanced = pesq(
    SR,
    enhanced_ref,
    enhanced_clean,
    "wb"
)


duration = len(enhanced) / SR

print("\n--- ENHANCED AUDIO ---")
print(f"SNR  : {snr_enhanced:.2f} dB")
print(f"STOI : {stoi_enhanced:.3f}")
print(f"PESQ : {pesq_enhanced:.3f}")
print(f"Audio duration : {duration:.2f} sec")
print(f"Load/evaluation time : {processing_time:.3f} sec")


print("\n" + "=" * 50)
print("IMPROVEMENT")
print("=" * 50)

print(f"SNR improvement  : {snr_enhanced - snr_noisy:+.2f} dB")
print(f"STOI improvement : {stoi_enhanced - stoi_noisy:+.3f}")
print(f"PESQ improvement : {pesq_enhanced - pesq_noisy:+.3f}")

print("\nEvaluation complete.")