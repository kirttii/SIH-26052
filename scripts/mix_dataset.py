import os
import glob
import random
import csv
import numpy as np
import soundfile as sf
import librosa

# =========================
# PATHS
# =========================

CLEAN_DIR = "dataset/clean_speech"
NOISE_DIR = "dataset/noise_16k"
OUT_DIR = "dataset/mixed"
RESULTS_DIR = "results"

SR = 16000

SNRS = [0, 5, 10, 15, 20]

random.seed(42)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# FIND FILES
# =========================

clean_files = sorted(
    glob.glob(os.path.join(CLEAN_DIR, "*.wav"))
)

noise_files = sorted(
    glob.glob(
        os.path.join(NOISE_DIR, "**", "*.wav"),
        recursive=True
    )
)

print("===================================")
print("DATASET MIXING")
print("===================================")
print("Clean files :", len(clean_files))
print("Noise files :", len(noise_files))
print("SNR levels  :", SNRS)
print()


# =========================
# RMS
# =========================

def rms(signal):
    return np.sqrt(np.mean(signal ** 2) + 1e-9)


# =========================
# MIX AUDIO
# =========================

def mix_audio(clean, noise, snr_db):

    # Repeat noise if shorter than speech
    if len(noise) < len(clean):
        repeats = int(
            np.ceil(len(clean) / len(noise))
        )
        noise = np.tile(noise, repeats)

    # Select random section of noise
    start = random.randint(
        0,
        len(noise) - len(clean)
    )

    noise = noise[
        start:start + len(clean)
    ]

    # Calculate desired noise RMS
    clean_rms = rms(clean)

    desired_noise_rms = (
        clean_rms /
        (10 ** (snr_db / 20))
    )

    noise_rms = rms(noise)

    noise = noise * (
        desired_noise_rms / noise_rms
    )

    # Mix
    noisy = clean + noise

    # Prevent clipping
    peak = np.max(np.abs(noisy))

    if peak > 0.99:
        noisy = noisy / peak * 0.99

    return noisy.astype(np.float32)


# =========================
# CLEAN OLD NOISY FILES
# =========================

old_noisy = glob.glob(
    os.path.join(OUT_DIR, "noisy_*.wav")
)

for file in old_noisy:
    os.remove(file)

print("Removed old noisy files:", len(old_noisy))
print()


# =========================
# MANIFEST
# =========================

manifest_path = os.path.join(
    RESULTS_DIR,
    "mixing_manifest.csv"
)

manifest_rows = []


# =========================
# CREATE DATASET
# =========================

total_noisy = 0

for i, clean_path in enumerate(clean_files):

    clean, _ = librosa.load(
        clean_path,
        sr=SR,
        mono=True
    )

    clean_id = f"{i:04d}"

    print(
        f"[{i+1}/{len(clean_files)}] "
        f"{os.path.basename(clean_path)}"
    )

    for snr in SNRS:

        # Select exact defence noise
        noise_path = random.choice(
            noise_files
        )

        noise, _ = librosa.load(
            noise_path,
            sr=SR,
            mono=True
        )

        noisy = mix_audio(
            clean,
            noise,
            snr
        )

        # Noise category
        noise_category = os.path.basename(
            os.path.dirname(noise_path)
        )

        noise_filename = os.path.basename(
            noise_path
        )

        noisy_filename = (
            f"noisy_{clean_id}_"
            f"{snr}dB_"
            f"{noise_category}.wav"
        )

        noisy_path = os.path.join(
            OUT_DIR,
            noisy_filename
        )

        sf.write(
            noisy_path,
            noisy,
            SR
        )

        # Store experiment information
        manifest_rows.append({
            "sample_id": clean_id,
            "clean_file": os.path.basename(clean_path),
            "noise_file": noise_filename,
            "noise_category": noise_category,
            "requested_snr_db": snr,
            "noisy_file": noisy_filename
        })

        total_noisy += 1


# =========================
# SAVE MANIFEST
# =========================

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "sample_id",
        "clean_file",
        "noise_file",
        "noise_category",
        "requested_snr_db",
        "noisy_file"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# =========================
# FINAL REPORT
# =========================

print()
print("===================================")
print("MIXING COMPLETE")
print("===================================")
print("Clean files :", len(clean_files))
print("Noisy files :", total_noisy)
print("Total WAV   :", total_noisy)
print("Manifest    :", manifest_path)
print("Output      :", OUT_DIR)