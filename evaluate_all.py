import time
import csv
import subprocess
import sys
import re
from pathlib import Path

import torch
import torchaudio
from pesq import pesq
from pystoi import stoi


# ============================================================
# CONFIGURATION
# ============================================================

MIXED_DIR = Path(r"dataset\mixed")
CLEAN_DIR = Path(r"dataset\clean_speech")

OUTPUT_DIR = Path(r"output\baseline")
CSV_FILE = Path(r"output\evaluation_all.csv")

MODEL = "DeepFilterNet"
TARGET_SR = 16000


# ============================================================
# TARGET VALUES FROM PROBLEM STATEMENT
# ============================================================

TARGET_SNR = 15.0
TARGET_STOI = 0.85
TARGET_PESQ = 2.5


# ============================================================
# FIND DEEPFILTERNET EXECUTABLE
# ============================================================

venv_scripts = Path(sys.executable).parent

deepfilter_exe = venv_scripts / "deep-filter-py.exe"

if not deepfilter_exe.exists():
    deepfilter_exe = venv_scripts / "deepFilter.exe"

if not deepfilter_exe.exists():

    print("ERROR: DeepFilterNet executable not found.")
    print("Checked:", venv_scripts)

    sys.exit(1)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):

    audio, sr = torchaudio.load(str(path))

    # Convert stereo to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    else:
        audio = audio.squeeze(0)

    return audio, sr


# ============================================================
# CALCULATE SNR
# ============================================================

def calculate_snr(clean_signal, test_signal):

    noise = clean_signal - test_signal

    signal_power = torch.sum(
        clean_signal ** 2
    )

    noise_power = torch.sum(
        noise ** 2
    )

    if noise_power == 0:

        return float("inf")

    snr = 10 * torch.log10(
        signal_power / noise_power
    )

    return snr.item()


# ============================================================
# FIND CLEAN FILE
# ============================================================

def find_clean_file(sample_id):

    # Example:
    # sample_id = 0000
    #
    # Possible clean file:
    # 1272-128104-0000.wav

    matches = list(
        CLEAN_DIR.glob(
            f"*-{sample_id}.wav"
        )
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# PARSE MIXED FILE NAME
# ============================================================

def parse_filename(filename):

    # Expected:
    #
    # noisy_0000_10dB_engine.wav

    pattern = (
        r"noisy_(\d+)_"
        r"(-?\d+)dB_"
        r"(.+)\.wav$"
    )

    match = re.match(
        pattern,
        filename,
        re.IGNORECASE
    )

    if not match:
        return None

    sample_id = match.group(1)
    snr_level = match.group(2)
    noise_type = match.group(3)

    return (
        sample_id,
        snr_level,
        noise_type
    )


# ============================================================
# GET ALL MIXED FILES
# ============================================================

mixed_files = sorted(
    MIXED_DIR.glob("*.wav")
)

if not mixed_files:

    print(
        "ERROR: No WAV files found in:",
        MIXED_DIR
    )

    sys.exit(1)


print("\n==========================================")
print(" DEEPFILTERNET BATCH EVALUATION")
print("==========================================")

print(
    "Mixed files found:",
    len(mixed_files)
)

print(
    "DeepFilterNet:",
    deepfilter_exe
)

print()


# ============================================================
# CSV COLUMNS
# ============================================================

fieldnames = [

    "sample_id",
    "noise_type",
    "input_snr_db",

    "clean_file",
    "noisy_file",
    "enhanced_file",

    "sample_rate",
    "duration_sec",

    "snr_noisy_db",
    "snr_enhanced_db",
    "snr_improvement_db",

    "stoi_noisy",
    "stoi_enhanced",
    "stoi_improvement",

    "pesq_noisy",
    "pesq_enhanced",
    "pesq_improvement",

    "deepfilter_latency_sec",
    "rtf",

    "snr_pass",
    "stoi_pass",
    "pesq_pass",
    "rtf_pass",

    "overall_pass"
]


# ============================================================
# OPEN MASTER CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()


    # ========================================================
    # PROCESS EVERY MIXED FILE
    # ========================================================

    for index, noisy_path in enumerate(mixed_files, start=1):

        parsed = parse_filename(
            noisy_path.name
        )

        if parsed is None:

            print(
                f"[{index}/{len(mixed_files)}] "
                f"SKIPPED: {noisy_path.name}"
            )

            continue


        sample_id, input_snr, noise_type = parsed


        print()
        print("------------------------------------------")
        print(
            f"[{index}/{len(mixed_files)}]"
        )

        print(
            "Sample:",
            sample_id
        )

        print(
            "Noise:",
            noise_type
        )

        print(
            "Input SNR:",
            input_snr,
            "dB"
        )

        print(
            "File:",
            noisy_path.name
        )


        # ====================================================
        # FIND CLEAN FILE
        # ====================================================

        clean_path = find_clean_file(
            sample_id
        )

        if clean_path is None:

            print(
                "WARNING: Clean file not found "
                f"for sample {sample_id}"
            )

            continue


        # ====================================================
        # RUN DEEPFILTERNET
        # ====================================================

        print(
            "Running DeepFilterNet..."
        )


        # Remove previously generated output
        # with the same input name if necessary.

        expected_output = (
            OUTPUT_DIR /
            (
                noisy_path.stem +
                "_DeepFilterNet.wav"
            )
        )


        start_time = time.perf_counter()


        result = subprocess.run(

            [
                str(deepfilter_exe),

                str(noisy_path),

                "--model-base-dir",
                MODEL,

                "--output-dir",
                str(OUTPUT_DIR)
            ],

            capture_output=True,
            text=True
        )


        end_time = time.perf_counter()


        deepfilter_time = (
            end_time - start_time
        )


        if result.returncode != 0:

            print(
                "ERROR: DeepFilterNet failed."
            )

            print(result.stderr)

            continue


        # ====================================================
        # FIND ENHANCED FILE
        # ====================================================

        # DeepFilterNet normally produces:
        #
        # noisy_0000_10dB_engine_DeepFilterNet.wav

        if expected_output.exists():

            enhanced_path = expected_output

        else:

            # Fallback: search for matching output
            candidates = list(
                OUTPUT_DIR.glob(
                    noisy_path.stem +
                    "_*.wav"
                )
            )

            if not candidates:

                print(
                    "ERROR: Enhanced output not found."
                )

                continue

            enhanced_path = max(
                candidates,
                key=lambda p: p.stat().st_mtime
            )


        print(
            "Processing time:",
            round(deepfilter_time, 4),
            "sec"
        )


        # ====================================================
        # LOAD AUDIO
        # ====================================================

        clean, sr_clean = load_audio(
            clean_path
        )

        noisy, sr_noisy = load_audio(
            noisy_path
        )

        enhanced, sr_enhanced = load_audio(
            enhanced_path
        )


        # ====================================================
        # RESAMPLE TO 16 kHz
        # ====================================================

        if sr_clean != TARGET_SR:

            clean = (
                torchaudio.functional.resample(
                    clean,
                    sr_clean,
                    TARGET_SR
                )
            )


        if sr_noisy != TARGET_SR:

            noisy = (
                torchaudio.functional.resample(
                    noisy,
                    sr_noisy,
                    TARGET_SR
                )
            )


        if sr_enhanced != TARGET_SR:

            enhanced = (
                torchaudio.functional.resample(
                    enhanced,
                    sr_enhanced,
                    TARGET_SR
                )
            )


        # ====================================================
        # SAME LENGTH
        # ====================================================

        length = min(
            len(clean),
            len(noisy),
            len(enhanced)
        )


        clean = clean[:length]
        noisy = noisy[:length]
        enhanced = enhanced[:length]


        duration = (
            length / TARGET_SR
        )


        # ====================================================
        # SNR
        # ====================================================

        snr_noisy = calculate_snr(
            clean,
            noisy
        )

        snr_enhanced = calculate_snr(
            clean,
            enhanced
        )


        snr_improvement = (
            snr_enhanced -
            snr_noisy
        )


        # ====================================================
        # STOI
        # ====================================================

        clean_np = clean.numpy()
        noisy_np = noisy.numpy()
        enhanced_np = enhanced.numpy()


        stoi_noisy = stoi(
            clean_np,
            noisy_np,
            TARGET_SR,
            extended=False
        )


        stoi_enhanced = stoi(
            clean_np,
            enhanced_np,
            TARGET_SR,
            extended=False
        )


        stoi_improvement = (
            stoi_enhanced -
            stoi_noisy
        )


        # ====================================================
        # PESQ
        # ====================================================

        try:

            pesq_noisy = pesq(
                TARGET_SR,
                clean_np,
                noisy_np,
                "wb"
            )

            pesq_enhanced = pesq(
                TARGET_SR,
                clean_np,
                enhanced_np,
                "wb"
            )

        except Exception as e:

            print(
                "PESQ error:",
                e
            )

            pesq_noisy = None
            pesq_enhanced = None


        if (
            pesq_noisy is not None
            and
            pesq_enhanced is not None
        ):

            pesq_improvement = (
                pesq_enhanced -
                pesq_noisy
            )

        else:

            pesq_improvement = None


        # ====================================================
        # RTF
        # ====================================================

        rtf = (
            deepfilter_time /
            duration
        )


        # ====================================================
        # TARGET CHECK
        # ====================================================

        snr_pass = (
            snr_enhanced >=
            TARGET_SNR
        )


        stoi_pass = (
            stoi_enhanced >=
            TARGET_STOI
        )


        pesq_pass = (

            pesq_enhanced is not None
            and
            pesq_enhanced >=
            TARGET_PESQ
        )


        rtf_pass = (
            rtf < 1
        )


        overall_pass = (
            snr_pass
            and
            stoi_pass
            and
            pesq_pass
            and
            rtf_pass
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()
        print("Results:")

        print(
            "SNR:",
            round(snr_noisy, 4),
            "->",
            round(snr_enhanced, 4),
            "dB"
        )

        print(
            "STOI:",
            round(stoi_noisy, 4),
            "->",
            round(stoi_enhanced, 4)
        )

        if pesq_noisy is not None:

            print(
                "PESQ:",
                round(pesq_noisy, 4),
                "->",
                round(pesq_enhanced, 4)
            )

        else:

            print(
                "PESQ: ERROR"
            )

        print(
            "Latency:",
            round(deepfilter_time, 4),
            "sec"
        )

        print(
            "RTF:",
            round(rtf, 4)
        )

        print(
            "Overall:",
            "PASS"
            if overall_pass
            else
            "FAIL"
        )


        # ====================================================
        # WRITE ONE CSV ROW
        # ====================================================

        row = {

            "sample_id":
                sample_id,

            "noise_type":
                noise_type,

            "input_snr_db":
                float(input_snr),

            "clean_file":
                str(clean_path),

            "noisy_file":
                str(noisy_path),

            "enhanced_file":
                str(enhanced_path),

            "sample_rate":
                TARGET_SR,

            "duration_sec":
                round(
                    duration,
                    4
                ),

            "snr_noisy_db":
                round(
                    snr_noisy,
                    4
                ),

            "snr_enhanced_db":
                round(
                    snr_enhanced,
                    4
                ),

            "snr_improvement_db":
                round(
                    snr_improvement,
                    4
                ),

            "stoi_noisy":
                round(
                    stoi_noisy,
                    4
                ),

            "stoi_enhanced":
                round(
                    stoi_enhanced,
                    4
                ),

            "stoi_improvement":
                round(
                    stoi_improvement,
                    4
                ),

            "pesq_noisy":
                (
                    round(
                        pesq_noisy,
                        4
                    )
                    if pesq_noisy is not None
                    else ""
                ),

            "pesq_enhanced":
                (
                    round(
                        pesq_enhanced,
                        4
                    )
                    if pesq_enhanced is not None
                    else ""
                ),

            "pesq_improvement":
                (
                    round(
                        pesq_improvement,
                        4
                    )
                    if pesq_improvement is not None
                    else ""
                ),

            "deepfilter_latency_sec":
                round(
                    deepfilter_time,
                    4
                ),

            "rtf":
                round(
                    rtf,
                    4
                ),

            "snr_pass":
                snr_pass,

            "stoi_pass":
                stoi_pass,

            "pesq_pass":
                pesq_pass,

            "rtf_pass":
                rtf_pass,

            "overall_pass":
                overall_pass
        }


        writer.writerow(row)

        csv_file.flush()


# ============================================================
# FINISHED
# ============================================================

print()
print("==========================================")
print(" BATCH EVALUATION COMPLETE")
print("==========================================")

print(
    "Total WAV files found:",
    len(mixed_files)
)

print(
    "CSV saved to:"
)

print(
    CSV_FILE
)

print("==========================================")