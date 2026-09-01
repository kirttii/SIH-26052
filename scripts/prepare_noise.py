import os
import glob
import torchaudio

SRC = "defence_noise/defence_noise"
DST = "dataset/noise_16k"

categories = [
    "airplane",
    "engine",
    "helicopter",
    "siren",
    "train",
    "wind"
]

os.makedirs(DST, exist_ok=True)

total = 0

for category in categories:
    src_dir = os.path.join(SRC, category)
    dst_dir = os.path.join(DST, category)

    os.makedirs(dst_dir, exist_ok=True)

    files = glob.glob(os.path.join(src_dir, "*.wav"))

    for i, file in enumerate(files):
        waveform, sr = torchaudio.load(file)

        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform, sr, 16000
            )

        filename = os.path.basename(file)
        output = os.path.join(dst_dir, filename)

        torchaudio.save(output, waveform, 16000)

        total += 1

    print(category, ":", len(files), "files converted")

print("\nTOTAL:", total)
print("Output:", DST)