import torchaudio

src = "audio/clean.flac"
dst = "audio/clean_48k.wav"

wav, sr = torchaudio.load(src)
print(f"Input: {sr} Hz, shape {wav.shape}")

if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)

resampler = torchaudio.transforms.Resample(sr, 48000)
wav_48k = resampler(wav)

torchaudio.save(dst, wav_48k, 48000)
print(f"Saved {dst} | 48000 Hz | {wav_48k.shape[1]/48000:.2f}s")