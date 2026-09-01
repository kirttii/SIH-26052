import torchaudio, torch, matplotlib.pyplot as plt

waveform, sr = torchaudio.load("audio/clean_48k.wav")
print(f"Loaded {sr} Hz")

n_fft = 1024
hop = 256
win = torch.hann_window(n_fft)

stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
mag = stft.abs()
mag_db = 20*torch.log10(mag + 1e-8)

plt.figure(figsize=(10,4))
plt.imshow(mag_db[0].numpy(), aspect='auto', origin='lower')
plt.title("Clean Spectrogram 48kHz")
plt.xlabel("Time frames")
plt.ylabel("Freq bins")
plt.colorbar(label="dB")
plt.tight_layout()
plt.savefig("output/spectrogram_clean.png", dpi=150)
print("Saved output/spectrogram_clean.png")