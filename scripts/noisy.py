import torchaudio, torch, os, random, math
from pathlib import Path

clean_path = "audio/clean_48k.wav"
clean, sr = torchaudio.load(clean_path)
print(f"Clean: {sr} Hz, {clean.shape}")

# Create fake defence noises if you don't have real files yet
os.makedirs("audio/noise", exist_ok=True)

# For now: generate 3 types of noise to test pipeline
# Later replace with real gunshot.wav, helicopter.wav, engine.wav
def make_noise(type, length):
    if type == "gunshot":
        # impulsive bursts
        noise = torch.zeros(1, length)
        for _ in range(5):
            idx = random.randint(0, length-2000)
            noise[0, idx:idx+1000] = torch.randn(1000) * 2
        return noise
    elif type == "helicopter":
        # low freq rumble 80Hz
        t = torch.arange(length)/sr
        return (torch.sin(2*math.pi*80*t) * 0.5).unsqueeze(0) + torch.randn(1,length)*0.1
    else: # engine
        t = torch.arange(length)/sr
        return (torch.sin(2*math.pi*150*t) * 0.3).unsqueeze(0) + torch.randn(1,length)*0.2

os.makedirs("audio/noisy", exist_ok=True)

snrs = [-5, 0, 5, 10, 20]
types = ["gunshot", "helicopter", "engine"]

for ntype in types:
    noise = make_noise(ntype, clean.shape[1])
    torchaudio.save(f"audio/noise/{ntype}_48k.wav", noise, sr)
    print(f"Saved audio/noise/{ntype}_48k.wav")

    for snr_db in snrs:
        # SNR mixing: clean / noisy
        clean_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = torch.sqrt(clean_power / (noise_power * (10**(snr_db/10))))
        noisy = clean + scale * noise
        out = f"audio/noisy/{ntype}_{snr_db}dB.wav"
        torchaudio.save(out, noisy, sr)
        print(f" -> {out} at {snr_db} dB")

print("Done! Check audio/noisy folder")