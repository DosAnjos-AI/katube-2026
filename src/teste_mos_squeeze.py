import torch
import torchaudio
from pathlib import Path

# Load SHEET predictor
print("Loading SHEET predictor...")
predictor = torch.hub.load("unilight/sheet:v0.1.0", "default", trust_repo=True)
predictor.model.cuda()

# Test audio - CAMINHO RELATIVO
audio_path = Path("../audios/EhzSC3LWez4/EhzSC3LWez4.flac")

# Load audio
waveform, sr = torchaudio.load(str(audio_path))
print(f"Original shape: {waveform.shape}")

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# 🔥 FIX: Remove channel dimension and move to GPU
waveform = waveform.squeeze(0).cuda()  # Shape: [8428287] on GPU
print(f"After squeeze + GPU: shape={waveform.shape}, device={waveform.device}")

# TEST: predict() with wav= argument
try:
    score = predictor.predict(wav=waveform)
    print(f"\n✅ SUCCESS! MOS Score: {score:.2f}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
