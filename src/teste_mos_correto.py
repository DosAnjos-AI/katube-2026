import torch
import torchaudio
from pathlib import Path

# Load SHEET predictor
print("Loading SHEET predictor...")
predictor = torch.hub.load("unilight/sheet:v0.1.0", "default", trust_repo=True)
predictor.model.cuda()  # Move to GPU

# Test audio - CAMINHO ABSOLUTO
audio_path = Path.home() / "katube_teste/katube-novo/audios/EhzSC3LWez4/EhzSC3LWez4.flac"
print(f"Audio path: {audio_path}")
print(f"File exists: {audio_path.exists()}")

if not audio_path.exists():
    print("❌ File not found!")
    exit(1)

# Method 1: wav_path com Path absoluto
try:
    score1 = predictor.predict(wav_path=str(audio_path))
    print(f"✅ Method 1 (wav_path): MOS = {score1:.2f}")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: Load audio and move to GPU
try:
    waveform, sr = torchaudio.load(str(audio_path))
    waveform = waveform.cuda()  # Move to GPU
    print(f"Waveform shape: {waveform.shape}, SR: {sr}, Device: {waveform.device}")
    
    # Try calling predictor directly
    score2 = predictor({"wav": waveform, "sr": sr})
    print(f"✅ Method 2 (dict): MOS = {score2:.2f}")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Method 3: Check predictor signature
print("\n📋 Predictor info:")
print(f"Type: {type(predictor)}")
print(f"Has __call__: {hasattr(predictor, '__call__')}")

# Inspect predict method
import inspect
print(f"\nPredict signature: {inspect.signature(predictor.predict)}")
