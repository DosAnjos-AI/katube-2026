# ~/katube_teste/katube-novo/test_sheet_api.py
import torch
import torchaudio

# Load SHEET predictor
predictor = torch.hub.load("unilight/sheet:v0.1.0", "default", trust_repo=True)

# Test audio
audio_path = "audios/EhzSC3LWez4/EhzSC3LWez4.flac"

# Method 1: wav_path
try:
    score1 = predictor.predict(wav_path=audio_path)
    print(f"✅ Method 1 (wav_path): {score1}")
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

# Method 2: waveform tensor
try:
    waveform, sr = torchaudio.load(audio_path)
    score2 = predictor.predict(waveform=waveform, sr=sr)
    print(f"✅ Method 2 (waveform): {score2}")
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

# Method 3: Direct call
try:
    waveform, sr = torchaudio.load(audio_path)
    score3 = predictor(waveform, sr)
    print(f"✅ Method 3 (direct call): {score3}")
except Exception as e:
    print(f"❌ Method 3 failed: {e}")

# Inspect predictor methods
print("\n📋 Available methods:")
print([m for m in dir(predictor) if not m.startswith('_')])