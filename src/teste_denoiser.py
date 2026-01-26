#!/usr/bin/env python3
"""
Teste com audio em CPU
"""

import torch
from df import enhance, init_df
import librosa

# Inicializa modelo
model, df_state, sr = init_df(post_filter=1, log_level="ERROR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Modelo na GPU
print(f"[INFO] Modelo em: {device}")

# Carrega audio
audio_path = "arquivos/temp/QN7gUP7nYhQ/02-segmentos_originais/QN7gUP7nYhQ_001.flac"
audio, _ = librosa.load(audio_path, sr=48000, mono=True)

# MANTÉM AUDIO EM CPU
audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # ← SEM .to(device)
print(f"[INFO] Audio tensor em: {audio_tensor.device}")

# Processa
print("[INFO] Processando...")
result = enhance(model, df_state, audio_tensor, atten_lim_db=0.95)

print(f"[OK] Sucesso! Result device: {result.device}")
print(f"[OK] Result shape: {result.shape}")

# Converte para numpy
result_np = result.cpu().numpy() if result.is_cuda else result.numpy()
print(f"[OK] Numpy shape: {result_np.shape}")