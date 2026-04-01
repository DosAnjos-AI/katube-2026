import logging
from pathlib import Path
from diarizer import EnhancedDiarizer
import logging
from pathlib import Path
from diarizer import EnhancedDiarizer
import torch  # ← ADICIONAR ESTA LINHA
import torchaudio  # ← ADICIONAR ESTA
logging.basicConfig(level=logging.DEBUG)

# Testar inicialização
diarizer = EnhancedDiarizer()

print(f"Pipeline object: {diarizer.pipeline}")
print(f"Pipeline is None: {diarizer.pipeline is None}")
print(f"Pipeline type: {type(diarizer.pipeline)}")
print(f"Device: {diarizer.device}")

# CAMINHO CORRETO (sem ...)
audio_path = Path("/home/anjos/Dropbox/PROJETO CEIA/Alcateia/Katube_2025_new/katube-novo/audios/EhzSC3LWez4/EhzSC3LWez4.flac")
print(f"\nCaminho do áudio: {audio_path}")
print(f"Arquivo existe? {audio_path.exists()}")

# Testar diarização
# Primeiro, testar manualmente o pipeline
print("\n=== TESTE MANUAL DO PIPELINE ===")
audio_path_test = Path("/home/anjos/Dropbox/PROJETO CEIA/Alcateia/Katube_2025_new/katube-novo/audios/EhzSC3LWez4/EhzSC3LWez4.flac")

# Carregar áudio
import torchaudio
waveform, sr = torchaudio.load(audio_path_test)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Rodar pipeline diretamente
audio_input = {"waveform": waveform, "sample_rate": sr}
result = diarizer.pipeline(audio_input)

print(f"Tipo do resultado: {type(result)}")
print(f"Atributos disponíveis: {dir(result)}")
print(f"Tem 'segments'? {hasattr(result, 'segments')}")
print(f"Tem 'labels'? {hasattr(result, 'labels')}")

# Ver o que tem dentro
if hasattr(result, '__dict__'):
    print(f"Conteúdo: {result.__dict__}")