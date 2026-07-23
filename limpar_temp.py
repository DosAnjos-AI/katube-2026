from pathlib import Path

AUDIOS_DIR = Path("/home/ubuntu/spotify2026/arquivos/audios")
TEMP_DIR   = Path("/home/ubuntu/spotify2026/arquivos/temp")

temp_pastas = {p.name for p in TEMP_DIR.iterdir() if p.is_dir()}

print(f"Pastas em temp:   {len(temp_pastas)}")
print(f"Pastas em audios: {len(list(AUDIOS_DIR.iterdir()))}")
print("Iniciando limpeza...\n")

removidas = 0
for nome in temp_pastas:
    alvo = AUDIOS_DIR / nome
    if alvo.is_dir():
        import shutil
        shutil.rmtree(alvo)
        removidas += 1

print(f"Pastas removidas:       {removidas}")
print(f"Pastas em audios agora: {len(list(AUDIOS_DIR.iterdir()))}")
