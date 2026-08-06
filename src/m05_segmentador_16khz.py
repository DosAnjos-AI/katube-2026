#!/usr/bin/env python3
"""
Module m05_segmentador_16khz.py
Converts audio segments to 16kHz mono when needed
Keeps the original format, copies the metadata JSON
"""

import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Tuple



# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Add root folder to the path to import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EXTENSOES_AUDIO

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def obter_sample_rate(caminho_audio: Path) -> int:
    """
    Gets the sample rate of an audio file using ffprobe

    Args:
        caminho_audio: Path of the audio file

    Returns:
        Sample rate in Hz (e.g., 16000, 44100, 48000)
        Returns 0 if there is an error
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(caminho_audio)
        ]
        resultado = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', check=True)
        return int(resultado.stdout.strip())
    except Exception as e:
        print(f"Erro ao obter SR de {caminho_audio.name}: {e}")
        return 0


def obter_canais(caminho_audio: Path) -> int:
    """
    Gets the number of channels of an audio file using ffprobe

    Args:
        caminho_audio: Path of the audio file

    Returns:
        Number of channels (1=mono, 2=stereo, etc)
        Returns 0 if there is an error
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=channels',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(caminho_audio)
        ]
        resultado = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', check=True)
        return int(resultado.stdout.strip())
    except Exception as e:
        print(f"Erro ao obter canais de {caminho_audio.name}: {e}")
        return 0


def converter_audio_16khz_mono(caminho_origem: Path, caminho_destino: Path) -> bool:
    """
    Converts audio to 16kHz mono, keeping the original format

    Args:
        caminho_origem: Path of the original file
        caminho_destino: Path of the destination file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', str(caminho_origem),
            '-ar', '16000',
            '-ac', '1',  # Force conversion to mono
            '-y',  # Overwrite without asking
            str(caminho_destino)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Erro ao converter {caminho_origem.name}: {e}")
        return False


def copiar_audio(caminho_origem: Path, caminho_destino: Path) -> bool:
    """
    Copies an audio file without conversion

    Args:
        caminho_origem: Path of the original file
        caminho_destino: Path of the destination file

    Returns:
        True if the copy succeeded, False otherwise
    """
    try:
        shutil.copy2(caminho_origem, caminho_destino)
        return True
    except Exception as e:
        print(f"Erro ao copiar {caminho_origem.name}: {e}")
        return False


def processar_audio(caminho_origem: Path, caminho_destino: Path) -> Tuple[bool, str]:
    """
    Processes an audio file: converts if necessary, otherwise copies

    Args:
        caminho_origem: Path of the original file
        caminho_destino: Path of the destination file

    Returns:
        Tuple (sucesso: bool, acao: str)
        acao can be: 'convertido', 'copiado', 'falhou'
    """
    sr_atual = obter_sample_rate(caminho_origem)
    canais_atual = obter_canais(caminho_origem)
    
    if sr_atual == 0 or canais_atual == 0:
        return False, 'falhou'
    
    if sr_atual == 16000 and canais_atual == 1:
        # Audio file is already 16kHz mono, just copy
        sucesso = copiar_audio(caminho_origem, caminho_destino)
        return sucesso, 'copiado' if sucesso else 'falhou'
    else:
        # Needs to convert to 16kHz mono
        sucesso = converter_audio_16khz_mono(caminho_origem, caminho_destino)
        return sucesso, 'convertido' if sucesso else 'falhou'


def listar_arquivos_audio(pasta: Path) -> List[Path]:
    """
    Lists all audio files in the folder

    Args:
        pasta: Path of the folder to search

    Returns:
        List of Paths of the audio files found
    """
    arquivos = []
    for arquivo in pasta.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_AUDIO:
            arquivos.append(arquivo)
    return sorted(arquivos)


def copiar_json(pasta_origem: Path, pasta_destino: Path, audio_id: str) -> bool:
    """
    Copies the metadata JSON file to the destination folder.
    Creates an additional copy in the 00-json_dinamico folder with a
    custom name.

    Args:
        pasta_origem: Path of the source folder
        pasta_destino: Path of the destination folder
        audio_id: Audio ID, used to name the tracking file

    Returns:
        True if both copies succeeded, False otherwise
    """
    try:
        arquivos_json = list(pasta_origem.glob('*.json'))
        if not arquivos_json:
            print("ERRO: Nenhum arquivo JSON encontrado na pasta origem")
            return False

        if len(arquivos_json) > 1:
            print(f"AVISO: Multiplos JSONs encontrados, copiando o primeiro: {arquivos_json[0].name}")

        json_origem = arquivos_json[0]

        # Copy 1: 03-segments_16khz folder with the original name
        json_destino = pasta_destino / json_origem.name
        shutil.copy2(json_origem, json_destino)
        print(f"JSON copiado: {json_origem.name}")

        # Copy 2: 00-json_dinamico folder with a custom name
        pasta_json_dinamico = pasta_destino.parent / "00-json_dinamico"
        pasta_json_dinamico.mkdir(parents=True, exist_ok=True)

        nome_acompanhamento = f"{audio_id}_segments_acompanhamento.json"
        json_acompanhamento = pasta_json_dinamico / nome_acompanhamento

        shutil.copy2(json_origem, json_acompanhamento)
        print(f"JSON acompanhamento copiado: {nome_acompanhamento}")

        return True

    except Exception as e:
        print(f"Erro ao copiar JSON: {e}")
        return False


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def processar_pasta(audio_id: str) -> bool:
    """
    Main function: processes all audio files in the input folder.
    Converts to 16kHz mono when necessary and copies the JSON.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the required output was produced (audio files + JSON),
        False if a precondition is missing, if the JSON could not be
        copied, or if no audio file reached the destination folder.
    """
    print("=" * 70)
    print("INICIANDO CONVERSAO DE AUDIOS PARA 16kHz MONO")
    print("=" * 70)

    # Validate paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    pasta_input = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "02-segmentos_originais"
    pasta_output = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "03-segments_16khz"

    if not pasta_input.exists():
        print(f"ERRO: Pasta de input nao existe: {pasta_input}")
        return False

    # Create output folder if it does not exist
    pasta_output.mkdir(parents=True, exist_ok=True)

    # Validate the JSON's existence before starting
    arquivos_json = list(pasta_input.glob('*.json'))
    if not arquivos_json:
        print("ERRO: Nenhum arquivo JSON encontrado na pasta origem")
        print("Processo abortado - JSON e obrigatorio")
        return False

    # List audio files
    arquivos_audio = listar_arquivos_audio(pasta_input)
    total_arquivos = len(arquivos_audio)

    print(f"\nArquivos encontrados: {total_arquivos}")
    print(f"Pasta origem: {pasta_input}")
    print(f"Pasta destino: {pasta_output}")
    print("-" * 70)

    # Counters
    convertidos = 0
    copiados = 0
    falhas = []

    # Process each file
    print("\nProcessando arquivos...")
    for idx, arquivo_origem in enumerate(arquivos_audio, 1):
        arquivo_destino = pasta_output / arquivo_origem.name
        
        print(f"[{idx}/{total_arquivos}] {arquivo_origem.name}...", end=" ")
        
        sucesso, acao = processar_audio(arquivo_origem, arquivo_destino)
        
        if acao == 'convertido':
            convertidos += 1
            print("convertido")
        elif acao == 'copiado':
            copiados += 1
            print("copiado")
        else:
            falhas.append(arquivo_origem)
            print("FALHOU")
    
    # Second attempt for files that failed
    if falhas:
        print("\n" + "=" * 70)
        print(f"SEGUNDA TENTATIVA - {len(falhas)} arquivo(s) com falha")
        print("=" * 70)
        
        falhas_finais = []
        
        for idx, arquivo_origem in enumerate(falhas, 1):
            arquivo_destino = pasta_output / arquivo_origem.name
            
            print(f"[{idx}/{len(falhas)}] {arquivo_origem.name}...", end=" ")
            
            sucesso, acao = processar_audio(arquivo_origem, arquivo_destino)
            
            if acao == 'convertido':
                convertidos += 1
                print("convertido")
            elif acao == 'copiado':
                copiados += 1
                print("copiado")
            else:
                falhas_finais.append(arquivo_origem.name)
                print("FALHOU NOVAMENTE")
        
        falhas = falhas_finais
    
    # Copy JSON (required: without it, the following modules have no metadata)
    print("\n" + "-" * 70)
    json_copiado = copiar_json(pasta_input, pasta_output, audio_id)

    # Final report
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 70)
    print(f"Total de arquivos: {total_arquivos}")
    print(f"Convertidos (SR/canais alterados): {convertidos}")
    print(f"Copiados (ja 16kHz mono): {copiados}")
    print(f"Falhas finais: {len(falhas)}")

    if falhas:
        print("\nArquivos que falharam apos 2 tentativas:")
        for nome in falhas:
            print(f"  - {nome}")

    print("=" * 70)

    if not json_copiado:
        print("ERRO: JSON obrigatorio nao foi copiado para a pasta de destino")
        return False

    # No audio in the output means the module delivered nothing
    if convertidos + copiados == 0:
        print("ERRO: Nenhum audio produzido em 03-segments_16khz")
        return False

    return True


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Direct execution requires audio_id as an argument - no fixed id in the code
    if len(sys.argv) != 2:
        print("Uso: python src/m05_segmentador_16khz.py <audio_id>")
        sys.exit(1)
    if not processar_pasta(sys.argv[1]):
        sys.exit(1)