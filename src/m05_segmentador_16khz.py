#!/usr/bin/env python3
"""
Modulo m05_segmentador_16khz.py
Converte segmentos de audio para 16kHz quando necessario
Mantem formato original, copia JSON de metadados
"""

import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Tuple

id_video= 'kzwB1kLsLes'

# ==============================================================================
# CONFIGURACAO DE CAMINHOS
# ==============================================================================

PASTA_INPUT = "arquivos/temp/"+id_video+"/02-segmentos_originais"  # Alterar para pasta de origem
PASTA_OUTPUT = "arquivos/temp/"+id_video+"/03-segments_16khz"  # Alterar para pasta destino

# ==============================================================================
# EXTENSOES DE AUDIO SUPORTADAS
# ==============================================================================

EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def obter_sample_rate(caminho_audio: Path) -> int:
    """
    Obtem o sample rate de um arquivo de audio usando ffprobe
    
    Args:
        caminho_audio: Path do arquivo de audio
        
    Returns:
        Sample rate em Hz (ex: 16000, 44100, 48000)
        Retorna 0 se houver erro
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
        resultado = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(resultado.stdout.strip())
    except Exception as e:
        print(f"Erro ao obter SR de {caminho_audio.name}: {e}")
        return 0


def converter_audio_16khz(caminho_origem: Path, caminho_destino: Path) -> bool:
    """
    Converte audio para 16kHz mantendo formato original
    
    Args:
        caminho_origem: Path do arquivo original
        caminho_destino: Path do arquivo destino
        
    Returns:
        True se conversao bem-sucedida, False caso contrario
    """
    try:
        cmd = [
            'ffmpeg',
            '-i', str(caminho_origem),
            '-ar', '16000',
            '-y',  # Sobrescrever sem perguntar
            str(caminho_destino)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except Exception as e:
        print(f"Erro ao converter {caminho_origem.name}: {e}")
        return False


def copiar_audio(caminho_origem: Path, caminho_destino: Path) -> bool:
    """
    Copia arquivo de audio sem conversao
    
    Args:
        caminho_origem: Path do arquivo original
        caminho_destino: Path do arquivo destino
        
    Returns:
        True se copia bem-sucedida, False caso contrario
    """
    try:
        shutil.copy2(caminho_origem, caminho_destino)
        return True
    except Exception as e:
        print(f"Erro ao copiar {caminho_origem.name}: {e}")
        return False


def processar_audio(caminho_origem: Path, caminho_destino: Path) -> Tuple[bool, str]:
    """
    Processa um arquivo de audio: converte se necessario ou copia
    
    Args:
        caminho_origem: Path do arquivo original
        caminho_destino: Path do arquivo destino
        
    Returns:
        Tupla (sucesso: bool, acao: str)
        acao pode ser: 'convertido', 'copiado', 'falhou'
    """
    sr_atual = obter_sample_rate(caminho_origem)
    
    if sr_atual == 0:
        return False, 'falhou'
    
    if sr_atual == 16000:
        # Audio ja esta em 16kHz, apenas copiar
        sucesso = copiar_audio(caminho_origem, caminho_destino)
        return sucesso, 'copiado' if sucesso else 'falhou'
    else:
        # Precisa converter para 16kHz
        sucesso = converter_audio_16khz(caminho_origem, caminho_destino)
        return sucesso, 'convertido' if sucesso else 'falhou'


def listar_arquivos_audio(pasta: Path) -> List[Path]:
    """
    Lista todos os arquivos de audio na pasta
    
    Args:
        pasta: Path da pasta para buscar
        
    Returns:
        Lista de Path dos arquivos de audio encontrados
    """
    arquivos = []
    for arquivo in pasta.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_AUDIO:
            arquivos.append(arquivo)
    return sorted(arquivos)


def copiar_json(pasta_origem: Path, pasta_destino: Path) -> bool:
    """
    Copia o arquivo JSON de metadados para pasta destino
    
    Args:
        pasta_origem: Path da pasta de origem
        pasta_destino: Path da pasta de destino
        
    Returns:
        True se copia bem-sucedida, False caso contrario
    """
    try:
        arquivos_json = list(pasta_origem.glob('*.json'))
        if not arquivos_json:
            print("ERRO: Nenhum arquivo JSON encontrado na pasta origem")
            return False
        
        if len(arquivos_json) > 1:
            print(f"AVISO: Multiplos JSONs encontrados, copiando o primeiro: {arquivos_json[0].name}")
        
        json_origem = arquivos_json[0]
        json_destino = pasta_destino / json_origem.name
        
        shutil.copy2(json_origem, json_destino)
        print(f"JSON copiado: {json_origem.name}")
        return True
        
    except Exception as e:
        print(f"Erro ao copiar JSON: {e}")
        return False


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def processar_pasta():
    """
    Funcao principal: processa todos os audios da pasta input
    Converte para 16kHz quando necessario e copia JSON
    """
    print("=" * 70)
    print("INICIANDO CONVERSAO DE AUDIOS PARA 16kHz")
    print("=" * 70)
    
    # Validar caminhos
    pasta_input = Path(PASTA_INPUT)
    pasta_output = Path(PASTA_OUTPUT)
    
    if not pasta_input.exists():
        print(f"ERRO: Pasta de input nao existe: {PASTA_INPUT}")
        return
    
    # Criar pasta output se nao existir
    pasta_output.mkdir(parents=True, exist_ok=True)
    
    # Validar existencia do JSON antes de iniciar
    arquivos_json = list(pasta_input.glob('*.json'))
    if not arquivos_json:
        print("ERRO: Nenhum arquivo JSON encontrado na pasta origem")
        print("Processo abortado - JSON e obrigatorio")
        return
    
    # Listar arquivos de audio
    arquivos_audio = listar_arquivos_audio(pasta_input)
    total_arquivos = len(arquivos_audio)
    
    print(f"\nArquivos encontrados: {total_arquivos}")
    print(f"Pasta origem: {pasta_input}")
    print(f"Pasta destino: {pasta_output}")
    print("-" * 70)
    
    # Contadores
    convertidos = 0
    copiados = 0
    falhas = []
    
    # Processar cada arquivo
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
    
    # Segunda tentativa para arquivos que falharam
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
    
    # Copiar JSON
    print("\n" + "-" * 70)
    copiar_json(pasta_input, pasta_output)
    
    # Relatorio final
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 70)
    print(f"Total de arquivos: {total_arquivos}")
    print(f"Convertidos (SR alterado): {convertidos}")
    print(f"Copiados (SR ja era 16kHz): {copiados}")
    print(f"Falhas finais: {len(falhas)}")
    
    if falhas:
        print("\nArquivos que falharam apos 2 tentativas:")
        for nome in falhas:
            print(f"  - {nome}")
    
    print("=" * 70)


# ==============================================================================
# EXECUCAO
# ==============================================================================

if __name__ == "__main__":
    processar_pasta()