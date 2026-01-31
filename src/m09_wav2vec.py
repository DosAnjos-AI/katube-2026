#!/usr/bin/env python3
"""
Modulo m09_wav2vec.py
Transcreve segmentos de audio usando wav2vec2 (lgris/wav2vec2-large-xlsr-open-brazilian-portuguese)
Adiciona campo 'stt_wav2vec' aos metadados JSON
"""

import sys
import json
from pathlib import Path
from typing import Dict, Set, Optional
import shutil

import torch
from transformers import pipeline

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STT_WAV2VEC2, PROJECT_ROOT
from m01_load_models import ModelManager


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# ID do video a processar
id_video = 'B4RgpqJhoIo'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / id_video / "03-segments_16khz"

# Caminhos de saida
PASTA_OUTPUT_STT = PROJECT_ROOT / "arquivos" / "temp" / id_video / "07-stt_wav2vec"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Extensoes de audio suportadas
EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Modelo wav2vec2
MODELO_WAV2VEC2 = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese"


# ==============================================================================
# FUNCOES DE DEVICE
# ==============================================================================

def obter_device() -> str:
    """
    Determina o device a ser usado baseado na configuracao
    
    Returns:
        str: 'cuda' ou 'cpu'
    """
    device_config = STT_WAV2VEC2.get("device", "auto").lower()
    
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device_config == "gpu":
        if not torch.cuda.is_available():
            print("AVISO: GPU solicitada mas CUDA nao disponivel. Usando CPU.")
            return "cpu"
        return "cuda"
    else:  # cpu
        return "cpu"


# ==============================================================================
# FUNCOES DE LEITURA DE JSON
# ==============================================================================

def carregar_json(caminho: Path) -> Optional[Dict]:
    """
    Carrega arquivo JSON
    
    Args:
        caminho: Path do arquivo JSON
        
    Returns:
        Dict com conteudo ou None se nao existir
    """
    if not caminho.exists():
        return None
    
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao ler {caminho}: {e}")
        return None


def obter_segmentos_elegiveis() -> tuple[Dict, Set[str]]:
    """
    Determina quais segmentos devem ser processados
    
    Returns:
        Tupla (dados_acompanhamento, set_de_chaves_elegiveis)
    """
    # Carregar arquivo de acompanhamento (obrigatorio)
    caminho_acompanhamento = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    dados_acompanhamento = carregar_json(caminho_acompanhamento)
    
    if not dados_acompanhamento:
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho_acompanhamento}")
    
    # Tentar carregar arquivo de filtro (opcional)
    caminho_filtro = PASTA_JSON_DINAMICO / f"{id_video}.json"
    dados_filtro = carregar_json(caminho_filtro)
    
    # Determinar segmentos elegiveis
    if dados_filtro:
        print(f"Arquivo de filtro encontrado: {caminho_filtro.name}")
        print(f"Processando apenas {len(dados_filtro)} segmentos filtrados")
        segmentos_elegiveis = set(dados_filtro.keys())
    else:
        print("Arquivo de filtro nao encontrado")
        print(f"Processando todos os {len(dados_acompanhamento)} segmentos")
        segmentos_elegiveis = set(dados_acompanhamento.keys())
    
    return dados_acompanhamento, segmentos_elegiveis


# ==============================================================================
# FUNCAO DE TRANSCRICAO
# ==============================================================================

def transcrever_segmentos(dados_acompanhamento: Dict, segmentos_elegiveis: Set[str]) -> tuple[Dict, Dict]:
    """
    Transcreve os segmentos elegiveis usando wav2vec2
    
    Args:
        dados_acompanhamento: Dicionario completo de segmentos
        segmentos_elegiveis: Set com chaves dos segmentos a processar
        
    Returns:
        Tupla (dados_acompanhamento_atualizado, dados_wav2vec_somente_elegiveis)
    """
    # Inicializar pipeline wav2vec2 usando ModelManager (singleton)
    print(f"\nCarregando modelo: {MODELO_WAV2VEC2}")
    manager = ModelManager()
    pipe = manager.get_wav2vec()
    
    # Device ja gerenciado pelo manager
    device = str(pipe.model.device)
    if 'cuda' in device:
        device = 'cuda'
        device_id = 0
    else:
        device = 'cpu'
        device_id = -1
    
    print(f"Usando device: {device}")
    print("Modelo carregado com sucesso\n")
    
    # Preparar outputs
    dados_acompanhamento_output = dados_acompanhamento.copy()
    dados_wav2vec_output = {}
    
    # Processar cada segmento
    total = len(dados_acompanhamento)
    processados = 0
    transcricoes_realizadas = 0
    
    for chave_segmento, metadados in dados_acompanhamento.items():
        processados += 1
        
        # Verificar se eh elegivel
        if chave_segmento not in segmentos_elegiveis:
            dados_acompanhamento_output[chave_segmento]["stt_wav2vec"] = None
            continue
        
        # Buscar arquivo de audio
        caminho_audio = PASTA_AUDIOS / chave_segmento
        
        if not caminho_audio.exists():
            print(f"[{processados}/{total}] AVISO: Audio nao encontrado: {chave_segmento}")
            dados_acompanhamento_output[chave_segmento]["stt_wav2vec"] = None
            continue
        
        # Transcrever
        try:
            resultado = pipe(str(caminho_audio))
            transcricao = resultado["text"]
            
            # Adicionar transcricao aos outputs
            dados_acompanhamento_output[chave_segmento]["stt_wav2vec"] = transcricao
            
            # Criar entrada para arquivo wav2vec (somente elegiveis)
            dados_wav2vec_output[chave_segmento] = metadados.copy()
            dados_wav2vec_output[chave_segmento]["stt_wav2vec"] = transcricao
            
            transcricoes_realizadas += 1
            print(f"[{processados}/{total}] Transcrito: {chave_segmento}")
            
        except Exception as e:
            print(f"[{processados}/{total}] ERRO ao transcrever {chave_segmento}: {e}")
            dados_acompanhamento_output[chave_segmento]["stt_wav2vec"] = None
    
    print(f"\nResumo: {transcricoes_realizadas} transcricoes realizadas de {len(segmentos_elegiveis)} elegiveis")
    
    return dados_acompanhamento_output, dados_wav2vec_output


# ==============================================================================
# FUNCOES DE SALVAMENTO
# ==============================================================================

def salvar_json(dados: Dict, caminho: Path):
    """
    Salva dicionario como JSON
    
    Args:
        dados: Dicionario a salvar
        caminho: Path do arquivo de destino
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    
    print(f"Salvo: {caminho}")


def salvar_outputs(dados_acompanhamento: Dict, dados_wav2vec: Dict):
    """
    Salva os arquivos de output nas pastas corretas
    
    Args:
        dados_acompanhamento: Dados completos com stt_wav2vec
        dados_wav2vec: Dados apenas dos segmentos elegiveis
    """
    print("\n" + "="*70)
    print("SALVANDO OUTPUTS")
    print("="*70)
    
    # Criar pasta de output STT
    PASTA_OUTPUT_STT.mkdir(parents=True, exist_ok=True)
    
    # Salvar em 07-stt_wav2vec
    caminho_acompanhamento_stt = PASTA_OUTPUT_STT / f"{id_video}_segments_acompanhamento.json"
    caminho_wav2vec_stt = PASTA_OUTPUT_STT / f"{id_video}_wav2vec.json"
    
    salvar_json(dados_acompanhamento, caminho_acompanhamento_stt)
    salvar_json(dados_wav2vec, caminho_wav2vec_stt)
    
    # Copiar para 00-json_dinamico (sobrescrever)
    print("\nCopiando para pasta dinamica (sobrescrever):")
    
    caminho_acompanhamento_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    caminho_wav2vec_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
    
    shutil.copy2(caminho_acompanhamento_stt, caminho_acompanhamento_dinamico)
    print(f"Copiado: {caminho_acompanhamento_dinamico}")
    
    shutil.copy2(caminho_wav2vec_stt, caminho_wav2vec_dinamico)
    print(f"Copiado: {caminho_wav2vec_dinamico}")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main():
    """
    Funcao principal de execucao
    """
    print("="*70)
    print("TRANSCRICAO COM WAV2VEC2")
    print("="*70)
    print(f"Video ID: {id_video}")
    print(f"Modelo: {MODELO_WAV2VEC2}\n")
    
    # Obter segmentos elegiveis
    dados_acompanhamento, segmentos_elegiveis = obter_segmentos_elegiveis()
    
    # Transcrever
    dados_acompanhamento_atualizado, dados_wav2vec = transcrever_segmentos(
        dados_acompanhamento,
        segmentos_elegiveis
    )
    
    # Salvar outputs
    salvar_outputs(dados_acompanhamento_atualizado, dados_wav2vec)
    
    print("\n" + "="*70)
    print("PROCESSAMENTO CONCLUIDO")
    print("="*70)


if __name__ == "__main__":
    main()