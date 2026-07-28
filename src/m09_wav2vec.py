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

# Modelo wav2vec2
MODELO_WAV2VEC2 = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese"


# ==============================================================================
# FUNCOES DE DEVICE
# ==============================================================================


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


def obter_segmentos_elegiveis(pasta_json_dinamico: Path, audio_id: str) -> tuple[Dict, Set[str]]:
    """
    Determina quais segmentos devem ser processados.

    Args:
        pasta_json_dinamico: Caminho para pasta 00-json_dinamico
        audio_id: ID do audio

    Returns:
        Tupla (dados_acompanhamento, set_de_chaves_elegiveis)
    """
    # Carregar arquivo de acompanhamento (obrigatorio)
    caminho_acompanhamento = pasta_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    dados_acompanhamento = carregar_json(caminho_acompanhamento)

    if not dados_acompanhamento:
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {caminho_acompanhamento}")

    # Tentar carregar arquivo de filtro (opcional)
    caminho_filtro = pasta_json_dinamico / f"{audio_id}.json"
    dados_filtro = carregar_json(caminho_filtro)
    
    # Determinar segmentos elegiveis
    # Distinguir "arquivo nao existe" (None) de "arquivo existe mas vazio" ({})
    if dados_filtro is None:
        print("Arquivo de filtro nao encontrado")
        print(f"Processando todos os {len(dados_acompanhamento)} segmentos")
        segmentos_elegiveis = set(dados_acompanhamento.keys())
    elif dados_filtro:
        print(f"Arquivo de filtro encontrado: {caminho_filtro.name}")
        print(f"Processando apenas {len(dados_filtro)} segmentos filtrados")
        segmentos_elegiveis = set(dados_filtro.keys())
    else:
        print(f"Arquivo de filtro encontrado mas sem segmentos elegiveis — nada a processar")
        segmentos_elegiveis = set()
    
    return dados_acompanhamento, segmentos_elegiveis


# ==============================================================================
# FUNCAO DE TRANSCRICAO
# ==============================================================================

def transcrever_segmentos(dados_acompanhamento: Dict, segmentos_elegiveis: Set[str], pasta_audios: Path) -> tuple[Dict, Dict]:
    """
    Transcreve os segmentos elegiveis usando wav2vec2
    
    Args:
        dados_acompanhamento: Dicionario completo de segmentos
        segmentos_elegiveis: Set com chaves dos segmentos a processar
        pasta_audios: Caminho para pasta com audios
        
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
        caminho_audio = pasta_audios / chave_segmento
        
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


def salvar_outputs(dados_acompanhamento: Dict, dados_wav2vec: Dict, pasta_output_stt: Path, pasta_output_json_dinamico: Path, audio_id: str):
    """
    Salva os arquivos de output nas pastas corretas.

    Args:
        dados_acompanhamento: Dados completos com stt_wav2vec
        dados_wav2vec: Dados apenas dos segmentos elegiveis
        pasta_output_stt: Caminho para pasta 07-stt_wav2vec
        pasta_output_json_dinamico: Caminho para pasta 00-json_dinamico
        audio_id: ID do audio
    """
    print("\n" + "="*70)
    print("SALVANDO OUTPUTS")
    print("="*70)

    # Criar pasta de output STT
    pasta_output_stt.mkdir(parents=True, exist_ok=True)

    # Salvar em 07-stt_wav2vec
    caminho_acompanhamento_stt = pasta_output_stt / f"{audio_id}_segments_acompanhamento.json"
    caminho_wav2vec_stt = pasta_output_stt / f"{audio_id}_wav2vec.json"

    salvar_json(dados_acompanhamento, caminho_acompanhamento_stt)
    salvar_json(dados_wav2vec, caminho_wav2vec_stt)

    # Copiar para 00-json_dinamico (sobrescrever)
    print("\nCopiando para pasta dinamica (sobrescrever):")

    caminho_acompanhamento_dinamico = pasta_output_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    caminho_wav2vec_dinamico = pasta_output_json_dinamico / f"{audio_id}.json"

    shutil.copy2(caminho_acompanhamento_stt, caminho_acompanhamento_dinamico)
    print(f"Copiado: {caminho_acompanhamento_dinamico}")

    shutil.copy2(caminho_wav2vec_stt, caminho_wav2vec_dinamico)
    print(f"Copiado: {caminho_wav2vec_dinamico}")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main(audio_id: str):
    """
    Funcao principal de execucao.

    Args:
        audio_id: ID do audio a processar
    """
    # Definir caminhos baseados no audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "03-segments_16khz"
    PASTA_OUTPUT_STT = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "07-stt_wav2vec"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO

    print("="*70)
    print("TRANSCRICAO COM WAV2VEC2")
    print("="*70)
    print(f"Audio ID: {audio_id}")
    print(f"Modelo: {MODELO_WAV2VEC2}\n")

    # Obter segmentos elegiveis
    dados_acompanhamento, segmentos_elegiveis = obter_segmentos_elegiveis(PASTA_JSON_DINAMICO, audio_id)

    # Transcrever
    dados_acompanhamento_atualizado, dados_wav2vec = transcrever_segmentos(
        dados_acompanhamento,
        segmentos_elegiveis,
        PASTA_AUDIOS
    )

    # Salvar outputs
    salvar_outputs(dados_acompanhamento_atualizado, dados_wav2vec, PASTA_OUTPUT_STT, PASTA_OUTPUT_JSON_DINAMICO, audio_id)
    
    print("\n" + "="*70)
    print("PROCESSAMENTO CONCLUIDO")
    print("="*70)


if __name__ == "__main__":
    main('CA6TSoMw86k')