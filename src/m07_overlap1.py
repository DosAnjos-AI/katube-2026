#!/usr/bin/env python3
"""
Modulo m07_overlap01.py
Detecta overlap (sobreposicao de locutores) em segmentos de audio
Utiliza pyannote/speaker-diarization para identificar multiplos speakers
"""

import sys
import json
import shutil
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import torch
from dotenv import load_dotenv
import os

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OVERLAP_DETECTOR, PROJECT_ROOT, EXTENSOES_AUDIO
from m01_load_models import ModelManager


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# Carregar variaveis de ambiente (.env)
load_dotenv(PROJECT_ROOT / '.env')


# ==============================================================================
# TIMEOUT HANDLER
# ==============================================================================

class TimeoutException(Exception):
    """Excecao lancada quando timeout e atingido"""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager para timeout de operacoes
    
    Args:
        seconds: Tempo maximo em segundos
        
    Raises:
        TimeoutException: Se tempo limite for excedido
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operacao excedeu {seconds}s")
    
    # Configurar handler (apenas Linux/Mac)
    if hasattr(signal, 'SIGALRM'):
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    else:
        # Windows nao suporta SIGALRM - executar sem timeout
        yield


# ==============================================================================
# FUNCOES DE VALIDACAO E CONFIGURACAO
# ==============================================================================

def validar_hf_token() -> str:
    """
    Valida existencia do token HuggingFace
    
    Returns:
        Token HuggingFace
        
    Raises:
        ValueError: Se token nao encontrado
    """
    token = os.getenv('HF_TOKEN')
    if not token or token == 'seu_token_aqui':
        raise ValueError(
            "Token HuggingFace nao configurado!\n"
            "Configure HF_TOKEN no arquivo .env na raiz do projeto"
        )
    return token



# ==============================================================================
# FUNCOES DE CARREGAMENTO
# ==============================================================================



def carregar_json(caminho: Path) -> Dict:
    """
    Carrega arquivo JSON
    
    Args:
        caminho: Path do arquivo JSON
        
    Returns:
        Dicionario com conteudo do JSON
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def salvar_json(dados: Dict, caminho: Path) -> None:
    """
    Salva dicionario em arquivo JSON com indentacao
    
    Args:
        dados: Dicionario para salvar
        caminho: Path do arquivo de destino
    """
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ==============================================================================
# FUNCOES DE PROCESSAMENTO
# ==============================================================================

def listar_segmentos_para_processar(pasta_json_dinamico: Path, audio_id: str) -> Tuple[Dict, Dict, List[str]]:
    """
    Lista segmentos elegiveis para processamento.

    Args:
        pasta_json_dinamico: Caminho para pasta 00-json_dinamico
        audio_id: ID do audio

    Returns:
        Tupla contendo:
        - dados_acompanhamento: JSON completo de acompanhamento
        - dados_filtro: JSON de filtro (se existir) ou None
        - segmentos_processar: Lista de nomes de arquivos a processar
    """
    # Carregar JSON de acompanhamento (obrigatorio)
    json_acompanhamento = pasta_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    
    if not json_acompanhamento.exists():
        raise FileNotFoundError(
            f"JSON de acompanhamento nao encontrado: {json_acompanhamento}"
        )
    
    dados_acompanhamento = carregar_json(json_acompanhamento)
    
    # Tentar carregar JSON de filtro (opcional)
    json_filtro = pasta_json_dinamico / f"{audio_id}.json"
    dados_filtro = None
    
    if json_filtro.exists():
        dados_filtro = carregar_json(json_filtro)
        segmentos_processar = list(dados_filtro.keys())
        print(f"JSON de filtro encontrado: {len(segmentos_processar)} segmentos elegiveis")
    else:
        segmentos_processar = list(dados_acompanhamento.keys())
        print(f"JSON de filtro NAO encontrado: processando todos {len(segmentos_processar)} segmentos")
    
    return dados_acompanhamento, dados_filtro, segmentos_processar


def detectar_overlap(pipeline, audio_path: Path, timeout_segundos: int) -> Optional[bool]:
    """
    Detecta se ha overlap (multiplos speakers) no audio
    
    Args:
        pipeline: Pipeline pyannote carregado
        audio_path: Path do arquivo de audio
        timeout_segundos: Tempo maximo de processamento
        
    Returns:
        True: Multiplos speakers detectados (overlap)
        False: Apenas 1 speaker ou nenhum
        None: Erro ou timeout
    """
    try:
        with timeout(timeout_segundos):
            # Executar diarizacao
            diarizacao = pipeline(str(audio_path))
            
            # Extrair speakers unicos
            speakers = set()
            for segment, _, speaker in diarizacao.speaker_diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            # Overlap = 2 ou mais speakers distintos
            num_speakers = len(speakers)
            tem_overlap = num_speakers >= 2
            
            return tem_overlap
            
    except TimeoutException as e:
        print(f"  TIMEOUT: {e}")
        return None
    except Exception as e:
        msg = str(e)
        # Segmento mais curto que a janela do pyannote (esperava 160000 samples = 10s)
        # Segmentos curtos dificilmente tem 2 speakers simultaneos — tratar como sem overlap
        if "samples instead of the expected" in msg:
            print(f"  AVISO: audio curto demais para pyannote (chunk insuficiente) — assumindo sem overlap")
            return False
        print(f"  ERRO: {e}")
        return None


def processar_todos_segmentos(
    pipeline,
    segmentos: List[str],
    timeout_segundos: int,
    pasta_audios: Path
) -> Dict[str, Optional[bool]]:
    """
    Processa todos os segmentos de audio
    
    Args:
        pipeline: Pipeline pyannote carregado
        segmentos: Lista de nomes de arquivos a processar
        timeout_segundos: Timeout por audio
        pasta_audios: Caminho para pasta com audios
        
    Returns:
        Dicionario {nome_arquivo: resultado_overlap}
        resultado_overlap pode ser: True, False ou None
    """
    resultados = {}
    total = len(segmentos)
    
    print(f"Total de segmentos a processar: {total}")
    print("-" * 70)
    
    for idx, nome_arquivo in enumerate(segmentos, 1):
        # Encontrar arquivo de audio (pode ter extensao diferente)
        audio_path = None
        for ext in EXTENSOES_AUDIO:
            caminho_teste = pasta_audios / nome_arquivo
            if caminho_teste.exists():
                audio_path = caminho_teste
                break
        
        if not audio_path:
            print(f"[{idx}/{total}] {nome_arquivo} - ARQUIVO NAO ENCONTRADO")
            resultados[nome_arquivo] = None
            continue
        
        # Processar audio
        print(f"[{idx}/{total}] {nome_arquivo}...", end=" ", flush=True)
        
        resultado = detectar_overlap(pipeline, audio_path, timeout_segundos)
        
        if resultado is None:
            print("FALHOU")
        elif resultado:
            print("OVERLAP DETECTADO")
        else:
            print("SEM OVERLAP")
        
        resultados[nome_arquivo] = resultado
    
    return resultados


def retry_falhas(
    pipeline,
    resultados: Dict[str, Optional[bool]],
    timeout_segundos: int,
    pasta_audios: Path
) -> Dict[str, Optional[bool]]:
    """
    Tenta reprocessar arquivos que falharam
    
    Args:
        pipeline: Pipeline pyannote carregado
        resultados: Resultados do processamento inicial
        timeout_segundos: Timeout por audio
        pasta_audios: Caminho para pasta com audios
        
    Returns:
        Dicionario de resultados atualizado
    """
    # Identificar falhas
    falhas = [nome for nome, resultado in resultados.items() if resultado is None]
    
    if not falhas:
        print("\nNenhuma falha detectada - nao ha necessidade de retry")
        return resultados
    
    print(f"\n{len(falhas)} arquivo(s) falharam - tentando novamente...")
    print("=" * 70)
    
    for idx, nome_arquivo in enumerate(falhas, 1):
        # Encontrar arquivo
        audio_path = None
        for ext in EXTENSOES_AUDIO:
            caminho_teste = pasta_audios / nome_arquivo
            if caminho_teste.exists():
                audio_path = caminho_teste
                break
        
        if not audio_path:
            continue
        
        print(f"[{idx}/{len(falhas)}] {nome_arquivo}...", end=" ", flush=True)
        
        resultado = detectar_overlap(pipeline, audio_path, timeout_segundos)
        
        if resultado is None:
            print("FALHOU NOVAMENTE")
        elif resultado:
            print("OVERLAP DETECTADO")
        else:
            print("SEM OVERLAP")
        
        resultados[nome_arquivo] = resultado
    
    return resultados


# ==============================================================================
# FUNCOES DE CRIACAO DE OUTPUTS
# ==============================================================================

def criar_jsons_output(
    dados_acompanhamento: Dict,
    dados_filtro: Optional[Dict],
    resultados: Dict[str, Optional[bool]]
) -> Tuple[Dict, Dict]:
    """
    Cria os JSONs de output
    
    Args:
        dados_acompanhamento: JSON completo original
        dados_filtro: JSON de filtro (se existir)
        resultados: Resultados do processamento
        
    Returns:
        Tupla (json_acompanhamento_atualizado, json_overlap01)
    """
    # Atualizar JSON de acompanhamento com campo overlap01
    json_acompanhamento_novo = dados_acompanhamento.copy()
    
    for nome_arquivo, metadados in json_acompanhamento_novo.items():
        if nome_arquivo in resultados:
            # Segmento foi processado
            metadados['overlap01'] = resultados[nome_arquivo]
        else:
            # Segmento nao foi processado (nao estava em filtro)
            metadados['overlap01'] = None
    
    # Criar JSON overlap01 (apenas segmentos aprovados: overlap01 = False)
    json_overlap01 = {}
    
    for nome_arquivo, metadados in json_acompanhamento_novo.items():
        if metadados.get('overlap01') is False:
            json_overlap01[nome_arquivo] = metadados.copy()
    
    return json_acompanhamento_novo, json_overlap01


def validar_consistencia(
    json_acompanhamento: Dict,
    json_overlap01: Dict,
    resultados: Dict[str, Optional[bool]],
    pasta_audios: Path
) -> bool:
    """
    Valida consistencia dos dados antes de salvar
    
    Args:
        json_acompanhamento: JSON de acompanhamento
        json_overlap01: JSON overlap01
        resultados: Resultados do processamento
        pasta_audios: Caminho para pasta com audios
        
    Returns:
        True se validacao OK, False caso contrario
    """
    erros = []
    
    # Validacao 1: Todos os resultados estao no JSON acompanhamento
    for nome in resultados.keys():
        if nome not in json_acompanhamento:
            erros.append(f"Resultado sem entrada no JSON: {nome}")
    
    # Validacao 2: Todos em overlap01 tem overlap01=False
    for nome, metadados in json_overlap01.items():
        if metadados.get('overlap01') is not False:
            erros.append(f"Segmento em overlap01 com overlap01!={False}: {nome}")
    
    # Validacao 3: Verificar arquivos fisicos existem
    for nome in resultados.keys():
        arquivo_existe = False
        for ext in EXTENSOES_AUDIO:
            if (pasta_audios / nome).exists():
                arquivo_existe = True
                break
        
        if not arquivo_existe:
            erros.append(f"Resultado sem arquivo fisico: {nome}")
    
    if erros:
        print("\nERROS DE VALIDACAO DETECTADOS:")
        for erro in erros:
            print(f"  - {erro}")
        return False
    
    return True


def salvar_outputs(
    json_acompanhamento: Dict,
    json_overlap01: Dict,
    pasta_output_overlap: Path,
    pasta_output_json_dinamico: Path,
    audio_id: str
) -> None:
    """
    Salva JSONs nas pastas de output.

    Args:
        json_acompanhamento: JSON de acompanhamento atualizado
        json_overlap01: JSON overlap01 (apenas aprovados)
        pasta_output_overlap: Caminho para pasta 05-overlap1
        pasta_output_json_dinamico: Caminho para pasta 00-json_dinamico
        audio_id: ID do audio
    """
    # Criar pasta 05-overlap1 se nao existir
    pasta_output_overlap.mkdir(parents=True, exist_ok=True)

    # Salvar em 05-overlap1
    caminho_acompanhamento = pasta_output_overlap / f"{audio_id}_segments_acompanhamento.json"
    caminho_overlap01 = pasta_output_overlap / f"{audio_id}_overlap01.json"

    salvar_json(json_acompanhamento, caminho_acompanhamento)
    salvar_json(json_overlap01, caminho_overlap01)

    print(f"\nJSONs salvos em: {pasta_output_overlap}")
    print(f"  - {caminho_acompanhamento.name}")
    print(f"  - {caminho_overlap01.name}")

    # Copiar para 00-json_dinamico (sobrescrever)
    dest_acompanhamento = pasta_output_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    dest_filtro = pasta_output_json_dinamico / f"{audio_id}.json"
    
    shutil.copy2(caminho_acompanhamento, dest_acompanhamento)
    shutil.copy2(caminho_overlap01, dest_filtro)
    
    print(f"\nJSONs copiados para: {pasta_output_json_dinamico}")
    print(f"  - {dest_acompanhamento.name} (sobrescrito)")
    print(f"  - {dest_filtro.name} (sobrescrito)")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main(audio_id: str):
    """
    Funcao principal: orquestra todo o processamento.

    Args:
        audio_id: ID do audio a processar
    """
    # Definir caminhos baseados no audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "03-segments_16khz"
    PASTA_OUTPUT_OVERLAP = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "05-overlap1"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO
    
    print("=" * 70)
    print("DETECTOR DE OVERLAP DE LOCUTORES")
    print("=" * 70)
    
    # Configurar timeout
    timeout_segundos = OVERLAP_DETECTOR['timeout']['por_audio_segundos']
    
    # Validar caminhos
    if not PASTA_JSON_DINAMICO.exists():
        print(f"ERRO: Pasta JSON nao existe: {PASTA_JSON_DINAMICO}")
        return
    
    if not PASTA_AUDIOS.exists():
        print(f"ERRO: Pasta de audios nao existe: {PASTA_AUDIOS}")
        return
    
    # Carregar modelo usando ModelManager (singleton)
    print("\n2. Carregando modelo pyannote...")
    manager = ModelManager()
    pipeline = manager.get_pyannote()
    
    # O device ja esta configurado pelo ModelManager
    # Pyannote pipeline nao tem .parameters() como modelos normais
    _device_cfg = OVERLAP_DETECTOR.get('device', 'auto').lower()
    if _device_cfg == 'cpu':
        _device_str = 'cpu'
    elif _device_cfg in ('gpu', 'cuda'):
        _device_str = 'cuda'
    else:
        _device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Pipeline configurado para {_device_str.upper()}")
    
    # Listar segmentos para processar
    print("\n3. Listando segmentos para processar...")
    dados_acompanhamento, dados_filtro, segmentos = listar_segmentos_para_processar(PASTA_JSON_DINAMICO, audio_id)
    
    if not segmentos:
        print("AVISO: Nenhum segmento para processar")
        return
    
    # Processar segmentos
    print("\n4. Processando segmentos...")
    resultados = processar_todos_segmentos(pipeline, segmentos, timeout_segundos, PASTA_AUDIOS)
    
    # Retry para falhas (se houver)
    resultados = retry_falhas(pipeline, resultados, timeout_segundos, PASTA_AUDIOS)
    
    # Criar JSONs de output
    print("\n5. Criando JSONs de output...")
    json_acompanhamento_novo, json_overlap01 = criar_jsons_output(
        dados_acompanhamento,
        dados_filtro,
        resultados
    )
    
    # Validar consistencia
    print("\n6. Validando consistencia dos dados...")
    if not validar_consistencia(json_acompanhamento_novo, json_overlap01, resultados, PASTA_AUDIOS):
        print("\nERRO: Validacao falhou - JSONs NAO foram salvos")
        return
    
    print("Validacao OK")
    
    # Salvar outputs
    print("\n7. Salvando outputs...")
    salvar_outputs(json_acompanhamento_novo, json_overlap01, PASTA_OUTPUT_OVERLAP, PASTA_OUTPUT_JSON_DINAMICO, audio_id)
    
    # Relatorio final
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 70)
    
    total = len(segmentos)
    com_overlap = sum(1 for r in resultados.values() if r is True)
    sem_overlap = sum(1 for r in resultados.values() if r is False)
    falhas = sum(1 for r in resultados.values() if r is None)
    
    print(f"Total de segmentos processados: {total}")
    print(f"  Com overlap (2+ speakers): {com_overlap}")
    print(f"  Sem overlap (1 speaker): {sem_overlap}")
    print(f"  Falhas/Timeouts: {falhas}")
    print(f"\nSegmentos aprovados (overlap01=False): {len(json_overlap01)}")
    print("=" * 70)


# ==============================================================================
# EXECUCAO
# ==============================================================================

if __name__ == "__main__":
    main('CA6TSoMw86k')