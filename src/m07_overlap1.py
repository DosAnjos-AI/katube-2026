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

from config import OVERLAP_DETECTOR, PROJECT_ROOT


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# Carregar variaveis de ambiente (.env)
load_dotenv(PROJECT_ROOT / '.env')

# ID do video a processar
id_video = '0aICqierMVA'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / id_video / "03-segments_16khz"

# Caminhos de saida
PASTA_OUTPUT_OVERLAP = PROJECT_ROOT / "arquivos" / "temp" / id_video / "05-overlap1"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Extensoes de audio suportadas
EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


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


def detectar_device(config_device: str) -> str:
    """
    Detecta dispositivo de processamento (GPU/CPU)
    
    Args:
        config_device: Configuracao do usuario ('auto', 'gpu', 'cpu')
        
    Returns:
        'cuda' ou 'cpu'
        
    Raises:
        RuntimeError: Se GPU solicitada mas nao disponivel
    """
    if config_device == 'cpu':
        return 'cpu'
    elif config_device == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU solicitada mas CUDA nao disponivel")
        return 'cuda'
    else:  # 'auto'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device auto-detectado: {device.upper()}")
        return device


# ==============================================================================
# FUNCOES DE CARREGAMENTO
# ==============================================================================

def carregar_modelo(device: str, hf_token: str):
    """
    Carrega modelo pyannote de diarizacao
    
    Args:
        device: 'cuda' ou 'cpu'
        hf_token: Token HuggingFace para autenticacao
        
    Returns:
        Pipeline pyannote carregado
    """
    from pyannote.audio import Pipeline
    
    modelo_nome = OVERLAP_DETECTOR['modelo']
    print(f"Carregando modelo: {modelo_nome}")
    print("AVISO: Primeira execucao pode demorar (download ~1-3GB)")
    
    pipeline = Pipeline.from_pretrained(
        modelo_nome,
        token=hf_token  # Atualizado: use_auth_token -> token
    )
    
    pipeline.to(torch.device(device))
    print(f"Modelo carregado em {device.upper()}")
    
    return pipeline


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

def listar_segmentos_para_processar() -> Tuple[Dict, Dict, List[str]]:
    """
    Lista segmentos elegiveis para processamento
    
    Returns:
        Tupla contendo:
        - dados_acompanhamento: JSON completo de acompanhamento
        - dados_filtro: JSON de filtro (se existir) ou None
        - segmentos_processar: Lista de nomes de arquivos a processar
    """
    # Carregar JSON de acompanhamento (obrigatorio)
    json_acompanhamento = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    
    if not json_acompanhamento.exists():
        raise FileNotFoundError(
            f"JSON de acompanhamento nao encontrado: {json_acompanhamento}"
        )
    
    dados_acompanhamento = carregar_json(json_acompanhamento)
    
    # Tentar carregar JSON de filtro (opcional)
    json_filtro = PASTA_JSON_DINAMICO / f"{id_video}.json"
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
        print(f"  ERRO: {e}")
        return None


def processar_todos_segmentos(
    pipeline,
    segmentos: List[str],
    timeout_segundos: int
) -> Dict[str, Optional[bool]]:
    """
    Processa todos os segmentos de audio
    
    Args:
        pipeline: Pipeline pyannote carregado
        segmentos: Lista de nomes de arquivos a processar
        timeout_segundos: Timeout por audio
        
    Returns:
        Dicionario {nome_arquivo: resultado_overlap}
    """
    resultados = {}
    total = len(segmentos)
    
    print(f"\nProcessando {total} segmentos de audio...")
    print("-" * 70)
    
    for idx, nome_arquivo in enumerate(segmentos, 1):
        # Encontrar arquivo de audio correspondente
        audio_path = None
        for ext in EXTENSOES_AUDIO:
            caminho_teste = PASTA_AUDIOS / nome_arquivo
            # Ajustar extensao se necessario
            if caminho_teste.suffix.lower() not in EXTENSOES_AUDIO:
                # Tentar com extensao do JSON
                caminho_teste = PASTA_AUDIOS / nome_arquivo
            
            if caminho_teste.exists():
                audio_path = caminho_teste
                break
        
        if not audio_path:
            print(f"[{idx}/{total}] {nome_arquivo}... AUDIO NAO ENCONTRADO")
            resultados[nome_arquivo] = None
            continue
        
        print(f"[{idx}/{total}] {nome_arquivo}...", end=" ", flush=True)
        
        # Detectar overlap
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
    timeout_segundos: int
) -> Dict[str, Optional[bool]]:
    """
    Tenta reprocessar segmentos que falharam
    
    Args:
        pipeline: Pipeline pyannote
        resultados: Resultados da primeira tentativa
        timeout_segundos: Timeout por audio
        
    Returns:
        Resultados atualizados
    """
    falhas = [nome for nome, res in resultados.items() if res is None]
    
    if not falhas:
        return resultados
    
    print("\n" + "=" * 70)
    print(f"SEGUNDA TENTATIVA - {len(falhas)} segmento(s) com falha")
    print("=" * 70)
    
    for idx, nome_arquivo in enumerate(falhas, 1):
        # Encontrar arquivo
        audio_path = None
        for ext in EXTENSOES_AUDIO:
            caminho_teste = PASTA_AUDIOS / nome_arquivo
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
    resultados: Dict[str, Optional[bool]]
) -> bool:
    """
    Valida consistencia dos dados antes de salvar
    
    Args:
        json_acompanhamento: JSON de acompanhamento
        json_overlap01: JSON overlap01
        resultados: Resultados do processamento
        
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
            if (PASTA_AUDIOS / nome).exists():
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
    json_overlap01: Dict
) -> None:
    """
    Salva JSONs nas pastas de output
    
    Args:
        json_acompanhamento: JSON de acompanhamento atualizado
        json_overlap01: JSON overlap01 (apenas aprovados)
    """
    # Criar pasta 05-overlap1 se nao existir
    PASTA_OUTPUT_OVERLAP.mkdir(parents=True, exist_ok=True)
    
    # Salvar em 05-overlap1
    caminho_acompanhamento = PASTA_OUTPUT_OVERLAP / f"{id_video}_segments_acompanhamento.json"
    caminho_overlap01 = PASTA_OUTPUT_OVERLAP / f"{id_video}_overlap01.json"
    
    salvar_json(json_acompanhamento, caminho_acompanhamento)
    salvar_json(json_overlap01, caminho_overlap01)
    
    print(f"\nJSONs salvos em: {PASTA_OUTPUT_OVERLAP}")
    print(f"  - {caminho_acompanhamento.name}")
    print(f"  - {caminho_overlap01.name}")
    
    # Copiar para 00-json_dinamico (sobrescrever)
    dest_acompanhamento = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    dest_filtro = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
    
    shutil.copy2(caminho_acompanhamento, dest_acompanhamento)
    shutil.copy2(caminho_overlap01, dest_filtro)
    
    print(f"\nJSONs copiados para: {PASTA_OUTPUT_JSON_DINAMICO}")
    print(f"  - {dest_acompanhamento.name} (sobrescrito)")
    print(f"  - {dest_filtro.name} (sobrescrito)")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main():
    """
    Funcao principal: orquestra todo o processamento
    """
    print("=" * 70)
    print("DETECTOR DE OVERLAP DE LOCUTORES")
    print("=" * 70)
    
    # Validar configuracoes
    print("\n1. Validando configuracoes...")
    hf_token = validar_hf_token()
    device = detectar_device(OVERLAP_DETECTOR['device'])
    timeout_segundos = OVERLAP_DETECTOR['timeout']['por_audio_segundos']
    
    # Validar caminhos
    if not PASTA_JSON_DINAMICO.exists():
        print(f"ERRO: Pasta JSON nao existe: {PASTA_JSON_DINAMICO}")
        return
    
    if not PASTA_AUDIOS.exists():
        print(f"ERRO: Pasta de audios nao existe: {PASTA_AUDIOS}")
        return
    
    # Carregar modelo
    print("\n2. Carregando modelo pyannote...")
    pipeline = carregar_modelo(device, hf_token)
    
    # Listar segmentos para processar
    print("\n3. Listando segmentos para processar...")
    dados_acompanhamento, dados_filtro, segmentos = listar_segmentos_para_processar()
    
    if not segmentos:
        print("AVISO: Nenhum segmento para processar")
        return
    
    # Processar segmentos
    print("\n4. Processando segmentos...")
    resultados = processar_todos_segmentos(pipeline, segmentos, timeout_segundos)
    
    # Retry para falhas (se houver)
    resultados = retry_falhas(pipeline, resultados, timeout_segundos)
    
    # Criar JSONs de output
    print("\n5. Criando JSONs de output...")
    json_acompanhamento_novo, json_overlap01 = criar_jsons_output(
        dados_acompanhamento,
        dados_filtro,
        resultados
    )
    
    # Validar consistencia
    print("\n6. Validando consistencia dos dados...")
    if not validar_consistencia(json_acompanhamento_novo, json_overlap01, resultados):
        print("\nERRO: Validacao falhou - JSONs NAO foram salvos")
        return
    
    print("Validacao OK")
    
    # Salvar outputs
    print("\n7. Salvando outputs...")
    salvar_outputs(json_acompanhamento_novo, json_overlap01)
    
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
    main()