#!/usr/bin/env python3
"""
Module m07_overlap01.py
Detects overlap (speaker overlap) in audio segments
Uses pyannote/speaker-diarization to identify multiple speakers
"""

import sys
import json
import shutil
import signal
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

from dotenv import load_dotenv

# Add root folder to the path to import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import OVERLAP_DETECTOR, PROJECT_ROOT, EXTENSOES_AUDIO
from m01_load_models import ModelManager


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load environment variables (.env)
load_dotenv(PROJECT_ROOT / '.env')


# ==============================================================================
# TIMEOUT HANDLER
# ==============================================================================

class TimeoutException(Exception):
    """Exception raised when the timeout is reached"""
    pass


@contextmanager
def timeout(seconds: int):
    """
    Context manager for operation timeouts

    Args:
        seconds: Maximum time in seconds

    Raises:
        TimeoutException: If the time limit is exceeded
    """
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operacao excedeu {seconds}s")
    
    # Configure handler (Linux/Mac only)
    if hasattr(signal, 'SIGALRM'):
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    else:
        # Windows does not support SIGALRM - run without timeout
        yield


# ==============================================================================
# LOADING FUNCTIONS
# ==============================================================================



def carregar_json(caminho: Path) -> Dict:
    """
    Loads a JSON file

    Args:
        caminho: Path of the JSON file

    Returns:
        Dictionary with the JSON content
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def salvar_json(dados: Dict, caminho: Path) -> None:
    """
    Saves a dictionary to a JSON file with indentation

    Args:
        dados: Dictionary to save
        caminho: Path of the destination file
    """
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# ==============================================================================
# PROCESSING FUNCTIONS
# ==============================================================================

def listar_segmentos_para_processar(pasta_json_dinamico: Path, audio_id: str) -> Tuple[Dict, Dict, List[str]]:
    """
    Lists segments eligible for processing.

    Args:
        pasta_json_dinamico: Path to the 00-json_dinamico folder
        audio_id: Audio ID

    Returns:
        Tuple containing:
        - dados_acompanhamento: full tracking JSON
        - dados_filtro: filter JSON (if it exists) or None
        - segmentos_processar: list of file names to process
    """
    # Load tracking JSON (required)
    json_acompanhamento = pasta_json_dinamico / f"{audio_id}_segments_acompanhamento.json"

    if not json_acompanhamento.exists():
        raise FileNotFoundError(
            f"JSON de acompanhamento nao encontrado: {json_acompanhamento}"
        )

    dados_acompanhamento = carregar_json(json_acompanhamento)

    # Try to load filter JSON (optional)
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


def _extrair_anotacao(resultado):
    """
    Returns the diarization Annotation, whatever the return format is.

    The format varies with the version and with the pyannote pipeline's
    mode:
    - an output object that carries the Annotation in the
      speaker_diarization field;
    - the Annotation itself, directly (pyannote 3.x and 4.x's legacy
      mode).

    The choice is made by inspecting the received object, never by
    version number. Order matters: if an object is an Annotation AND has
    the field, the field wins.

    Args:
        resultado: Object returned by the pipeline call

    Returns:
        Annotation with the diarization tracks

    Raises:
        TypeError: If the object is neither of the two known formats
    """
    from pyannote.core import Annotation

    if hasattr(resultado, 'speaker_diarization'):
        return resultado.speaker_diarization

    if isinstance(resultado, Annotation):
        return resultado

    atributos = sorted(a for a in dir(resultado) if not a.startswith('_'))
    raise TypeError(
        "Retorno do pipeline pyannote em formato desconhecido. "
        f"Tipo recebido: {type(resultado).__name__}. "
        f"pyannote.audio instalado: {importlib.metadata.version('pyannote.audio')}. "
        f"Atributos publicos: {atributos}"
    )


def detectar_overlap(pipeline, audio_path: Path, timeout_segundos: int) -> Optional[bool]:
    """
    Detects whether there is overlap (multiple speakers) in the audio

    Args:
        pipeline: Loaded pyannote pipeline
        audio_path: Path of the audio file
        timeout_segundos: Maximum processing time

    Returns:
        True: Multiple speakers detected (overlap)
        False: Only 1 speaker or none
        None: Error or timeout
    """
    try:
        with timeout(timeout_segundos):
            # Run diarization
            diarizacao = pipeline(str(audio_path))
            anotacao = _extrair_anotacao(diarizacao)

            # Extract unique speakers
            speakers = set()
            for segment, _, speaker in anotacao.itertracks(yield_label=True):
                speakers.add(speaker)
            # Overlap = 2 or more distinct speakers
            num_speakers = len(speakers)
            tem_overlap = num_speakers >= 2
            
            return tem_overlap
            
    except TimeoutException as e:
        print(f"  TIMEOUT: {e}")
        return None
    except Exception as e:
        msg = str(e)
        # Segment shorter than the pyannote window (expected 160000 samples = 10s)
        # Short segments rarely have 2 simultaneous speakers — treat as no overlap
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
    Processes all audio segments

    Args:
        pipeline: Loaded pyannote pipeline
        segmentos: List of file names to process
        timeout_segundos: Timeout per audio file
        pasta_audios: Path to the folder with audio files

    Returns:
        Dictionary {file_name: overlap_result}
        overlap_result can be: True, False or None
    """
    resultados = {}
    total = len(segmentos)
    
    print(f"Total de segmentos a processar: {total}")
    print("-" * 70)
    
    for idx, nome_arquivo in enumerate(segmentos, 1):
        # Find audio file (may have a different extension)
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

        # Process audio
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
    Tries to reprocess files that failed

    Args:
        pipeline: Loaded pyannote pipeline
        resultados: Results from the initial processing
        timeout_segundos: Timeout per audio file
        pasta_audios: Path to the folder with audio files

    Returns:
        Updated results dictionary
    """
    # Identify failures
    falhas = [nome for nome, resultado in resultados.items() if resultado is None]
    
    if not falhas:
        print("\nNenhuma falha detectada - nao ha necessidade de retry")
        return resultados
    
    print(f"\n{len(falhas)} arquivo(s) falharam - tentando novamente...")
    print("=" * 70)
    
    for idx, nome_arquivo in enumerate(falhas, 1):
        # Find file
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
# OUTPUT CREATION FUNCTIONS
# ==============================================================================

def criar_jsons_output(
    dados_acompanhamento: Dict,
    dados_filtro: Optional[Dict],
    resultados: Dict[str, Optional[bool]]
) -> Tuple[Dict, Dict]:
    """
    Creates the output JSONs

    Args:
        dados_acompanhamento: original full JSON
        dados_filtro: filter JSON (if it exists)
        resultados: processing results

    Returns:
        Tuple (updated_json_acompanhamento, json_overlap01)
    """
    # Update tracking JSON with the overlap01 field
    json_acompanhamento_novo = dados_acompanhamento.copy()

    for nome_arquivo, metadados in json_acompanhamento_novo.items():
        if nome_arquivo in resultados:
            # Segment was processed
            metadados['overlap01'] = resultados[nome_arquivo]
        else:
            # Segment was not processed (was not in the filter)
            metadados['overlap01'] = None

    # Create overlap01 JSON (approved segments only: overlap01 = False)
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
    Validates data consistency before saving

    Args:
        json_acompanhamento: tracking JSON
        json_overlap01: overlap01 JSON
        resultados: processing results
        pasta_audios: path to the folder with audio files

    Returns:
        True if validation passed, False otherwise
    """
    erros = []
    
    # Validation 1: All results are in the tracking JSON
    for nome in resultados.keys():
        if nome not in json_acompanhamento:
            erros.append(f"Resultado sem entrada no JSON: {nome}")

    # Validation 2: Everything in overlap01 has overlap01=False
    for nome, metadados in json_overlap01.items():
        if metadados.get('overlap01') is not False:
            erros.append(f"Segmento em overlap01 com overlap01!={False}: {nome}")

    # Validation 3: Check that physical files exist
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
    Saves JSONs to the output folders.

    Args:
        json_acompanhamento: updated tracking JSON
        json_overlap01: overlap01 JSON (approved only)
        pasta_output_overlap: path to the 05-overlap1 folder
        pasta_output_json_dinamico: path to the 00-json_dinamico folder
        audio_id: audio ID
    """
    # Create 05-overlap1 folder if it does not exist
    pasta_output_overlap.mkdir(parents=True, exist_ok=True)

    # Save in 05-overlap1
    caminho_acompanhamento = pasta_output_overlap / f"{audio_id}_segments_acompanhamento.json"
    caminho_overlap01 = pasta_output_overlap / f"{audio_id}_overlap01.json"

    salvar_json(json_acompanhamento, caminho_acompanhamento)
    salvar_json(json_overlap01, caminho_overlap01)

    print(f"\nJSONs salvos em: {pasta_output_overlap}")
    print(f"  - {caminho_acompanhamento.name}")
    print(f"  - {caminho_overlap01.name}")

    # Copy to 00-json_dinamico (overwrite)
    dest_acompanhamento = pasta_output_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    dest_filtro = pasta_output_json_dinamico / f"{audio_id}.json"
    
    shutil.copy2(caminho_acompanhamento, dest_acompanhamento)
    shutil.copy2(caminho_overlap01, dest_filtro)
    
    print(f"\nJSONs copiados para: {pasta_output_json_dinamico}")
    print(f"  - {dest_acompanhamento.name} (sobrescrito)")
    print(f"  - {dest_filtro.name} (sobrescrito)")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(audio_id: str) -> bool:
    """
    Main function: orchestrates the entire processing.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the module completed (including when there is no
        eligible segment), False if a precondition is missing or
        validation failed and the JSONs were not saved.
    """
    # Define paths based on audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "03-segments_16khz"
    PASTA_OUTPUT_OVERLAP = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "05-overlap1"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO

    print("=" * 70)
    print("DETECTOR DE OVERLAP DE LOCUTORES")
    print("=" * 70)

    # Configure timeout
    timeout_segundos = OVERLAP_DETECTOR['timeout']['por_audio_segundos']

    # This module processes ONE segment at a time, with an individual
    # timeout: there is no batch path. The config field IS now read and
    # rejected when it asks for something the module does not do -
    # previously it was silently ignored, and anyone who configured
    # batch 8 had no way of noticing.
    batch_size = OVERLAP_DETECTOR['batch']['batch_size']
    if batch_size != 1:
        print(f"ERRO: OVERLAP_DETECTOR['batch']['batch_size'] = {batch_size!r}")
        print("Este modulo processa um segmento por vez - o unico valor "
              "suportado e 1")
        return False

    # Validate paths
    if not PASTA_JSON_DINAMICO.exists():
        print(f"ERRO: Pasta JSON nao existe: {PASTA_JSON_DINAMICO}")
        return False

    if not PASTA_AUDIOS.exists():
        print(f"ERRO: Pasta de audios nao existe: {PASTA_AUDIOS}")
        return False

    # Load model using ModelManager (singleton)
    print("\n2. Carregando modelo pyannote...")
    manager = ModelManager()
    pipeline = manager.get_pyannote()

    # The device is decided by the ModelManager (m01). Here we only
    # announce the device the pipeline is actually using - no resolving
    # the device a second time, which was how the log could announce
    # CUDA with the model on CPU.
    print(f"Pipeline carregado no dispositivo: {pipeline.device}")

    # List segments to process
    print("\n3. Listando segmentos para processar...")
    dados_acompanhamento, dados_filtro, segmentos = listar_segmentos_para_processar(PASTA_JSON_DINAMICO, audio_id)

    if not segmentos:
        # The funnel may have failed everything earlier: not a failure of this module
        print("AVISO: Nenhum segmento para processar")
        return True

    # Process segments
    print("\n4. Processando segmentos...")
    resultados = processar_todos_segmentos(pipeline, segmentos, timeout_segundos, PASTA_AUDIOS)

    # Retry for failures (if any)
    resultados = retry_falhas(pipeline, resultados, timeout_segundos, PASTA_AUDIOS)

    # Create output JSONs
    print("\n5. Criando JSONs de output...")
    json_acompanhamento_novo, json_overlap01 = criar_jsons_output(
        dados_acompanhamento,
        dados_filtro,
        resultados
    )

    # Validate consistency
    print("\n6. Validando consistencia dos dados...")
    if not validar_consistencia(json_acompanhamento_novo, json_overlap01, resultados, PASTA_AUDIOS):
        print("\nERRO: Validacao falhou - JSONs NAO foram salvos")
        return False

    print("Validacao OK")

    # Save outputs
    print("\n7. Salvando outputs...")
    salvar_outputs(json_acompanhamento_novo, json_overlap01, PASTA_OUTPUT_OVERLAP, PASTA_OUTPUT_JSON_DINAMICO, audio_id)

    # Final report
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

    return True


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Direct execution requires audio_id as an argument - there is no
    # fixed id in the code. Same pattern as m15_cleanup.py.
    if len(sys.argv) != 2:
        print("Uso: python src/m07_overlap1.py <audio_id>")
        sys.exit(1)

    sys.exit(0 if main(sys.argv[1]) else 1)
