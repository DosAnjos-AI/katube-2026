#!/usr/bin/env python3
"""
main.py - Main Orchestrator for the katube-2026 Pipeline
Processes audio files, generating datasets for TTS/STT
"""

import sys
import logging
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
from dotenv import load_dotenv

# Add root and src folders to the path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

# Load .env before any torch/CUDA import
# CUDA_VISIBLE_DEVICES="" in .env forces CPU on this machine
load_dotenv(PROJECT_ROOT / '.env')

from config import MASTER, EXTENSOES_AUDIO, CSV_CONCLUIDOS

# Import pipeline modules (all located in ./src/)
from m00_nomeacao import preparar_entrada
from m02_diretorios import criar_diretorios
from m04_segmentador_audio_vad import executar_segmentacao_vad
from m05_segmentador_16khz import processar_pasta
from m06_mos_filter import processar_mos
from m07_overlap1 import main as processar_overlap
from m08_whisper import main as processar_whisper
from m09_wav2vec import main as processar_wav2vec
from m10_texto_normalizador import processar_normalizacao
from m11_validador_similaridade import processar_validacao
from m12_denoiser_deepfilternet3 import main as processar_denoiser
from m13_normalizador_audio import main as processar_normalizador_audio
from m14_metadados import processar_metadados, CSV_CONCLUIDOS_HEADER
from m15_cleanup import executar_cleanup


# ==============================================================================
# LOG CONFIGURATION
# ==============================================================================

def configurar_logger(audio_id: str) -> logging.Logger:
    """
    Configures the logger for the specific audio file.

    Args:
        audio_id: Audio ID

    Returns:
        Configured logger
    """
    # Create logs folder
    pasta_logs = PROJECT_ROOT / "dataset" / "log"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    # Log file
    arquivo_log = pasta_logs / f"{audio_id}.log"

    # Configure logger
    logger = logging.getLogger(f"pipeline_{audio_id}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(arquivo_log, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def listar_ids_disponiveis() -> List[str]:
    """
    Lists available audio ids in ./arquivos/audios/.

    Each subfolder must contain an audio file with the same name as the
    folder (any supported extension): arquivos/audios/{audio_id}/{audio_id}.{ext}

    Returns:
        List of audio_ids found, sorted alphabetically
    """
    pasta_audios = PROJECT_ROOT / "arquivos" / "audios"

    if not pasta_audios.exists():
        return []

    ids = []
    for subpasta in pasta_audios.iterdir():
        if not subpasta.is_dir():
            continue
        audio_id = subpasta.name
        # Checks whether an audio file with the audio_id name exists
        tem_audio = any(
            (subpasta / f"{audio_id}{ext}").exists()
            for ext in EXTENSOES_AUDIO
        )
        if tem_audio:
            ids.append(audio_id)
        else:
            print(f"  AVISO: pasta '{audio_id}' nao contem '{audio_id}.<ext>' — ignorada")

    return sorted(ids)


def carregar_ids_concluidos() -> Optional[Set[str]]:
    """
    Loads, ONCE per run, the set of already-completed ids.

    The source is the completed-audio CSV (config.CSV_CONCLUIDOS), written
    by m14 as the last step for each audio file that finished. An
    in-memory set, not a per-audio scan: the lookup stays O(1) even as the
    file grows batch after batch.

    A missing file is legitimate - it is the machine's first run, and no
    audio has been completed yet. Returns an empty set.

    Returns:
        The set of ids, or None if the file exists and CANNOT BE READ.
        None forces the caller to abort: returning an empty set in that
        case would make the pipeline silently reprocess the entire
        dataset.
    """
    if not CSV_CONCLUIDOS.exists():
        print(f"CSV dos concluidos ainda nao existe ({CSV_CONCLUIDOS}) - "
              "nenhum audio foi concluido nesta maquina")
        return set()

    ids: Set[str] = set()
    linhas_ignoradas = 0

    try:
        with open(CSV_CONCLUIDOS, 'r', encoding='utf-8') as f:
            for numero, linha in enumerate(f, 1):
                texto = linha.rstrip('\n')

                if numero == 1:
                    if texto == CSV_CONCLUIDOS_HEADER:
                        continue
                    # An unexpected header cannot pass unnoticed: either the
                    # file is in another format, or it lost its first line
                    print(f"AVISO: primeira linha de {CSV_CONCLUIDOS} nao e o "
                          f"header esperado ({CSV_CONCLUIDOS_HEADER!r}) - "
                          f"tratada como dado")

                campo_id = texto.split('|')[0].strip()
                if not campo_id:
                    linhas_ignoradas += 1
                    continue
                ids.add(campo_id)

    except OSError as e:
        print(f"ERRO ao ler o CSV dos concluidos '{CSV_CONCLUIDOS}': {e}")
        return None

    if linhas_ignoradas:
        print(f"AVISO: {linhas_ignoradas} linha(s) sem id em {CSV_CONCLUIDOS} "
              "foram ignoradas")

    print(f"Audios ja concluidos (de {CSV_CONCLUIDOS.name}): {len(ids)}")
    return ids


def calcular_duracao_audios_aprovados(ids_processados: List[str]) -> float:
    """
    Calculates the total duration of approved audio files ONLY for the
    given IDs.

    READS THE JSONS FROM dataset/historico_dataset/. This history stopped
    being the deduplication mechanism (which today is the completed-audio
    CSV), but it CONTINUES TO BE WRITTEN by m14 - this is where the
    `duracao_audios_aprovados_segundos` column of
    processamento_metadados.csv comes from, besides serving as a backup to
    rebuild the dataset without rerunning the models. "Removing
    deduplication from the history" does NOT mean "stop writing the
    JSONs": without them, this count silently returns 0.00, because a
    missing file falls into the `if` below and is ignored.

    Args:
        ids_processados: List of IDs processed IN THIS RUN

    Returns:
        Total duration in seconds
    """
    duracao_total = 0.0

    for audio_id in ids_processados:
        historico_path = PROJECT_ROOT / "dataset" / "historico_dataset" / f"{audio_id}.json"
        
        if historico_path.exists():
            try:
                with open(historico_path, 'r', encoding='utf-8') as f:
                    dados = json.load(f)

                # Sum only valid (non-null) durations
                for metadados in dados.values():
                    duracao = metadados.get('duracao')
                    if duracao is not None:
                        duracao_total += duracao

            except Exception as e:
                # History exists but cannot be read: it does not invalidate
                # the already-saved dataset, but it cannot vanish from the report
                print(f"AVISO: falha ao ler historico de '{audio_id}' "
                      f"({historico_path}): {e}")
                print("       Duracao deste audio ficou de fora do total")

    return duracao_total


def obter_proximo_id_csv() -> int:
    """
    Gets the next sequential ID for the CSV

    Returns:
        Next available ID (1 if the file does not exist)
    """
    csv_path = PROJECT_ROOT / "dataset" / "processamento_metadados.csv"
    
    if not csv_path.exists():
        return 1
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            ids = [int(row['id']) for row in reader if row['id'].isdigit()]
            return max(ids) + 1 if ids else 1
    except Exception as e:
        # CSV exists but cannot be read: going back to 1 collides with ids
        # already saved, so the warning needs to show up
        print(f"AVISO: falha ao ler {csv_path}: {e}")
        print("       Proximo id do CSV de metricas volta a 1 - risco de id repetido")
        return 1


def salvar_metadados_csv(
    duracao_total: float,
    total_audios: int,
    processados: int,
    pulados: int,
    erros: int,
    duracao_audios_aprovados: float,
    tempos_modulos: Dict[str, Optional[float]]
):
    """
    Saves processing metadata to CSV (APPEND mode)
    ONLY with data from the current iteration

    Args:
        duracao_total: Total processing duration in seconds
        total_audios: Total audio files found IN THIS RUN
        processados: Audio files processed successfully IN THIS RUN
        pulados: Audio files skipped IN THIS RUN
        erros: Audio files with errors IN THIS RUN
        duracao_audios_aprovados: Total duration of approved audio files IN THIS RUN
        tempos_modulos: Dictionary with the time for each module IN THIS RUN
    """
    csv_path = PROJECT_ROOT / "dataset" / "processamento_metadados.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check whether the file exists, to create the header
    arquivo_existe = csv_path.exists()

    # Get next ID
    registro_id = obter_proximo_id_csv()

    # End date/time
    data_hora_fim = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Calculate percentages
    percentuais = {}
    for modulo, tempo in tempos_modulos.items():
        if tempo is not None and duracao_total > 0:
            percentuais[modulo] = round((tempo / duracao_total) * 100, 1)
        else:
            percentuais[modulo] = None

    # Prepare data row
    linha = {
        'id': registro_id,
        'data_hora_fim': data_hora_fim,
        'duracao_total_segundos': round(duracao_total, 2),
        'total_audios': total_audios,
        'processados_sucesso': processados,
        'pulados_processados': pulados,
        'erros': erros,
        'duracao_audios_aprovados_segundos': round(duracao_audios_aprovados, 2),
        
        # M02
        'm02_criar_diretorios_segundos': round(tempos_modulos.get('m02', 0) or 0, 2) if tempos_modulos.get('m02') is not None else 'null',
        'm02_criar_diretorios_percentual': percentuais.get('m02') if percentuais.get('m02') is not None else 'null',

        # M04 VAD
        'm04_segmentacao_vad_segundos': round(tempos_modulos.get('m04_vad', 0) or 0, 2) if tempos_modulos.get('m04_vad') is not None else 'null',
        'm04_segmentacao_vad_percentual': percentuais.get('m04_vad') if percentuais.get('m04_vad') is not None else 'null',
        
        # M05
        'm05_conversao_16khz_segundos': round(tempos_modulos.get('m05', 0) or 0, 2) if tempos_modulos.get('m05') is not None else 'null',
        'm05_conversao_16khz_percentual': percentuais.get('m05') if percentuais.get('m05') is not None else 'null',
        
        # M06
        'm06_filtro_mos_segundos': round(tempos_modulos.get('m06', 0) or 0, 2) if tempos_modulos.get('m06') is not None else 'null',
        'm06_filtro_mos_percentual': percentuais.get('m06') if percentuais.get('m06') is not None else 'null',
        
        # M07
        'm07_deteccao_overlap_segundos': round(tempos_modulos.get('m07', 0) or 0, 2) if tempos_modulos.get('m07') is not None else 'null',
        'm07_deteccao_overlap_percentual': percentuais.get('m07') if percentuais.get('m07') is not None else 'null',
        
        # M08
        'm08_transcricao_whisper_segundos': round(tempos_modulos.get('m08', 0) or 0, 2) if tempos_modulos.get('m08') is not None else 'null',
        'm08_transcricao_whisper_percentual': percentuais.get('m08') if percentuais.get('m08') is not None else 'null',
        
        # M09
        'm09_transcricao_wav2vec_segundos': round(tempos_modulos.get('m09', 0) or 0, 2) if tempos_modulos.get('m09') is not None else 'null',
        'm09_transcricao_wav2vec_percentual': percentuais.get('m09') if percentuais.get('m09') is not None else 'null',
        
        # M10
        'm10_normalizacao_texto_segundos': round(tempos_modulos.get('m10', 0) or 0, 2) if tempos_modulos.get('m10') is not None else 'null',
        'm10_normalizacao_texto_percentual': percentuais.get('m10') if percentuais.get('m10') is not None else 'null',
        
        # M11
        'm11_validacao_similaridade_segundos': round(tempos_modulos.get('m11', 0) or 0, 2) if tempos_modulos.get('m11') is not None else 'null',
        'm11_validacao_similaridade_percentual': percentuais.get('m11') if percentuais.get('m11') is not None else 'null',
        
        # M12
        'm12_denoiser_deepfilternet_segundos': round(tempos_modulos.get('m12', 0) or 0, 2) if tempos_modulos.get('m12') is not None else 'null',
        'm12_denoiser_deepfilternet_percentual': percentuais.get('m12') if percentuais.get('m12') is not None else 'null',
        
        # M13
        'm13_normalizacao_audio_sox_segundos': round(tempos_modulos.get('m13', 0) or 0, 2) if tempos_modulos.get('m13') is not None else 'null',
        'm13_normalizacao_audio_sox_percentual': percentuais.get('m13') if percentuais.get('m13') is not None else 'null',
        
        # M14
        'm14_geracao_metadados_segundos': round(tempos_modulos.get('m14', 0) or 0, 2) if tempos_modulos.get('m14') is not None else 'null',
        'm14_geracao_metadados_percentual': percentuais.get('m14') if percentuais.get('m14') is not None else 'null',
        
        # M15
        'm15_cleanup_segundos': round(tempos_modulos.get('m15', 0) or 0, 2) if tempos_modulos.get('m15') is not None else 'null',
        'm15_cleanup_percentual': percentuais.get('m15') if percentuais.get('m15') is not None else 'null',
    }
    
    # Headers
    fieldnames = list(linha.keys())
    
    # Write in APPEND mode
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')

        # Write header only if the file does not exist
        if not arquivo_existe:
            writer.writeheader()
        
        writer.writerow(linha)
    
    print(f"\nMetadados salvos em: {csv_path}")
    print(f"ID do registro: {registro_id}")


# ==============================================================================
# PROCESSING PIPELINE
# ==============================================================================

def executar_pipeline(audio_id: str, logger: logging.Logger, tempos_modulos: Dict[str, Optional[float]]) -> bool:
    """
    Runs the full pipeline for one audio file.

    Args:
        audio_id: ID of the audio file to process
        logger: Configured logger
        tempos_modulos: Dictionary to store execution times

    Returns:
        True on success, False on error
    """
    logger.info("="*80)
    logger.info(f"INICIANDO PIPELINE PARA AUDIO: {audio_id}")
    logger.info("="*80)

    try:
        # ======================================================================
        # M02 - CREATE DIRECTORIES (REQUIRED)
        # ======================================================================
        logger.info("[M02] Criando estrutura de diretorios...")
        inicio = time.time()
        # The specs come from the SOURCE, probed before the conversion to
        # WAV. They are the only memory of where the audio came from: from
        # here on only WAV circulates.
        specs_origem = criar_diretorios(audio_id)
        tempos_modulos['m02'] = time.time() - inicio
        if specs_origem is None:
            logger.error("[M02] FALHOU - entrada ausente ou conversao falhou. Abortando pipeline")
            return False
        logger.info(f"[M02] Concluido ({tempos_modulos['m02']:.2f}s)")

        # ======================================================================
        # M04 - SEGMENTATION (CONDITIONAL)
        # ======================================================================
        modo_segmentacao = MASTER.get('segmentacao', '')

        if modo_segmentacao == 'vad':
            logger.info("[M04-VAD] Executando segmentacao por VAD...")
            inicio = time.time()
            tem_segmentos = executar_segmentacao_vad(audio_id, specs_origem)
            tempos_modulos['m04_vad'] = time.time() - inicio
            if not tem_segmentos:
                logger.warning("[M04-VAD] Nenhum segmento valido encontrado — audio descartado")
                logger.info(f"[M04-VAD] Concluido ({tempos_modulos['m04_vad']:.2f}s)")
                return False
            logger.info(f"[M04-VAD] Concluido ({tempos_modulos['m04_vad']:.2f}s)")

        elif modo_segmentacao == '':
            logger.info("[SEGMENTACAO] Pulando (audio ja segmentado)")
        else:
            logger.error(f"[SEGMENTACAO] Modo invalido: {modo_segmentacao}")
            return False

        # ======================================================================
        # M05 - 16KHZ CONVERSION (REQUIRED)
        # ======================================================================
        logger.info("[M05] Convertendo para 16kHz mono...")
        inicio = time.time()
        sucesso = processar_pasta(audio_id)
        tempos_modulos['m05'] = time.time() - inicio
        if not sucesso:
            logger.error("[M05] FALHOU - conversao 16kHz nao produziu saida. Abortando pipeline")
            return False
        logger.info(f"[M05] Concluido ({tempos_modulos['m05']:.2f}s)")

        # ======================================================================
        # M06 - MOS FILTER (CONDITIONAL)
        # ======================================================================
        if MASTER.get('mos_filter', False):
            logger.info("[M06] Aplicando filtro MOS...")
            inicio = time.time()
            sucesso = processar_mos(audio_id)
            tempos_modulos['m06'] = time.time() - inicio
            if not sucesso:
                logger.error("[M06] FALHOU - Abortando pipeline")
                return False
            logger.info(f"[M06] Concluido ({tempos_modulos['m06']:.2f}s)")
        else:
            logger.info("[M06] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M07 - OVERLAP DETECTION (CONDITIONAL)
        # ======================================================================
        if MASTER.get('overlap', False):
            logger.info("[M07] Detectando overlap de locutores...")
            inicio = time.time()
            sucesso = processar_overlap(audio_id)
            tempos_modulos['m07'] = time.time() - inicio
            if not sucesso:
                logger.error("[M07] FALHOU - Abortando pipeline")
                return False
            logger.info(f"[M07] Concluido ({tempos_modulos['m07']:.2f}s)")
        else:
            logger.info("[M07] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M08 - WHISPER TRANSCRIPTION (CONDITIONAL)
        # ======================================================================
        if MASTER.get('transcricao_whisper', False):
            logger.info("[M08] Transcrevendo com Whisper...")
            inicio = time.time()
            sucesso = processar_whisper(audio_id)
            tempos_modulos['m08'] = time.time() - inicio
            if not sucesso:
                logger.error("[M08] FALHOU - Abortando pipeline")
                return False
            logger.info(f"[M08] Concluido ({tempos_modulos['m08']:.2f}s)")
        else:
            logger.info("[M08] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M09 - WAV2VEC TRANSCRIPTION (CONDITIONAL)
        # ======================================================================
        if MASTER.get('transcricao_wav2vec', False):
            logger.info("[M09] Transcrevendo com Wav2Vec...")
            inicio = time.time()
            sucesso = processar_wav2vec(audio_id)
            tempos_modulos['m09'] = time.time() - inicio
            if not sucesso:
                logger.error("[M09] FALHOU - Abortando pipeline")
                return False
            logger.info(f"[M09] Concluido ({tempos_modulos['m09']:.2f}s)")
        else:
            logger.info("[M09] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M10 - TEXT NORMALIZATION (REQUIRED)
        # ======================================================================
        logger.info("[M10] Normalizando textos...")
        inicio = time.time()
        sucesso = processar_normalizacao(audio_id)
        tempos_modulos['m10'] = time.time() - inicio
        if not sucesso:
            logger.error("[M10] FALHOU - Abortando pipeline")
            return False
        logger.info(f"[M10] Concluido ({tempos_modulos['m10']:.2f}s)")

        # ======================================================================
        # M11 - SIMILARITY VALIDATION (REQUIRED)
        # ======================================================================
        logger.info("[M11] Validando similaridade (WER, CER, Levenshtein normalizado)...")
        inicio = time.time()
        sucesso = processar_validacao(audio_id)
        tempos_modulos['m11'] = time.time() - inicio
        if not sucesso:
            logger.error("[M11] FALHOU - Abortando pipeline")
            return False
        logger.info(f"[M11] Concluido ({tempos_modulos['m11']:.2f}s)")

        # ======================================================================
        # M12 - DENOISER (CONDITIONAL)
        # ======================================================================
        if MASTER.get('Denoiser', False):
            logger.info("[M12] Aplicando DeepFilterNet3...")
            inicio = time.time()
            sucesso = processar_denoiser(audio_id)
            tempos_modulos['m12'] = time.time() - inicio
            if not sucesso:
                logger.error("[M12] FALHOU - Abortando pipeline")
                return False
            logger.info(f"[M12] Concluido ({tempos_modulos['m12']:.2f}s)")
        else:
            logger.info("[M12] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M13 - AUDIO NORMALIZATION (REQUIRED)
        # ======================================================================
        logger.info("[M13] Normalizando audios com SoX...")
        inicio = time.time()
        sucesso = processar_normalizador_audio(audio_id)
        tempos_modulos['m13'] = time.time() - inicio
        if not sucesso:
            logger.error("[M13] FALHOU - Abortando pipeline")
            return False
        logger.info(f"[M13] Concluido ({tempos_modulos['m13']:.2f}s)")

        # ======================================================================
        # M14 - METADATA (REQUIRED)
        # ======================================================================
        logger.info("[M14] Gerando metadados CSV...")
        inicio = time.time()
        resultado_m14 = processar_metadados(audio_id)
        tempos_modulos['m14'] = time.time() - inicio

        if not resultado_m14['sucesso']:
            logger.error(f"[M14] FALHA: {resultado_m14['motivo_falha']}")
            logger.error("="*80)
            logger.error(f"PIPELINE INTERROMPIDA NO M14 PARA: {audio_id}")
            logger.error("="*80)
            return False

        for aviso in resultado_m14['avisos']:
            logger.warning(f"[M14] {aviso}")

        logger.info(
            f"[M14] Concluido ({tempos_modulos['m14']:.2f}s) - "
            f"{resultado_m14['n_persistidos']} linha(s) persistida(s) no dataset.csv"
        )

        # ======================================================================
        # M15 - CLEANUP (CONDITIONAL)
        # ======================================================================
        modo_cleanup = MASTER.get('cleanup', 'none')
        if modo_cleanup != 'none':
            logger.info(f"[M15] Executando cleanup (modo: {modo_cleanup})...")
            inicio = time.time()
            cleanup_ok = executar_cleanup(audio_id)
            tempos_modulos['m15'] = time.time() - inicio
            # Cleanup is a special case: the dataset is already saved, so a
            # failure here is a warning and does not invalidate the
            # processed audio
            if not cleanup_ok:
                logger.warning("[M15] Cleanup falhou - dataset ja gravado, pipeline segue")
            logger.info(f"[M15] Concluido ({tempos_modulos['m15']:.2f}s)")
        else:
            logger.info("[M15] Pulando (cleanup desabilitado)")

        # ======================================================================
        # FINALIZATION
        # ======================================================================
        logger.info("="*80)
        logger.info(f"PIPELINE CONCLUIDA COM SUCESSO PARA: {audio_id}")
        logger.info("="*80)

        return True

    except Exception as e:
        logger.error("="*80)
        logger.error(f"ERRO CRITICO NA PIPELINE: {str(e)}")
        logger.error("="*80)
        logger.exception("Traceback completo:")
        return False


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main function of the orchestrator
    """
    print("\n" + "="*80)
    print("KATUBE-2026 PIPELINE ORQUESTRADOR")
    print("="*80)
    print(f"Data/Hora Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Mark the start of the total processing
    inicio_geral = time.time()

    # Deduplication: the already-completed ids, read ONCE and used at both
    # guard points (m00 and the loop below). A read error ABORTS - with an
    # empty set, everything would be reprocessed without anyone noticing.
    ids_concluidos = carregar_ids_concluidos()
    if ids_concluidos is None:
        print("\nERRO ao ler o CSV dos concluidos - execucao abortada")
        print("Sem ele nao ha como saber o que ja foi processado, e seguir "
              "duplicaria o dataset. Corrija o arquivo e rode de novo.")
        return

    # Step 00: name and move the material from arquivos/input/ to
    # arquivos/audios/, which is what the listing below scans.
    # The predicate is injected: the rule keeps living in only one place.
    if not preparar_entrada(lambda audio_id: audio_id in ids_concluidos):
        print("\nERRO na etapa de nomeacao - execucao abortada")
        print("Nenhum audio foi processado. Verifique as mensagens [M00] acima.")
        return

    # List available IDs
    ids_disponiveis = listar_ids_disponiveis()

    if not ids_disponiveis:
        print("\nNENHUM AUDIO ENCONTRADO em ./arquivos/audios/")
        print("Adicione pastas no formato: arquivos/audios/{audio_id}/{audio_id}.<ext>")
        return

    print(f"\nAudios encontrados: {len(ids_disponiveis)}")
    for audio_id in ids_disponiveis:
        print(f"  - {audio_id}")
    print()

    # Counters FOR THIS RUN
    total_nesta_execucao = 0
    processados_nesta_execucao = 0
    pulados_nesta_execucao = 0
    erros_nesta_execucao = 0

    # List of IDs processed SUCCESSFULLY IN THIS RUN
    ids_processados_com_sucesso = []

    # Dictionary to accumulate times for all modules IN THIS RUN
    tempos_modulos_acumulados = {
        'm02': 0.0, 'm04_vad': None,
        'm05': 0.0, 'm06': None, 'm07': None, 'm08': None, 'm09': None,
        'm10': 0.0, 'm11': 0.0, 'm12': None, 'm13': 0.0, 'm14': 0.0, 'm15': None
    }

    for idx, audio_id in enumerate(ids_disponiveis, 1):
        print(f"\n[{idx}/{len(ids_disponiveis)}] Processando: {audio_id}")
        print("-"*80)

        # Check whether it was already completed BEFORE
        if audio_id in ids_concluidos:
            print(f"Audio {audio_id} ja concluido (consta de {CSV_CONCLUIDOS.name})")

            # Run cleanup if configured
            modo_cleanup = MASTER.get('cleanup', 'none')
            if modo_cleanup in ['all', 'input']:
                print(f"Executando cleanup (modo: {modo_cleanup})...")
                logger_pulado = configurar_logger(audio_id)
                if not executar_cleanup(audio_id):
                    logger_pulado.warning(
                        "[M15] Cleanup falhou para audio ja concluido - nada a reverter"
                    )

            pulados_nesta_execucao += 1
            continue

        # Audio file was NOT completed - count IN THIS RUN
        total_nesta_execucao += 1

        # Configure logger
        logger = configurar_logger(audio_id)

        # Dictionary for this audio file's times
        tempos_modulos_audio = {}

        # Run the pipeline
        sucesso = executar_pipeline(audio_id, logger, tempos_modulos_audio)

        if sucesso:
            processados_nesta_execucao += 1
            ids_processados_com_sucesso.append(audio_id)

            # Accumulate times ONLY for those processed successfully
            for modulo, tempo in tempos_modulos_audio.items():
                if tempo is not None:
                    if tempos_modulos_acumulados[modulo] is None:
                        tempos_modulos_acumulados[modulo] = 0.0
                    tempos_modulos_acumulados[modulo] += tempo
        else:
            erros_nesta_execucao += 1
            print(f"\nERRO ao processar {audio_id} - verifique o log")
    
    # Calculate the total duration OF THIS RUN
    duracao_total = time.time() - inicio_geral

    # Calculate the duration of approved audio files ONLY FROM THIS RUN
    duracao_audios_aprovados = calcular_duracao_audios_aprovados(ids_processados_com_sucesso)

    # Save metadata to CSV (APPEND) - ONLY data FROM THIS RUN
    salvar_metadados_csv(
        duracao_total=duracao_total,
        total_audios=total_nesta_execucao,
        processados=processados_nesta_execucao,
        pulados=pulados_nesta_execucao,
        erros=erros_nesta_execucao,
        duracao_audios_aprovados=duracao_audios_aprovados,
        tempos_modulos=tempos_modulos_acumulados
    )
    
    # Final report
    print("\n" + "="*80)
    print("PROCESSAMENTO FINALIZADO")
    print("="*80)
    print(f"Total de audios NESTA EXECUCAO: {total_nesta_execucao}")
    print(f"  Processados com sucesso: {processados_nesta_execucao}")
    print(f"  Pulados (ja concluidos): {pulados_nesta_execucao}")
    print(f"  Erros: {erros_nesta_execucao}")
    print(f"Duracao total: {duracao_total/60:.2f} minutos")
    print(f"Duracao audios aprovados: {duracao_audios_aprovados/60:.2f} minutos")
    print("="*80 + "\n")


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessamento interrompido pelo usuario (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERRO CRITICO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)