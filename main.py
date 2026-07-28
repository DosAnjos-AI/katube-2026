#!/usr/bin/env python3
"""
main.py - Orquestrador Principal do Pipeline katube-2026
Processa audios gerando datasets para TTS/STT
"""

import sys
import logging
import csv
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Adicionar pasta raiz e src ao path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))

# Carregar .env antes de qualquer import de torch/CUDA
# CUDA_VISIBLE_DEVICES="" no .env forca CPU nesta maquina
load_dotenv(PROJECT_ROOT / '.env')

from config import MASTER, EXTENSOES_AUDIO

# Importar modulos do pipeline (todos estao em ./src/)
from m02_diretorios import criar_diretorios
from m04_segmentador_audio_vad import executar_segmentacao_vad
from m05_segmentador_16khz import processar_pasta
from m06_mos_filter import processar_mos
from m07_overlap1 import main as processar_overlap
from m08_whisper import main as processar_whisper
from m09_wav2vec import main as processar_wav2vec
from m10_texto_normalizador import processar_normalizacao
from m11_validador_levenshtein import processar_validacao
from m12_denoiser_deepfilternet3 import main as processar_denoiser
from m13_normalizador_audio import main as processar_normalizador_audio
from m14_metadados import processar_metadados
from m15_cleanup import executar_cleanup


# ==============================================================================
# CONFIGURACAO DE LOGS
# ==============================================================================

def configurar_logger(audio_id: str) -> logging.Logger:
    """
    Configura logger para o audio especifico.

    Args:
        audio_id: ID do audio

    Returns:
        Logger configurado
    """
    # Criar pasta de logs
    pasta_logs = PROJECT_ROOT / "dataset" / "log"
    pasta_logs.mkdir(parents=True, exist_ok=True)

    # Arquivo de log
    arquivo_log = pasta_logs / f"{audio_id}.log"

    # Configurar logger
    logger = logging.getLogger(f"pipeline_{audio_id}")
    logger.setLevel(logging.INFO)

    # Remover handlers existentes
    logger.handlers.clear()

    # Handler para arquivo
    file_handler = logging.FileHandler(arquivo_log, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formato
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
# FUNCOES AUXILIARES
# ==============================================================================

def listar_ids_disponiveis() -> List[str]:
    """
    Lista IDs de audios disponiveis em ./arquivos/audios/.

    Cada subpasta deve conter um arquivo de audio com o mesmo nome da pasta
    (qualquer extensao suportada): arquivos/audios/{audio_id}/{audio_id}.{ext}

    Returns:
        Lista de audio_ids encontrados, ordenada alfabeticamente
    """
    pasta_audios = PROJECT_ROOT / "arquivos" / "audios"

    if not pasta_audios.exists():
        return []

    ids = []
    for subpasta in pasta_audios.iterdir():
        if not subpasta.is_dir():
            continue
        audio_id = subpasta.name
        # Verifica se existe arquivo de audio com o nome do audio_id
        tem_audio = any(
            (subpasta / f"{audio_id}{ext}").exists()
            for ext in EXTENSOES_AUDIO
        )
        if tem_audio:
            ids.append(audio_id)
        else:
            print(f"  AVISO: pasta '{audio_id}' nao contem '{audio_id}.<ext>' — ignorada")

    return sorted(ids)


def verificar_processado(audio_id: str) -> bool:
    """
    Verifica se audio ja foi processado checando historico.

    Args:
        audio_id: ID do audio

    Returns:
        True se ja processado, False caso contrario
    """
    arquivo_historico = PROJECT_ROOT / "dataset" / "historico_dataset" / f"{audio_id}.json"
    return arquivo_historico.exists()


def calcular_duracao_audios_aprovados(ids_processados: List[str]) -> float:
    """
    Calcula duracao total dos audios aprovados APENAS dos IDs fornecidos.

    Args:
        ids_processados: Lista de IDs processados NESTA EXECUCAO

    Returns:
        Duracao total em segundos
    """
    duracao_total = 0.0

    for audio_id in ids_processados:
        historico_path = PROJECT_ROOT / "dataset" / "historico_dataset" / f"{audio_id}.json"
        
        if historico_path.exists():
            try:
                with open(historico_path, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                
                # Somar apenas duracoes validas (nao null)
                for metadados in dados.values():
                    duracao = metadados.get('duracao')
                    if duracao is not None:
                        duracao_total += duracao
                        
            except Exception:
                # Ignorar erros silenciosamente
                pass
    
    return duracao_total


def obter_proximo_id_csv() -> int:
    """
    Obtem o proximo ID sequencial para o CSV
    
    Returns:
        Proximo ID disponivel (1 se arquivo nao existe)
    """
    csv_path = PROJECT_ROOT / "dataset" / "processamento_metadados.csv"
    
    if not csv_path.exists():
        return 1
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='|')
            ids = [int(row['id']) for row in reader if row['id'].isdigit()]
            return max(ids) + 1 if ids else 1
    except Exception:
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
    Salva metadados do processamento em CSV (modo APPEND)
    APENAS com dados da iteracao atual
    
    Args:
        duracao_total: Duracao total do processamento em segundos
        total_audios: Total de audios encontrados NESTA EXECUCAO
        processados: Audios processados com sucesso NESTA EXECUCAO
        pulados: Audios pulados NESTA EXECUCAO
        erros: Audios com erro NESTA EXECUCAO
        duracao_audios_aprovados: Duracao total dos audios aprovados NESTA EXECUCAO
        tempos_modulos: Dicionario com tempos de cada modulo NESTA EXECUCAO
    """
    csv_path = PROJECT_ROOT / "dataset" / "processamento_metadados.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Verificar se arquivo existe para criar header
    arquivo_existe = csv_path.exists()
    
    # Obter proximo ID
    registro_id = obter_proximo_id_csv()
    
    # Data/hora fim
    data_hora_fim = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calcular percentuais
    percentuais = {}
    for modulo, tempo in tempos_modulos.items():
        if tempo is not None and duracao_total > 0:
            percentuais[modulo] = round((tempo / duracao_total) * 100, 1)
        else:
            percentuais[modulo] = None
    
    # Preparar linha de dados
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
    
    # Escrever em modo APPEND
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        
        # Escrever header apenas se arquivo nao existe
        if not arquivo_existe:
            writer.writeheader()
        
        writer.writerow(linha)
    
    print(f"\nMetadados salvos em: {csv_path}")
    print(f"ID do registro: {registro_id}")


# ==============================================================================
# PIPELINE DE PROCESSAMENTO
# ==============================================================================

def executar_pipeline(audio_id: str, logger: logging.Logger, tempos_modulos: Dict[str, Optional[float]]) -> bool:
    """
    Executa pipeline completa para um audio.

    Args:
        audio_id: ID do audio a processar
        logger: Logger configurado
        tempos_modulos: Dicionario para armazenar tempos de execucao

    Returns:
        True se sucesso, False se erro
    """
    logger.info("="*80)
    logger.info(f"INICIANDO PIPELINE PARA AUDIO: {audio_id}")
    logger.info("="*80)

    try:
        # ======================================================================
        # M02 - CRIAR DIRETORIOS (OBRIGATORIO)
        # ======================================================================
        logger.info("[M02] Criando estrutura de diretorios...")
        inicio = time.time()
        criar_diretorios(audio_id)
        tempos_modulos['m02'] = time.time() - inicio
        logger.info(f"[M02] Concluido ({tempos_modulos['m02']:.2f}s)")

        # ======================================================================
        # M04 - SEGMENTACAO (CONDICIONAL)
        # ======================================================================
        modo_segmentacao = MASTER.get('segmentacao', '')

        if modo_segmentacao == 'vad':
            logger.info("[M04-VAD] Executando segmentacao por VAD...")
            inicio = time.time()
            tem_segmentos = executar_segmentacao_vad(audio_id)
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
        # M05 - CONVERSAO 16KHZ (OBRIGATORIO)
        # ======================================================================
        logger.info("[M05] Convertendo para 16kHz mono...")
        inicio = time.time()
        processar_pasta(audio_id)
        tempos_modulos['m05'] = time.time() - inicio
        logger.info(f"[M05] Concluido ({tempos_modulos['m05']:.2f}s)")

        # ======================================================================
        # M06 - FILTRO MOS (CONDICIONAL)
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
        # M07 - DETECCAO OVERLAP (CONDICIONAL)
        # ======================================================================
        if MASTER.get('overlap', False):
            logger.info("[M07] Detectando overlap de locutores...")
            inicio = time.time()
            processar_overlap(audio_id)
            tempos_modulos['m07'] = time.time() - inicio
            logger.info(f"[M07] Concluido ({tempos_modulos['m07']:.2f}s)")
        else:
            logger.info("[M07] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M08 - TRANSCRICAO WHISPER (CONDICIONAL)
        # ======================================================================
        if MASTER.get('transcricao_whisper', False):
            logger.info("[M08] Transcrevendo com Whisper...")
            inicio = time.time()
            processar_whisper(audio_id)
            tempos_modulos['m08'] = time.time() - inicio
            logger.info(f"[M08] Concluido ({tempos_modulos['m08']:.2f}s)")
        else:
            logger.info("[M08] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M09 - TRANSCRICAO WAV2VEC (CONDICIONAL)
        # ======================================================================
        if MASTER.get('transcricao_wav2vec', False):
            logger.info("[M09] Transcrevendo com Wav2Vec...")
            inicio = time.time()
            processar_wav2vec(audio_id)
            tempos_modulos['m09'] = time.time() - inicio
            logger.info(f"[M09] Concluido ({tempos_modulos['m09']:.2f}s)")
        else:
            logger.info("[M09] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M10 - NORMALIZACAO TEXTO (OBRIGATORIO)
        # ======================================================================
        logger.info("[M10] Normalizando textos...")
        inicio = time.time()
        processar_normalizacao(audio_id)
        tempos_modulos['m10'] = time.time() - inicio
        logger.info(f"[M10] Concluido ({tempos_modulos['m10']:.2f}s)")

        # ======================================================================
        # M11 - VALIDACAO SIMILARIDADE (OBRIGATORIO)
        # ======================================================================
        logger.info("[M11] Validando similaridade (Levenshtein)...")
        inicio = time.time()
        processar_validacao(audio_id)
        tempos_modulos['m11'] = time.time() - inicio
        logger.info(f"[M11] Concluido ({tempos_modulos['m11']:.2f}s)")

        # ======================================================================
        # M12 - DENOISER (CONDICIONAL)
        # ======================================================================
        if MASTER.get('Denoiser', False):
            logger.info("[M12] Aplicando DeepFilterNet3...")
            inicio = time.time()
            processar_denoiser(audio_id)
            tempos_modulos['m12'] = time.time() - inicio
            logger.info(f"[M12] Concluido ({tempos_modulos['m12']:.2f}s)")
        else:
            logger.info("[M12] Pulando (desabilitado no MASTER)")

        # ======================================================================
        # M13 - NORMALIZACAO AUDIO (OBRIGATORIO)
        # ======================================================================
        logger.info("[M13] Normalizando audios com SoX...")
        inicio = time.time()
        processar_normalizador_audio(audio_id)
        tempos_modulos['m13'] = time.time() - inicio
        logger.info(f"[M13] Concluido ({tempos_modulos['m13']:.2f}s)")

        # ======================================================================
        # M14 - METADADOS (OBRIGATORIO)
        # ======================================================================
        logger.info("[M14] Gerando metadados CSV...")
        inicio = time.time()
        processar_metadados(audio_id)
        tempos_modulos['m14'] = time.time() - inicio
        logger.info(f"[M14] Concluido ({tempos_modulos['m14']:.2f}s)")

        # ======================================================================
        # M15 - CLEANUP (CONDICIONAL)
        # ======================================================================
        modo_cleanup = MASTER.get('cleanup', 'none')
        if modo_cleanup != 'none':
            logger.info(f"[M15] Executando cleanup (modo: {modo_cleanup})...")
            inicio = time.time()
            executar_cleanup(audio_id)
            tempos_modulos['m15'] = time.time() - inicio
            logger.info(f"[M15] Concluido ({tempos_modulos['m15']:.2f}s)")
        else:
            logger.info("[M15] Pulando (cleanup desabilitado)")

        # ======================================================================
        # FINALIZACAO
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
# FUNCAO PRINCIPAL
# ==============================================================================

def main():
    """
    Funcao principal do orquestrador
    """
    print("\n" + "="*80)
    print("KATUBE-2026 PIPELINE ORQUESTRADOR")
    print("="*80)
    print(f"Data/Hora Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Marcar inicio do processamento total
    inicio_geral = time.time()

    # Listar IDs disponiveis
    ids_disponiveis = listar_ids_disponiveis()

    if not ids_disponiveis:
        print("\nNENHUM AUDIO ENCONTRADO em ./arquivos/audios/")
        print("Adicione pastas no formato: arquivos/audios/{audio_id}/{audio_id}.<ext>")
        return

    print(f"\nAudios encontrados: {len(ids_disponiveis)}")
    for audio_id in ids_disponiveis:
        print(f"  - {audio_id}")
    print()

    # Contadores DESTA EXECUCAO
    total_nesta_execucao = 0
    processados_nesta_execucao = 0
    pulados_nesta_execucao = 0
    erros_nesta_execucao = 0

    # Lista de IDs processados COM SUCESSO NESTA EXECUCAO
    ids_processados_com_sucesso = []

    # Dicionario para acumular tempos de todos os modulos DESTA EXECUCAO
    tempos_modulos_acumulados = {
        'm02': 0.0, 'm04_vad': None,
        'm05': 0.0, 'm06': None, 'm07': None, 'm08': None, 'm09': None,
        'm10': 0.0, 'm11': 0.0, 'm12': None, 'm13': 0.0, 'm14': 0.0, 'm15': None
    }

    for idx, audio_id in enumerate(ids_disponiveis, 1):
        print(f"\n[{idx}/{len(ids_disponiveis)}] Processando: {audio_id}")
        print("-"*80)

        # Verificar se ja foi processado ANTES
        if verificar_processado(audio_id):
            print(f"Audio {audio_id} ja processado (encontrado em historico)")

            # Executar cleanup se configurado
            modo_cleanup = MASTER.get('cleanup', 'none')
            if modo_cleanup in ['all', 'input']:
                print(f"Executando cleanup (modo: {modo_cleanup})...")
                logger_temp = configurar_logger(audio_id)
                executar_cleanup(audio_id)

            pulados_nesta_execucao += 1
            continue

        # Audio NAO estava processado - contar NESTA EXECUCAO
        total_nesta_execucao += 1

        # Configurar logger
        logger = configurar_logger(audio_id)

        # Dicionario para tempos deste audio
        tempos_modulos_audio = {}

        # Executar pipeline
        sucesso = executar_pipeline(audio_id, logger, tempos_modulos_audio)

        if sucesso:
            processados_nesta_execucao += 1
            ids_processados_com_sucesso.append(audio_id)

            # Acumular tempos APENAS dos processados com sucesso
            for modulo, tempo in tempos_modulos_audio.items():
                if tempo is not None:
                    if tempos_modulos_acumulados[modulo] is None:
                        tempos_modulos_acumulados[modulo] = 0.0
                    tempos_modulos_acumulados[modulo] += tempo
        else:
            erros_nesta_execucao += 1
            print(f"\nERRO ao processar {audio_id} - verifique o log")
    
    # Calcular duracao total DESTA EXECUCAO
    duracao_total = time.time() - inicio_geral
    
    # Calcular duracao dos audios aprovados APENAS DESTA EXECUCAO
    duracao_audios_aprovados = calcular_duracao_audios_aprovados(ids_processados_com_sucesso)
    
    # Salvar metadados em CSV (APPEND) - APENAS dados DESTA EXECUCAO
    salvar_metadados_csv(
        duracao_total=duracao_total,
        total_audios=total_nesta_execucao,
        processados=processados_nesta_execucao,
        pulados=pulados_nesta_execucao,
        erros=erros_nesta_execucao,
        duracao_audios_aprovados=duracao_audios_aprovados,
        tempos_modulos=tempos_modulos_acumulados
    )
    
    # Relatorio final
    print("\n" + "="*80)
    print("PROCESSAMENTO FINALIZADO")
    print("="*80)
    print(f"Total de audios NESTA EXECUCAO: {total_nesta_execucao}")
    print(f"  Processados com sucesso: {processados_nesta_execucao}")
    print(f"  Pulados (ja processados): {pulados_nesta_execucao}")
    print(f"  Erros: {erros_nesta_execucao}")
    print(f"Duracao total: {duracao_total/60:.2f} minutos")
    print(f"Duracao audios aprovados: {duracao_audios_aprovados/60:.2f} minutos")
    print("="*80 + "\n")


# ==============================================================================
# EXECUCAO
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