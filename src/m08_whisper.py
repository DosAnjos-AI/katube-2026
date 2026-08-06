#!/usr/bin/env python3
"""
Module m08_whisper.py
Transcribes audio segments using Whisper (distil-whisper-large-v3-ptbr)
Adds the 'stt_whisper' field to the JSON metadata
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import pipeline
import librosa

# Add root folder to the path to import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STT_WHISPER, PROJECT_ROOT, EXTENSOES_AUDIO
from m01_load_models import ModelManager


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Whisper model
MODELO_WHISPER = "freds0/distil-whisper-large-v3-ptbr"


# ==============================================================================
# MODEL AND DEVICE FUNCTIONS
# ==============================================================================


def calcular_batch_size_auto(device: str) -> int:
    """
    Calculates the automatic batch_size based on available VRAM

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        Optimized batch size
    """
    if device == 'cpu':
        return 1
    
    try:
        # Get total available VRAM
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Conservative calculation based on empirical tests
        # distil-whisper-large-v3: ~2.5GB base + ~0.4GB per additional audio file
        if vram_total_gb >= 20:
            return 16
        elif vram_total_gb >= 12:
            return 8
        elif vram_total_gb >= 8:
            return 4
        elif vram_total_gb >= 4:
            return 2
        else:
            return 1
    except:
        # Safe fallback
        return 4


def obter_batch_size(device: str) -> int:
    """
    Gets the final batch_size, considering config and device

    Args:
        device: 'cuda' or 'cpu'

    Returns:
        Batch size to use
    """
    # CPU always uses batch_size=1
    if device == 'cpu':
        batch_config = STT_WHISPER['batch']['batch_size']
        if batch_config != 1:
            print(f"CPU detectada: batch_size ajustado de {batch_config} para 1 automaticamente")
        return 1

    # GPU: use config or calculate automatically
    batch_config = STT_WHISPER['batch']['batch_size']
    
    if batch_config == 'auto':
        batch_size = calcular_batch_size_auto(device)
        print(f"Batch size automatico calculado: {batch_size}")
        return batch_size
    else:
        print(f"Batch size configurado: {batch_config}")
        return int(batch_config)




# ==============================================================================
# JSON FUNCTIONS
# ==============================================================================

def carregar_json(caminho: Path) -> Optional[Dict]:
    """
    Loads a JSON file

    Args:
        caminho: Path of the JSON file

    Returns:
        Dictionary with data, or None on error
    """
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"ERRO ao carregar {caminho.name}: {e}")
        return None


def salvar_json(dados: Dict, caminho: Path) -> bool:
    """
    Saves data to a JSON file

    Args:
        dados: Dictionary to save
        caminho: Path of the destination file

    Returns:
        True on success, False on error
    """
    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"ERRO ao salvar {caminho.name}: {e}")
        return False


def carregar_metadados(pasta_json_dinamico: Path, audio_id: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Loads the metadata JSON files.

    Args:
        pasta_json_dinamico: Path to the 00-json_dinamico folder
        audio_id: Audio ID

    Returns:
        Tuple (json_filtrado, json_acompanhamento)
        json_filtrado can be None if it does not exist
        json_acompanhamento must exist (required)
    """
    # Tracking file (required)
    arquivo_acompanhamento = pasta_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    json_acompanhamento = carregar_json(arquivo_acompanhamento)

    if json_acompanhamento is None:
        print(f"ERRO CRITICO: Arquivo obrigatorio nao encontrado: {arquivo_acompanhamento.name}")
        return None, None

    # Filtered file (optional)
    arquivo_filtrado = pasta_json_dinamico / f"{audio_id}.json"
    json_filtrado = carregar_json(arquivo_filtrado)
    
    if json_filtrado is None:
        print(f"Arquivo de filtro nao encontrado: {arquivo_filtrado.name}")
        print("Processando TODOS os segmentos do arquivo de acompanhamento")
    else:
        print(f"Arquivo de filtro encontrado: {arquivo_filtrado.name}")
        print(f"Processando APENAS segmentos filtrados ({len(json_filtrado)} segmentos)")
    
    return json_filtrado, json_acompanhamento


def determinar_segmentos_elegiveis(json_filtrado: Optional[Dict], 
                                   json_acompanhamento: Dict) -> List[str]:
    """
    Determines which segments should be processed

    Args:
        json_filtrado: JSON with filtered segments (or None)
        json_acompanhamento: JSON with all segments

    Returns:
        List of file names eligible for processing
    """
    if json_filtrado is not None:
        # Use only segments from the filtered file
        return list(json_filtrado.keys())
    else:
        # Use all segments from the tracking file
        return list(json_acompanhamento.keys())


# ==============================================================================
# AUDIO FUNCTIONS
# ==============================================================================

def listar_arquivos_audio_elegiveis(pasta: Path, 
                                    segmentos_elegiveis: List[str]) -> List[Path]:
    """
    Lists audio files that are in the eligible list

    Args:
        pasta: Path of the folder with audio files
        segmentos_elegiveis: List of eligible file names

    Returns:
        List of Paths of the files found
    """
    arquivos = []
    segmentos_set = set(segmentos_elegiveis)
    
    for arquivo in pasta.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_AUDIO:
            if arquivo.name in segmentos_set:
                arquivos.append(arquivo)
    
    return sorted(arquivos)


def transcrever_batch(pipe: pipeline, 
                     arquivos_audio: List[Path],
                     batch_size: int) -> Dict[str, str]:
    """
    Transcribes a batch of audio files

    Args:
        pipe: Configured Whisper pipeline
        arquivos_audio: List of file paths
        batch_size: Batch size

    Returns:
        Dictionary {file_name: transcription}
    """
    resultados = {}
    total = len(arquivos_audio)
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = arquivos_audio[i:i + batch_size]
        batch_atual = min(i + batch_size, total)

        print(f"Processando batch [{i+1}-{batch_atual}/{total}]...")

        # Load the batch's audio files
        audios = []
        nomes = []
        for arquivo in batch:
            try:
                # Load audio at 16kHz (Whisper's sample rate)
                audio, _ = librosa.load(str(arquivo), sr=16000, mono=True)
                audios.append(audio)
                nomes.append(arquivo.name)
            except Exception as e:
                print(f"  ERRO ao carregar {arquivo.name}: {e}")
                resultados[arquivo.name] = None

        # Transcribe batch
        if audios:
            try:
                # Pipeline accepts a list of arrays
                outputs = pipe(audios, generate_kwargs={"language": "pt", "task": "transcribe"})

                # Extract transcriptions
                for nome, output in zip(nomes, outputs):
                    transcricao = output['text'].strip()
                    resultados[nome] = transcricao
                    print(f"  {nome}: OK")

            except Exception as e:
                print(f"  ERRO no batch: {e}")
                # Mark all of the batch as failed
                for nome in nomes:
                    if nome not in resultados:
                        resultados[nome] = None
    
    return resultados


def processar_transcricoes(pipe: pipeline,
                          arquivos_audio: List[Path],
                          batch_size: int) -> Dict[str, str]:
    """
    Processes all transcriptions with progress tracking

    Args:
        pipe: Whisper pipeline
        arquivos_audio: List of files to transcribe
        batch_size: Batch size

    Returns:
        Dictionary {file_name: transcription}
    """
    print(f"\nIniciando transcricao de {len(arquivos_audio)} arquivos...")
    print(f"Batch size: {batch_size}")
    print("-" * 70)
    
    resultados = transcrever_batch(pipe, arquivos_audio, batch_size)
    
    # Statistics
    total = len(resultados)
    sucesso = sum(1 for v in resultados.values() if v is not None)
    falhas = total - sucesso
    
    print("-" * 70)
    print(f"Transcricao concluida: {sucesso}/{total} sucesso, {falhas} falhas")
    
    return resultados


# ==============================================================================
# UPDATE AND SAVE FUNCTIONS
# ==============================================================================

def atualizar_json_com_transcricoes(json_dados: Dict,
                                    transcricoes: Dict[str, str]) -> Dict:
    """
    Adds the stt_whisper field to the metadata

    Args:
        json_dados: Original metadata dictionary
        transcricoes: Dictionary {file_name: transcription}

    Returns:
        Updated dictionary
    """
    json_atualizado = json_dados.copy()
    
    for nome_arquivo, transcricao in transcricoes.items():
        if nome_arquivo in json_atualizado:
            json_atualizado[nome_arquivo]['stt_whisper'] = transcricao
    
    return json_atualizado


def adicionar_transcricoes_null(json_dados: Dict,
                                segmentos_processados: List[str]) -> Dict:
    """
    Adds stt_whisper=null for segments that were not processed

    Args:
        json_dados: Metadata dictionary
        segmentos_processados: List of segments that were processed

    Returns:
        Updated dictionary
    """
    json_atualizado = json_dados.copy()
    processados_set = set(segmentos_processados)
    
    for nome_arquivo in json_atualizado.keys():
        if nome_arquivo not in processados_set:
            if 'stt_whisper' not in json_atualizado[nome_arquivo]:
                json_atualizado[nome_arquivo]['stt_whisper'] = None
    
    return json_atualizado


def salvar_outputs(json_filtrado: Optional[Dict],
                  json_acompanhamento: Dict,
                  segmentos_elegiveis: List[str],
                  transcricoes: Dict[str, str],
                  pasta_output_stt: Path,
                  pasta_output_json_dinamico: Path,
                  audio_id: str) -> bool:
    """
    Saves the updated JSONs to the output folders.

    Args:
        json_filtrado: original filtered JSON (or None)
        json_acompanhamento: original tracking JSON
        segmentos_elegiveis: list of segments that were eligible
        transcricoes: dictionary with transcriptions
        pasta_output_stt: path to the 06-stt_whisper folder
        pasta_output_json_dinamico: path to the 00-json_dinamico folder
        audio_id: audio ID

    Returns:
        True on success, False on error
    """
    # Create output folder if it does not exist
    pasta_output_stt.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("SALVANDO OUTPUTS")
    print("=" * 70)

    # 1. Update tracking JSON (all segments)
    json_acomp_atualizado = atualizar_json_com_transcricoes(
        json_acompanhamento,
        transcricoes
    )
    # Add null for unprocessed segments
    json_acomp_atualizado = adicionar_transcricoes_null(
        json_acomp_atualizado,
        segmentos_elegiveis
    )

    # Save in 06-stt_whisper
    arquivo_acomp_output = pasta_output_stt / f"{audio_id}_segments_acompanhamento.json"
    if not salvar_json(json_acomp_atualizado, arquivo_acomp_output):
        return False
    print(f"Salvo: {arquivo_acomp_output}")

    # Copy to 00-json_dinamico (overwrite)
    arquivo_acomp_dinamico = pasta_output_json_dinamico / f"{audio_id}_segments_acompanhamento.json"
    if not salvar_json(json_acomp_atualizado, arquivo_acomp_dinamico):
        return False
    print(f"Sobrescrito: {arquivo_acomp_dinamico}")

    # 2. If a filtered JSON exists, update and save
    if json_filtrado is not None:
        json_filtrado_atualizado = atualizar_json_com_transcricoes(
            json_filtrado,
            transcricoes
        )

        # Save in 06-stt_whisper
        arquivo_filtrado_output = pasta_output_stt / f"{audio_id}_whisper.json"
        if not salvar_json(json_filtrado_atualizado, arquivo_filtrado_output):
            return False
        print(f"Salvo: {arquivo_filtrado_output}")

        # Copy to 00-json_dinamico as {id}.json (overwrite)
        arquivo_filtrado_dinamico = pasta_output_json_dinamico / f"{audio_id}.json"
        if not salvar_json(json_filtrado_atualizado, arquivo_filtrado_dinamico):
            return False
        print(f"Sobrescrito: {arquivo_filtrado_dinamico}")
    
    print("=" * 70)
    return True


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main(audio_id: str) -> bool:
    """
    Main function: orchestrates the entire transcription flow.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the transcriptions were saved (or there was no eligible
        segment), False if a precondition is missing or saving failed.
    """
    # Define paths based on audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "03-segments_16khz"
    PASTA_OUTPUT_STT = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "06-stt_whisper"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO

    print("=" * 70)
    print("MODULO 08: TRANSCRICAO WHISPER")
    print("=" * 70)
    print(f"Audio ID: {audio_id}")
    print(f"Modelo: {MODELO_WHISPER}")

    # 1. Load model using ModelManager (singleton)
    print("\nCarregando modelo Whisper...")
    manager = ModelManager()
    pipe = manager.get_whisper()

    # Device and dtype already managed by the manager
    # Get the model's device for logs and batch_size
    device = str(pipe.model.device)
    if 'cuda' in device:
        device = 'cuda'
    print(f"Pipeline carregado em {device.upper()}")

    # Get batch_size
    batch_size = obter_batch_size(device)

    # 2. Load metadata
    print("\n" + "=" * 70)
    print("CARREGANDO METADADOS")
    print("=" * 70)
    json_filtrado, json_acompanhamento = carregar_metadados(PASTA_JSON_DINAMICO, audio_id)
    
    if json_acompanhamento is None:
        print("ERRO: Nao foi possivel carregar metadados. Abortando.")
        return False

    # 4. Determine eligible segments
    segmentos_elegiveis = determinar_segmentos_elegiveis(
        json_filtrado,
        json_acompanhamento
    )
    print(f"\nSegmentos elegiveis para processamento: {len(segmentos_elegiveis)}")

    # 5. List eligible audio files
    print("\n" + "=" * 70)
    print("LISTANDO ARQUIVOS DE AUDIO")
    print("=" * 70)
    arquivos_audio = listar_arquivos_audio_elegiveis(
        PASTA_AUDIOS,
        segmentos_elegiveis
    )

    if not arquivos_audio:
        # The funnel may have failed everything earlier: not a failure of this module
        print("AVISO: Nenhum arquivo de audio elegivel encontrado")
        print("Verifique se os arquivos existem em:", PASTA_AUDIOS)
        return True

    print(f"Arquivos encontrados: {len(arquivos_audio)}/{len(segmentos_elegiveis)}")

    # 6. Process transcriptions
    print("\n" + "=" * 70)
    print("PROCESSANDO TRANSCRICOES")
    print("=" * 70)
    transcricoes = processar_transcricoes(pipe, arquivos_audio, batch_size)

    # 7. Save outputs
    sucesso = salvar_outputs(
        json_filtrado,
        json_acompanhamento,
        segmentos_elegiveis,
        transcricoes,
        PASTA_OUTPUT_STT,
        PASTA_OUTPUT_JSON_DINAMICO,
        audio_id
    )

    # 8. Final report
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 70)
    if sucesso:
        print("Status: SUCESSO")
        print(f"Transcritos: {len([t for t in transcricoes.values() if t is not None])}")
        print(f"Outputs salvos em: {PASTA_OUTPUT_STT}")
        print(f"JSONs atualizados em: {PASTA_OUTPUT_JSON_DINAMICO}")
    else:
        print("Status: ERRO ao salvar outputs")
    print("=" * 70)

    return sucesso


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Direct execution requires audio_id as an argument - there is no
    # fixed id in the code. Same pattern as m15_cleanup.py.
    if len(sys.argv) != 2:
        print("Uso: python src/m08_whisper.py <audio_id>")
        sys.exit(1)

    sys.exit(0 if main(sys.argv[1]) else 1)
