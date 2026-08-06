"""
Module 06: MOS Filter (Mean Opinion Score - Audio Quality)
Evaluates audio segment quality using the SQUIM model (Speech Quality and Intelligibility Measures)
Classifies segments as low, medium or high quality based on configurable thresholds
"""

import torch
import torchaudio
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import time
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent))

from config import MOS_FILTER, PROJECT_ROOT
from m01_load_models import ModelManager


# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== HELPER FUNCTIONS ====================


def calcular_batch_size(device: torch.device) -> int:
    """
    Calculates the batch size based on configuration and device.

    Args:
        device: Device being used

    Returns:
        int: Batch size to use
    """
    batch_size_config = MOS_FILTER['batch']['batch_size']
    
    if batch_size_config == 'auto':
        if device.type == 'cuda':
            batch_size = 8
            logger.info(f"Batch size automático (GPU): {batch_size}")
        else:
            batch_size = 1
            logger.info(f"Batch size automático (CPU): {batch_size}")
    else:
        batch_size = int(batch_size_config)
        logger.info(f"Batch size configurado: {batch_size}")
    
    return batch_size


def carregar_json_input(json_path: Path) -> Dict:
    """
    Loads and validates the input JSON.

    Args:
        json_path: Path to the input JSON

    Returns:
        Dict: Loaded JSON data

    Raises:
        FileNotFoundError: If the JSON does not exist
        json.JSONDecodeError: If the JSON is corrupted
        ValueError: If required fields are missing
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON de entrada não encontrado: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON corrompido: {json_path}", e.doc, e.pos)
    
    # Validates required fields in each segment
    campos_obrigatorios = ['tempo_inicio', 'tempo_fim', 'duracao']
    
    for nome_arquivo, metadados in dados.items():
        for campo in campos_obrigatorios:
            if campo not in metadados:
                raise ValueError(f"Campo obrigatório '{campo}' ausente em '{nome_arquivo}'")
    
    logger.info(f"JSON de entrada carregado: {len(dados)} segmentos")
    return dados


def preparar_audio(audio_path: Path, device: torch.device) -> torch.Tensor:
    """
    Loads and prepares audio for SQUIM processing.

    Args:
        audio_path: Path to the audio file
        device: Device to place the tensor on

    Returns:
        torch.Tensor: Prepared audio (1, samples) at 16kHz
    """
    # Load audio (should already be at 16kHz)
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if necessary
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Check sample rate (should be 16kHz)
    if sr != 16000:
        logger.warning(f"Sample rate inesperado: {sr}Hz (esperado 16kHz) - {audio_path.name}")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    # Adjust to exactly 12s (192000 samples @ 16kHz)
    target_samples = 192000

    if audio.shape[1] < target_samples:
        # Pad if too short
        padding = target_samples - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, padding), value=0.0)
    elif audio.shape[1] > target_samples:
        # Truncate if too long
        audio = audio[:, :target_samples]
    
    return audio.to(device)


def calcular_mos_batch(
    model: torch.nn.Module,
    audios: List[torch.Tensor],
    device: torch.device
) -> List[Dict[str, float]]:
    """
    Calculates MOS for a batch of audio files.

    Args:
        model: SQUIM model
        audios: List of audio tensors
        device: Processing device

    Returns:
        List[Dict]: List of dictionaries with metrics for each audio file
    """
    # Stack audio files into a batch (batch, 192000)
    batch = torch.cat(audios, dim=0).to(device)

    # Process batch
    with torch.no_grad():
        stoi, pesq, si_sdr = model(batch)

    # Convert results to a list of dictionaries
    resultados = []
    for i in range(len(audios)):
        resultados.append({
            'mos_score': float(pesq[i].cpu().item()),      # PESQ (1-5) as the main score
            'mos_stoi': float(stoi[i].cpu().item()),       # STOI (0-1)
            'mos_si_sdr': float(si_sdr[i].cpu().item())    # SI-SDR (dB)
        })
    
    return resultados


def classificar_qualidade(mos_score: float) -> str:
    """
    Classifies audio quality based on the MOS score (PESQ).

    Args:
        mos_score: PESQ score (1-5)

    Returns:
        str: 'baixa', 'media' or 'alta'
    """
    min_threshold = MOS_FILTER['thresholds']['min_threshold']
    max_threshold = MOS_FILTER['thresholds']['max_threshold']
    
    if mos_score < min_threshold:
        return 'baixa'
    elif mos_score >= max_threshold:
        return 'alta'
    else:
        return 'media'


def processar_segmentos(
    input_dir: Path,
    dados_json: Dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int
) -> Dict:
    """
    Processes all segments, calculating MOS scores.

    Args:
        input_dir: Directory with the input audio files
        dados_json: Original JSON data
        model: SQUIM model
        device: Processing device
        batch_size: Batch size

    Returns:
        Dict: JSON data updated with MOS scores
    """
    total_segmentos = len(dados_json)
    logger.info(f"Processando {total_segmentos} segmentos (batch_size={batch_size})")
    
    # Prepare results structure
    resultados = {}

    # List of files to process
    arquivos = list(dados_json.keys())

    # Process in batches
    for i in range(0, len(arquivos), batch_size):
        batch_files = arquivos[i:i+batch_size]
        batch_audios = []
        batch_names = []

        # Load the batch's audio files
        for nome_arquivo in batch_files:
            audio_path = input_dir / nome_arquivo
            
            if not audio_path.exists():
                logger.error(f"Arquivo de áudio não encontrado: {audio_path}")
                raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
            
            audio = preparar_audio(audio_path, device)
            batch_audios.append(audio)
            batch_names.append(nome_arquivo)
        
        # Calculate MOS for the batch
        batch_resultados = calcular_mos_batch(model, batch_audios, device)

        # Update data with results
        for nome_arquivo, mos_metrics in zip(batch_names, batch_resultados):
            # Copy original data
            resultados[nome_arquivo] = dados_json[nome_arquivo].copy()

            # Add MOS metrics
            resultados[nome_arquivo].update(mos_metrics)

            # Add quality classification
            mos_score = mos_metrics['mos_score']
            resultados[nome_arquivo]['mos_qualidade'] = classificar_qualidade(mos_score)

        # Progress log
        processados = min(i + batch_size, total_segmentos)
        logger.info(f"Processados {processados}/{total_segmentos} segmentos")
    
    return resultados


def filtrar_segmentos_aprovados(dados_completos: Dict) -> Dict:
    """
    Filters only segments with medium or high quality.

    Args:
        dados_completos: Data with all segments

    Returns:
        Dict: Only approved segments (medium/high)
    """
    aprovados = {}
    
    for nome_arquivo, metadados in dados_completos.items():
        qualidade = metadados.get('mos_qualidade')
        if qualidade in ['media', 'alta']:
            aprovados[nome_arquivo] = metadados
    
    total = len(dados_completos)
    aprovados_count = len(aprovados)
    rejeitados = total - aprovados_count
    
    logger.info(f"Segmentos aprovados: {aprovados_count}/{total} ({rejeitados} rejeitados)")
    
    return aprovados


def salvar_json_com_validacao(dados: Dict, output_path: Path) -> bool:
    """
    Saves the JSON and validates it after writing.

    Args:
        dados: Data to save
        output_path: Output path

    Returns:
        bool: True if saved and validated successfully
    """
    try:
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)

        # Post-write validation: tries to load
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        
        logger.info(f"JSON salvo e validado: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar/validar JSON {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def gerar_estatisticas(dados_completos: Dict) -> None:
    """
    Generates and logs processing statistics.

    Args:
        dados_completos: Data with all processed segments
    """
    total = len(dados_completos)
    alta = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'alta')
    media = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'media')
    baixa = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'baixa')
    
    logger.info("=" * 60)
    logger.info("ESTATISTICAS DE QUALIDADE MOS")
    logger.info("=" * 60)
    logger.info(f"Total de segmentos: {total}")
    logger.info(f"Alta qualidade: {alta} ({alta/total*100:.1f}%)")
    logger.info(f"Media qualidade: {media} ({media/total*100:.1f}%)")
    logger.info(f"Baixa qualidade (rejeitados): {baixa} ({baixa/total*100:.1f}%)")
    logger.info(f"Aprovados (media+alta): {alta+media} ({(alta+media)/total*100:.1f}%)")
    logger.info("=" * 60)


# ==================== MAIN FUNCTION ====================

def processar_mos(audio_id: str) -> bool:
    """
    Processes MOS for all segments of an audio file.

    Args:
        audio_id: Audio ID

    Returns:
        bool: True if processed successfully, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info(f"INICIANDO PROCESSAMENTO MOS - Audio ID: {audio_id}")
        logger.info("=" * 60)

        # Define paths
        input_dir = PROJECT_ROOT / 'arquivos' / 'temp' / audio_id / '03-segments_16khz'
        output_dir = PROJECT_ROOT / 'arquivos' / 'temp' / audio_id / '04-mos_score'

        json_input_path = input_dir / f"{audio_id}_segments_originais.json"
        json_acompanhamento_path = output_dir / f"{audio_id}_segments_acompanhamento.json"
        json_mos_path = output_dir / f"{audio_id}_segments_mos.json"
        
        # Check whether it should process (overwrite)
        sobrescrever = MOS_FILTER['comportamento']['sobrescrever']

        if json_acompanhamento_path.exists() and not sobrescrever:
            logger.info(f"MOS ja processado para {audio_id} e sobrescrever=False. Pulando...")
            return True

        # Create output directory if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load input JSON
        logger.info(f"Carregando JSON de entrada: {json_input_path.name}")
        dados_json = carregar_json_input(json_input_path)

        # Detect device and configure batch size
        # Get model from the manager (singleton)
        manager = ModelManager()
        model = manager.get_squim()

        # Device already managed by the manager
        device = next(model.parameters()).device
        batch_size = calcular_batch_size(device)

        # Process all segments
        start_time = time.time()
        dados_completos = processar_segmentos(
            input_dir=input_dir,
            dados_json=dados_json,
            model=model,
            device=device,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        logger.info(f"Processamento concluído em {elapsed:.2f}s")
        
        # Generate statistics
        gerar_estatisticas(dados_completos)

        # Save tracking JSON (all segments)
        logger.info("Salvando JSON de acompanhamento...")
        if not salvar_json_com_validacao(dados_completos, json_acompanhamento_path):
            logger.error("Falha ao salvar JSON de acompanhamento")
            return False

        # Filter approved segments (medium/high)
        dados_aprovados = filtrar_segmentos_aprovados(dados_completos)

        # Save MOS JSON (approved only)
        logger.info("Salvando JSON MOS (segmentos aprovados)...")
        if not salvar_json_com_validacao(dados_aprovados, json_mos_path):
            logger.error("Falha ao salvar JSON MOS")
            return False

        # Copy JSONs to the 00-json_dinamico folder
        logger.info("Copiando JSONs para 00-json_dinamico...")
        json_dinamico_dir = PROJECT_ROOT / 'arquivos' / 'temp' / audio_id / '00-json_dinamico'

        # Copy tracking file (same name)
        shutil.copy2(
            json_acompanhamento_path,
            json_dinamico_dir / f"{audio_id}_segments_acompanhamento.json"
        )

        # Copy mos (renames to {id}.json)
        shutil.copy2(
            json_mos_path,
            json_dinamico_dir / f"{audio_id}.json"
        )

        logger.info("JSONs copiados para 00-json_dinamico")

        logger.info("=" * 60)
        logger.info(f"MOS processado com sucesso: {audio_id}")
        logger.info("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        return False
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON corrompido ou inválido: {e}")
        return False
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Erro inesperado ao processar MOS: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== MANUAL TEST ====================

if __name__ == "__main__":
    sucesso = processar_mos("CA6TSoMw86k")
    
    if sucesso:
        logger.info("Processamento MOS concluido com sucesso")
    else:
        logger.error("Erro ao processar MOS")