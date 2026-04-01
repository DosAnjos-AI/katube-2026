"""
Utilitário para padronização de nomenclatura de arquivos na pipeline.
"""
import re
from pathlib import Path
from typing import Optional

def extract_base_name(file_path: Path) -> str:
    """
    Extrai o nome base do arquivo removendo sufixos de etapa.
    
    Args:
        file_path: Caminho do arquivo
        
    Returns:
        Nome base do arquivo
    """
    name = file_path.stem
    
    # Remove sufixos de etapa conhecidos
    suffixes_to_remove = [
        r'_download$',
        r'_normalized$',
        r'_segment_\d+$',
        r'_chunk_\d+$',
        r'_mos_approved_\d+$',
        r'_mos_rejected_\d+$',
        r'_diarized_\d+$',
        r'_overlap_\d+$',
        r'_speaker_\d+_\d+$',
        r'_stt_whisper_\d+$',
        r'_stt_wav2vec2_\d+$',
        r'_validated_\d+$',
        r'_rejected_\d+$',
        r'_denoised_\d+$',
        r'_\d+\.\d+_\d+\.\d+$',  # timestamps
        r'_SPEAKER_\d+$',
        r'_SSPEAKER_\d+$'
    ]
    
    for suffix in suffixes_to_remove:
        name = re.sub(suffix, '', name)
    
    return name

def generate_standard_name(base_name: str, stage: str, index: Optional[int] = None, 
                          speaker_id: Optional[str] = None, timestamps: Optional[tuple] = None) -> str:
    """
    Gera nome padronizado para arquivo baseado na etapa.
    
    Args:
        base_name: Nome base do arquivo
        stage: Etapa da pipeline (download, normalized, segment, etc.)
        index: Índice do arquivo (opcional)
        speaker_id: ID do speaker (opcional)
        timestamps: Tupla (start, end) em segundos (opcional)
        
    Returns:
        Nome padronizado do arquivo
    """
    name_parts = [base_name]
    
    # Adiciona sufixo da etapa
    if stage == "download":
        name_parts.append("_download")
    elif stage == "normalized":
        name_parts.append("_normalized")
    elif stage == "segment":
        if index is not None:
            name_parts.append(f"_segment_{index:03d}")
    elif stage == "chunk":
        if index is not None:
            name_parts.append(f"_chunk_{index:02d}")
    elif stage == "mos_approved":
        if index is not None:
            name_parts.append(f"_mos_approved_{index:03d}")
    elif stage == "mos_intermediate":
        if index is not None:
            name_parts.append(f"_mos_intermediate_{index:03d}")
    elif stage == "mos_rejected":
        if index is not None:
            name_parts.append(f"_mos_rejected_{index:03d}")
    elif stage == "diarized":
        if index is not None:
            name_parts.append(f"_diarized_{index:03d}")
    elif stage == "overlap":
        if index is not None:
            name_parts.append(f"_overlap_{index:03d}")
    elif stage == "speaker":
        if speaker_id is not None and index is not None:
            name_parts.append(f"_speaker_{speaker_id}_{index:03d}")
    elif stage == "stt_whisper":
        if index is not None:
            name_parts.append(f"_stt_whisper_{index:03d}")
    elif stage == "stt_wav2vec2":
        if index is not None:
            name_parts.append(f"_stt_wav2vec2_{index:03d}")
    elif stage == "validated":
        if index is not None:
            name_parts.append(f"_validated_{index:03d}")
    elif stage == "rejected":
        if index is not None:
            name_parts.append(f"_rejected_{index:03d}")
    elif stage == "denoised":
        if index is not None:
            name_parts.append(f"_denoised_{index:03d}")
    elif stage == "sox_normalized":
        if index is not None:
            name_parts.append(f"_sox_normalized_{index:03d}")
    elif stage == "final":
        if index is not None:
            name_parts.append(f"_final_{index:03d}")
    
    # Adiciona timestamps se fornecidos
    if timestamps and len(timestamps) == 2:
        start, end = timestamps
        name_parts.append(f"_{start:.2f}_{end:.2f}")
    
    return "".join(name_parts)

def get_stage_from_filename(filename: str) -> str:
    """
    Identifica a etapa baseada no nome do arquivo.
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Nome da etapa identificada
    """
    if "_download" in filename:
        return "download"
    elif "_normalized" in filename:
        return "normalized"
    elif "_segment_" in filename:
        return "segment"
    elif "_chunk_" in filename:
        return "chunk"
    elif "_mos_approved_" in filename:
        return "mos_approved"
    elif "_mos_intermediate_" in filename:
        return "mos_intermediate"
    elif "_mos_rejected_" in filename:
        return "mos_rejected"
    elif "_diarized_" in filename:
        return "diarized"
    elif "_overlap_" in filename:
        return "overlap"
    elif "_speaker_" in filename:
        return "speaker"
    elif "_stt_whisper_" in filename:
        return "stt_whisper"
    elif "_stt_wav2vec2_" in filename:
        return "stt_wav2vec2"
    elif "_validated_" in filename:
        return "validated"
    elif "_rejected_" in filename:
        return "rejected"
    elif "_denoised_" in filename:
        return "denoised"
    elif "_sox_normalized_" in filename:
        return "sox_normalized"
    elif "_final_" in filename:
        return "final"
    else:
        return "unknown"

def standardize_existing_filename(filename: str, target_stage: str, index: Optional[int] = None) -> str:
    """
    Padroniza um nome de arquivo existente para uma nova etapa.
    
    Args:
        filename: Nome atual do arquivo
        target_stage: Etapa de destino
        index: Índice do arquivo (opcional)
        
    Returns:
        Nome padronizado
    """
    base_name = extract_base_name(Path(filename))
    return generate_standard_name(base_name, target_stage, index)
