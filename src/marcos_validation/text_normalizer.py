#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalizador de Texto para Validação STT
Versão robusta com normalização avançada + mapeamento inteligente de arquivos
Filosofia KISS - simples e funcional, sem emojis
"""

import os
import re
import json
import unicodedata
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Mapeamento de caracteres especiais para português
CHARS_MAP = str.maketrans({
    'ï': 'i', 'ù': 'u', 'ö': 'o', 'î': 'i', 'ñ': 'n',
    'ë': 'e', 'ì': 'i', 'ò': 'o', 'ů': 'u', 'ẽ': 'e',
    'ü': 'u', 'è': 'e', 'æ': 'a', 'å': 'a', 'ø': 'o',
    'þ': 't', 'ð': 'd', 'ß': 's', 'ł': 'l', 'đ': 'd',
    'ć': 'c', 'č': 'c', 'š': 's', 'ž': 'z', 'ý': 'y'
})

def apply_char_mapping(text: str) -> str:
    """
    Aplica mapeamento de caracteres especiais
    
    Args:
        text: Texto para aplicar mapeamento
        
    Returns:
        Texto com caracteres mapeados
    """
    return text.translate(CHARS_MAP)


def number_to_words_pt(num: int) -> str:
    """
    Converte número para extenso em português
    Suporta números de 0 até 999.999.999
    
    Args:
        num: Número inteiro para converter
        
    Returns:
        Número por extenso
    """
    if num == 0:
        return 'zero'
    
    # Unidades
    unidades = ['', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
    
    # 10 a 19
    especiais = ['dez', 'onze', 'doze', 'treze', 'quatorze', 'quinze', 
                 'dezesseis', 'dezessete', 'dezoito', 'dezenove']
    
    # Dezenas
    dezenas = ['', '', 'vinte', 'trinta', 'quarenta', 'cinquenta',
               'sessenta', 'setenta', 'oitenta', 'noventa']
    
    # Centenas
    centenas = ['', 'cento', 'duzentos', 'trezentos', 'quatrocentos', 
                'quinhentos', 'seiscentos', 'setecentos', 'oitocentos', 'novecentos']
    
    def converter_ate_999(n):
        if n == 0:
            return ''
        elif n < 10:
            return unidades[n]
        elif n < 20:
            return especiais[n - 10]
        elif n < 100:
            dez = n // 10
            uni = n % 10
            if uni == 0:
                return dezenas[dez]
            return f"{dezenas[dez]} e {unidades[uni]}"
        else:  # n < 1000
            cen = n // 100
            resto = n % 100
            if n == 100:
                return 'cem'
            elif resto == 0:
                return centenas[cen]
            return f"{centenas[cen]} e {converter_ate_999(resto)}"
    
    if num < 1000:
        return converter_ate_999(num)
    elif num < 1000000:
        milhares = num // 1000
        resto = num % 1000
        if milhares == 1:
            mil_text = 'mil'
        else:
            mil_text = f"{converter_ate_999(milhares)} mil"
        
        if resto == 0:
            return mil_text
        return f"{mil_text} e {converter_ate_999(resto)}"
    else:
        milhoes = num // 1000000
        resto = num % 1000000
        if milhoes == 1:
            milhao_text = 'um milhao'
        else:
            milhao_text = f"{converter_ate_999(milhoes)} milhoes"
        
        if resto == 0:
            return milhao_text
        elif resto < 1000:
            return f"{milhao_text} e {converter_ate_999(resto)}"
        else:
            return f"{milhao_text} {number_to_words_pt(resto)}"


def ordinal_to_words_pt(num: int, gender: str = 'm') -> str:
    """
    Converte número ordinal para extenso
    
    Args:
        num: Número ordinal
        gender: 'm' para masculino, 'f' para feminino
        
    Returns:
        Ordinal por extenso
    """
    ordinais_m = {
        1: 'primeiro', 2: 'segundo', 3: 'terceiro', 4: 'quarto', 5: 'quinto',
        6: 'sexto', 7: 'setimo', 8: 'oitavo', 9: 'nono', 10: 'decimo',
        11: 'decimo primeiro', 12: 'decimo segundo', 13: 'decimo terceiro',
        14: 'decimo quarto', 15: 'decimo quinto', 16: 'decimo sexto',
        17: 'decimo setimo', 18: 'decimo oitavo', 19: 'decimo nono',
        20: 'vigesimo', 30: 'trigesimo', 40: 'quadragesimo',
        50: 'quinquagesimo', 60: 'sexagesimo', 70: 'septuagesimo',
        80: 'octogesimo', 90: 'nonagesimo', 100: 'centesimo'
    }
    
    ordinais_f = {k: v.replace('o', 'a') for k, v in ordinais_m.items()}
    ordinais = ordinais_f if gender == 'f' else ordinais_m
    
    return ordinais.get(num, f"{num}o")


def advanced_number_to_text(text: str) -> str:
    """
    Conversão avançada de números para texto
    Suporta: decimais, percentuais, datas, horas, moedas, ordinais
    
    Args:
        text: Texto com números
        
    Returns:
        Texto com números convertidos
    """
    # Ordinais (1º, 2ª, 3º, etc.)
    def replace_ordinal(match):
        num = int(match.group(1))
        gender = 'f' if match.group(2) == 'ª' else 'm'
        return ordinal_to_words_pt(num, gender)
    
    text = re.sub(r'(\d+)[ºª]', replace_ordinal, text)
    
    # Decimais (ex: 3.14, 2,5)
    def replace_decimal(match):
        inteiro = int(match.group(1))
        decimal = match.group(2)
        int_text = number_to_words_pt(inteiro)
        dec_text = ' '.join([number_to_words_pt(int(d)) for d in decimal])
        return f"{int_text} virgula {dec_text}"
    
    text = re.sub(r'(\d+)[,\.](\d+)', replace_decimal, text)
    
    # Percentuais
    text = re.sub(r'(\d+)%', lambda m: f"{number_to_words_pt(int(m.group(1)))} por cento", text)
    
    # Números inteiros
    def replace_integer(match):
        num = int(match.group(0))
        return number_to_words_pt(num)
    
    text = re.sub(r'\b\d+\b', replace_integer, text)
    
    return text


def remove_html_tags(text: str) -> str:
    """
    Remove tags HTML do texto
    
    Args:
        text: Texto com possíveis tags HTML
        
    Returns:
        Texto limpo
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def text_cleaning(text: str) -> str:
    """
    Limpeza avançada do texto
    
    Args:
        text: Texto para limpar
        
    Returns:
        Texto limpo
    """
    # Remove HTML
    text = remove_html_tags(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Aplica mapeamento de caracteres
    text = apply_char_mapping(text)
    
    # Remove pontuação (mantém espaços)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normaliza espaços
    text = ' '.join(text.split())
    
    return text.strip()


def normalize_text(text: str) -> Optional[str]:
    """
    Normalização completa do texto
    
    Args:
        text: Texto para normalizar
        
    Returns:
        Texto normalizado ou None se vazio
    """
    if not text or text.strip() == "":
        return None
    
    # Converte números para texto
    text = advanced_number_to_text(text)
    
    # Aplica limpeza
    normalized = text_cleaning(text)
    
    return normalized if normalized else None


def extract_base_info(filename: str) -> Tuple[str, str, str]:
    """
    Extrai informações base do nome do arquivo
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Tupla (video_id, segment_number, subsegment_number)
    """
    pattern = r'^([^_]+)_segment_(\d+)(?:.*?_(\d+))?\.'
    match = re.match(pattern, filename)
    
    if match:
        video_id = match.group(1)
        segment_num = match.group(2)
        subseg_num = match.group(3) if match.group(3) else "001"
        return video_id, segment_num, subseg_num
    
    return "", "", ""
def extract_flac_timestamp(filename: str) -> float:
    """
    Extrai timestamp inicial do arquivo FLAC
    
    Args:
        filename: Nome do arquivo FLAC
        
    Returns:
        Timestamp inicial em segundos
        
    Exemplo:
        EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac -> 1.43
        EhzSC3LWez4_segment_000_SPEAKER_00_125.67_189.23.flac -> 125.67
    """
    try:
        # Remove extensao e divide por underscore
        parts = filename.replace('.flac', '').split('_')
        # Penultimo elemento e o timestamp inicial
        timestamp_start = float(parts[-2])
        return timestamp_start
    except (ValueError, IndexError) as e:
        logger.warning(f"Erro ao extrair timestamp de {filename}: {e}")
        return 0.0


def extract_flac_info(filename: str) -> Tuple[str, str]:
    """
    Extrai informacoes de arquivos FLAC do segments_aprovados
    
    Args:
        filename: Nome do arquivo FLAC
        
    Returns:
        Tupla (video_id, segment_number)
    """
    # Exemplo: EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac
    pattern = r'^([^_]+)_segment_(\d+)_'
    match = re.match(pattern, filename)
    
    if match:
        video_id = match.group(1)
        segment_num = match.group(2)
        return video_id, segment_num
    
    return "", ""


def group_flac_files_by_segment(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Agrupa arquivos FLAC por segmento base e ordena por timestamp
    
    Args:
        files: Lista de arquivos FLAC
        
    Returns:
        Dicionario agrupado por segment base, com FLACs ordenados por timestamp
    """
    grouped = defaultdict(list)
    
    for file_path in files:
        video_id, segment_num = extract_flac_info(file_path.name)
        
        if video_id and segment_num:
            base_key = f"{video_id}_segment_{segment_num}"
            grouped[base_key].append(file_path)
    
    # Ordena arquivos dentro de cada grupo por timestamp inicial
    for key in grouped:
        grouped[key].sort(key=lambda x: extract_flac_timestamp(x.name))
    
    return dict(grouped)

def extract_flac_info(filename: str) -> Tuple[str, str]:
    """
    Extrai informacoes de arquivos FLAC do segments_aprovados
    
    Args:
        filename: Nome do arquivo FLAC
        
    Returns:
        Tupla (video_id, segment_number)
    """
    # Exemplo: EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac
    pattern = r'^([^_]+)_segment_(\d+)_'
    match = re.match(pattern, filename)
    
    if match:
        video_id = match.group(1)
        segment_num = match.group(2)
        return video_id, segment_num
    
    return "", ""


def group_flac_files_by_segment(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Agrupa arquivos FLAC por segmento base (ignora timestamps)
    
    Args:
        files: Lista de arquivos FLAC
        
    Returns:
        Dicionario agrupado por segment base
    """
    grouped = defaultdict(list)
    
    for file_path in files:
        video_id, segment_num = extract_flac_info(file_path.name)
        
        if video_id and segment_num:
            base_key = f"{video_id}_segment_{segment_num}"
            grouped[base_key].append(file_path)
    
    # Ordena arquivos dentro de cada grupo pelo nome completo
    for key in grouped:
        grouped[key].sort(key=lambda x: x.name)
    
    return dict(grouped)

def group_files_by_segment(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Agrupa arquivos por segmento base
    
    Args:
        files: Lista de arquivos
        
    Returns:
        Dicionário agrupado por segment base
    """
    grouped = defaultdict(list)
    
    for file_path in files:
        video_id, segment_num, _ = extract_base_info(file_path.name)
        
        if video_id and segment_num:
            base_key = f"{video_id}_segment_{segment_num}"
            grouped[base_key].append(file_path)
    
    # Ordena arquivos dentro de cada grupo
    for key in grouped:
        grouped[key].sort(key=lambda x: x.name)
    
    return dict(grouped)
def map_txt_to_flac(whisper_files: List[Path], 
                    wav2vec2_files: List[Path],
                    flac_files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """
    Mapeia arquivos .txt aos .flac correspondentes
    
    Logica:
    1. Agrupa TXTs por segment (tem _NNN no final)
    2. Agrupa FLACs por segment (ordena por timestamp)
    3. Mapeia por indice: TXT _001 -> FLAC indice 0 (menor timestamp)
    
    Args:
        whisper_files: Lista de arquivos whisper
        wav2vec2_files: Lista de arquivos wav2vec2
        flac_files: Lista de arquivos flac
        
    Returns:
        Dicionario de mapeamento com validacao de consistencia
    """
    # Agrupa TXTs normalmente (tem _NNN no final)
    whisper_groups = group_files_by_segment(whisper_files)
    wav2vec2_groups = group_files_by_segment(wav2vec2_files)
    
    # Agrupa FLACs e ordena por timestamp
    flac_groups = group_flac_files_by_segment(flac_files)
    
    mappings = {}
    
    all_keys = set(whisper_groups.keys()) | set(wav2vec2_groups.keys()) | set(flac_groups.keys())
    
    for base_key in sorted(all_keys):
        whisper_list = whisper_groups.get(base_key, [])
        wav2vec2_list = wav2vec2_groups.get(base_key, [])
        flac_list = flac_groups.get(base_key, [])
        
        # Validacao: verifica se quantidades batem
        if len(whisper_list) != len(wav2vec2_list):
            logger.warning(f"Inconsistencia em {base_key}: "
                         f"{len(whisper_list)} whisper vs {len(wav2vec2_list)} wav2vec2")
        
        if len(whisper_list) != len(flac_list):
            logger.warning(f"Inconsistencia em {base_key}: "
                         f"{len(whisper_list)} TXTs vs {len(flac_list)} FLACs")
        
        # Usa o minimo para evitar index out of range
        min_len = min(len(whisper_list), len(wav2vec2_list), len(flac_list))
        
        if min_len == 0:
            logger.warning(f"Grupo vazio ignorado: {base_key}")
            continue
        
        # Mapeia por indice ordenado
        for i in range(min_len):
            # Extrai numero do arquivo whisper (que e a fonte da verdade)
            whisper_name = whisper_list[i].name
            # Exemplo: EhzSC3LWez4_segment_000_stt_whisper_001.txt
            match = re.search(r'_(\d+)\.txt$', whisper_name)
            if match:
                subseg_num = match.group(1)
            else:
                subseg_num = f"{i+1:03d}"
            
            key = f"{base_key}_stt_{subseg_num}"
            
            # Validacao: verifica se wav2vec2 tem mesmo numero
            wav2vec2_name = wav2vec2_list[i].name
            if f"_{subseg_num}.txt" not in wav2vec2_name:
                logger.warning(f"Numeracao inconsistente: {whisper_name} vs {wav2vec2_name}")
            
            mappings[key] = {
                "whisper": whisper_list[i],
                "wav2vec2": wav2vec2_list[i],
                "flac": flac_list[i]
            }
            
            logger.debug(f"Mapeamento criado: {key}")
            logger.debug(f"  Whisper: {whisper_list[i].name}")
            logger.debug(f"  WAV2VEC2: {wav2vec2_list[i].name}")
            logger.debug(f"  FLAC: {flac_list[i].name}")
    
    return mappings


def read_text_file(filepath: Path) -> Optional[str]:
    """
    Lê arquivo de texto com tratamento de erro
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Conteúdo do arquivo ou None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Erro ao ler {filepath.name}: {e}")
        return None


def process_stt_results(session_dir: str) -> Dict:
    """
    Processa resultados STT de uma sessão
    
    Args:
        session_dir: Diretório da sessão
        
    Returns:
        Dicionário com resultados do processamento
    """
    session_path = Path(session_dir)
    
    if not session_path.exists():
        error_msg = f"Diretório da sessão não encontrado: {session_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Processando sessão: {session_path}")
    
    # Define caminhos
    stt_results_dir = session_path / "stt_results" / "stt_results"
    whisper_dir = stt_results_dir / "STT-whisper"
    wav2vec2_dir = stt_results_dir / "STT-wav2vec2"
    segments_dir = session_path / "stt_ready"
    
    # Verifica diretórios
    if not whisper_dir.exists() or not wav2vec2_dir.exists():
        error_msg = f"Diretórios STT não encontrados em {stt_results_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    if not segments_dir.exists():
        error_msg = f"Diretório stt_ready não encontrado: {segments_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Coleta arquivos
    whisper_files = list(whisper_dir.glob("*.txt"))
    wav2vec2_files = list(wav2vec2_dir.glob("*.txt"))
    # Busca recursiva em subpastas speaker_XX
    flac_files = []
    for speaker_dir in segments_dir.iterdir():
        if speaker_dir.is_dir() and speaker_dir.name.startswith('speaker_'):
            flac_files.extend(list(speaker_dir.glob("*.flac")))
    
    logger.info(f"Encontrados {len(whisper_files)} arquivos Whisper")
    logger.info(f"Encontrados {len(wav2vec2_files)} arquivos WAV2VEC2")
    logger.info(f"Encontrados {len(flac_files)} arquivos FLAC")
    
    if not whisper_files and not wav2vec2_files:
        error_msg = "Nenhum arquivo .txt STT encontrado"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Mapeia arquivos
    mappings = map_txt_to_flac(whisper_files, wav2vec2_files, flac_files)
    
    if not mappings:
        error_msg = "Nenhum mapeamento válido criado"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Criados {len(mappings)} mapeamentos")
    
    # Processa cada mapeamento
    normalized_pairs = {}
    
    for key, files in mappings.items():
        whisper_path = files["whisper"]
        wav2vec2_path = files["wav2vec2"]
        flac_path = files["flac"]
        
        pair_data = {
            "txt_whisper": whisper_path.name if whisper_path else None,
            "txt_wav2vec2": wav2vec2_path.name if wav2vec2_path else None,
            "flac_file": flac_path.name if flac_path else None,
            "whisper_original": "",
            "whisper_normalized": "",
            "wav2vec2_original": "",
            "wav2vec2_normalized": ""
        }
        
        # Lê e normaliza whisper
        if whisper_path and whisper_path.exists():
            whisper_text = read_text_file(whisper_path)
            if whisper_text:
                pair_data["whisper_original"] = whisper_text
                pair_data["whisper_normalized"] = normalize_text(whisper_text)
        
        # Lê e normaliza wav2vec2
        if wav2vec2_path and wav2vec2_path.exists():
            wav2vec2_text = read_text_file(wav2vec2_path)
            if wav2vec2_text:
                pair_data["wav2vec2_original"] = wav2vec2_text
                pair_data["wav2vec2_normalized"] = normalize_text(wav2vec2_text)
        
        normalized_pairs[key] = pair_data
    
    # Extrai video_id
    video_id = list(mappings.keys())[0].split('_')[0] if mappings else "unknown"
    
    # Cria JSON de saída
    output_data = {
        "video_id": video_id,
        "session_dir": str(session_path),
        "total_segments": len(normalized_pairs),
        "normalized_pairs": normalized_pairs
    }
    
    # Salva JSON
# Salva JSON dentro de stt_results/validation_results
    validation_results_dir = stt_results_dir / "validation_results"
    validation_results_dir.mkdir(parents=True, exist_ok=True)
    output_file = validation_results_dir / f"{video_id}_normalized_text.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Salvo texto normalizado em: {output_file}")
        
        return {
            "success": True,
            "output_file": str(output_file),
            "output_files": [str(output_file)],
            "total_segments": len(normalized_pairs),
            "total_videos": 1,
            "video_id": video_id
        }
        
    except Exception as e:
        error_msg = f"Erro ao salvar JSON: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


def find_all_sessions(base_dir: str = "../../audios_baixados/output") -> List[Path]:
    """
    Encontra todas as sessões no diretório base
    
    Args:
        base_dir: Diretório base
        
    Returns:
        Lista de diretórios de sessão
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.warning(f"Diretório base não encontrado: {base_dir}")
        return []
    
    sessions = []
    for item in base_path.iterdir():
        if item.is_dir():
            stt_dir = item / "stt_results"
            if stt_dir.exists():
                sessions.append(item)
    
    return sorted(sessions)


def process_all_sessions(base_dir: str = "../../audios_baixados/output") -> Dict:
    """
    Processa todas as sessões no diretório base
    
    Args:
        base_dir: Diretório base
        
    Returns:
        Dicionário com resultados de todas as sessões
    """
    logger.info("Buscando sessões automaticamente...")
    sessions = find_all_sessions(base_dir)
    
    if not sessions:
        error_msg = f"Nenhuma sessão encontrada em {base_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Encontradas {len(sessions)} sessões para processar")
    
    results = {
        "success": True,
        "total_sessions": len(sessions),
        "processed_sessions": [],
        "failed_sessions": []
    }
    
    for session_path in sessions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando sessão: {session_path.name}")
        logger.info(f"{'='*60}")
        
        result = process_stt_results(str(session_path))
        
        if result["success"]:
            results["processed_sessions"].append({
                "session_name": session_path.name,
                "session_path": str(session_path),
                "output_file": result["output_file"],
                "total_segments": result["total_segments"],
                "video_id": result["video_id"]
            })
        else:
            results["failed_sessions"].append({
                "session_name": session_path.name,
                "session_path": str(session_path),
                "error": result.get("error")
            })
    
    # Resumo
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info(f"{'='*60}")
    logger.info(f"Total de sessões: {results['total_sessions']}")
    logger.info(f"Processadas com sucesso: {len(results['processed_sessions'])}")
    logger.info(f"Falharam: {len(results['failed_sessions'])}")
    
    return results


def main():
    """Ponto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Normaliza textos STT para validação'
    )
    
    parser.add_argument(
        'session_dir',
        nargs='?',
        help='Caminho do diretório da sessão (opcional com --all)'
    )
    
    parser.add_argument(
        '--base-dir',
        default='../../audios_baixados/output',
        help='Diretório base para descoberta automática de sessões'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Processa todas as sessões no diretório base'
    )
    
    args = parser.parse_args()
    
    logger.info("NORMALIZADOR DE TEXTO PARA VALIDACAO STT")
    logger.info("="*50)
    
    # Processa todas as sessões ou sessão específica
    if args.all or args.session_dir is None:
        logger.info("Modo: Processamento automático de todas as sessões")
        result = process_all_sessions(args.base_dir)
        
        if result["success"]:
            logger.info("\nProcessamento concluído!")
            logger.info(f"Sessões processadas: {len(result['processed_sessions'])}")
            if result['failed_sessions']:
                logger.warning(f"Sessões com erro: {len(result['failed_sessions'])}")
        else:
            logger.error(f"Erro: {result.get('error')}")
    
    else:
        logger.info(f"Modo: Processamento de sessão específica")
        result = process_stt_results(args.session_dir)
        
        if result["success"]:
            logger.info("Processamento concluído com sucesso!")
            logger.info(f"Arquivo de saída: {result['output_file']}")
        else:
            logger.error(f"Erro durante processamento: {result.get('error')}")


if __name__ == "__main__":
    main()