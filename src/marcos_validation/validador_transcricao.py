#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validador de Texto STT usando Distancia Levenshtein
Versao simplificada que trabalha com JSON de entrada e saida
Filosofia KISS - simples e funcional, sem emojis
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict
from textdistance import levenshtein

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def validate_normalized_texts(input_json_path: str) -> Dict:
    """
    Valida textos normalizados calculando similaridade Levenshtein
    Adiciona scores MOS dos segmentos originais
    
    Args:
        input_json_path: Caminho para o arquivo JSON com textos normalizados
        
    Returns:
        Dicionario com resultados da validacao
    """
    input_path = Path(input_json_path)
    
    # Valida arquivo de entrada
    if not input_path.exists():
        error_msg = f"Arquivo de entrada nao encontrado: {input_json_path}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Processando arquivo: {input_path}")
    
    try:
        # Le JSON de entrada
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_id = data.get('video_id', 'unknown')
        session_dir = data.get('session_dir', '')
        total_segments = data.get('total_segments', 0)
        normalized_pairs = data.get('normalized_pairs', {})
        
        if not normalized_pairs:
            error_msg = "Nenhum par normalizado encontrado no JSON"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Total de segmentos: {total_segments}")
        
        # Carregar scores MOS
        mos_scores_dict = {}
        if session_dir:
            mos_json_path = Path(session_dir) / "segments" / f"{video_id}_mos_scores.json"
            
            if mos_json_path.exists():
                try:
                    with open(mos_json_path, 'r', encoding='utf-8') as f:
                        mos_scores_dict = json.load(f)
                    logger.info(f"MOS scores carregados: {len(mos_scores_dict)} segmentos")
                except Exception as e:
                    logger.warning(f"Erro ao carregar MOS scores: {e}")
            else:
                logger.warning(f"Arquivo MOS nao encontrado: {mos_json_path}")
        
        # Processa cada par calculando similaridade e adicionando MOS
        validated_count = 0
        similarities = []
        mos_scores_found = 0
        
        for segment_id, pair_data in normalized_pairs.items():
            whisper_norm = pair_data.get('whisper_normalized', '').strip()
            wav2vec2_norm = pair_data.get('wav2vec2_normalized', '').strip()
            
            # Verifica se textos sao validos
            if not whisper_norm or not wav2vec2_norm:
                logger.warning(f"Textos vazios em {segment_id}, pulando")
                pair_data['levenshtein_similarity'] = 0.0
                pair_data['mos_score'] = None
                continue
            
            # Calcula similaridade Levenshtein normalizada
            similarity = levenshtein.normalized_similarity(whisper_norm, wav2vec2_norm)
            
            # Adiciona campo de similaridade
            pair_data['levenshtein_similarity'] = round(similarity, 6)
            
            # Extrair prefixo do segment_id para buscar MOS
            # Ex: "EhzSC3LWez4_segment_000_stt_001" -> "EhzSC3LWez4_segment_000"
            prefix = segment_id
            if '_stt_' in segment_id:
                prefix = segment_id.split('_stt_')[0]
            
            # Buscar MOS score pelo prefixo
            mos_score = mos_scores_dict.get(prefix, None)
            pair_data['mos_score'] = mos_score
            
            if mos_score is not None:
                mos_scores_found += 1
                logger.debug(f"{segment_id}: similarity={similarity:.4f}, mos={mos_score}")
            else:
                logger.warning(f"{segment_id}: MOS nao encontrado para prefixo '{prefix}'")
            
            validated_count += 1
            similarities.append(similarity)
        
        # Calcula estatisticas
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        min_similarity = min(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        
        logger.info(f"Validados: {validated_count}/{total_segments}")
        logger.info(f"MOS encontrados: {mos_scores_found}/{validated_count}")
        logger.info(f"Similaridade media: {avg_similarity:.4f}")
        logger.info(f"Similaridade minima: {min_similarity:.4f}")
        logger.info(f"Similaridade maxima: {max_similarity:.4f}")
        
        # Prepara JSON de saida com TODOS os campos originais + levenshtein_similarity + mos_score
        output_data = {
            "video_id": video_id,
            "session_dir": session_dir,
            "total_segments": total_segments,
            "validated_segments": validated_count,
            "mos_scores_found": mos_scores_found,
            "average_similarity": round(avg_similarity, 6),
            "min_similarity": round(min_similarity, 6),
            "max_similarity": round(max_similarity, 6),
            "normalized_pairs": normalized_pairs
        }
        
        # Define nome e caminho do arquivo de saida
        output_filename = f"{video_id}_text_validation.json"
        output_path = input_path.parent / output_filename
        
        # Salva JSON de saida
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Arquivo de validacao salvo em: {output_path}")
        
        return {
            "success": True,
            "input_file": str(input_path),
            "output_file": str(output_path),
            "video_id": video_id,
            "total_segments": total_segments,
            "validated_segments": validated_count,
            "mos_scores_found": mos_scores_found,
            "average_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }
        
    except json.JSONDecodeError as e:
        error_msg = f"Erro ao decodificar JSON: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    except Exception as e:
        error_msg = f"Erro durante validacao: {e}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

def validate_all_sessions(base_dir: str = "../../audios_baixados/output") -> Dict:
    """
    Valida todas as sessoes encontradas no diretorio base
    
    Args:
        base_dir: Diretorio base com sessoes
        
    Returns:
        Dicionario com resultados de todas as validacoes
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        error_msg = f"Diretorio base nao encontrado: {base_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Busca todos os arquivos *_normalized_text.json
    normalized_files = list(base_path.rglob("*_normalized_text.json"))
    
    if not normalized_files:
        error_msg = f"Nenhum arquivo normalizado encontrado em {base_dir}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    logger.info(f"Encontrados {len(normalized_files)} arquivos para validar")
    
    results = {
        "success": True,
        "total_files": len(normalized_files),
        "processed_files": [],
        "failed_files": []
    }
    
    for normalized_file in normalized_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {normalized_file.name}")
        logger.info(f"{'='*60}")
        
        result = validate_normalized_texts(str(normalized_file))
        
        if result["success"]:
            results["processed_files"].append({
                "file": str(normalized_file),
                "output": result["output_file"],
                "video_id": result["video_id"],
                "avg_similarity": result["average_similarity"]
            })
        else:
            results["failed_files"].append({
                "file": str(normalized_file),
                "error": result.get("error")
            })
    
    # Resumo
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO DA VALIDACAO")
    logger.info(f"{'='*60}")
    logger.info(f"Total de arquivos: {results['total_files']}")
    logger.info(f"Processados com sucesso: {len(results['processed_files'])}")
    logger.info(f"Falharam: {len(results['failed_files'])}")
    
    return results


def main():
    """Ponto de entrada principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Valida textos STT normalizados usando Levenshtein'
    )
    
    parser.add_argument(
        'input_json',
        nargs='?',
        help='Caminho do arquivo JSON normalizado (opcional com --all)'
    )
    
    parser.add_argument(
        '--base-dir',
        default='../../audios_baixados/output',
        help='Diretorio base para buscar arquivos (com --all)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Processa todos os arquivos normalizados no diretorio base'
    )
    
    args = parser.parse_args()
    
    logger.info("VALIDADOR DE TEXTO STT - LEVENSHTEIN")
    logger.info("="*50)
    
    if args.all or args.input_json is None:
        logger.info("Modo: Validacao de todos os arquivos")
        result = validate_all_sessions(args.base_dir)
        
        if result["success"]:
            logger.info("\nValidacao concluida!")
            logger.info(f"Arquivos processados: {len(result['processed_files'])}")
            if result['failed_files']:
                logger.warning(f"Arquivos com erro: {len(result['failed_files'])}")
        else:
            logger.error(f"Erro: {result.get('error')}")
    
    else:
        logger.info(f"Modo: Validacao de arquivo especifico")
        result = validate_normalized_texts(args.input_json)
        
        if result["success"]:
            logger.info("Validacao concluida com sucesso!")
            logger.info(f"Arquivo de saida: {result['output_file']}")
            logger.info(f"Similaridade media: {result['average_similarity']:.4f}")
        else:
            logger.error(f"Erro: {result.get('error')}")


if __name__ == "__main__":
    main()