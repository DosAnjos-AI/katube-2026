#!/usr/bin/env python3
"""
Modulo m11_validador_levenshtein.py
Valida similaridade entre transcricoes STT usando metricas baseadas em Levenshtein
Adiciona campos de similaridade aos metadados JSON
"""

import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import Levenshtein

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SIMILARITY_VALIDATOR, TEXT_NORMALIZER


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# ID do video a processar
id_video = 'QN7gUP7nYhQ'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
ARQUIVO_ACOMPANHAMENTO = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
ARQUIVO_FILTRADO = PASTA_JSON_DINAMICO / f"{id_video}.json"

# Caminhos de saida
PASTA_OUTPUT_VALIDACAO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "09-validacao_levenshtein"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Campos de texto a comparar (todos opcionais)
CAMPOS_STT = ["stt_leg_normalizado", "stt_whisper_normalizado", "stt_wav2vec_normalizado"]

# Configuracoes do validador
SIMILARITY_THRESHOLD = SIMILARITY_VALIDATOR["similarity_threshold"]
METRIC_TYPE = SIMILARITY_VALIDATOR["metric_type"]


# ==============================================================================
# FUNCOES DE NORMALIZACAO DE TEXTO
# ==============================================================================

def normalizar_para_comparacao(texto: str) -> str:
    """
    Normaliza texto para comparacao seguindo configuracoes do TEXT_NORMALIZER
    
    Args:
        texto: Texto original
        
    Returns:
        Texto normalizado para comparacao
    """
    if not texto:
        return ""
    
    texto_norm = texto
    
    # Remove pontuacao que afeta diccao (se configurado)
    if TEXT_NORMALIZER.get("remove_punctuation_diction", False):
        pontuacao = ".,;!?_"
        for char in pontuacao:
            texto_norm = texto_norm.replace(char, "")
    
    # Remove acentuacao grafica (se configurado)
    if TEXT_NORMALIZER.get("remove_accents_graphic", False):
        import unicodedata
        texto_norm = ''.join(
            c for c in unicodedata.normalize('NFD', texto_norm)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Normaliza espacos multiplos
    texto_norm = ' '.join(texto_norm.split())
    
    # Lowercase para comparacao case-insensitive
    texto_norm = texto_norm.lower()
    
    return texto_norm


# ==============================================================================
# FUNCOES DE CALCULO DE SIMILARIDADE
# ==============================================================================

def calcular_wer(referencia: str, hipotese: str) -> float:
    """
    Calcula Word Error Rate (WER) como metrica de similaridade
    
    WER = 1 - (distancia_levenshtein / num_palavras_referencia)
    Retorna valor entre 0.0 (totalmente diferente) e 1.0 (identico)
    
    Args:
        referencia: Texto de referencia
        hipotese: Texto a comparar
        
    Returns:
        Similaridade WER normalizada (0.0 a 1.0)
    """
    palavras_ref = referencia.split()
    palavras_hip = hipotese.split()
    
    if not palavras_ref:
        return 1.0 if not palavras_hip else 0.0
    
    distancia = Levenshtein.distance(' '.join(palavras_ref), ' '.join(palavras_hip))
    max_len = max(len(' '.join(palavras_ref)), len(' '.join(palavras_hip)))
    
    if max_len == 0:
        return 1.0
    
    similaridade = 1.0 - (distancia / max_len)
    return max(0.0, min(1.0, similaridade))


def calcular_cer(referencia: str, hipotese: str) -> float:
    """
    Calcula Character Error Rate (CER) como metrica de similaridade
    
    CER = 1 - (distancia_levenshtein / num_caracteres_referencia)
    Retorna valor entre 0.0 (totalmente diferente) e 1.0 (identico)
    
    Args:
        referencia: Texto de referencia
        hipotese: Texto a comparar
        
    Returns:
        Similaridade CER normalizada (0.0 a 1.0)
    """
    if not referencia:
        return 1.0 if not hipotese else 0.0
    
    distancia = Levenshtein.distance(referencia, hipotese)
    max_len = max(len(referencia), len(hipotese))
    
    if max_len == 0:
        return 1.0
    
    similaridade = 1.0 - (distancia / max_len)
    return max(0.0, min(1.0, similaridade))


def calcular_levenshtein_normalizado(referencia: str, hipotese: str) -> float:
    """
    Calcula distancia de Levenshtein normalizada como metrica de similaridade
    
    Similaridade = 1 - (distancia_levenshtein / max_length)
    Retorna valor entre 0.0 (totalmente diferente) e 1.0 (identico)
    
    Args:
        referencia: Texto de referencia
        hipotese: Texto a comparar
        
    Returns:
        Similaridade normalizada (0.0 a 1.0)
    """
    if not referencia and not hipotese:
        return 1.0
    
    distancia = Levenshtein.distance(referencia, hipotese)
    max_len = max(len(referencia), len(hipotese))
    
    if max_len == 0:
        return 1.0
    
    similaridade = 1.0 - (distancia / max_len)
    return max(0.0, min(1.0, similaridade))


def calcular_similaridade(texto1: str, texto2: str, metrica: str) -> float:
    """
    Calcula similaridade entre dois textos usando metrica especificada
    
    Args:
        texto1: Primeiro texto
        texto2: Segundo texto
        metrica: Tipo de metrica ("wer", "cer", "levenshtein_norm")
        
    Returns:
        Score de similaridade (0.0 a 1.0)
    """
    # Normaliza textos para comparacao
    texto1_norm = normalizar_para_comparacao(texto1)
    texto2_norm = normalizar_para_comparacao(texto2)
    
    if metrica == "wer":
        return calcular_wer(texto1_norm, texto2_norm)
    elif metrica == "cer":
        return calcular_cer(texto1_norm, texto2_norm)
    elif metrica == "levenshtein_norm":
        return calcular_levenshtein_normalizado(texto1_norm, texto2_norm)
    else:
        raise ValueError(f"Metrica invalida: {metrica}. Use 'wer', 'cer' ou 'levenshtein_norm'")


# ==============================================================================
# FUNCOES DE VALIDACAO
# ==============================================================================

def validar_segmento(dados_segmento: Dict) -> Dict:
    """
    Valida similaridade entre transcricoes STT de um segmento
    
    Args:
        dados_segmento: Dicionario com metadados do segmento
        
    Returns:
        Dicionario com campos de similaridade calculados
    """
    # Identifica campos STT disponiveis
    campos_disponiveis = [campo for campo in CAMPOS_STT if campo in dados_segmento and dados_segmento[campo]]
    
    # Inicializa campos de saida
    resultado = {
        "sim_leg_whisper": None,
        "sim_leg_wav2vec": None,
        "sim_whisper_wav2vec": None,
        "nota_similaridade": None,
        "status_similaridade": None,
        "metrica_similaridade": METRIC_TYPE
    }
    
    # Verifica elegibilidade (minimo 2 campos)
    if len(campos_disponiveis) < 2:
        return resultado
    
    # Lista para armazenar todas as similaridades calculadas
    similaridades = []
    
    # Calcula similaridades par-a-par
    if "stt_leg_normalizado" in campos_disponiveis and "stt_whisper_normalizado" in campos_disponiveis:
        resultado["sim_leg_whisper"] = calcular_similaridade(
            dados_segmento["stt_leg_normalizado"],
            dados_segmento["stt_whisper_normalizado"],
            METRIC_TYPE
        )
        similaridades.append(resultado["sim_leg_whisper"])
    
    if "stt_leg_normalizado" in campos_disponiveis and "stt_wav2vec_normalizado" in campos_disponiveis:
        resultado["sim_leg_wav2vec"] = calcular_similaridade(
            dados_segmento["stt_leg_normalizado"],
            dados_segmento["stt_wav2vec_normalizado"],
            METRIC_TYPE
        )
        similaridades.append(resultado["sim_leg_wav2vec"])
    
    if "stt_whisper_normalizado" in campos_disponiveis and "stt_wav2vec_normalizado" in campos_disponiveis:
        resultado["sim_whisper_wav2vec"] = calcular_similaridade(
            dados_segmento["stt_whisper_normalizado"],
            dados_segmento["stt_wav2vec_normalizado"],
            METRIC_TYPE
        )
        similaridades.append(resultado["sim_whisper_wav2vec"])
    
    # Calcula media das similaridades
    if similaridades:
        resultado["nota_similaridade"] = sum(similaridades) / len(similaridades)
        
        # Define status baseado no threshold
        if resultado["nota_similaridade"] >= SIMILARITY_THRESHOLD:
            resultado["status_similaridade"] = "aprovado"
        else:
            resultado["status_similaridade"] = "reprovado"
    
    return resultado


# ==============================================================================
# FUNCOES DE PROCESSAMENTO
# ==============================================================================

def processar_validacao():
    """
    Processa validacao de similaridade para todos os segmentos elegiveis
    """
    print(f"\n{'='*70}")
    print(f"INICIANDO VALIDACAO DE SIMILARIDADE - Video: {id_video}")
    print(f"{'='*70}\n")
    
    # Cria pasta de saida
    PASTA_OUTPUT_VALIDACAO.mkdir(parents=True, exist_ok=True)
    
    # Carrega arquivo de acompanhamento (obrigatorio)
    if not ARQUIVO_ACOMPANHAMENTO.exists():
        raise FileNotFoundError(f"Arquivo obrigatorio nao encontrado: {ARQUIVO_ACOMPANHAMENTO}")
    
    with open(ARQUIVO_ACOMPANHAMENTO, 'r', encoding='utf-8') as f:
        dados_acompanhamento = json.load(f)
    
    print(f"[INFO] Arquivo acompanhamento carregado: {len(dados_acompanhamento)} segmentos")
    
    # Carrega arquivo filtrado (opcional)
    segmentos_elegiveis = None
    if ARQUIVO_FILTRADO.exists():
        with open(ARQUIVO_FILTRADO, 'r', encoding='utf-8') as f:
            dados_filtrados = json.load(f)
        segmentos_elegiveis = set(dados_filtrados.keys())
        print(f"[INFO] Arquivo filtrado carregado: {len(segmentos_elegiveis)} segmentos elegiveis")
    else:
        print(f"[INFO] Arquivo filtrado nao encontrado - processando todos os segmentos")
    
    # Processa validacao
    total_processados = 0
    total_aprovados = 0
    total_reprovados = 0
    total_nao_elegiveis = 0
    
    for segment_id, dados_segmento in dados_acompanhamento.items():
        # Verifica elegibilidade baseada no arquivo filtrado
        if segmentos_elegiveis is not None and segment_id not in segmentos_elegiveis:
            # Segmento nao elegivel - adiciona campos null
            dados_segmento.update({
                "sim_leg_whisper": None,
                "sim_leg_wav2vec": None,
                "sim_whisper_wav2vec": None,
                "nota_similaridade": None,
                "status_similaridade": None,
                "metrica_similaridade": None
            })
            total_nao_elegiveis += 1
            continue
        
        # Valida segmento
        resultado_validacao = validar_segmento(dados_segmento)
        dados_segmento.update(resultado_validacao)
        
        # Contabiliza resultados
        if resultado_validacao["status_similaridade"] == "aprovado":
            total_aprovados += 1
        elif resultado_validacao["status_similaridade"] == "reprovado":
            total_reprovados += 1
        else:
            total_nao_elegiveis += 1
        
        total_processados += 1
    
    # Salva arquivo de acompanhamento atualizado (OUTPUT 01)
    arquivo_acomp_output = PASTA_OUTPUT_VALIDACAO / f"{id_video}_segments_acompanhamento.json"
    with open(arquivo_acomp_output, 'w', encoding='utf-8') as f:
        json.dump(dados_acompanhamento, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVE] Acompanhamento atualizado: {arquivo_acomp_output}")
    
    # Cria arquivo validado (apenas segmentos aprovados) (OUTPUT 01)
    dados_validados = {
        seg_id: dados 
        for seg_id, dados in dados_acompanhamento.items() 
        if dados.get("status_similaridade") == "aprovado"
    }
    
    arquivo_validado_output = PASTA_OUTPUT_VALIDACAO / f"{id_video}_validado.json"
    with open(arquivo_validado_output, 'w', encoding='utf-8') as f:
        json.dump(dados_validados, f, ensure_ascii=False, indent=2)
    
    print(f"[SAVE] Arquivo validado criado: {arquivo_validado_output} ({len(dados_validados)} aprovados)")
    
    # Copia arquivos para pasta json_dinamico (OUTPUT 02)
    shutil.copy2(arquivo_acomp_output, PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json")
    print(f"[COPY] Acompanhamento copiado para: {PASTA_OUTPUT_JSON_DINAMICO}")
    
    shutil.copy2(
        arquivo_validado_output,
        PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
    )
    print(f"[COPY] Validado copiado para: {PASTA_OUTPUT_JSON_DINAMICO / f'{id_video}.json'}")
    
    # Relatorio final
    print(f"\n{'='*70}")
    print(f"VALIDACAO CONCLUIDA")
    print(f"{'='*70}")
    print(f"Metrica utilizada: {METRIC_TYPE}")
    print(f"Threshold de aprovacao: {SIMILARITY_THRESHOLD}")
    print(f"\nSegmentos processados: {total_processados}")
    print(f"  - Aprovados: {total_aprovados}")
    print(f"  - Reprovados: {total_reprovados}")
    print(f"  - Nao elegiveis (< 2 campos STT): {total_nao_elegiveis}")
    print(f"\nTotal de segmentos no arquivo: {len(dados_acompanhamento)}")
    print(f"{'='*70}\n")


# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    try:
        processar_validacao()
    except Exception as e:
        print(f"\n[ERRO] Falha na validacao: {str(e)}")
        raise