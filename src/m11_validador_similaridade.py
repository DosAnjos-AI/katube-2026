#!/usr/bin/env python3
"""
Module m11_validador_similaridade.py
Validates similarity between the STT transcriptions (whisper x wav2vec)

ALWAYS calculates all three metrics, each in its own established
convention:

  wer              error rate per WORD,       0 = perfect, no upper bound
  cer              error rate per CHARACTER,  0 = perfect
  levenshtein_norm normalized similarity,     1 = identical, range 0 to 1

Since the conventions point in different directions, each metric has
its own comparator against its own threshold. The segment is only
approved if it passes all THREE thresholds.
"""

import sys
import json
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List
import Levenshtein

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SIMILARITY_VALIDATOR, TEXT_NORMALIZER


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# Limiares, um por metrica - cada um comparado na direcao da sua convencao
LIMIAR_WER = SIMILARITY_VALIDATOR["limiar_wer"]
LIMIAR_CER = SIMILARITY_VALIDATOR["limiar_cer"]
LIMIAR_LEVENSHTEIN_NORM = SIMILARITY_VALIDATOR["limiar_levenshtein_norm"]


# ==============================================================================
# FUNCOES DE NORMALIZACAO DE TEXTO
# ==============================================================================

def normalizar_para_comparacao(texto: str) -> str:
    """
    Normalizes text for comparison, following the TEXT_NORMALIZER
    settings

    Args:
        texto: Original text

    Returns:
        Text normalized for comparison
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
# FUNCOES DE CALCULO DAS TRES METRICAS
# ==============================================================================

def calcular_wer(referencia: str, hipotese: str) -> float:
    """
    Calculates the Word Error Rate (WER) - error rate per WORD.

    WER = edit_distance_in_WORDS / number_of_words_in_the_reference

    The distance is calculated over the LIST of words, not over the
    string: Levenshtein.distance accepts sequences, so each word counts
    as a single token. Swapping an entire word costs 1, whatever its
    length. That is what sets WER apart from CER.

    Convention: 0.0 = identical transcriptions. There is NO upper bound
    - when the recognizer inserts more words than the reference has,
    the value goes above 1.0.

    Args:
        referencia: Reference text (already normalized)
        hipotese: Text to compare (already normalized)

    Returns:
        Error rate per word (0.0 = perfect, no upper bound)
    """
    palavras_ref = referencia.split()
    palavras_hip = hipotese.split()

    # Referencia vazia: nao ha por quantas palavras dividir. Hipotese tambem
    # vazia e acerto perfeito; hipotese com conteudo e erro total.
    if not palavras_ref:
        return 0.0 if not palavras_hip else 1.0

    distancia = Levenshtein.distance(palavras_ref, palavras_hip)

    return distancia / len(palavras_ref)


def calcular_cer(referencia: str, hipotese: str) -> float:
    """
    Calculates the Character Error Rate (CER) - error rate per
    CHARACTER.

    CER = edit_distance_in_CHARACTERS / number_of_characters_in_the_reference

    Convention: 0.0 = identical transcriptions. It also has no upper
    bound, for the same reason as WER.

    Args:
        referencia: Reference text (already normalized)
        hipotese: Text to compare (already normalized)

    Returns:
        Error rate per character (0.0 = perfect, no upper bound)
    """
    if not referencia:
        return 0.0 if not hipotese else 1.0

    distancia = Levenshtein.distance(referencia, hipotese)

    return distancia / len(referencia)


def calcular_levenshtein_normalizado(referencia: str, hipotese: str) -> float:
    """
    Calculates the normalized Levenshtein similarity over CHARACTERS.

    Similarity = 1 - (levenshtein_distance / max_length)

    Convention that is the INVERSE of the other two: 1.0 = identical,
    0.0 = completely different. Closed range between 0.0 and 1.0.

    Args:
        referencia: Reference text (already normalized)
        hipotese: Text to compare (already normalized)

    Returns:
        Normalized similarity (1.0 = identical)
    """
    max_len = max(len(referencia), len(hipotese))

    # Ambos vazios: identicos por definicao
    if max_len == 0:
        return 1.0

    distancia = Levenshtein.distance(referencia, hipotese)

    return 1.0 - (distancia / max_len)


def calcular_tres_metricas(texto1: str, texto2: str) -> Dict[str, float]:
    """
    Calculates all three metrics at once for the pair of texts.

    Normalization runs a single time and feeds all three.

    Args:
        texto1: First text (reference)
        texto2: Second text (hypothesis)

    Returns:
        Dictionary with the three metrics, each in its own convention
    """
    texto1_norm = normalizar_para_comparacao(texto1)
    texto2_norm = normalizar_para_comparacao(texto2)

    return {
        "wer": calcular_wer(texto1_norm, texto2_norm),
        "cer": calcular_cer(texto1_norm, texto2_norm),
        "levenshtein_norm": calcular_levenshtein_normalizado(texto1_norm, texto2_norm),
    }


# ==============================================================================
# FUNCOES DE VALIDACAO
# ==============================================================================

def motivos_de_reprovacao(wer: float, cer: float, levenshtein_norm: float) -> List[str]:
    """
    Checks the three metrics against their thresholds and returns the
    list of rejection reasons, already formatted for the log.

    Each metric is compared in the direction of its own convention:
      wer              passes if <= threshold (error rate: lower is better)
      cer              passes if <= threshold (error rate: lower is better)
      levenshtein_norm passes if >= threshold (similarity: higher is better)

    Returns:
        Empty list when the segment passes all three thresholds
    """
    motivos = []

    if wer > LIMIAR_WER:
        motivos.append(f"wer={wer:.4f} (limiar <= {LIMIAR_WER})")

    if cer > LIMIAR_CER:
        motivos.append(f"cer={cer:.4f} (limiar <= {LIMIAR_CER})")

    if levenshtein_norm < LIMIAR_LEVENSHTEIN_NORM:
        motivos.append(
            f"levenshtein_norm={levenshtein_norm:.4f} "
            f"(limiar >= {LIMIAR_LEVENSHTEIN_NORM})"
        )

    return motivos


def validar_segmento(dados_segmento: Dict) -> Dict:
    """
    Validates the similarity between a segment's STT transcriptions.

    Eligibility criterion: whisper and wav2vec are both required. If
    either one is missing or empty, the segment is not eligible and all
    similarity fields come out as None.

    Approval criterion: passing all THREE thresholds.

    Args:
        dados_segmento: Dictionary with the segment's metadata

    Returns:
        Dictionary with the three metrics and the status, plus the list
        of rejection reasons under the '_motivos' key - this key is
        consumed by the caller for the log and never reaches the JSON
    """
    resultado = {
        "sim_whisper_wav2vec_wer": None,
        "sim_whisper_wav2vec_cer": None,
        "sim_whisper_wav2vec_levenshtein_norm": None,
        "status_similaridade": None,
    }

    texto_whisper = dados_segmento.get("stt_whisper_normalizado")
    texto_wav2vec = dados_segmento.get("stt_wav2vec_normalizado")

    # Verifica elegibilidade: as duas transcricoes sao obrigatorias
    if not texto_whisper or not texto_wav2vec:
        return {**resultado, "_motivos": []}

    metricas = calcular_tres_metricas(texto_whisper, texto_wav2vec)

    resultado["sim_whisper_wav2vec_wer"] = metricas["wer"]
    resultado["sim_whisper_wav2vec_cer"] = metricas["cer"]
    resultado["sim_whisper_wav2vec_levenshtein_norm"] = metricas["levenshtein_norm"]

    motivos = motivos_de_reprovacao(
        metricas["wer"], metricas["cer"], metricas["levenshtein_norm"]
    )

    resultado["status_similaridade"] = "reprovado" if motivos else "aprovado"

    return {**resultado, "_motivos": motivos}


# ==============================================================================
# FUNCOES DE PROCESSAMENTO
# ==============================================================================

def processar_validacao(audio_id: str) -> bool:
    """
    Processes similarity validation for all eligible segments.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the validation JSONs were written. A missing
        precondition propagates an exception (FileNotFoundError).
    """
    # Definir caminhos baseados no audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    ARQUIVO_ACOMPANHAMENTO = PASTA_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json"
    ARQUIVO_FILTRADO = PASTA_JSON_DINAMICO / f"{audio_id}.json"
    PASTA_OUTPUT_VALIDACAO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "09-validacao_similaridade"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO

    print(f"\n{'='*70}")
    print(f"INICIANDO VALIDACAO DE SIMILARIDADE - Audio: {audio_id}")
    print(f"{'='*70}\n")

    print(f"[INFO] Limiares: wer <= {LIMIAR_WER} | cer <= {LIMIAR_CER} | "
          f"levenshtein_norm >= {LIMIAR_LEVENSHTEIN_NORM}")
    print(f"[INFO] O segmento so e aprovado se passar nos tres")

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

    # Quantas vezes cada metrica reprovou. Um segmento pode falhar em mais de
    # uma, entao a soma destes tres pode passar do total de reprovados.
    reprovados_por_metrica = {"wer": 0, "cer": 0, "levenshtein_norm": 0}

    print()

    for segment_id, dados_segmento in dados_acompanhamento.items():
        # Verifica elegibilidade baseada no arquivo filtrado
        if segmentos_elegiveis is not None and segment_id not in segmentos_elegiveis:
            # Segmento nao elegivel - adiciona campos null
            dados_segmento.update({
                "sim_whisper_wav2vec_wer": None,
                "sim_whisper_wav2vec_cer": None,
                "sim_whisper_wav2vec_levenshtein_norm": None,
                "status_similaridade": None,
            })
            total_nao_elegiveis += 1
            print(f"[NAO ELEGIVEL] {segment_id}: descartado por filtro a montante do m11")
            continue

        # Valida segmento
        resultado_validacao = validar_segmento(dados_segmento)
        motivos = resultado_validacao.pop("_motivos")
        dados_segmento.update(resultado_validacao)

        status = resultado_validacao["status_similaridade"]
        valor_wer = resultado_validacao["sim_whisper_wav2vec_wer"]
        valor_cer = resultado_validacao["sim_whisper_wav2vec_cer"]
        valor_lev = resultado_validacao["sim_whisper_wav2vec_levenshtein_norm"]

        if status == "aprovado":
            total_aprovados += 1
            print(f"[APROVADO]  {segment_id}: wer={valor_wer:.4f} "
                  f"cer={valor_cer:.4f} levenshtein_norm={valor_lev:.4f}")
        elif status == "reprovado":
            total_reprovados += 1
            for motivo in motivos:
                reprovados_por_metrica[motivo.split("=")[0]] += 1
            print(f"[REPROVADO] {segment_id}: reprovado por {', '.join(motivos)} "
                  f"| valores completos: wer={valor_wer:.4f} "
                  f"cer={valor_cer:.4f} levenshtein_norm={valor_lev:.4f}")
        else:
            total_nao_elegiveis += 1
            print(f"[NAO ELEGIVEL] {segment_id}: falta transcricao whisper ou wav2vec")

        total_processados += 1

    # Salva arquivo de acompanhamento atualizado (OUTPUT 01)
    arquivo_acomp_output = PASTA_OUTPUT_VALIDACAO / f"{audio_id}_segments_acompanhamento.json"
    with open(arquivo_acomp_output, 'w', encoding='utf-8') as f:
        json.dump(dados_acompanhamento, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] Acompanhamento atualizado: {arquivo_acomp_output}")

    # Cria arquivo validado (apenas segmentos aprovados) (OUTPUT 01)
    dados_validados = {
        seg_id: dados
        for seg_id, dados in dados_acompanhamento.items()
        if dados.get("status_similaridade") == "aprovado"
    }

    arquivo_validado_output = PASTA_OUTPUT_VALIDACAO / f"{audio_id}_validado.json"
    with open(arquivo_validado_output, 'w', encoding='utf-8') as f:
        json.dump(dados_validados, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Arquivo validado criado: {arquivo_validado_output} ({len(dados_validados)} aprovados)")

    # Copia arquivos para pasta json_dinamico (OUTPUT 02)
    shutil.copy2(arquivo_acomp_output, PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json")
    print(f"[COPY] Acompanhamento copiado para: {PASTA_OUTPUT_JSON_DINAMICO}")

    shutil.copy2(
        arquivo_validado_output,
        PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}.json"
    )
    print(f"[COPY] Validado copiado para: {PASTA_OUTPUT_JSON_DINAMICO / f'{audio_id}.json'}")

    # Relatorio final
    print(f"\n{'='*70}")
    print(f"VALIDACAO CONCLUIDA")
    print(f"{'='*70}")
    print(f"Limiares aplicados:")
    print(f"  - wer              <= {LIMIAR_WER}   (taxa de erro por palavra)")
    print(f"  - cer              <= {LIMIAR_CER}   (taxa de erro por caractere)")
    print(f"  - levenshtein_norm >= {LIMIAR_LEVENSHTEIN_NORM}   (similaridade normalizada)")
    print(f"\nSegmentos processados: {total_processados}")
    print(f"  - Aprovados: {total_aprovados}")
    print(f"  - Reprovados: {total_reprovados}")
    print(f"  - Nao elegiveis (falta whisper ou wav2vec): {total_nao_elegiveis}")
    print(f"\nReprovacoes por metrica (um segmento pode falhar em mais de uma):")
    print(f"  - por wer: {reprovados_por_metrica['wer']}")
    print(f"  - por cer: {reprovados_por_metrica['cer']}")
    print(f"  - por levenshtein_norm: {reprovados_por_metrica['levenshtein_norm']}")
    print(f"\nTotal de segmentos no arquivo: {len(dados_acompanhamento)}")
    print(f"{'='*70}\n")

    return True


# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERRO] Uso: python m11_validador_similaridade.py <audio_id>")
        sys.exit(1)

    try:
        processar_validacao(sys.argv[1])
    except Exception as e:
        print(f"\n[ERRO] Falha na validacao: {str(e)}")
        raise
