#!/usr/bin/env python3
"""
Modulo m11_validador_similaridade.py
Valida similaridade entre as transcricoes STT (whisper x wav2vec)

Calcula SEMPRE as tres metricas, cada uma na sua convencao consagrada:

  wer              taxa de erro por PALAVRA,    0 = perfeito, sem teto superior
  cer              taxa de erro por CARACTERE,  0 = perfeito
  levenshtein_norm similaridade normalizada,    1 = identico, faixa 0 a 1

Como as convencoes apontam em direcoes diferentes, cada metrica tem o seu
proprio comparador contra o proprio limiar. O segmento so e aprovado se
passar nos TRES limiares.
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
    Calcula Word Error Rate (WER) - taxa de erro por PALAVRA.

    WER = distancia_de_edicao_em_PALAVRAS / numero_de_palavras_da_referencia

    A distancia e calculada sobre a LISTA de palavras, nao sobre a string:
    Levenshtein.distance aceita sequencias, entao cada palavra conta como um
    token unico. Trocar uma palavra inteira custa 1, qualquer que seja o
    tamanho dela. E o que separa o WER do CER.

    Convencao: 0.0 = transcricoes identicas. NAO ha teto superior - quando o
    reconhecedor insere mais palavras do que a referencia tem, o valor passa
    de 1.0.

    Args:
        referencia: Texto de referencia (ja normalizado)
        hipotese: Texto a comparar (ja normalizado)

    Returns:
        Taxa de erro por palavra (0.0 = perfeito, sem limite superior)
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
    Calcula Character Error Rate (CER) - taxa de erro por CARACTERE.

    CER = distancia_de_edicao_em_CARACTERES / numero_de_caracteres_da_referencia

    Convencao: 0.0 = transcricoes identicas. Tambem nao tem teto superior,
    pela mesma razao do WER.

    Args:
        referencia: Texto de referencia (ja normalizado)
        hipotese: Texto a comparar (ja normalizado)

    Returns:
        Taxa de erro por caractere (0.0 = perfeito, sem limite superior)
    """
    if not referencia:
        return 0.0 if not hipotese else 1.0

    distancia = Levenshtein.distance(referencia, hipotese)

    return distancia / len(referencia)


def calcular_levenshtein_normalizado(referencia: str, hipotese: str) -> float:
    """
    Calcula similaridade normalizada de Levenshtein sobre CARACTERES.

    Similaridade = 1 - (distancia_levenshtein / comprimento_maximo)

    Convencao INVERSA a das outras duas: 1.0 = identico, 0.0 = totalmente
    diferente. Faixa fechada entre 0.0 e 1.0.

    Args:
        referencia: Texto de referencia (ja normalizado)
        hipotese: Texto a comparar (ja normalizado)

    Returns:
        Similaridade normalizada (1.0 = identico)
    """
    max_len = max(len(referencia), len(hipotese))

    # Ambos vazios: identicos por definicao
    if max_len == 0:
        return 1.0

    distancia = Levenshtein.distance(referencia, hipotese)

    return 1.0 - (distancia / max_len)


def calcular_tres_metricas(texto1: str, texto2: str) -> Dict[str, float]:
    """
    Calcula as tres metricas de uma vez para o par de textos.

    A normalizacao roda uma unica vez e alimenta as tres.

    Args:
        texto1: Primeiro texto (referencia)
        texto2: Segundo texto (hipotese)

    Returns:
        Dicionario com as tres metricas, cada uma na sua convencao
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
    Verifica as tres metricas contra os seus limiares e devolve a lista de
    motivos de reprovacao, ja formatados para log.

    Cada metrica e comparada na direcao da sua propria convencao:
      wer              passa se <= limiar (taxa de erro: menor e melhor)
      cer              passa se <= limiar (taxa de erro: menor e melhor)
      levenshtein_norm passa se >= limiar (similaridade: maior e melhor)

    Returns:
        Lista vazia quando o segmento passa nos tres limiares
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
    Valida a similaridade entre as transcricoes STT de um segmento.

    Criterio de elegibilidade: whisper e wav2vec sao ambos obrigatorios. Se
    qualquer um dos dois estiver ausente ou vazio, o segmento nao e elegivel
    e todos os campos de similaridade saem em None.

    Criterio de aprovacao: passar nos TRES limiares.

    Args:
        dados_segmento: Dicionario com metadados do segmento

    Returns:
        Dicionario com as tres metricas e o status, mais a lista de motivos
        de reprovacao sob a chave '_motivos' - essa chave e consumida pelo
        chamador para o log e nunca chega ao JSON
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
    Processa validacao de similaridade para todos os segmentos elegiveis.

    Args:
        audio_id: ID do audio a processar

    Returns:
        True se os JSONs de validacao foram gravados. Pre-condicao
        ausente propaga excecao (FileNotFoundError).
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
