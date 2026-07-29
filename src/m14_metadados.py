#!/usr/bin/env python3
"""
Modulo m14_metadados.py
Gera metadados finais do dataset em formato CSV
Copia arquivo de acompanhamento JSON para historico

Arquivo gerenciado:
  dataset.csv — uma linha por segmento entregue, de todos os audio_ids

Regras de escrita (SEGURANCA MAXIMA DO dataset.csv):

1. APPEND PURO. O modulo so CRIA o arquivo, quando ele ainda nao existe, e
   faz APPEND das linhas da rodada ao final. Nunca reescreve o arquivo
   inteiro, nunca remove linha, nunca apaga audio do disco. Remover linha
   do dataset e privilegio exclusivo do usuario, manualmente.

2. LINHA REPETIDA NAO E PROBLEMA AQUI. A deduplicacao nao pertence ao CSV:
   o main barra o audio ja processado na ENTRADA, pelo historico
   (dataset/historico_dataset/{id}.json), antes de qualquer modulo rodar.
   O CSV acompanha o lote; o historico e quem cobre o audio que reaparece
   em outro lote, sem aquele CSV por perto.

3. HEADER FIXADO NA CRIACAO. O header e o da rodada que criou o arquivo.
   Campo do lote que nao esta no header nao entra, e e avisado
   nominalmente; coluna do header ausente no lote vira 'null'. E a
   consequencia direta do append puro: sem reescrita, nao ha como
   acrescentar coluna as linhas antigas. Solucao-ponte — o esquema fixo do
   CSV resolve isso de vez.

4. RETORNO EXPLICITO. processar_metadados devolve um dicionario com
   sucesso, contagem e motivo de falha. Quem chama verifica.
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


# ==============================================================================
# CONFIGURACAO DE PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# CONFIGURACAO DE INPUTS/OUTPUTS
# ==============================================================================

CSV_SEPARATOR = '|'
CSV_ENCODING  = 'utf-8'

COLUNAS_FIXAS = ['arquivo_nome', 'caminho']

CAMPOS_EXCLUIDOS = [
    'stt_whisper_normalizado',
    'stt_wav2vec_normalizado',
]


# ==============================================================================
# FUNCOES AUXILIARES — DADOS
# ==============================================================================

def construir_caminho_audio(nome_arquivo: str, audio_id: str) -> str:
    """Constroi caminho relativo do audio a partir do audio_id."""
    return f"./audio_dataset/{audio_id}/{nome_arquivo}"


def pasta_audios_do_id(audio_id: str) -> Path:
    """Pasta onde vivem os .flac entregues de um audio_id."""
    return PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id


def converter_bool_para_str(valor: Any) -> Any:
    """Converte booleanos Python para string 'True'/'False' capitalizada."""
    if isinstance(valor, bool):
        return str(valor)
    return valor


def carregar_json(caminho: Path) -> Optional[Dict[str, Any]]:
    """Carrega arquivo JSON. Retorna None se o arquivo nao existir."""
    if not caminho.exists():
        return None
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def obter_todas_colunas(dados_json: Dict[str, Any]) -> List[str]:
    """
    Coleta todas as colunas unicas do JSON, mantendo ordem de primeira aparicao.
    Exclui os campos definidos em CAMPOS_EXCLUIDOS.
    Retorna: colunas fixas + campos do JSON filtrados.
    """
    colunas     = COLUNAS_FIXAS.copy()
    colunas_set = set(COLUNAS_FIXAS)

    for segmento_data in dados_json.values():
        for chave in segmento_data.keys():
            if chave not in colunas_set and chave not in CAMPOS_EXCLUIDOS:
                colunas.append(chave)
                colunas_set.add(chave)

    return colunas


def preparar_linha_csv(nome_arquivo: str,
                       dados_segmento: Dict[str, Any],
                       colunas: List[str],
                       audio_id: str) -> Dict[str, Any]:
    """
    Prepara um dicionario de linha para o CSV.
    Todas as colunas sao preenchidas:
      - colunas fixas com seus valores calculados
      - campos presentes no JSON com seus valores (bool convertido)
      - campos ausentes com 'null'
    """
    linha: Dict[str, Any] = {}
    for col in colunas:
        if col == 'arquivo_nome':
            linha[col] = nome_arquivo
        elif col == 'caminho':
            linha[col] = construir_caminho_audio(nome_arquivo, audio_id)
        elif col in dados_segmento:
            linha[col] = converter_bool_para_str(dados_segmento[col])
        else:
            linha[col] = 'null'
    return linha


def ler_header_csv(caminho: Path) -> List[str]:
    """
    Le APENAS o header do CSV (primeira linha).
    Uso de RAM: O(1) — nao carrega nenhuma linha de dados.
    Retorna lista vazia quando o arquivo nao existe ou esta vazio.
    """
    if not caminho.exists():
        return []
    with open(caminho, 'r', encoding=CSV_ENCODING, newline='') as f:
        primeira_linha = f.readline().strip()
    if not primeira_linha:
        return []
    return primeira_linha.split(CSV_SEPARATOR)


# ==============================================================================
# ESCRITA DO CSV — SO CRIACAO E APPEND
# ==============================================================================

def escrever_linhas(caminho_csv: Path,
                    colunas: List[str],
                    linhas_novas: List[Dict[str, Any]],
                    escrever_header: bool) -> str:
    """
    Grava as linhas da rodada no fim do CSV.

    O arquivo e aberto em modo 'a': ele e criado se nao existir e JAMAIS e
    truncado. Nenhuma linha ja gravada e lida, movida ou removida. O header
    so e escrito quando o arquivo esta sendo criado agora.

    As linhas chegam prontas de preparar_linha_csv, que monta o dicionario
    apenas com as chaves de 'colunas': e ele quem descarta o campo do lote
    fora do header e quem preenche com 'null' a coluna do header ausente no
    lote. O extrasaction e o restval do DictWriter sao redundancia
    defensiva, nunca exercitada por este fluxo. O aviso nominal dos campos
    descartados e emitido pelo chamador.

    Returns:
        Modo usado: 'criacao' ou 'append'
    """
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_csv, 'a', encoding=CSV_ENCODING, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=colunas,
            delimiter=CSV_SEPARATOR,
            extrasaction='ignore',
            restval='null',
        )
        if escrever_header:
            writer.writeheader()
        writer.writerows(linhas_novas)

    return 'criacao' if escrever_header else 'append'


# ==============================================================================
# HISTORICO
# ==============================================================================

def copiar_json_historico(origem: Path, destino: Path) -> None:
    """
    Copia o JSON de acompanhamento para o historico (sobrescreve se ja existe).

    O historico e o marcador de audio concluido: sua presenca faz o main
    pular o audio na entrada da proxima rodada. Por isso so e chamado
    depois das linhas estarem efetivadas no CSV.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino)
    print(f"  JSON de historico copiado: {destino}")


# ==============================================================================
# RESULTADO
# ==============================================================================

def montar_resultado(audio_id: str,
                     sucesso: bool,
                     motivo_falha: Optional[str] = None,
                     n_persistidos: int = 0,
                     avisos: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Monta o resultado devolvido por processar_metadados.

    Campos:
        sucesso       — False so em falha real (JSON de entrada ausente,
                        pasta de audio ausente havendo segmentos a entregar).
                        Lote sem segmento aprovado e sucesso-com-aviso.
        motivo_falha  — texto do erro quando sucesso e False
        n_persistidos — linhas gravadas no CSV nesta rodada
        avisos        — mensagens de degradacao do lote
    """
    return {
        'audio_id': audio_id,
        'sucesso': sucesso,
        'motivo_falha': motivo_falha,
        'n_persistidos': n_persistidos,
        'avisos': avisos if avisos is not None else [],
    }


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def processar_metadados(audio_id: str) -> Dict[str, Any]:
    """
    Processa metadados e gera outputs:
      1. Faz append das linhas do lote no dataset.csv (criando-o se preciso)
      2. Copia o JSON de acompanhamento para o historico

    O historico e gravado por ultimo, so depois das linhas efetivadas: se a
    rodada morrer antes daqui, o audio nao fica marcado como processado e a
    proxima execucao o reprocessa.

    Args:
        audio_id: ID do audio a processar

    Returns:
        Dicionario de resultado (ver montar_resultado). O chamador DEVE
        verificar a chave 'sucesso'.
    """
    # --- Definir caminhos ---
    PASTA_JSON_DINAMICO       = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    ARQUIVO_JSON_FILTRADO     = PASTA_JSON_DINAMICO / f"{audio_id}.json"
    ARQUIVO_JSON_ACOMPANHAMENTO = PASTA_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json"

    PASTA_DATASET         = PROJECT_ROOT / "dataset"
    ARQUIVO_CSV_DATASET   = PASTA_DATASET / "dataset.csv"
    PASTA_HISTORICO       = PASTA_DATASET / "historico_dataset"
    ARQUIVO_JSON_HISTORICO = PASTA_HISTORICO / f"{audio_id}.json"

    print(f"Processando metadados para audio: {audio_id}")
    print("-" * 80)

    # --- Carregar JSON (prioridade: filtrado, fallback: acompanhamento) ---
    dados_json = carregar_json(ARQUIVO_JSON_FILTRADO)

    if dados_json is None:
        print(f"  Arquivo filtrado nao encontrado: {ARQUIVO_JSON_FILTRADO}")
        print(f"  Usando arquivo de acompanhamento: {ARQUIVO_JSON_ACOMPANHAMENTO}")
        dados_json = carregar_json(ARQUIVO_JSON_ACOMPANHAMENTO)

        if dados_json is None:
            motivo = f"JSON de entrada ausente: {ARQUIVO_JSON_ACOMPANHAMENTO}"
            print(f"  ERRO: {motivo}")
            print("-" * 80)
            return montar_resultado(audio_id, sucesso=False, motivo_falha=motivo)
    else:
        print(f"  Usando arquivo filtrado: {ARQUIVO_JSON_FILTRADO}")

    nomes_json = list(dados_json.keys())
    print(f"  Total de segmentos no JSON: {len(nomes_json)}")
    print("-" * 80)

    avisos: List[str] = []

    # --- Lote sem segmento aprovado: nada a gravar, historico nao avanca ---
    # Nao e falha do modulo (o funil pode reprovar tudo), mas o audio fica
    # sem marca de concluido e sera reprocessado na proxima rodada.
    if not nomes_json:
        aviso = "Nenhum segmento aprovado — nada gravado no CSV e historico nao atualizado"
        avisos.append(aviso)
        print(f"  {aviso}")
        print("-" * 80)
        return montar_resultado(audio_id, sucesso=True, n_persistidos=0, avisos=avisos)

    # --- Pasta de audios ausente havendo segmentos e falha real ---
    # (significa que o m13 nao produziu nada)
    if not pasta_audios_do_id(audio_id).is_dir():
        motivo = f"Pasta de audios do dataset nao existe: {pasta_audios_do_id(audio_id)}"
        print(f"  ERRO: {motivo}")
        print("-" * 80)
        return montar_resultado(audio_id, sucesso=False, motivo_falha=motivo)

    # --- Determinar colunas ---
    # Header existente manda: sem reescrita, nao ha como acrescentar coluna
    # as linhas ja gravadas. Campo novo do lote e ignorado, com aviso.
    colunas_json       = obter_todas_colunas(dados_json)
    colunas_existentes = ler_header_csv(ARQUIVO_CSV_DATASET)           # O(1) de RAM

    if colunas_existentes:
        colunas = colunas_existentes
        ignoradas = [c for c in colunas_json if c not in set(colunas_existentes)]
        if ignoradas:
            aviso = (f"{len(ignoradas)} campo(s) do lote fora do header do CSV, "
                     f"nao gravado(s): {', '.join(ignoradas)}")
            avisos.append(aviso)
            print(f"  AVISO: {aviso}")
    else:
        colunas = colunas_json
        print(f"  CSV sera criado com {len(colunas)} colunas")

    # --- Preparar linhas do lote atual ---
    # RAM usada aqui: apenas o lote, nunca o CSV inteiro
    linhas_novas = [
        preparar_linha_csv(nome, dados_json[nome], colunas, audio_id)
        for nome in nomes_json
    ]

    # --- Gravar no CSV (criacao ou append; nunca reescrita) ---
    print("Gravando linhas no CSV...")
    modo = escrever_linhas(
        ARQUIVO_CSV_DATASET, colunas, linhas_novas,
        escrever_header=not colunas_existentes,
    )

    print(f"  Arquivo CSV: {ARQUIVO_CSV_DATASET}")
    print(f"  Modo de escrita: {modo}")
    print(f"  Linhas gravadas nesta rodada: {len(linhas_novas)}")
    print("-" * 80)

    # --- Copiar JSON para historico (por ultimo: marca o audio como concluido) ---
    print("Copiando JSON para historico...")
    copiar_json_historico(ARQUIVO_JSON_ACOMPANHAMENTO, ARQUIVO_JSON_HISTORICO)
    print("-" * 80)

    print("Processamento concluido.")

    return montar_resultado(
        audio_id, sucesso=True,
        n_persistidos=len(linhas_novas),
        avisos=avisos,
    )


# ==============================================================================
# EXECUCAO
# ==============================================================================
if __name__ == "__main__":
    processar_metadados('exemplo_audio_id')
