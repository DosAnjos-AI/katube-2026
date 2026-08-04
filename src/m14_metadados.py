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

3. SCHEMA FIXO. As colunas sao SEMPRE as de SCHEMA_DATASET, na ordem
   declarada, qualquer que seja a configuracao da rodada. Denoiser
   desligado, SoX que nao rodou, modulo pulado: a coluna continua la,
   vazia. O que varia e o preenchimento, nunca o conjunto de colunas.
   Campo do JSON fora do schema nao entra, e e avisado nominalmente.

4. HEADER DIVERGENTE FALHA ALTO. Se o arquivo ja existe com header
   diferente do schema, o modulo se recusa a gravar. Fazer append de 31
   campos sob um header de outro tamanho corromperia o arquivo em silencio.

5. RETORNO EXPLICITO. processar_metadados devolve um dicionario com
   sucesso, contagem e motivo de falha. Quem chama verifica.
"""

import sys
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


# ==============================================================================
# CONFIGURACAO DE PATHS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_NOMEACAO


# ==============================================================================
# CONFIGURACAO DE INPUTS/OUTPUTS
# ==============================================================================

CSV_SEPARATOR = '|'
CSV_ENCODING  = 'utf-8'

# ------------------------------------------------------------------------------
# SCHEMA FIXO DO dataset.csv - as mesmas colunas SEMPRE, na ordem declarada
# ------------------------------------------------------------------------------
# As tres primeiras identificam o audio e o segmento, em granularidades
# diferentes e sem ambiguidade:
#   nome_original      - caminho de origem em arquivos/input/ (do AUDIO,
#                        repetido nas n linhas dele)
#   nome_processado    - id do audio: nome desempatado ou hash (do AUDIO)
#   nome_arquivo_audio - {id}_{numeracao}.{formato de saida} (do SEGMENTO)
SCHEMA_DATASET = [
    'nome_original',
    'nome_processado',
    'nome_arquivo_audio',
    'caminho',
    'tempo_inicio',
    'tempo_fim',
    'duracao',
    'vad',
    'origem_codec',
    'origem_bitrate',
    'origem_sample_rate',
    'mos_score',
    'mos_stoi',
    'mos_si_sdr',
    'mos_qualidade',
    'overlap01',
    'stt_whisper',
    'stt_wav2vec',
    'sim_whisper_wav2vec_wer',
    'sim_whisper_wav2vec_cer',
    'sim_whisper_wav2vec_levenshtein_norm',
    'status_similaridade',
    'utilizou_denoiser',
    'sox_sample_rate',
    'sox_bit_depth',
    'sox_channels',
    'sox_output_format',
    'sox_normalize_method',
    'sox_target_level_db',
    'utilizou_sox',
    'datetime_processado',
]

# Colunas calculadas pelo m14, que nunca vem do JSON de acompanhamento
COLUNAS_CALCULADAS = {
    'nome_original',
    'nome_processado',
    'nome_arquivo_audio',
    'caminho',
    'datetime_processado',
}

# ------------------------------------------------------------------------------
# VALOR DE AUSENCIA
# ------------------------------------------------------------------------------
# Regra geral: campo VAZIO. E o padrao de CSV para nulo - pandas, polars e o
# csv do Python leem como nulo sem tratamento. 'NULL', 'None' e 'N/A' virariam
# TEXTO, e uma coluna numerica inteira passaria a ser lida como string.
#
# Excecao, para os booleanos abaixo: ausencia vira False, o que mantem a
# invariante de booleano nunca nulo. Nos tres, False significa "nao usei":
# nao segmentei por VAD, nao passei pelo denoiser, nao passei pelo SoX.
#
# ATENCAO - `overlap01` NAO esta nesta lista, DE PROPOSITO. Nele, False nao
# significa "nao usei": significa "NAO HA SOBREPOSICAO", ou seja, o segmento
# passou no teste. Se o m07 nao rodar (MASTER['overlap'] = False), carimbar
# False seria falsear aprovacao de um teste que nunca aconteceu. Por isso ele
# cai na regra geral e sai vazio. Nao "conserte" esta excecao.
COLUNAS_BOOLEANAS = {'vad', 'utilizou_denoiser', 'utilizou_sox'}

# Campos que o JSON traz e que ficam FORA do CSV POR DECISAO, nao por
# esquecimento. Sao insumo interno do m11 (comparacao das transcricoes), nao
# dado de dataset. Ficam nesta lista para nao poluirem o aviso de campo
# inesperado - o aviso existe para descobrir campo NOVO que ninguem previu.
CAMPOS_EXCLUIDOS = {
    'stt_whisper_normalizado',
    'stt_wav2vec_normalizado',
}


# ==============================================================================
# FUNCOES AUXILIARES — DADOS
# ==============================================================================

def construir_caminho_audio(nome_arquivo: str, audio_id: str) -> str:
    """Constroi caminho relativo do audio a partir do audio_id."""
    return f"./audio_dataset/{audio_id}/{nome_arquivo}"


def pasta_audios_do_id(audio_id: str) -> Path:
    """Pasta onde vivem os .flac entregues de um audio_id."""
    return PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id


def formatar_valor(valor: Any, coluna: str) -> Any:
    """
    Converte um valor do JSON para o que vai na celula do CSV.

    Booleano vira string 'True'/'False' capitalizada. Ausencia (None) segue a
    regra de COLUNAS_BOOLEANAS: 'False' para os tres booleanos, celula vazia
    para todo o resto.
    """
    if valor is None:
        return 'False' if coluna in COLUNAS_BOOLEANAS else ''
    if isinstance(valor, bool):
        return str(valor)
    return valor


def agora_iso() -> str:
    """
    Momento atual em ISO 8601 com fuso, precisao de segundos.

    O fuso vem do sistema operacional (astimezone sem argumento), NUNCA de
    constante no codigo: um servidor em Frankfurt tem que gravar '+02:00'
    sozinho. Sem milissegundos - as linhas de um mesmo audio sao escritas no
    mesmo instante, e a precisao extra nao distinguiria nada.
    """
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def carregar_json(caminho: Path) -> Optional[Dict[str, Any]]:
    """Carrega arquivo JSON. Retorna None se o arquivo nao existir."""
    if not caminho.exists():
        return None
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def buscar_nome_original(audio_id: str) -> Optional[str]:
    """
    Procura no CSV auxiliar da nomeacao o caminho de origem deste audio_id.

    O arquivo e varrido linha a linha (O(1) de RAM) e a PRIMEIRA ocorrencia
    vence: o CSV e append puro, entao a linha mais antiga e a do move que de
    fato aconteceu.

    Returns:
        O caminho relativo de origem, ou None quando nao ha registro - o que
        acontece com audio colocado a mao em arquivos/audios/, sem passar
        pelo m00. O chamador transforma isso em aviso, nunca em silencio.
    """
    if not CSV_NOMEACAO.exists():
        return None

    try:
        with open(CSV_NOMEACAO, 'r', encoding=CSV_ENCODING) as f:
            for linha in f:
                campos = linha.rstrip('\n').split(CSV_SEPARATOR)
                if len(campos) >= 2 and campos[0] == audio_id:
                    return campos[1]
    except OSError as e:
        print(f"  AVISO: falha ao ler o CSV auxiliar '{CSV_NOMEACAO}': {e}")
        return None

    return None


def preparar_linha_csv(nome_arquivo: str,
                       dados_segmento: Dict[str, Any],
                       audio_id: str,
                       nome_original: str,
                       momento: str) -> Dict[str, Any]:
    """
    Prepara um dicionario de linha para o CSV, com TODAS as colunas do
    SCHEMA_DATASET preenchidas:
      - colunas calculadas pelo m14 com seus valores
      - campos presentes no JSON com seus valores (bool convertido)
      - campos ausentes com o valor de ausencia (vazio, ou 'False' nos
        booleanos declarados)
    """
    linha: Dict[str, Any] = {}
    for col in SCHEMA_DATASET:
        if col == 'nome_original':
            linha[col] = nome_original
        elif col == 'nome_processado':
            linha[col] = audio_id
        elif col == 'nome_arquivo_audio':
            linha[col] = nome_arquivo
        elif col == 'caminho':
            linha[col] = construir_caminho_audio(nome_arquivo, audio_id)
        elif col == 'datetime_processado':
            linha[col] = momento
        else:
            linha[col] = formatar_valor(dados_segmento.get(col), col)
    return linha


def campos_fora_do_schema(dados_json: Dict[str, Any]) -> List[str]:
    """
    Lista os campos do JSON que nao tem coluna no schema, na ordem de
    primeira aparicao. Sao os campos que NAO serao gravados - o chamador
    avisa nominalmente, para que a perda nunca seja silenciosa.
    """
    fora: List[str] = []
    conhecidos = set(SCHEMA_DATASET) | CAMPOS_EXCLUIDOS

    for segmento_data in dados_json.values():
        for chave in segmento_data.keys():
            if chave not in conhecidos:
                fora.append(chave)
                conhecidos.add(chave)

    return fora


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
                    linhas_novas: List[Dict[str, Any]],
                    escrever_header: bool) -> str:
    """
    Grava as linhas da rodada no fim do CSV.

    O arquivo e aberto em modo 'a': ele e criado se nao existir e JAMAIS e
    truncado. Nenhuma linha ja gravada e lida, movida ou removida. O header
    so e escrito quando o arquivo esta sendo criado agora, e e sempre o
    SCHEMA_DATASET.

    As linhas chegam prontas de preparar_linha_csv, com exatamente as chaves
    do schema. O extrasaction e o restval do DictWriter sao redundancia
    defensiva, nunca exercitada por este fluxo.

    Returns:
        Modo usado: 'criacao' ou 'append'
    """
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_csv, 'a', encoding=CSV_ENCODING, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SCHEMA_DATASET,
            delimiter=CSV_SEPARATOR,
            extrasaction='ignore',
            restval='',
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

    # --- Guarda do schema: header divergente FALHA ALTO ---
    # As colunas sao sempre as do SCHEMA_DATASET. Se o arquivo ja existe com
    # outro header, fazer append por cima desalinharia todas as linhas novas
    # em relacao ao cabecalho - corrupcao silenciosa.
    colunas_existentes = ler_header_csv(ARQUIVO_CSV_DATASET)           # O(1) de RAM

    if colunas_existentes and colunas_existentes != SCHEMA_DATASET:
        motivo = (
            f"Header do CSV diverge do schema fixo. "
            f"Arquivo: {len(colunas_existentes)} coluna(s); "
            f"schema: {len(SCHEMA_DATASET)} coluna(s). "
            f"Nada foi gravado. Arquive '{ARQUIVO_CSV_DATASET}' antes de rodar "
            f"de novo - o append puro nao migra header."
        )
        print(f"  ERRO: {motivo}")
        print("-" * 80)
        return montar_resultado(audio_id, sucesso=False, motivo_falha=motivo)

    # --- Campo do JSON sem coluna no schema: descartado COM aviso nominal ---
    ignoradas = campos_fora_do_schema(dados_json)
    if ignoradas:
        aviso = (f"{len(ignoradas)} campo(s) do lote fora do schema do CSV, "
                 f"nao gravado(s): {', '.join(ignoradas)}")
        avisos.append(aviso)
        print(f"  AVISO: {aviso}")

    # --- Procedencia do audio (coluna nome_original) ---
    nome_original = buscar_nome_original(audio_id)
    if nome_original is None:
        aviso = (f"Sem registro em '{CSV_NOMEACAO}' para o id '{audio_id}' — "
                 f"coluna 'nome_original' gravada vazia nas "
                 f"{len(nomes_json)} linha(s)")
        avisos.append(aviso)
        print(f"  AVISO: {aviso}")
        nome_original = ''

    # --- Preparar linhas do lote atual ---
    # RAM usada aqui: apenas o lote, nunca o CSV inteiro.
    # O momento e calculado UMA vez: as linhas de um mesmo audio sao mesmo
    # escritas no mesmo instante, e um relogio por linha so daria a ilusao de
    # precisao.
    momento = agora_iso()
    linhas_novas = [
        preparar_linha_csv(nome, dados_json[nome], audio_id, nome_original, momento)
        for nome in nomes_json
    ]

    # --- Gravar no CSV (criacao ou append; nunca reescrita) ---
    print("Gravando linhas no CSV...")
    modo = escrever_linhas(
        ARQUIVO_CSV_DATASET, linhas_novas,
        escrever_header=not colunas_existentes,
    )

    print(f"  Arquivo CSV: {ARQUIVO_CSV_DATASET}")
    print(f"  Modo de escrita: {modo} ({len(SCHEMA_DATASET)} colunas do schema fixo)")
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
