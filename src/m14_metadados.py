#!/usr/bin/env python3
"""
Module m14_metadados.py
Generates the dataset's final metadata in CSV format
Copies the tracking JSON file to the history

File managed:
  dataset.csv — one row per delivered segment, from all audio_ids

Writing rules (MAXIMUM SAFETY OF dataset.csv):

1. PURE APPEND. The module only CREATES the file, when it does not yet
   exist, and APPENDS the run's rows at the end. It never rewrites the
   whole file, never removes a row, never deletes audio from disk.
   Removing a row from the dataset is the user's exclusive privilege,
   done manually.

2. A REPEATED ROW IS NOT A PROBLEM HERE. Deduplication does not belong
   to dataset.csv: main blocks the already-completed audio file at the
   ENTRY point, via the completed-audio CSV (config.CSV_CONCLUIDOS),
   before any module runs. This module is the one that writes to that
   CSV, as the LAST step - see registrar_concluido.

3. FIXED SCHEMA. The columns are ALWAYS those of SCHEMA_DATASET, in the
   declared order, whatever the run's configuration is. Denoiser off,
   SoX that did not run, a skipped module: the column stays there,
   empty. What varies is the fill, never the set of columns. A JSON
   field outside the schema does not get in, and is warned about by
   name.

4. DIVERGENT HEADER FAILS LOUD. If the file already exists with a
   header different from the schema, the module refuses to write.
   Appending 31 fields under a header of a different size would
   silently corrupt the file.

5. EXPLICIT RETURN. processar_metadados returns a dictionary with
   success, count and failure reason. The caller checks it.
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

from config import CSV_CONCLUIDOS, CSV_NOMEACAO


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
    """Builds the audio's relative path from the audio_id."""
    return f"./audio_dataset/{audio_id}/{nome_arquivo}"


def pasta_audios_do_id(audio_id: str) -> Path:
    """Folder where the delivered .flac files of an audio_id live."""
    return PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id


def formatar_valor(valor: Any, coluna: str) -> Any:
    """
    Converts a JSON value into what goes in the CSV cell.

    A boolean becomes the capitalized string 'True'/'False'. Absence
    (None) follows the COLUNAS_BOOLEANAS rule: 'False' for the three
    booleans, an empty cell for everything else.
    """
    if valor is None:
        return 'False' if coluna in COLUNAS_BOOLEANAS else ''
    if isinstance(valor, bool):
        return str(valor)
    return valor


def agora_iso() -> str:
    """
    Current moment in ISO 8601 with timezone, second precision.

    The timezone comes from the operating system (astimezone with no
    argument), NEVER from a constant in the code: a server in Frankfurt
    has to write '+02:00' on its own. No milliseconds - the rows of the
    same audio file are written at the same instant, and the extra
    precision would not distinguish anything.
    """
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def carregar_json(caminho: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON file. Returns None if the file does not exist."""
    if not caminho.exists():
        return None
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def buscar_nome_original(audio_id: str) -> Optional[str]:
    """
    Looks up this audio_id's source path in the auxiliary naming CSV.

    The file is scanned line by line (O(1) of RAM) and the FIRST
    occurrence wins: the CSV is pure append, so the oldest row is the
    one from the move that actually happened.

    Returns:
        The relative source path, or None when there is no record -
        which happens with an audio file placed by hand in
        arquivos/audios/, without going through m00. The caller turns
        this into a warning, never into silence.
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
    Prepares a row dictionary for the CSV, with ALL SCHEMA_DATASET
    columns filled in:
      - columns calculated by m14, with their values
      - fields present in the JSON, with their values (bool converted)
      - missing fields with the absence value (empty, or 'False' for
        the declared booleans)
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
    Lists the JSON fields that have no column in the schema, in order
    of first appearance. These are the fields that will NOT be written
    - the caller warns about them by name, so the loss is never silent.
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
    Reads ONLY the CSV header (first line).
    RAM usage: O(1) — does not load any data row.
    Returns an empty list when the file does not exist or is empty.
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
    Writes the run's rows at the end of the CSV.

    The file is opened in 'a' mode: it is created if it does not exist
    and is NEVER truncated. No row already written is read, moved or
    removed. The header is only written when the file is being created
    right now, and it is always SCHEMA_DATASET.

    The rows arrive ready from preparar_linha_csv, with exactly the
    schema's keys. DictWriter's extrasaction and restval are defensive
    redundancy, never exercised by this flow.

    Returns:
        Mode used: 'criacao' or 'append'
    """
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    # lineterminator explicito: o padrao do modulo csv e '\r\n', que deixaria
    # este CSV em CRLF enquanto o nomeacao.csv e o concluidos.csv saem em LF.
    with open(caminho_csv, 'a', encoding=CSV_ENCODING, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SCHEMA_DATASET,
            delimiter=CSV_SEPARATOR,
            extrasaction='ignore',
            restval='',
            lineterminator='\n',
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
    Copies the tracking JSON to the history (overwrites if it already
    exists).

    The history is NO LONGER the marker for a completed audio file -
    that role belongs to the completed-audio CSV (see
    registrar_concluido). What it is today:

    1. BACKUP OF THE PROCESSED INFORMATION. If dataset.csv is lost, the
       dataset can be rebuilt from these JSONs WITHOUT rerunning the
       models. That alone justifies its cost.

    2. SOURCE OF THE RUN'S APPROVED DURATION. calcular_duracao_audios_aprovados
       (main.py) reads these JSONs to sum up the `duracao` field of
       each segment, and that is where the
       `duracao_audios_aprovados_segundos` column of
       processamento_metadados.csv comes from.

    ATTENTION: removing deduplication from the history does NOT mean
    stopping the writing of the JSONs. If this copy stops happening,
    item 2's count starts returning zero WITH NO ERROR AT ALL - a
    missing file is ignored there. Do not remove this call.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino)
    print(f"  JSON de historico copiado: {destino}")


# ==============================================================================
# CSV DOS CONCLUIDOS — O MARCADOR DA DEDUPLICACAO
# ==============================================================================

CSV_CONCLUIDOS_HEADER = 'nome_processado|nome_original|datetime_concluido'


def registrar_concluido(audio_id: str, nome_original: str) -> bool:
    """
    Appends ONE row to the completed-audio CSV, in PURE APPEND.

    This is the marker that blocks the audio file at the entry point of
    the next run, and that is why it is the LAST thing m14 does: when
    this row is written, the segments are already in
    dataset/audio_dataset/{id}/, the rows are already in dataset.csv
    and the JSON backup is already in the history. An audio file that
    broke before this point is NOT registered and WILL be reprocessed -
    that is the reason for the order.

    Never reads the file, never rewrites an existing row, never deletes
    anything. The header is only written when the file is born.

    Returns:
        True if the row is saved, False if it failed (with a log
        entry). The caller turns the failure into a warning: the
        dataset.csv rows are already saved and would not be undone, but
        the audio file is not marked and will go back to being
        processed - which needs to show up in the log.
    """
    CSV_CONCLUIDOS.parent.mkdir(parents=True, exist_ok=True)

    escrever_header = not CSV_CONCLUIDOS.exists()

    try:
        with open(CSV_CONCLUIDOS, 'a', encoding=CSV_ENCODING) as f:
            if escrever_header:
                f.write(CSV_CONCLUIDOS_HEADER + '\n')
            f.write(f"{audio_id}{CSV_SEPARATOR}{nome_original}"
                    f"{CSV_SEPARATOR}{agora_iso()}\n")
    except OSError as e:
        print(f"  ERRO ao gravar o CSV dos concluidos '{CSV_CONCLUIDOS}': {e}")
        return False

    return True


# ==============================================================================
# RESULTADO
# ==============================================================================

def montar_resultado(audio_id: str,
                     sucesso: bool,
                     motivo_falha: Optional[str] = None,
                     n_persistidos: int = 0,
                     avisos: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Builds the result returned by processar_metadados.

    Fields:
        sucesso       — False only on a real failure (missing input
                        JSON, missing audio folder while there are
                        segments to deliver). A batch with no approved
                        segment is a success-with-warning.
        motivo_falha  — error text when sucesso is False
        n_persistidos — rows written to the CSV in this run
        avisos        — batch degradation messages
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
    Processes metadata and generates outputs, IN THIS ORDER:
      1. Appends the batch's rows to dataset.csv (creating it if needed)
      2. Copies the tracking JSON to the history (backup)
      3. Registers the audio file in the completed-audio CSV (the
         deduplication marker)

    The order is the mechanism, not a detail: step 3 is the last one
    precisely so that a run that dies before it leaves the audio file
    WITHOUT a completed mark, and the next execution reprocesses it in
    full.

    A batch with no approved segment returns before step 1 and
    therefore never reaches step 3 - the audio file is not marked, on
    purpose.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        Result dictionary (see montar_resultado). The caller MUST check
        the 'sucesso' key.
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

    # --- Lote sem segmento aprovado: nada a gravar, nada marcado ---
    # Nao e falha do modulo (o funil pode reprovar tudo). O retorno acontece
    # ANTES dos tres passos de escrita: nem dataset.csv, nem backup do JSON,
    # nem registro no CSV dos concluidos. O audio fica sem marca e sera
    # reprocessado na proxima rodada.
    if not nomes_json:
        aviso = ("Nenhum segmento aprovado — nada gravado no dataset.csv, "
                 "audio nao registrado como concluido")
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

    # --- Copiar JSON para historico (backup, NAO marcador - ver a funcao) ---
    print("Copiando JSON para historico...")
    copiar_json_historico(ARQUIVO_JSON_ACOMPANHAMENTO, ARQUIVO_JSON_HISTORICO)
    print("-" * 80)

    # --- Registrar o audio como CONCLUIDO (ultimo passo, sempre) ---
    # Aqui, e so aqui, valem as tres coisas ao mesmo tempo: segmentos em
    # dataset/audio_dataset/{id}/, linhas no dataset.csv, backup do JSON no
    # historico. E o unico instante do projeto em que o audio pode ser dado
    # por concluido.
    if registrar_concluido(audio_id, nome_original):
        print(f"  Registrado como concluido em: {CSV_CONCLUIDOS}")
    else:
        aviso = (f"Falha ao registrar '{audio_id}' em '{CSV_CONCLUIDOS}' — as "
                 f"{len(linhas_novas)} linha(s) JA estao no dataset.csv, mas o "
                 f"audio nao ficou marcado e SERA reprocessado na proxima "
                 f"rodada, duplicando essas linhas")
        avisos.append(aviso)
        print(f"  AVISO: {aviso}")
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
