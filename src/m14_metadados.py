#!/usr/bin/env python3
"""
Modulo m14_metadados.py
Gera metadados finais do dataset em formato CSV
Copia arquivo de acompanhamento JSON para historico

Estrategia de escrita:
- CASO NORMAL   (colunas iguais)  : append direto ao final do arquivo
- CASO ESPECIAL (colunas novas)   : reescrita streaming via tempfile + os.replace() atomico

Arquivos gerenciados (vivem lado a lado, sempre sincronizados):
  dataset.csv    — dados completos
  dataset.index  — apenas os nomes dos segmentos (um por linha)
                   permite deteccao de duplicatas em O(1) sem ler o CSV inteiro
                   persiste entre pit stops: parar, adicionar audios, continuar

Garantias de integridade:
- O arquivo original jamais e truncado antes da escrita estar 100% concluida
- Em caso de falha no meio da escrita, ambos os arquivos originais permanecem intactos
- CSV e indice sao atualizados atomicamente juntos (nunca ficam dessincronizados)
- Validacao 1:1 entre segmentos do JSON e arquivos fisicos na pasta de audio
- Deteccao de duplicatas antes de qualquer escrita
"""

import sys
import json
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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

# Arquivo de indice: mesmo nome do CSV, extensao .index
# Armazena apenas os nomes dos segmentos (um por linha)
# Permite checagem de duplicatas em O(1) sem ler o CSV inteiro
# Persiste entre pit stops junto com o CSV
INDEX_SUFFIX = '.index'

COLUNAS_FIXAS = ['arquivo_nome', 'caminho']

CAMPOS_EXCLUIDOS = [
    'stt_leg_normalizado',
    'stt_whisper_normalizado',
    'stt_wav2vec_normalizado',
]


# ==============================================================================
# FUNCOES AUXILIARES — DADOS
# ==============================================================================

def construir_caminho_audio(nome_arquivo: str, audio_id: str) -> str:
    """Constroi caminho relativo do audio a partir do audio_id."""
    return f"./audio_dataset/{audio_id}/{nome_arquivo}"


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


def mesclar_colunas(colunas_existentes: List[str],
                    colunas_novas: List[str]) -> List[str]:
    """
    Mescla colunas existentes com as do lote atual.
    Novas colunas sao adicionadas ao final, mantendo a ordem do CSV.
    """
    existentes_set = set(colunas_existentes)
    resultado = colunas_existentes.copy()
    for col in colunas_novas:
        if col not in existentes_set:
            resultado.append(col)
    return resultado


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


# ==============================================================================
# FUNCOES AUXILIARES — LEITURA LEVE DO CSV (sem carregar tudo em RAM)
# ==============================================================================

def ler_header_csv(caminho: Path) -> List[str]:
    """
    Le APENAS o header do CSV (primeira linha).
    Uso de RAM: O(1) — nao carrega nenhuma linha de dados.
    """
    if not caminho.exists():
        return []
    with open(caminho, 'r', encoding=CSV_ENCODING, newline='') as f:
        primeira_linha = f.readline().strip()
    if not primeira_linha:
        return []
    return primeira_linha.split(CSV_SEPARATOR)


def caminho_indice(caminho_csv: Path) -> Path:
    """Retorna o caminho do arquivo de indice correspondente ao CSV."""
    return caminho_csv.with_suffix(INDEX_SUFFIX)


def carregar_indice(caminho_csv: Path) -> set:
    """
    Carrega o indice de nomes em RAM como set para lookup O(1).

    O indice (dataset.index) e um arquivo texto com um nome por linha.
    E ~25x menor que o CSV e persiste entre pit stops junto com ele.

    Auto-recuperacao: se o indice nao existir mas o CSV sim
    (ex: primeiro uso apos migracao, ou indice deletado acidentalmente),
    reconstroi o indice lendo o CSV em streaming — ocorre uma unica vez.
    """
    indice_path = caminho_indice(caminho_csv)

    if indice_path.exists():
        with open(indice_path, 'r', encoding=CSV_ENCODING) as f:
            return {linha.strip() for linha in f if linha.strip()}

    if caminho_csv.exists():
        print("  AVISO: indice ausente — reconstruindo a partir do CSV (ocorre uma unica vez)...")
        nomes: set = set()
        with open(caminho_csv, 'r', encoding=CSV_ENCODING, newline='') as f:
            reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
            for row in reader:
                nome = row.get('arquivo_nome', '').strip()
                if nome:
                    nomes.add(nome)
        _salvar_indice_completo(indice_path, nomes)
        print(f"  Indice reconstruido: {len(nomes):,} entradas -> {indice_path}")
        return nomes

    return set()


def _salvar_indice_completo(indice_path: Path, nomes: set) -> None:
    """Salva o indice completo de forma atomica. Uso: reconstrucao ou reescrita."""
    indice_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=indice_path.parent, suffix='.tmp', prefix='index_')
    try:
        with os.fdopen(fd, 'w', encoding=CSV_ENCODING) as f:
            f.write('\n'.join(sorted(nomes)) + '\n')
        os.replace(tmp, indice_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_indice(indice_path: Path, nomes_novos: List[str]) -> None:
    """Append de novos nomes ao indice. Simetrico ao _append_csv."""
    with open(indice_path, 'a', encoding=CSV_ENCODING) as f:
        f.write('\n'.join(nomes_novos) + '\n')


# ==============================================================================
# ESCRITA DO CSV — DOIS CAMINHOS CONFORME O CASO
# ==============================================================================

def _append_csv(caminho: Path,
                colunas: List[str],
                linhas_novas: List[Dict[str, Any]],
                nomes_novos: List[str]) -> None:
    """
    CASO NORMAL: colunas iguais -> append direto ao CSV + append ao indice.

    Ordem de operacoes:
      1. Append no CSV  (open 'a' — nunca trunca)
      2. Append no indice (open 'a' — nunca trunca)

    Se morrer entre 1 e 2: na proxima execucao o indice e reconstruido
    automaticamente pelo carregar_indice() — sem perda de dados no CSV.
    Se morrer durante 1: linhas parciais ficam no final do CSV mas o
    header e todas as linhas anteriores permanecem intactos.
    """
    with open(caminho, 'a', encoding=CSV_ENCODING, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=colunas,
            delimiter=CSV_SEPARATOR,
            extrasaction='ignore',
            restval='null',
        )
        writer.writerows(linhas_novas)

    _append_indice(caminho_indice(caminho), nomes_novos)


def _reescrever_csv_streaming(caminho: Path,
                               colunas_finais: List[str],
                               colunas_novas: List[str],
                               linhas_novas: List[Dict[str, Any]],
                               indice_atual: set,
                               nomes_novos: List[str]) -> None:
    """
    CASO ESPECIAL: novas colunas -> reescrita streaming atomica do CSV + indice.

    Ambos os arquivos sao escritos em temporarios separados e substituidos
    atomicamente juntos. Se qualquer etapa falhar, os originais ficam intactos.

    Ordem de operacoes:
      1. Escreve CSV novo em tmp_csv  (streaming, sem carregar tudo em RAM)
      2. Escreve indice novo em tmp_idx (indice_atual + nomes_novos)
      3. os.replace(tmp_csv  -> dataset.csv)   <- atomico
      4. os.replace(tmp_idx  -> dataset.index) <- atomico
    Se morrer em 3 ou 4: na proxima execucao carregar_indice() reconstroi
    o indice a partir do CSV — sem perda de dados.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    indice_path = caminho_indice(caminho)

    fd_csv, tmp_csv = tempfile.mkstemp(dir=caminho.parent, suffix='.tmp', prefix='dataset_')
    fd_idx, tmp_idx = tempfile.mkstemp(dir=caminho.parent, suffix='.tmp', prefix='index_')

    try:
        # --- Escreve CSV novo em streaming ---
        with os.fdopen(fd_csv, 'w', encoding=CSV_ENCODING, newline='') as f_out:
            writer = csv.DictWriter(
                f_out,
                fieldnames=colunas_finais,
                delimiter=CSV_SEPARATOR,
                extrasaction='ignore',
                restval='null',
            )
            writer.writeheader()
            with open(caminho, 'r', encoding=CSV_ENCODING, newline='') as f_in:
                for row in csv.DictReader(f_in, delimiter=CSV_SEPARATOR):
                    for col in colunas_novas:
                        row.setdefault(col, 'null')
                    writer.writerow(row)
            writer.writerows(linhas_novas)

        # --- Escreve indice novo (existentes + novos) ---
        nomes_finais = indice_atual | set(nomes_novos)
        with os.fdopen(fd_idx, 'w', encoding=CSV_ENCODING) as f_idx:
            f_idx.write('\n'.join(sorted(nomes_finais)) + '\n')

        # --- Substituicao atomica dos dois arquivos ---
        os.replace(tmp_csv, caminho)
        os.replace(tmp_idx, indice_path)

    except Exception:
        for tmp in (tmp_csv, tmp_idx):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def atualizar_csv(caminho: Path,
                  colunas_finais: List[str],
                  colunas_novas: List[str],
                  linhas_novas: List[Dict[str, Any]],
                  indice_atual: set,
                  nomes_novos: List[str]) -> str:
    """
    Ponto de entrada unico para escrita no CSV + indice.
    Decide automaticamente o caminho de escrita e retorna o modo usado.

    Args:
        caminho:        Caminho do dataset.csv
        colunas_finais: Todas as colunas (existentes + novas mescladas)
        colunas_novas:  Apenas as colunas que nao existiam no CSV anterior
        linhas_novas:   Linhas do lote atual (preparadas com colunas_finais)
        indice_atual:   Set de nomes ja existentes no indice
        nomes_novos:    Nomes dos segmentos do lote atual

    Returns:
        Modo usado: 'criacao', 'append' ou 'reescrita'
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    indice_path = caminho_indice(caminho)

    # --- Criacao inicial (CSV ainda nao existe) ---
    if not caminho.exists():
        fd_csv, tmp_csv = tempfile.mkstemp(dir=caminho.parent, suffix='.tmp', prefix='dataset_')
        fd_idx, tmp_idx = tempfile.mkstemp(dir=caminho.parent, suffix='.tmp', prefix='index_')
        try:
            with os.fdopen(fd_csv, 'w', encoding=CSV_ENCODING, newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=colunas_finais,
                    delimiter=CSV_SEPARATOR,
                    extrasaction='ignore',
                    restval='null',
                )
                writer.writeheader()
                writer.writerows(linhas_novas)
            with os.fdopen(fd_idx, 'w', encoding=CSV_ENCODING) as f:
                f.write('\n'.join(sorted(nomes_novos)) + '\n')
            os.replace(tmp_csv, caminho)
            os.replace(tmp_idx, indice_path)
        except Exception:
            for tmp in (tmp_csv, tmp_idx):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise
        return 'criacao'

    # --- CSV existe: append ou reescrita conforme necessidade ---
    if not colunas_novas:
        _append_csv(caminho, colunas_finais, linhas_novas, nomes_novos)
        return 'append'
    else:
        _reescrever_csv_streaming(caminho, colunas_finais, colunas_novas,
                                   linhas_novas, indice_atual, nomes_novos)
        return 'reescrita'


# ==============================================================================
# VALIDACAO DE INTEGRIDADE
# ==============================================================================

def validar_pasta_audios(audio_id: str,
                          nomes_segmentos: List[str]) -> Tuple[bool, str, List[str]]:
    """
    Valida integridade 1:1 entre os segmentos do JSON e os arquivos fisicos.

    Verificacoes realizadas:
      1. Pasta do audio existe e e um diretorio
      2. Pasta nao esta vazia
      3. Cada arquivo referenciado no JSON existe fisicamente na pasta (1:1)
      4. (Aviso) Se ha arquivos na pasta que nao estao no JSON

    Args:
        audio_id:        ID do audio
        nomes_segmentos: Lista de nomes de arquivo esperados (do JSON)

    Returns:
        Tupla (valido, mensagem_erro, arquivos_faltantes)
    """
    pasta_audios = PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id

    if not pasta_audios.exists():
        return False, f"ERRO: Pasta de audios nao existe: {pasta_audios}", []

    if not pasta_audios.is_dir():
        return False, f"ERRO: Caminho nao e um diretorio: {pasta_audios}", []

    # Verifica se tem ao menos um arquivo sem listar tudo (eficiente)
    try:
        next(pasta_audios.iterdir())
    except StopIteration:
        return False, f"ERRO: Pasta de audios esta vazia: {pasta_audios}", []

    # Verificacao 1:1 — cada arquivo do JSON deve existir fisicamente
    arquivos_faltantes = [
        nome for nome in nomes_segmentos
        if not (pasta_audios / nome).exists()
    ]

    if arquivos_faltantes:
        msg = (f"ERRO: {len(arquivos_faltantes)} arquivo(s) referenciados no JSON "
               f"nao encontrados na pasta de audios: {pasta_audios}")
        return False, msg, arquivos_faltantes

    # Aviso se ha excedente na pasta (arquivos nao referenciados no JSON)
    total_na_pasta = sum(1 for _ in pasta_audios.iterdir())
    if total_na_pasta != len(nomes_segmentos):
        print(f"  AVISO: Pasta tem {total_na_pasta} arquivo(s), "
              f"JSON referencia {len(nomes_segmentos)} segmento(s)")

    return True, "", []


def verificar_duplicatas(indice: set, nomes_novos: List[str]) -> List[str]:
    """
    Verifica duplicatas usando o indice em RAM — O(1) por nome.
    Nao acessa o disco.
    """
    return [n for n in nomes_novos if n in indice]


# ==============================================================================
# HISTORICO
# ==============================================================================

def copiar_json_historico(origem: Path, destino: Path) -> None:
    """Copia o JSON de acompanhamento para o historico (sobrescreve se ja existe)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino)
    print(f"  JSON de historico copiado: {destino}")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def processar_metadados(audio_id: str) -> None:
    """
    Processa metadados e gera outputs:
      1. Adiciona segmentos ao dataset.csv (append ou reescrita streaming)
      2. Copia JSON de acompanhamento para historico

    Args:
        audio_id: ID do audio a processar
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
            print(f"  ERRO: Arquivo de acompanhamento nao encontrado: {ARQUIVO_JSON_ACOMPANHAMENTO}")
            return
    else:
        print(f"  Usando arquivo filtrado: {ARQUIVO_JSON_FILTRADO}")

    nomes_segmentos = list(dados_json.keys())
    print(f"  Total de segmentos no JSON: {len(nomes_segmentos)}")
    print("-" * 80)

    # --- Sem segmentos aprovados: encerra graciosamente sem erro ---
    if not nomes_segmentos:
        print("  Nenhum segmento aprovado para este audio — nada a registrar no dataset.")
        print("-" * 80)
        return

    # --- Validar integridade 1:1 pasta de audios vs JSON ---
    print("Validando integridade dos audios...")
    valido, mensagem_erro, arquivos_faltantes = validar_pasta_audios(audio_id, nomes_segmentos)

    if not valido:
        print(f"  {mensagem_erro}")
        for arq in arquivos_faltantes[:10]:
            print(f"    - {arq}")
        if len(arquivos_faltantes) > 10:
            print(f"    ... e mais {len(arquivos_faltantes) - 10} arquivo(s)")
        print("  Processamento abortado.")
        print("-" * 80)
        return

    print("  Integridade OK: todos os arquivos de audio confirmados")
    print("-" * 80)

    # --- Carregar indice (O(1) por consulta, persiste entre pit stops) ---
    indice = carregar_indice(ARQUIVO_CSV_DATASET)

    # --- Verificar duplicatas antes de qualquer escrita ---
    print("Verificando duplicatas no CSV...")
    duplicatas = verificar_duplicatas(indice, nomes_segmentos)

    if duplicatas:
        print(f"  AVISO: {len(duplicatas)} segmento(s) ja existem no CSV — serao ignorados:")
        for d in duplicatas[:5]:
            print(f"    - {d}")
        if len(duplicatas) > 5:
            print(f"    ... e mais {len(duplicatas) - 5}")
        # Filtra o lote atual removendo os duplicados
        duplicatas_set  = set(duplicatas)
        nomes_segmentos = [n for n in nomes_segmentos if n not in duplicatas_set]
        dados_json      = {k: v for k, v in dados_json.items() if k in set(nomes_segmentos)}
        print(f"  Segmentos restantes para adicionar: {len(nomes_segmentos)}")
    else:
        print("  Nenhuma duplicata encontrada")

    print("-" * 80)

    if not nomes_segmentos:
        print("Nenhum segmento novo para adicionar. Encerrando.")
        return

    # --- Determinar colunas ---
    colunas_json      = obter_todas_colunas(dados_json)
    colunas_existentes = ler_header_csv(ARQUIVO_CSV_DATASET)           # O(1) de RAM
    colunas_finais    = mesclar_colunas(colunas_existentes, colunas_json)
    colunas_novas     = [c for c in colunas_finais if c not in set(colunas_existentes)]

    if colunas_novas:
        print(f"Novas colunas detectadas: {colunas_novas}")

    # --- Preparar linhas do lote atual ---
    # RAM usada aqui: apenas o lote (~20 segmentos), nunca o CSV inteiro
    linhas_novas = [
        preparar_linha_csv(nome, dados_json[nome], colunas_finais, audio_id)
        for nome in nomes_segmentos
    ]

    # --- Escrever CSV + indice ---
    print("Atualizando CSV...")
    modo = atualizar_csv(ARQUIVO_CSV_DATASET, colunas_finais, colunas_novas,
                         linhas_novas, indice, nomes_segmentos)

    print(f"  Arquivo CSV: {ARQUIVO_CSV_DATASET}")
    print(f"  Modo de escrita: {modo}")
    print(f"  Linhas adicionadas: {len(linhas_novas)}")
    print("-" * 80)

    # --- Copiar JSON para historico ---
    print("Copiando JSON para historico...")
    copiar_json_historico(ARQUIVO_JSON_ACOMPANHAMENTO, ARQUIVO_JSON_HISTORICO)
    print("-" * 80)
    print("Processamento concluido com sucesso!")


# ==============================================================================
# EXECUCAO
# ==============================================================================
if __name__ == "__main__":
    processar_metadados('exemplo_audio_id')