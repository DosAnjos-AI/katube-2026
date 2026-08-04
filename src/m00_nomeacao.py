#!/usr/bin/env python3
"""
Modulo m00_nomeacao.py - Porta de entrada da pipeline

Roda UMA vez por execucao do main.py, antes de listar os audios. Varre
`arquivos/input/` recursivamente, resolve um id para cada audio encontrado e
MOVE o arquivo para `arquivos/audios/{id}/{id}.<formato original>`, que e a
estrutura que o resto da pipeline consome.

Nao converte formato: o arquivo chega ao destino no formato original. A
conversao para WAV e do m02.

ATENCAO - A PASTA DE INPUT E ESVAZIADA: o arquivo e MOVIDO, nao copiado.
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# ==============================================================================
# CONFIGURACAO DE CAMINHOS
# ==============================================================================

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import NOMEACAO

# Cabecalho do CSV auxiliar do modo hash
CSV_HASH_HEADER = 'hash|nome_desempatado|caminho_relativo_origem'


# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def _nome_valido(nome: str) -> bool:
    """
    Diz se um nome de arquivo serve como id (nome de pasta e nome de arquivo).

    Recusa o que quebra o caminho ('', '.', '..', separadores) e o que
    corromperia uma linha do dataset.csv, que usa '|' como separador.
    Nome invalido e RECUSADO com log, nunca "consertado" em silencio.
    """
    if not nome or nome in ('.', '..'):
        return False
    return not any(c in nome for c in ('/', '\\', '|'))


def _mover_seguro(origem: Path, destino: Path) -> bool:
    """
    Move origem para destino sem jamais deixar arquivo incompleto com o nome
    final, e sem jamais apagar a origem antes de o destino estar pronto.

    O nome final so aparece depois de o conteudo estar completo - e o que
    permite a guarda 1 confiar num simples exists(). O rename dentro do mesmo
    sistema de arquivos e atomico; entre sistemas diferentes ele levanta
    OSError (EXDEV) e o caminho de copia assume.

    Returns:
        True se o arquivo esta no destino, False se falhou (com log).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + '.parcial')

    try:
        try:
            origem.rename(parcial)
        except OSError:
            # Sistemas de arquivos diferentes: copia, efetiva o nome final e
            # so entao apaga a origem
            shutil.copy2(origem, parcial)
            os.replace(parcial, destino)
            origem.unlink()
            return True

        os.replace(parcial, destino)
        return True

    except OSError as e:
        print(f"[M00] ERRO ao mover '{origem}' -> '{destino}': {e}")

        if parcial.exists():
            if origem.exists():
                # A origem sobreviveu: o parcial e lixo e pode sair
                try:
                    parcial.unlink()
                except OSError as e_limpeza:
                    print(f"[M00] ERRO ao remover parcial '{parcial}': {e_limpeza}")
            else:
                # A origem ja saiu do lugar: o parcial e a UNICA copia do audio
                print(f"[M00] ATENCAO: a origem nao existe mais e o conteudo "
                      f"esta em '{parcial}'. NAO apague esse arquivo - mova-o "
                      f"para '{destino}' na mao antes de rodar de novo.")

        return False


def _escrever_csv_hash(linhas: List[Tuple[str, str, str]]) -> bool:
    """
    Grava a relacao hash <-> nome no CSV auxiliar, com escrita atomica.

    O arquivo acumula entre execucoes: le o que ja existe, funde as linhas
    novas por hash e regrava inteiro em .tmp + os.replace (mesmo padrao do
    m13). O arquivo nunca existe pela metade.

    Linha malformada no arquivo existente NAO e descartada nem reescrita: a
    funcao recusa mexer no arquivo e devolve False, para que o Mestre decida.

    Returns:
        True se o CSV esta gravado, False se falhou (com log).
    """
    caminho = PROJECT_ROOT / "dataset" / "nomeacao_hash.csv"
    caminho.parent.mkdir(parents=True, exist_ok=True)

    registros = {}

    if caminho.exists():
        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                for numero, linha in enumerate(f, 1):
                    linha = linha.rstrip('\n')
                    if not linha:
                        continue
                    if numero == 1 and linha == CSV_HASH_HEADER:
                        continue
                    campos = linha.split('|')
                    if len(campos) != 3:
                        print(f"[M00] ERRO: linha {numero} de '{caminho}' esta "
                              f"malformada ({len(campos)} campos, esperado 3): "
                              f"{linha!r}")
                        print("[M00] O CSV auxiliar NAO foi alterado. "
                              "Corrija a linha antes de rodar de novo.")
                        return False
                    registros[campos[0]] = (campos[0], campos[1], campos[2])
        except OSError as e:
            print(f"[M00] ERRO ao ler o CSV auxiliar '{caminho}': {e}")
            return False

    for registro in linhas:
        registros[registro[0]] = registro

    temporario = caminho.with_name(caminho.name + '.tmp')
    try:
        with open(temporario, 'w', encoding='utf-8') as f:
            f.write(CSV_HASH_HEADER + '\n')
            for chave in sorted(registros):
                f.write('|'.join(registros[chave]) + '\n')
        os.replace(temporario, caminho)
    except OSError as e:
        print(f"[M00] ERRO ao gravar o CSV auxiliar '{caminho}': {e}")
        return False

    print(f"[M00] CSV auxiliar: {len(registros)} registro(s) em '{caminho}'")
    return True


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def preparar_entrada(ja_processado: Callable[[str], bool]) -> bool:
    """
    Nomeia e move o material de `arquivos/input/` para `arquivos/audios/`.

    Args:
        ja_processado: predicado que diz se um id ja consta do historico. Vem
            injetado pelo main.py para que a regra do historico continue
            morando num lugar so.

    Returns:
        True se toda a varredura terminou sem perder arquivo, False se algum
        move falhou ou se o CSV auxiliar nao pode ser gravado. Lote vazio nao
        e erro.
    """
    print("\n" + "="*80)
    print("[M00] NOMEACAO - PORTA DE ENTRADA")
    print("="*80)

    modo = NOMEACAO['modo']
    if modo not in ('nome_original', 'hash_md5'):
        print(f"[M00] ERRO: NOMEACAO['modo'] invalido: {modo!r}. "
              "Valores aceitos: 'nome_original', 'hash_md5'")
        return False

    formatos = {ext.lower() for ext in NOMEACAO['formatos_entrada']}

    pasta_input = PROJECT_ROOT / "arquivos" / "input"
    pasta_input.mkdir(parents=True, exist_ok=True)

    pasta_audios = PROJECT_ROOT / "arquivos" / "audios"
    pasta_audios.mkdir(parents=True, exist_ok=True)

    print(f"[M00] Modo: {modo}")
    print(f"[M00] Formatos aceitos: {sorted(formatos)}")
    print(f"[M00] Varrendo (recursivo): {pasta_input}")

    # Ordem alfabetica do caminho relativo completo: regra fixa que garante
    # que a mesma pasta de entrada produza sempre os mesmos ids
    candidatos = sorted(
        (p for p in pasta_input.rglob('*') if p.is_file()),
        key=lambda p: p.relative_to(pasta_input).as_posix()
    )

    encontrados = 0
    recusados_formato = 0
    recusados_nome = 0
    pulados_guarda1 = 0
    pulados_guarda2 = 0
    movidos = 0
    erros = 0

    # Nome base -> quantas vezes ja apareceu, para o desempate _002, _003...
    ocorrencias = {}
    linhas_csv = []

    for origem in candidatos:
        relativo = origem.relative_to(pasta_input).as_posix()
        encontrados += 1

        extensao = origem.suffix.lower()
        if extensao not in formatos:
            print(f"[M00] IGNORADO (formato fora da lista): {relativo}")
            recusados_formato += 1
            continue

        nome_base = origem.stem
        if not _nome_valido(nome_base) or '|' in relativo:
            print(f"[M00] RECUSADO (nome nao serve como id): {relativo}")
            recusados_nome += 1
            continue

        # O contador avanca ANTES das guardas: pular um audio ja movido nao
        # pode renumerar os seguintes, senao a segunda execucao produz outros ids
        numero = ocorrencias.get(nome_base, 0) + 1
        ocorrencias[nome_base] = numero
        nome_desempatado = nome_base if numero == 1 else f"{nome_base}_{numero:03d}"

        if modo == 'hash_md5':
            audio_id = hashlib.md5(nome_desempatado.encode('utf-8')).hexdigest()
        else:
            audio_id = nome_desempatado

        pasta_destino = pasta_audios / audio_id
        destino = pasta_destino / f"{audio_id}{extensao}"

        # Guarda 1 - retomada apos quebra: qualquer formato aceito ja presente
        # na pasta do id significa que este audio ja foi movido
        ja_no_destino = sorted(
            ext for ext in formatos if (pasta_destino / f"{audio_id}{ext}").exists()
        )
        if ja_no_destino:
            print(f"[M00] PULADO (guarda 1 - '{audio_id}{ja_no_destino[0]}' "
                  f"ja esta em arquivos/audios/{audio_id}/): {relativo}")
            pulados_guarda1 += 1
            continue

        # Guarda 2 - audio ja processado antes: nao entra de novo
        if ja_processado(audio_id):
            print(f"[M00] PULADO (guarda 2 - id '{audio_id}' ja consta do "
                  f"historico): {relativo}")
            pulados_guarda2 += 1
            continue

        if not _mover_seguro(origem, destino):
            erros += 1
            continue

        movidos += 1
        print(f"[M00] MOVIDO: {relativo} -> arquivos/audios/{audio_id}/{destino.name}")

        if modo == 'hash_md5':
            linhas_csv.append((audio_id, nome_desempatado, relativo))

    csv_ok = True
    if modo == 'hash_md5':
        csv_ok = _escrever_csv_hash(linhas_csv)

    print("-"*80)
    print(f"[M00] Arquivos encontrados          : {encontrados}")
    print(f"[M00] Movidos para arquivos/audios/ : {movidos}")
    print(f"[M00] Ignorados (formato fora)      : {recusados_formato}")
    print(f"[M00] Recusados (nome invalido)     : {recusados_nome}")
    print(f"[M00] Pulados (guarda 1 - ja movido): {pulados_guarda1}")
    print(f"[M00] Pulados (guarda 2 - historico): {pulados_guarda2}")
    print(f"[M00] Erros ao mover                : {erros}")
    print("="*80)

    return erros == 0 and csv_ok
