#!/usr/bin/env python3
"""
Module m00_nomeacao.py - Pipeline entry point

Runs ONCE per execution of main.py, before listing the audio files. Scans
`arquivos/input/` recursively, resolves an id for each audio file found and
MOVES the file to `arquivos/audios/{id}/{id}.<original format>`, which is
the structure the rest of the pipeline consumes.

Does not convert format: the file arrives at the destination in its
original format. Conversion to WAV belongs to m02.

The id comes from NOMEACAO['modo']: in 'hash_md5' mode it is the MD5 of the
file's BYTES (see _md5_do_conteudo); in 'nome_original' mode it is the name
without the extension, tie-broken with _002/_003 when it repeats in the
batch. The tie-break exists ONLY in 'nome_original' mode - see the id block
in preparar_entrada.

Two guards block a repeated audio file, both with a named log entry and a
count: guard 1, the audio file is already in arquivos/audios/{id}/ (resume
after a break); guard 2, the id is already in the completed-audio CSV
(config.CSV_CONCLUIDOS), checked via the predicate that main injects.

Also writes, in BOTH naming modes, the auxiliary provenance CSV
(config.CSV_NOMEACAO): this is what m14 uses to fill the `nome_original`
column of dataset.csv. The write is PURE APPEND, one row per moved audio
file, done right after the move - see _registrar_no_csv.

ATTENTION - THE INPUT FOLDER IS EMPTIED: the file is MOVED, not copied.
"""

import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Add root folder to the path to import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CSV_NOMEACAO, NOMEACAO

# Header of the auxiliary CSV, written in BOTH naming modes
CSV_NOMEACAO_HEADER = 'nome_processado|nome_original|datetime_movido'


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _nome_valido(nome: str) -> bool:
    """
    Says whether a file name works as an id (folder name and file name).

    Rejects anything that breaks the path ('', '.', '..', separators) and
    anything that would corrupt a dataset.csv row, which uses '|' as the
    separator. An invalid name is REJECTED with a log entry, never
    silently "fixed".
    """
    if not nome or nome in ('.', '..'):
        return False
    return not any(c in nome for c in ('/', '\\', '|'))


def _mover_seguro(origem: Path, destino: Path) -> bool:
    """
    Moves origem to destino without ever leaving an incomplete file under
    the final name, and without ever deleting origem before destino is
    ready.

    The final name only appears once the content is complete - that is
    what lets guard 1 rely on a simple exists(). The rename within the
    same file system is atomic; across different file systems it raises
    OSError (EXDEV) and the copy path takes over.

    Returns:
        True if the file is at destino, False if it failed (with a log
        entry).
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + '.parcial')

    try:
        try:
            origem.rename(parcial)
        except OSError:
            # Different file systems: copy, commit the final name and
            # only then delete the source
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
                # The source survived: the partial file is garbage and can go
                try:
                    parcial.unlink()
                except OSError as e_limpeza:
                    print(f"[M00] ERRO ao remover parcial '{parcial}': {e_limpeza}")
            else:
                # The source has already left its place: the partial file is
                # the ONLY copy of the audio
                print(f"[M00] ATENCAO: a origem nao existe mais e o conteudo "
                      f"esta em '{parcial}'. NAO apague esse arquivo - mova-o "
                      f"para '{destino}' na mao antes de rodar de novo.")

        return False


def _md5_do_conteudo(caminho: Path) -> str:
    """
    MD5 of the file's BYTES, read in 1 MB blocks.

    This is the id for 'hash_md5' mode. The hash is of the CONTENT, never
    the name: that is what makes two different audio files with the same
    name both get in, and makes the same file pasted into another folder
    be recognized as a repeat.

    Reading in blocks keeps RAM usage at O(1) - a 1 GB audio file takes up
    1 MB of memory here, not 1 GB.

    Returns:
        The 32 hexadecimal characters of the hash, or '' if the file
        cannot be read (with a log entry). An empty string NEVER becomes
        an id: the caller treats it as an error and does not move the
        file.
    """
    resumo = hashlib.md5()

    try:
        with open(caminho, 'rb') as f:
            for bloco in iter(lambda: f.read(1024 * 1024), b''):
                resumo.update(bloco)
    except OSError as e:
        print(f"[M00] ERRO ao ler '{caminho}' para calcular o hash: {e}")
        return ''

    return resumo.hexdigest()


def _agora_iso() -> str:
    """
    Current moment in ISO 8601 with timezone, second precision.

    The timezone comes from the operating system (astimezone with no
    argument), NEVER from a constant in the code: a server in Frankfurt
    has to write '+02:00' on its own, with no one editing the project.
    """
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _registrar_no_csv(nome_processado: str, nome_original: str) -> bool:
    """
    Appends ONE row to the auxiliary CSV, in PURE APPEND.

    Never reads the file, never rewrites an existing row, never loses a
    record. The header is only written when the file is born.

    The function is called right after each successful move, not at the
    end of the scan: if the process dies mid-batch, the provenance of the
    audio files already moved is already on disk. Losing this
    relationship would leave an audio file orphaned of its origin,
    sitting in arquivos/audios/, with no way to fill the `nome_original`
    column of dataset.csv.

    Protection against a duplicate row does NOT live here: it lives in
    guard 1 (exists() in arquivos/audios/{id}/), which blocks an
    already-moved audio file before it reaches this point.

    Returns:
        True if the row is saved, False if it failed (with a log entry).
    """
    CSV_NOMEACAO.parent.mkdir(parents=True, exist_ok=True)

    escrever_header = not CSV_NOMEACAO.exists()

    try:
        with open(CSV_NOMEACAO, 'a', encoding='utf-8') as f:
            if escrever_header:
                f.write(CSV_NOMEACAO_HEADER + '\n')
            f.write(f"{nome_processado}|{nome_original}|{_agora_iso()}\n")
    except OSError as e:
        print(f"[M00] ERRO ao gravar o CSV auxiliar '{CSV_NOMEACAO}': {e}")
        return False

    return True


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def preparar_entrada(ja_concluido: Callable[[str], bool]) -> bool:
    """
    Names and moves material from `arquivos/input/` to `arquivos/audios/`.

    Args:
        ja_concluido: predicate that says whether an id is already in the
            completed-audio CSV (config.CSV_CONCLUIDOS). Injected by
            main.py so that the deduplication rule keeps living in a
            single place.

    Returns:
        True if the whole scan finished without losing a file, False if
        some move failed or the auxiliary CSV could not be written. An
        empty batch is not an error.
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

    # Alphabetical order of the full relative path: fixed rule that ensures
    # the same input folder always produces the same ids
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

    # Base name -> how many times it has already appeared, for the
    # _002, _003... tie-break
    # Used ONLY in 'nome_original' mode - see the id block, further below
    ocorrencias = {}
    csv_ok = True

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

        if modo == 'hash_md5':
            # CONTENT id. There is no tie-break in this mode, and it is not
            # an oversight: two files with the same name are already
            # distinguished by their bytes, and if the bytes are the same
            # they are the same audio and should really collide - guard 1
            # correctly blocks the second one.
            audio_id = _md5_do_conteudo(origem)
            if not audio_id:
                print(f"[M00] ERRO (hash nao pode ser calculado, arquivo NAO "
                      f"movido): {relativo}")
                erros += 1
                continue
        else:
            # NAME id, with tie-break. It only exists here: in this mode
            # the tie-broken name IS the audio_id, and two 'entrevista'
            # from the same batch would compete for the same folder
            # without it.
            # The counter advances BEFORE the guards: skipping an
            # already-moved audio file cannot renumber the following
            # ones, or the second run would produce different ids.
            numero = ocorrencias.get(nome_base, 0) + 1
            ocorrencias[nome_base] = numero
            audio_id = nome_base if numero == 1 else f"{nome_base}_{numero:03d}"

        pasta_destino = pasta_audios / audio_id
        destino = pasta_destino / f"{audio_id}{extensao}"

        # Guard 1 - resume after a break: any accepted format already
        # present in the id's folder means this audio file has already
        # been moved
        ja_no_destino = sorted(
            ext for ext in formatos if (pasta_destino / f"{audio_id}{ext}").exists()
        )
        if ja_no_destino:
            print(f"[M00] PULADO (guarda 1 - '{audio_id}{ja_no_destino[0]}' "
                  f"ja esta em arquivos/audios/{audio_id}/): {relativo}")
            pulados_guarda1 += 1
            continue

        # Guard 2 - audio file already COMPLETED before: does not get in
        # again. The source is the completed-audio CSV
        # (config.CSV_CONCLUIDOS), checked via the predicate that main
        # injects
        if ja_concluido(audio_id):
            print(f"[M00] PULADO (guarda 2 - id '{audio_id}' ja consta do CSV "
                  f"dos concluidos): {relativo}")
            pulados_guarda2 += 1
            continue

        if not _mover_seguro(origem, destino):
            erros += 1
            continue

        movidos += 1
        print(f"[M00] MOVIDO: {relativo} -> arquivos/audios/{audio_id}/{destino.name}")

        # Provenance recorded immediately after the move, in both modes
        if not _registrar_no_csv(audio_id, relativo):
            csv_ok = False

    print("-"*80)
    print(f"[M00] Arquivos encontrados          : {encontrados}")
    print(f"[M00] Movidos para arquivos/audios/ : {movidos}")
    print(f"[M00] Ignorados (formato fora)      : {recusados_formato}")
    print(f"[M00] Recusados (nome invalido)     : {recusados_nome}")
    print(f"[M00] Pulados (guarda 1 - ja movido): {pulados_guarda1}")
    print(f"[M00] Pulados (guarda 2 - concluido): {pulados_guarda2}")
    print(f"[M00] Erros ao mover                : {erros}")
    print(f"[M00] CSV auxiliar                  : {CSV_NOMEACAO}")
    print("="*80)

    return erros == 0 and csv_ok
