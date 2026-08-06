# arquivos/input/

Paste here the audio files you want to process. This is the pipeline's
entry point.

## ATTENTION - ALWAYS PASTE A COPY, NEVER THE ONLY COPY

Files pasted here are **MOVED**, not copied. This folder is
**emptied** on every run of `main.py`.

If you paste the only copy you have of an audio file here, it leaves this
folder and goes to `arquivos/audios/{id}/`. If something goes wrong after
that, the original material is no longer where it came from.

**Always keep the original somewhere else.**

## How to paste

You can paste loose files or entire folders, at any number of subfolder
levels - the scan is recursive:

    arquivos/input/
    ├── audio_solto.flac
    └── lote_de_janeiro/
        ├── entrevista.mp3
        └── programa_02/
            └── entrevista.wav

## Accepted formats

Defined in `config.py`, in the `NOMEACAO['formatos_entrada']` field. A
file with an extension outside this list is **ignored with a log
warning** and stays where it is - it is neither deleted nor moved.

## What name each audio file receives

Decided in `config.py`, in the `NOMEACAO['modo']` field:

- `"hash_md5"` (recommended) - the id is the MD5 of the file's
  **content**. Two audio files with different content and the same name
  both get in; the same file pasted into another folder is recognized as
  a repeat.
- `"nome_original"` - the id is the file name without the extension.

In both modes, the `id <-> source path` relationship is kept in
`dataset/nomeacao.csv`.

Repeated names in different folders get a suffix: `entrevista`,
`entrevista_002`, `entrevista_003`, in alphabetical order of the path -
**only in `nome_original` mode**. In hash mode there is no suffix: the
content already distinguishes the files.

## Audio already completed

An audio file whose id is already in `dataset/concluidos.csv` **is not
moved** and stays here, named in the log with the guard that blocked it.
Same for an audio file already in `arquivos/audios/{id}/`.

`dataset/historico_dataset/` **does not** take part in this: it is a
backup of the tracking JSON, not a deduplication marker.

The content of this directory is not versioned (see .gitignore).
This README exists to preserve the folder in the repository.
