# arquivos/audios/

Input directory for the pipeline. Holds the folders with the audio files
that will be processed.

## Structure

    arquivos/audios/{id}/{id}.<format>

- First level: one folder per audio file, named with the {id}.
- Inside it: the audio file, using the same {id} as the base name.
- The format is variable (.mp3, .flac, .wav, .ogg, among others).

## Attention

Do not assume any format as default when implementing reading or
scanning. The extension must always be detected, never hardcoded.

The content of this directory is not versioned (see .gitignore).
This README exists to preserve the folder in the repository.
