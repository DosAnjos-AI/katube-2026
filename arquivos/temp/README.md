# arquivos/temp/

Diretório de trabalho intermediário da pipeline. Guarda a cadeia de subpastas
gerada pelos módulos durante o processamento.

## Estrutura

    arquivos/temp/{id}/<subpastas numeradas por etapa>

Exemplo de subpastas: 01-arquivos_originais, 06-stt_whisper, 07-stt_wav2vec.

## Ciclo de vida

Conteúdo transitório, removido por limpar_temp.py entre lotes.

O conteúdo deste diretório não é versionado (ver .gitignore).
Este README existe para preservar a pasta no repositório.
