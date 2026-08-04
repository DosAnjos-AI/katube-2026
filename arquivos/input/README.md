# arquivos/input/

Cole aqui os áudios que você quer processar. É a porta de entrada da
pipeline.

## ATENÇÃO - COLE SEMPRE UMA CÓPIA, NUNCA A ÚNICA CÓPIA

Os arquivos colados aqui são **MOVIDOS**, não copiados. Esta pasta é
**esvaziada** a cada execução do `main.py`.

Se você colar aqui a única cópia que tem de um áudio, ela sai daqui e vai
para `arquivos/audios/{id}/`. Se algo der errado depois disso, o material
original não está mais no lugar de onde veio.

**Sempre mantenha o original em outro lugar.**

## Como colar

Pode colar arquivos soltos ou pastas inteiras, em quantos níveis de subpasta
quiser - a varredura é recursiva:

    arquivos/input/
    ├── audio_solto.flac
    └── lote_de_janeiro/
        ├── entrevista.mp3
        └── programa_02/
            └── entrevista.wav

## Formatos aceitos

Definidos em `config.py`, no campo `NOMEACAO['formatos_entrada']`. Arquivo
com extensão fora dessa lista é **ignorado com aviso no log** e fica onde
está - não é apagado nem movido.

## Que nome cada áudio recebe

Decidido em `config.py`, no campo `NOMEACAO['modo']`:

- `"nome_original"` - o id é o nome do arquivo sem a extensão.
- `"hash_md5"` - o id é o MD5 do nome.

Nos dois modos, a relação `id <-> caminho de origem` fica em
`dataset/nomeacao.csv`.

Nomes repetidos em pastas diferentes ganham sufixo: `entrevista`,
`entrevista_002`, `entrevista_003`, na ordem alfabética do caminho.

## Áudio já processado

Áudio cujo id já consta de `dataset/historico_dataset/` **não é movido** e
fica parado aqui, com aviso no log. Mesma coisa para áudio que já está em
`arquivos/audios/{id}/`.

O conteúdo deste diretório não é versionado (ver .gitignore).
Este README existe para preservar a pasta no repositório.
