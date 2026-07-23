# arquivos/audios/

Diretório de entrada da pipeline. Guarda as pastas com os áudios que serão
processados.

## Estrutura

    arquivos/audios/{id}/{id}.<formato>

- Primeiro nível: uma pasta por áudio, nomeada com o {id}.
- Dentro dela: o arquivo de áudio usando o mesmo {id} como nome base.
- O formato é variável (.mp3, .flac, .wav, .ogg, entre outros).

## Atenção

Não assumir nenhum formato como padrão ao implementar leitura ou varredura.
A extensão deve ser sempre detectada, nunca fixada em código.

O conteúdo deste diretório não é versionado (ver .gitignore).
Este README existe para preservar a pasta no repositório.
