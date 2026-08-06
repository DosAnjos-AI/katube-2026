# katube-2026 — Pipeline de Processamento de Áudio

Pipeline modular que transforma áudios brutos, de **qualquer origem**, em um
**dataset limpo e transcrito para treino de modelos TTS/STT** em Português
Brasileiro.

A entrada é uma coletânea de áudios; a saída é um `dataset.csv` com segmentos
curtos, normalizados, transcritos por dois modelos STT e validados por
similaridade, mais os arquivos de áudio correspondentes. **As características
do áudio final não são fixas**: taxa de amostragem, profundidade de bits,
número de canais, formato do arquivo, método de normalização de volume e nível
alvo são todos definidos no bloco `SOX_NORMALIZER` do [config.py](config.py) —
ver [Saídas (o dataset)](#saídas-o-dataset).

---

## Visão geral

```
Áudios brutos (.ogg/.flac/...) colados em arquivos/input/
        │
        ▼
┌──────────────────────── main.py (orquestrador) ────────────────────────┐
│  M00 ── nomeação: varre input/, resolve o id e MOVE para audios/       │
│         │                                                              │
│         ▼                                                              │
│  arquivos/audios/{audio_id}/{audio_id}.ext                             │
│         │                                                              │
│         ▼                                                              │
│  M02 → M04 → M05 → M06 → M07 → M08 → M09 → M10 → M11 → M12 → M13 → M14 → M15  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
dataset/dataset.csv  +  dataset/audio_dataset/{audio_id}/*.{formato de saída}
```

Cada áudio é identificado por um **`audio_id`**, decidido pelo M00: o hash MD5
do **conteúdo** do arquivo (modo recomendado) ou o nome original — ver
`NOMEACAO` em [config.py](config.py). Todo o estado intermediário de um áudio vive em
`arquivos/temp/{audio_id}/`, organizado em subpastas numeradas que espelham as
etapas da pipeline.

> **ATENÇÃO — `arquivos/input/` é esvaziada a cada execução.** O M00 **move** os
> arquivos, não copia. **Cole sempre uma CÓPIA em `arquivos/input/`, nunca a única
> cópia dos áudios** — eles são consumidos no processamento. Ver
> [arquivos/input/README.md](arquivos/input/README.md).

---

## Diagrama de fluxo

![Diagrama do fluxo da pipeline Katube VAD 2026: da ingestão dos áudios em arquivos/input/ até o dataset.csv final, passando pelos módulos M00 a M15](Alcateia_-_Fluxo_Katube_VAD_2026.svg)

O desenho acima é uma exportação do board no Miro, que é a fonte sempre atualizada:
https://miro.com/app/board/uXjVG9eNQ_g=/?focusWidget=3458764660637824545

---

## Estrutura do projeto

```
katube-2026/
├── main.py                  # Orquestrador: roda M02→M15 para cada áudio
├── config.py                # TODA a configuração (bloco MASTER + params por módulo)
├── requirements-servidor.txt # Versões fixadas do ambiente de referência
├── .env.example             # Modelo do .env (token HuggingFace)
├── INSTALL.md               # Montagem e verificação do ambiente
├── Alcateia_-_Fluxo_Katube_VAD_2026.svg  # Diagrama do fluxo
├── src/                     # Módulos da pipeline (m00–m15)
│   ├── m00_nomeacao.py               # Porta de entrada: input/ → audios/{id}/{id}.ext
│   ├── m01_load_models.py            # Singleton de modelos de IA (carrega 1x)
│   ├── m02_diretorios.py             # Cria pastas do áudio e converte a entrada para WAV
│   ├── m04_segmentador_audio_vad.py  # Segmentação por VAD (Silero)
│   ├── m05_segmentador_16khz.py      # Conversão para 16 kHz mono
│   ├── m06_mos_filter.py             # Filtro de qualidade MOS (SQUIM)
│   ├── m07_overlap1.py               # Detecção de sobreposição de locutores (pyannote)
│   ├── m08_whisper.py                # STT com distil-Whisper PT-BR
│   ├── m09_wav2vec.py                # STT com Wav2Vec2 PT-BR
│   ├── m10_texto_normalizador.py     # Normalização de texto
│   ├── m11_validador_similaridade.py # Validação de similaridade (WER, CER, Levenshtein)
│   ├── m12_denoiser_deepfilternet3.py# Denoising (DeepFilterNet3)
│   ├── m13_normalizador_audio.py     # Normalização de áudio (SoX)
│   ├── m14_metadados.py              # Escreve dataset.csv (append) + histórico
│   └── m15_cleanup.py                # Limpeza de temporários/input
├── arquivos/
│   ├── input/                        # ENTRADA: cole aqui (é ESVAZIADA a cada execução)
│   ├── audios/{audio_id}/            # Pasta de trabalho: um áudio por pasta
│   └── temp/{audio_id}/              # Estado intermediário (ver abaixo)
└── dataset/
    ├── sumary_results.py             # Sumarização de resultados do dataset
    ├── dataset.csv                   # SAÍDA: metadados de cada segmento aprovado
    ├── nomeacao.csv                  # Procedência: id ↔ caminho de origem
    ├── concluidos.csv                # Deduplicação: quem terminou a pipeline
    ├── processamento_metadados.csv   # Registro de cada rodada: duração e tempo por módulo
    ├── audio_dataset/{audio_id}/     # SAÍDA: segmentos de áudio finais
    ├── historico_dataset/{id}.json   # Backup do JSON de acompanhamento por áudio
    └── log/{audio_id}.log            # Log detalhado por áudio
```

### Subpastas de `arquivos/temp/{audio_id}/`

Criadas por **M02** e consumidas/preenchidas pelos módulos seguintes:

| Pasta | Conteúdo | Módulo |
|-------|----------|--------|
| `00-json_dinamico/`     | JSON de acompanhamento (estado de cada segmento) | todos |
| `01-arquivos_originais/`| Cópia do áudio de entrada | M02 |
| `02-segmentos_originais/`| Segmentos no sample rate original | M04 |
| `03-segments_16khz/`    | Segmentos convertidos para 16 kHz | M05 |
| `04-mos_score/`         | Resultados do filtro MOS | M06 |
| `05-overlap1/`          | Resultados de detecção de overlap | M07 |
| `06-stt_whisper/`       | Transcrições Whisper | M08 |
| `07-stt_wav2vec/`       | Transcrições Wav2Vec2 | M09 |
| `08-normalizador_texto/`| Textos normalizados | M10 |
| `09-validacao_similaridade/`| Métricas de similaridade | M11 |
| `10-denoiser/`          | Áudios após denoising | M12 |
| `11-normalizador_audio/`| Áudios finais normalizados (SoX) | M13 |

O **`00-json_dinamico/{audio_id}_segments_acompanhamento.json`** é o coração da
pipeline: cada módulo lê e enriquece esse JSON com seus campos, e ao final o M14 o
converte em linhas do `dataset.csv`.

---

## A pipeline passo a passo

O orquestrador ([main.py](main.py)) roda o **M00 uma única vez**, depois percorre
todos os `audio_id` em `arquivos/audios/`, pula os que já têm histórico, e executa
as etapas abaixo para cada novo áudio. Etapas marcadas **(condicional)** só rodam
conforme o bloco `MASTER` em [config.py](config.py).

### M00 — Nomeação *(obrigatório, roda 1x por execução)*
Varre `arquivos/input/` **recursivamente**, filtra pelos formatos de
`NOMEACAO['formatos_entrada']`, resolve o `audio_id` de cada arquivo e o **MOVE**
para `arquivos/audios/{audio_id}/{audio_id}.ext`.

- **Id**: conforme `NOMEACAO['modo']`, sempre **determinístico** (a mesma entrada
  produz sempre o mesmo id):
  - **`hash_md5` — RECOMENDADO.** O id é o MD5 dos **bytes do arquivo**. Três
    ganhos de uma vez: (1) dois áudios de **conteúdo diferente** com o mesmo
    nome geram ids diferentes e **ambos entram** — material novo não se perde
    mais; (2) o **mesmo** arquivo colado em outra pasta gera o **mesmo** id e é
    reconhecido como repetido, então a retomada após quebra sobrevive até à
    renomeação da pasta de origem; (3) id seguro por construção — 32 caracteres
    hexadecimais, sem espaço, acento ou o `|` que quebraria a linha do CSV.
  - **`nome_original`**: o id é o nome sem extensão. Legível, mas colide entre
    lotes — dois lotes com `entrevista.flac` disputam o mesmo id e o segundo é
    barrado **mesmo tendo conteúdo diferente**.
- **Limitação do modo hash**: ele detecta **arquivo idêntico**, não "mesmo
  conteúdo sonoro". O mesmo áudio reexportado, com metadado diferente ou
  convertido de formato, gera hash diferente e **passa como novo**.
- **Por que não hash de metadados** (timestamp, tamanho): timestamp **não
  sobrevive** a cópia, download, extração de zip, sincronização de nuvem ou
  transferência para o servidor. O mesmo áudio chegaria à AWS com id diferente,
  garantindo duplicata em todo transporte — que é justamente o fluxo do projeto.
- **Custo do hash**: medido em 0,003% de uma rodada em CPU (73 ms para 13,4 MB),
  projetado em ~0,03% em GPU. É I/O de disco, não computação de modelo, então
  não acelera na GPU — mas parte de um patamar irrelevante.
- **Nomes repetidos** em pastas diferentes ganham sufixo `_002`, `_003`, na ordem
  alfabética do caminho relativo — **apenas no modo `nome_original`**. No modo
  hash não há desempate: dois arquivos de mesmo nome já se distinguem pelo
  conteúdo, e se o conteúdo for igual são o mesmo áudio e devem mesmo colidir.
- **Não move** o que já está em `arquivos/audios/{id}/` (guarda 1) nem o que já
  consta de `dataset/concluidos.csv` (guarda 2). O arquivo fica parado na
  `input/`, nomeado no log, com a guarda que o barrou e a contagem no rodapé.
- Arquivo de formato não aceito é **ignorado com aviso e contagem**, e não sai
  do lugar.
- Grava a procedência de cada áudio movido em `dataset/nomeacao.csv` (separador
  `|`, **append puro**, nos dois modos). É a origem da coluna `nome_original` do
  `dataset.csv`.

**A pasta `arquivos/input/` é esvaziada a cada execução — cole sempre uma cópia.**

### M02 — Criar diretórios *(obrigatório)*
Cria `arquivos/temp/{audio_id}/` com todas as subpastas numeradas e **converte** o
áudio de entrada para **WAV** em `01-arquivos_originais/`, preservando sample rate,
canais e profundidade de bits do original (24 bits viram `pcm_s24le`, não são
truncados). WAV é o formato interno da pipeline: daí em diante o formato de
entrada não circula mais, o que permite aceitar formatos que o SoX não lê
(`m4a`, `aac`, `wma`). Também garante a existência de `dataset/audio_dataset/`,
`dataset/historico_dataset/` e `dataset/log/`.

### M04 — Segmentação *(condicional)*
Quebra o áudio longo em segmentos curtos de fala. Modo definido por
`MASTER['segmentacao']`:
- **`'vad'`** (padrão): usa **Silero-VAD** para detectar fala/silêncio e cortar em
  pausas naturais. Parâmetros em `SEGMENTADOR_AUDIO_VAD` (threshold de voz, durações
  mín./máx. de segmento, padding nos cortes). Se nenhum segmento válido é encontrado,
  o áudio é **descartado**.
- **`''`**: pula (áudio já vem segmentado).

Saída: segmentos em `02-segmentos_originais/`.

### M05 — Conversão 16 kHz *(obrigatório)*
Converte cada segmento para **16 kHz mono**, formato esperado pelos modelos de IA a
jusante. Saída em `03-segments_16khz/`.

### M06 — Filtro MOS *(condicional — `mos_filter`)*
Avalia a qualidade percebida de cada segmento com o modelo **SQUIM** (MOS, STOI,
SI-SDR). Classifica em `alta`/`media`/`baixa` segundo `MOS_FILTER['thresholds']` e
**descarta** segmentos abaixo de `min_threshold`. Essa classificação também decide
o que o denoiser (M12) processa.

### M07 — Detecção de overlap *(condicional — `overlap`)*
Usa **pyannote** (`speaker-diarization-3.1`) para detectar sobreposição de locutores
(duas pessoas falando ao mesmo tempo). Segmentos com overlap recebem flag para serem
filtrados — áudio de treino TTS/STT deve ter um único falante por segmento.

### M08 — Transcrição Whisper *(condicional — `transcricao_whisper`)*
Transcreve cada segmento com **`freds0/distil-whisper-large-v3-ptbr`**. Campo
`stt_whisper`.

### M09 — Transcrição Wav2Vec2 *(condicional — `transcricao_wav2vec`)*
Transcreve com **`lgris/wav2vec2-large-xlsr-open-brazilian-portuguese`**. Campo
`stt_wav2vec`. Ter **duas transcrições independentes** permite validar a qualidade
por concordância (M11).

### M10 — Normalização de texto *(obrigatório)*
Normaliza as transcrições segundo `TEXT_NORMALIZER`: remove
pontuação que afeta dicção e, opcionalmente, acentuação gráfica. Produz os campos
`*_normalizado` usados apenas para comparação.

### M11 — Validação de similaridade *(obrigatório)*
Compara as duas transcrições normalizadas entre si (Whisper × Wav2Vec) calculando
**sempre as três métricas** (`SIMILARITY_VALIDATOR`), cada uma na sua convenção:

| Métrica | O que mede | Direção | Passa se | Limiar |
|---|---|---|---|---|
| `wer` | taxa de erro por **palavra** (sem teto superior) | 0 = perfeito | `<=` | `limiar_wer` (0.20) |
| `cer` | taxa de erro por **caractere** | 0 = perfeito | `<=` | `limiar_cer` (0.15) |
| `levenshtein_norm` | similaridade normalizada 0–1 | 1 = idêntico | `>=` | `limiar_levenshtein_norm` (0.85) |

O segmento só é aprovado se passar nos **três** limiares — alta divergência entre dois
STTs indica transcrição ruim ou áudio problemático. Cada reprovação é registrada no log
com a métrica que reprovou e o seu valor. As duas transcrições são obrigatórias: sem uma
delas o segmento não é elegível. Campos `sim_whisper_wav2vec_wer`,
`sim_whisper_wav2vec_cer`, `sim_whisper_wav2vec_levenshtein_norm`,
`status_similaridade`.

### M12 — Denoiser *(condicional — `Denoiser`)*
Aplica **DeepFilterNet3** para remover ruído. Processa apenas as faixas de qualidade
listadas em `DEEPFILTERNET_DENOISER['mos_quality_filter']` (ex.: só `media`),
preservando os originais. Saída em `10-denoiser/`.

### M13 — Normalização de áudio (SoX) *(obrigatório)*
Padroniza o áudio final com **SoX**. **Todas as características da saída vêm de
`SOX_NORMALIZER`, no [config.py](config.py)** — nenhuma é fixa no código:
`sample_rate`, `bit_depth`, `channels`, `output_format`, `normalize_method`,
`target_level_db`, além da remoção de silêncio nas pontas. Cada campo está
comentado no arquivo, com as opções aceitas e o efeito de cada uma. Saída em
`11-normalizador_audio/`.

### M14 — Metadados *(obrigatório)*
Converte o JSON de acompanhamento em linhas do **`dataset/dataset.csv`** (separador
`|`). Garantias importantes:
- Valida 1:1 entre segmentos do JSON e arquivos físicos em `audio_dataset/`.
- Escrita **append puro**: as linhas do lote são acrescentadas ao CSV (que é criado
  se ainda não existir). Não há reescrita nem truncamento. Quando o lote traz uma
  coluna que o cabeçalho já gravado não tem, o cabeçalho existente prevalece e os
  campos descartados são avisados nominalmente.
- A deduplicação entre execuções é feita pelo **`dataset/concluidos.csv`**: o
  `main.py` barra na entrada o áudio cujo id já conste dele.
- Copia o JSON para `historico_dataset/{audio_id}.json`. Isso é **backup**, não
  deduplicação: permite reconstruir o dataset sem rodar os modelos de novo, e é
  a fonte da duração aprovada da rodada.
- **Registra o áudio em `dataset/concluidos.csv` como último passo**, depois de
  os segmentos estarem em `audio_dataset/` e as linhas no `dataset.csv`. Áudio
  que quebrou no meio não fica registrado e **por isso pode ser reprocessado**.

### M15 — Cleanup *(condicional — `cleanup`)*
Remove pastas temporárias e/ou de entrada conforme `MASTER['cleanup']`:
`'all'` (temp + entrada), `'input'`, `'temp'` ou `'none'`.

> **Cuidado com o nome:** aqui `'input'` significa **`arquivos/audios/{audio_id}/`**,
> a pasta de trabalho do áudio — **não** `arquivos/input/`, onde você cola o material.
> A pasta `arquivos/input/` nunca é apagada pelo M15.

Ao final, o `main.py` grava em `dataset/processamento_metadados.csv` um registro com
duração total, contagem de áudios (processados/pulados/erros) e o **tempo gasto em
cada módulo** (absoluto e percentual).

---

## Saídas (o dataset)

**`dataset/dataset.csv`** — uma linha por segmento aprovado. **Schema fixo de 31
colunas**, na ordem do cabeçalho:

```
nome_original | nome_processado | nome_arquivo_audio | caminho |
tempo_inicio | tempo_fim | duracao | vad |
origem_codec | origem_bitrate | origem_sample_rate |
mos_score | mos_stoi | mos_si_sdr | mos_qualidade | overlap01 |
stt_whisper | stt_wav2vec | sim_whisper_wav2vec_wer |
sim_whisper_wav2vec_cer | sim_whisper_wav2vec_levenshtein_norm |
status_similaridade |
utilizou_denoiser | sox_sample_rate | sox_bit_depth | sox_channels |
sox_output_format | sox_normalize_method | sox_target_level_db |
utilizou_sox | datetime_processado
```

**As colunas são sempre estas, qualquer que seja a configuração.** Denoiser
desligado, SoX que não rodou, módulo pulado: a coluna continua lá, vazia. O que
varia é o preenchimento, nunca o conjunto de colunas. O schema vive em
`SCHEMA_DATASET`, no M14.

As três primeiras colunas identificam a origem, em granularidades diferentes:

| Coluna | Conteúdo | Granularidade |
|---|---|---|
| `nome_original` | caminho relativo de origem dentro de `arquivos/input/` | do **áudio** (repetido nas n linhas) |
| `nome_processado` | id do áudio: nome desempatado ou hash | do **áudio** (repetido nas n linhas) |
| `nome_arquivo_audio` | `{id}_{numeração}.{formato de saída}` | do **segmento** |

No modo `nome_original`, as duas primeiras ficam com o mesmo valor de base —
duplicação esperada, não defeito.

**Valor de ausência:** célula **vazia** — é o que pandas, polars e o `csv` do
Python leem como nulo sem tratamento nenhum. Exceção para três booleanos (`vad`,
`utilizou_denoiser`, `utilizou_sox`), onde a ausência vira `False`, que ali
significa "não usei". `overlap01` **não** é exceção: `False` nele significa "não
há sobreposição", ou seja, aprovado — carimbá-lo sem o M07 ter rodado falsearia
o dado.

**`datetime_processado`** — momento em que a linha foi escrita, em ISO 8601 com
fuso e precisão de segundos (`2026-08-04T15:32:07-03:00`). O fuso vem do sistema
operacional, nunca de constante no código: um servidor em Frankfurt grava
`+02:00` sozinho.

O M14 só **cria** e faz **append**. Se o arquivo já existir com header diferente
do schema, ele se recusa a gravar e devolve falha — append de 31 campos sob outro
cabeçalho corromperia o arquivo em silêncio. Para migrar, arquive o CSV antigo.

**`dataset/audio_dataset/{audio_id}/`** — os arquivos de áudio finais
referenciados pela coluna `caminho`. A extensão é a de
`SOX_NORMALIZER['output_format']`, e as demais características do arquivo
(taxa de amostragem, profundidade de bits, canais, normalização de volume)
saem dos outros campos do mesmo bloco. As colunas `sox_*` do `dataset.csv`
registram, linha a linha, os valores efetivamente aplicados — é lá que se lê o
que foi produzido, não em constante do código.

**`dataset/nomeacao.csv`** — a procedência de cada áudio
(`nome_processado | nome_original | datetime_movido`, separador `|`). Escrito pelo
M00 em **append puro** nos **dois** modos de nomeação, uma linha por áudio movido,
gravada logo após o move. É dele que sai a coluna `nome_original` do
`dataset.csv`. Nunca reescreve linha existente. O caminho é declarado em
`config.CSV_NOMEACAO`.

**`dataset/concluidos.csv`** — os áudios que **terminaram** a pipeline
(`nome_processado | nome_original | datetime_concluido`, separador `|`). É a
**única fonte da deduplicação**. Escrito pelo M14 em **append puro**, como
último passo de cada áudio: quando a linha aparece, os segmentos já estão em
`audio_dataset/`, as linhas já estão no `dataset.csv` e o backup do JSON já está
no histórico. Não existe, em lugar nenhum do projeto, código que apague linha ou
arquivo daqui — **reprocessar um áudio já concluído é ação manual**: apague a
linha correspondente com um editor, fora da pipeline. Não há campo de
configuração para "reprocessar mesmo assim", e isso é decisão: como o
`dataset.csv` é append puro e nada nele é removido, um botão desses seria um
botão de gerar duplicata. O caminho é declarado em `config.CSV_CONCLUIDOS`.

---

## Configuração

Toda a configuração fica em [config.py](config.py). O ponto de partida é o **bloco
`MASTER`**, que liga/desliga as etapas condicionais:

```python
MASTER = {
    'segmentacao': 'vad',        # 'vad' | '' (já segmentado)
    'mos_filter': True,
    'overlap': True,
    'transcricao_whisper': True,
    'transcricao_wav2vec': True,
    'Denoiser': True,
    'cleanup': 'all',            # 'all' | 'input' | 'temp' | 'none'
}
```

O bloco **`NOMEACAO`** governa a porta de entrada (M00):

```python
NOMEACAO = {
    'modo': 'nome_original',     # 'hash_md5' (recomendado) | 'nome_original'
    'formatos_entrada': {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'},
}
```

`formatos_entrada` é a **fonte única** de formato de entrada do projeto:
`EXTENSOES_AUDIO`, que os módulos importam, é derivada dele.

Cada módulo tem seu próprio dicionário de parâmetros (`SEGMENTADOR_AUDIO_VAD`,
`MOS_FILTER`, `OVERLAP_DETECTOR`, `STT_WHISPER`, `STT_WAV2VEC2`, `TEXT_NORMALIZER`,
`SIMILARITY_VALIDATOR`, `DEEPFILTERNET_DENOISER`, `SOX_NORMALIZER`) — todos
extensamente comentados no arquivo.

### Modelos de IA

Carregados como **singleton** por [m01_load_models.py](src/m01_load_models.py) (uma
vez só, reutilizados entre áudios):

| Etapa | Modelo |
|-------|--------|
| Whisper (M08)   | `freds0/distil-whisper-large-v3-ptbr` |
| Wav2Vec2 (M09)  | `lgris/wav2vec2-large-xlsr-open-brazilian-portuguese` |
| Overlap (M07)   | `pyannote/speaker-diarization-3.1` |
| MOS (M06)       | TorchAudio SQUIM |
| Denoiser (M12)  | DeepFilterNet3 |

### Variáveis de ambiente (`.env` na raiz do projeto)

- **Token HuggingFace** — necessário para o modelo pyannote (M07).
- `CUDA_VISIBLE_DEVICES=""` força execução em CPU (carregado antes de importar torch).

---

## Como executar

### Pré-requisitos

Montagem do ambiente, versões e verificação: **[INSTALL.md](INSTALL.md)**.

Em resumo: env conda ativado, `ffmpeg`/`ffprobe`/`sox` instalados **pelo
sistema** (não pelo conda) e `.env` preenchido a partir de
[.env.example](.env.example) com o token HuggingFace.

### 1. Ingestão dos áudios
Cole os áudios em **`arquivos/input/`** — arquivos soltos ou pastas inteiras, em
quantos níveis de subpasta quiser. O M00 cuida do resto na próxima execução do
`main.py`: varre, nomeia e move para `arquivos/audios/{audio_id}/{audio_id}.ext`.

> **COLE SEMPRE UMA CÓPIA, NUNCA A ÚNICA CÓPIA DOS ÁUDIOS.** Os arquivos são
> **movidos**, não copiados: `arquivos/input/` é esvaziada a cada execução e o
> material sai de lá. Mantenha sempre o original em outro lugar.

Quem preferir pode continuar colando direto em
`arquivos/audios/{audio_id}/{audio_id}.ext` — o M00 não atrapalha o que já está lá.

### 2. Rodar a pipeline
```bash
conda activate katube-2026
cd <raiz do projeto>
python main.py
```
O `main.py` processa todos os áudios pendentes em `arquivos/audios/`, pulando os já
registrados em `dataset/concluidos.csv`.

### 3. Conferir o resultado

`dataset/sumary_results.py` lê o `dataset.csv` e resume a rodada: duração total
em horas, contagem de segmentos, duração média, mínima e máxima, e quantos
segmentos repetidos existem.

---

## Notas de design

- **Idempotência**: o `dataset/concluidos.csv` permite parar e retomar a
  qualquer momento sem reprocessar nem duplicar segmentos. A linha nele é o que
  marca um áudio como concluído, e o `main.py` barra o áudio repetido na
  entrada, antes de qualquer módulo rodar. Como a linha só é escrita **depois**
  de tudo estar gravado, o áudio interrompido no meio volta a ser processado
  inteiro — a retomada não depende de adivinhar onde a rodada parou.
- **O histórico (`historico_dataset/`) não é deduplicação**: é backup. Os JSONs
  continuam sendo escritos, e servem para reconstruir o dataset sem rodar os
  modelos de novo e para somar a duração aprovada da rodada.
- **Estado dirigido por JSON**: cada módulo enriquece o
  `*_segments_acompanhamento.json`; o pipeline é, em essência, um enriquecimento
  progressivo desse documento até o M14 materializá-lo no CSV.
- **Filtros em cascata**: segmentos são descartados ao longo do caminho (VAD vazio →
  MOS baixo → overlap → baixa similaridade), de modo que só o material de boa
  qualidade chega ao dataset final.
- **Escrita resiliente**: o M14 nunca trunca o CSV — a gravação é **append puro**
  (ou criação, se o arquivo não existir). O histórico é copiado por último, só
  depois das linhas efetivadas: se algo falhar no meio, o áudio não fica marcado
  como concluído.
