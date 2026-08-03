# spotify2026 — Pipeline de Processamento de Áudio (katube-2026)

Pipeline modular que transforma áudios brutos (segmentos do Spotify) em um **dataset
limpo e transcrito para treino de modelos TTS/STT** em Português Brasileiro.

A entrada é uma coleção de áudios; a saída é um `dataset.csv` com segmentos curtos,
normalizados, transcritos por dois modelos STT, validados por similaridade e com
áudios finais em FLAC 16 kHz mono.

---

## Visão geral

```
Áudios brutos (.ogg/.flac/...)
        │
        ▼
[ nomecao_spotify.py ]  ── ingestão: hash MD5 + relacao_ids.csv
        │
        ▼
arquivos/audios/{audio_id}/{audio_id}.ext
        │
        ▼
┌──────────────────────── main.py (orquestrador) ────────────────────────┐
│  M02 → M04 → M05 → M06 → M07 → M08 → M09 → M10 → M11 → M12 → M13 → M14 → M15  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
dataset/dataset.csv  +  dataset/audio_dataset/{audio_id}/*.flac
```

Cada áudio é identificado por um **`audio_id`** (hash MD5). Todo o estado intermediário
de um áudio vive em `arquivos/temp/{audio_id}/`, organizado em subpastas numeradas que
espelham as etapas da pipeline.

---

## Estrutura do projeto

```
spotify2026/
├── main.py                  # Orquestrador: roda M02→M15 para cada áudio
├── config.py                # TODA a configuração (bloco MASTER + params por módulo)
├── pipeline_spotify.sh      # Driver em lote: ingestão + main.py + arquivamento por iteração
├── watchdog_spotify2026.sh  # Reexecuta main.py até esvaziar a fila (audios == temp)
├── limpar_temp.py           # Utilitário de limpeza manual de temporários
├── sumary_results.py        # Sumarização de resultados do dataset
├── src/                     # Módulos da pipeline (m01–m15)
│   ├── m01_load_models.py            # Singleton de modelos de IA (carrega 1x)
│   ├── m02_diretorios.py             # Cria estrutura de pastas do áudio
│   ├── m04_segmentador_audio_vad.py  # Segmentação por VAD (Silero)
│   ├── m05_segmentador_16khz.py      # Conversão para 16 kHz mono
│   ├── m06_mos_filter.py             # Filtro de qualidade MOS (SQUIM)
│   ├── m07_overlap1.py               # Detecção de sobreposição de locutores (pyannote)
│   ├── m08_whisper.py                # STT com distil-Whisper PT-BR
│   ├── m09_wav2vec.py                # STT com Wav2Vec2 PT-BR
│   ├── m10_texto_normalizador.py     # Normalização de texto
│   ├── m11_validador_levenshtein.py  # Validação de similaridade (WER/CER)
│   ├── m12_denoiser_deepfilternet3.py# Denoising (DeepFilterNet3)
│   ├── m13_normalizador_audio.py     # Normalização de áudio (SoX)
│   ├── m14_metadados.py              # Escreve dataset.csv (append) + histórico
│   └── m15_cleanup.py                # Limpeza de temporários/input
├── arquivos/
│   ├── audios/{audio_id}/            # ENTRADA: um áudio por pasta
│   └── temp/{audio_id}/              # Estado intermediário (ver abaixo)
└── dataset/
    ├── dataset.csv                   # SAÍDA: metadados de cada segmento aprovado
    ├── relacao_ids.csv               # Mapa hash ↔ nome/origem original
    ├── audio_dataset/{audio_id}/     # SAÍDA: segmentos .flac finais
    ├── historico_dataset/{id}.json   # JSON de acompanhamento por áudio processado
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
| `09-validacao_levenshtein/`| Notas de similaridade | M11 |
| `10-denoiser/`          | Áudios após denoising | M12 |
| `11-normalizador_audio/`| Áudios finais normalizados (SoX) | M13 |

O **`00-json_dinamico/{audio_id}_segments_acompanhamento.json`** é o coração da
pipeline: cada módulo lê e enriquece esse JSON com seus campos, e ao final o M14 o
converte em linhas do `dataset.csv`.

---

## A pipeline passo a passo

O orquestrador ([main.py](main.py)) percorre todos os `audio_id` em
`arquivos/audios/`, pula os que já têm histórico, e executa as etapas abaixo para
cada novo áudio. Etapas marcadas **(condicional)** só rodam conforme o bloco
`MASTER` em [config.py](config.py).

### M02 — Criar diretórios *(obrigatório)*
Cria `arquivos/temp/{audio_id}/` com todas as subpastas numeradas e copia o áudio
de entrada para `01-arquivos_originais/`. Também garante a existência de
`dataset/audio_dataset/`, `dataset/historico_dataset/` e `dataset/log/`.

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
Compara as duas transcrições normalizadas entre si (Whisper × Wav2Vec) usando
**WER/CER/Levenshtein** (`SIMILARITY_VALIDATOR`). Segmentos com similaridade abaixo
do `similarity_threshold` são marcados como reprovados — alta divergência entre dois
STTs indica transcrição ruim ou áudio problemático. As duas transcrições são
obrigatórias: sem uma delas o segmento não é elegível. Campos `sim_whisper_wav2vec`,
`nota_similaridade`, `status_similaridade`.

### M12 — Denoiser *(condicional — `Denoiser`)*
Aplica **DeepFilterNet3** para remover ruído. Processa apenas as faixas de qualidade
listadas em `DEEPFILTERNET_DENOISER['mos_quality_filter']` (ex.: só `media`),
preservando os originais. Saída em `10-denoiser/`.

### M13 — Normalização de áudio (SoX) *(obrigatório)*
Padroniza o áudio final com **SoX** segundo `SOX_NORMALIZER`: sample rate (16 kHz),
bit depth (16), mono, formato **FLAC**, normalização de volume (RMS −3 dB) e remoção
de silêncio nas pontas. Saída em `11-normalizador_audio/`.

### M14 — Metadados *(obrigatório)*
Converte o JSON de acompanhamento em linhas do **`dataset/dataset.csv`** (separador
`|`). Garantias importantes:
- Valida 1:1 entre segmentos do JSON e arquivos físicos em `audio_dataset/`.
- Escrita **append puro**: as linhas do lote são acrescentadas ao CSV (que é criado
  se ainda não existir). Não há reescrita nem truncamento. Quando o lote traz uma
  coluna que o cabeçalho já gravado não tem, o cabeçalho existente prevalece e os
  campos descartados são avisados nominalmente.
- A deduplicação entre execuções é feita pelo **histórico** (`historico_dataset/`),
  não por índice: o `main.py` barra na entrada o áudio que já tem histórico.
- Copia o JSON para `historico_dataset/{audio_id}.json` (marca o áudio como
  processado e evita reprocessá-lo).

### M15 — Cleanup *(condicional — `cleanup`)*
Remove pastas temporárias e/ou de input conforme `MASTER['cleanup']`:
`'all'` (temp + input), `'input'`, `'temp'` ou `'none'`.

Ao final, o `main.py` grava em `dataset/processamento_metadados.csv` um registro com
duração total, contagem de áudios (processados/pulados/erros) e o **tempo gasto em
cada módulo** (absoluto e percentual).

---

## Saídas (o dataset)

**`dataset/dataset.csv`** — uma linha por segmento aprovado. As 29 colunas de
hoje, na ordem do cabeçalho:

```
arquivo_nome | caminho | tempo_inicio | tempo_fim | duracao | texto | vad |
origem_codec | origem_bitrate | origem_sample_rate |
mos_score | mos_stoi | mos_si_sdr | mos_qualidade | overlap01 |
stt_whisper | stt_wav2vec | sim_whisper_wav2vec |
nota_similaridade | status_similaridade | metrica_similaridade |
utilizou_denoiser | sox_sample_rate | sox_bit_depth | sox_channels |
sox_output_format | sox_normalize_method | sox_target_level_db |
utilizou_sox
```

O cabeçalho não é fixo no código: o M14 o monta a partir das chaves do JSON de
acompanhamento, com `arquivo_nome` e `caminho` como únicas colunas fixas. As
transcrições ficam em `stt_whisper` e `stt_wav2vec`; a coluna `texto` é um campo
reservado e hoje sai sempre vazia.

**`dataset/audio_dataset/{audio_id}/*.flac`** — os arquivos de áudio finais
referenciados pela coluna `caminho`.

**`dataset/relacao_ids.csv`** — mapeia cada `audio_id` (hash) de volta ao nome e
caminho do arquivo original (`hash | nome_audio | caminho_origem | caminho_destino`).

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
- Ambiente conda **`katube-2026`** ativado.
- `ffmpeg` e `sox` instalados.
- `.env` configurado (token HuggingFace).

### 1. Ingestão dos áudios
Coloque os áudios de entrada em `arquivos/audios/{audio_id}/{audio_id}.ext`, ou use o
script de ingestão [Dataset_Spotify_Processado/nomecao_spotify.py](../Dataset_Spotify_Processado/nomecao_spotify.py),
que busca áudios recursivamente, renomeia cada um pelo **hash MD5** do caminho,
descarta arquivos abaixo de 11 KB e gera o `relacao_ids.csv`.

### 2. Rodar a pipeline
```bash
conda activate katube-2026
cd /home/ubuntu/spotify2026
python main.py
```
O `main.py` processa todos os áudios pendentes em `arquivos/audios/`, pulando os já
registrados em `historico_dataset/`.

### Execução em lote / não supervisionada

- **[pipeline_spotify.sh](pipeline_spotify.sh)** — roda iterações completas:
  para cada iteração `N` executa a ingestão (`nomecao_spotify{N}.py`), roda o
  `main.py`, e arquiva resultados (`dataset/`, `output.log`, `tempo.txt`) em
  `Dataset_Spotify_Processado/{N}/`, limpando o ambiente para a próxima.

- **[watchdog_spotify2026.sh](watchdog_spotify2026.sh)** — reexecuta `main.py` em
  loop até a fila esvaziar (condição de parada: nº de pastas em `audios/` ≤ nº em
  `temp/`). Usa lockfile em `/tmp` para impedir instâncias concorrentes. Útil para
  retomar automaticamente após falhas/interrupções.

---

## Notas de design

- **Idempotência**: o histórico (`historico_dataset/`) permite parar e retomar a
  qualquer momento sem reprocessar nem duplicar segmentos. A presença de
  `historico_dataset/{audio_id}.json` é o que marca um áudio como concluído, e o
  `main.py` barra o áudio repetido na entrada, antes de qualquer módulo rodar.
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
