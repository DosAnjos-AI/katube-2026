# katube2026

Pipeline automatizado para download de áudios do YouTube e geração de datasets de alta qualidade para treinamento de modelos TTS/STT em Português Brasileiro.

Desenvolvido no âmbito do **Projeto CEIA / Alcateia** (CEIA/UFG).

Comparativo Katube2025 vs Katube2026: https://www.notion.so/Alcateia-Katube-30ec77fe3fa08054ae70c3493eed2817

---

## Pipeline

O pipeline possui dois modos de segmentação, configuráveis via `config.py`. Após a segmentação, o fluxo é idêntico para ambos.

**Modo VAD** — segmentação automática por detecção de voz (sem legenda):
```
Áudio → Segmentação (VAD) → MOS → Overlap → Whisper + wav2vec → Normalização Texto
→ Similaridade (whisper vs wav2vec) → Denoiser (opcional) → Normalização Áudio (SOX) → Dataset
```

**Modo LEG** — segmentação guiada por legenda (requer `.txt`):
```
Áudio + Legenda → Segmentação (LEG) → MOS → Overlap → Whisper + wav2vec → Normalização Texto
→ Similaridade média (whisper vs wav2vec, whisper vs leg, wav2vec vs leg) → Denoiser (opcional) → Normalização Áudio (SOX) → Dataset
```

**Filtros aplicados:**
- MOS < 2.5 → segmento rejeitado
- Overlap detectado (2+ vozes) → segmento rejeitado
- Similaridade < 0.80 → segmento rejeitado
- Denoiser aplicado apenas em segmentos com MOS classificado como `media` (configurável)

---

## Estrutura do Projeto

```
katube2026/
├── arquivos/
│   ├── audios/                  # áudios brutos a processar (ver seção "Como usar")
│   ├── links_download/          # CSVs de links, logs e histórico do downloader
│   └── temp/                    # arquivos temporários durante processamento
│
├── dataset/
│   ├── audio_dataset/           # segmentos aprovados: {id}/{id}_001.flac ...
│   ├── historico_dataset/       # metadados por vídeo: {id}.json (não apagar)
│   ├── log/                     # logs de processamento por vídeo: {id}.log
│   ├── dataset.csv              # dataset consolidado (gerado automaticamente)
│   └── sumary_results.py        # estatísticas e métricas do CSV gerado
│
├── src/                         # módulos do pipeline (m00 a m15)
├── config.py                    # configuração de módulos e parâmetros
└── main.py                      # ponto de entrada
```

> **Importante:** nunca apague o conteúdo de `dataset/historico_dataset/`. Ele é usado como controle de vídeos já processados — sem ele, o pipeline reprocessará os mesmos IDs.

---

## Como Usar

### 1. Preparar os áudios

Coloque os arquivos na seguinte estrutura dentro de `arquivos/audios/`:

```
arquivos/audios/
└── {id}/
    ├── {id}.flac        # áudio (formatos suportados: flac, wav, mp3, m4a, opus)
    └── {id}.txt         # legenda — obrigatória apenas no modo LEG
```

Cada vídeo deve ter sua própria pasta nomeada com o `{id}`.

### 2. Configurar

Edite o `config.py`. O ponto de partida é o bloco `MASTER`:

```python
MASTER = {
    'downloader':          False,     # baixar áudios via yt-dlp
    'segmentacao':         'vad',     # 'vad', 'legenda' ou '' (já segmentado)
    'mos_filter':          True,
    'overlap':             True,
    'transcricao_whisper': True,
    'transcricao_wav2vec': True,
    'Denoiser':            True,
    'cleanup':             'all',     # 'all', 'input', 'temp', 'none'
}
```

Cada módulo possui seu próprio bloco de configuração no mesmo arquivo com parâmetros detalhados e documentados.

### 3. Executar

```bash
python main.py
```

O `dataset.csv` é gerado e atualizado automaticamente em `dataset/` a cada vídeo concluído.

---

## Dataset Gerado

Cada linha do `dataset.csv` representa um segmento de áudio aprovado.

| Coluna | Descrição |
|---|---|
| `arquivo_nome` / `caminho` | identificação e caminho do segmento |
| `tempo_inicio` / `tempo_fim` / `duracao` | posição temporal no áudio original |
| `texto` | transcrição final validada |
| `legenda` | texto da legenda original (vazio no modo VAD) |
| `vad` | `True` se segmentado via VAD |
| `mos_score` / `mos_qualidade` | nota de qualidade SQUIM (`baixa`, `media`, `alta`) |
| `overlap01` | `True` se overlap detectado (segmento rejeitado) |
| `stt_whisper` / `stt_wav2vec` | transcrições brutas de cada modelo |
| `sim_leg_whisper` / `sim_leg_wav2vec` / `sim_whisper_wav2vec` | similaridade por par (vazio no modo VAD exceto whisper vs wav2vec) |
| `nota_similaridade` | média final de similaridade |
| `status_similaridade` | `aprovado` ou `rejeitado` |
| `metrica_similaridade` | métrica usada: `wer`, `cer` ou `levenshtein_norm` |
| `utilizou_denoiser` | `True` se passou pelo DeepFilterNet3 |
| `sox_*` | parâmetros de normalização aplicados pelo SOX |

---

## Modelos Utilizados

| Módulo | Modelo |
|---|---|
| MOS / Qualidade | `torch.hub` SQUIM |
| Overlap | `pyannote/speaker-diarization-3.1` (requer token HuggingFace) |
| STT 01 | `freds0/distil-whisper-large-v3-ptbr` |
| STT 02 | `lgris/wav2vec2-large-xlsr-open-brazilian-portuguese` |
| Denoiser | DeepFilterNet3 |
