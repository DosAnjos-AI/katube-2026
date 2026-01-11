# Analisador Espectral de Áudio

Script independente para detectar upsampling artificial em arquivos de áudio.

## 📋 Requisitos

### Instalação de dependências:

```bash
pip install librosa numpy matplotlib soundfile scipy
```

## ⚙️ Configuração

Edite as configurações no início do arquivo `main.py`:

```python
# Caminhos (relativo ou absoluto)
INPUT_DIR = "audios"  # Pasta com seus áudios
OUTPUT_DIR = "analise_espectral"  # Onde salvar resultados

# Parâmetros de análise
ENERGY_THRESHOLD_DB = -60  # Threshold para detectar corte

# Visualização
SHOW_PLOTS = True  # Mostrar gráficos na tela
SAVE_PLOTS = True  # Salvar PNG
PAUSE_BETWEEN_FILES = False  # Pausar entre arquivos
DPI = 150  # Qualidade das imagens
```

## 🚀 Uso

### Execução simples:

```bash
python main.py
```

### Estrutura de pastas:

**Antes:**
```
projeto/
├── main.py
└── audios/
    ├── audio1.flac
    ├── audio2.mp3
    └── audio3.wav
```

**Depois:**
```
projeto/
├── main.py
├── audios/
└── analise_espectral/
    ├── audio1_analise/
    │   ├── espectrograma.png
    │   ├── espectro_frequencia.png
    │   ├── analise_bandas.png
    │   └── resultado.json
    ├── audio2_analise/
    │   └── ...
    └── resumo_geral.json
```

## 📊 Interpretação dos Resultados

### JSON (resultado.json):

```json
{
  "arquivo": "audio1.flac",
  "sample_rate_declarado": 48000,
  "sample_rate_efetivo": 24000,
  "frequencia_corte_hz": 12000,
  "status": "upsampled",  // ou "real"
  "qualidade_real": "24kHz",
  "canais": 2,
  "duracao_segundos": 351.2,
  "energia_por_banda": {
    "0-8kHz": 0.92,
    "8-16kHz": 0.45,
    "16-24kHz": 0.02
  }
}
```

### Gráficos:

1. **espectrograma.png**: Visualização tempo x frequência (similar Audacity)
2. **espectro_frequencia.png**: Mostra onde está o corte de frequência
3. **analise_bandas.png**: Energia por faixa de frequência (barras)

### Status:

- **"real"**: Áudio tem qualidade correspondente ao sample rate declarado
- **"upsampled"**: Áudio foi artificialmente expandido (interpolação)

### Exemplo de Upsampling:

```
Sample Rate Declarado: 48000 Hz (48 kHz)
Sample Rate Efetivo: 24000 Hz (24 kHz)
Frequência de Corte: 12000 Hz
Status: UPSAMPLED
Qualidade Real: 24kHz
```

**Significado**: Arquivo está em 48 kHz mas só tem frequências até 12 kHz (qualidade real de 24 kHz).

## 🔧 Ajustes Comuns

### Mudar threshold de detecção:

```python
ENERGY_THRESHOLD_DB = -70  # Mais rigoroso
ENERGY_THRESHOLD_DB = -50  # Menos rigoroso
```

### Não mostrar gráficos (apenas salvar):

```python
SHOW_PLOTS = False
```

### Pausar entre arquivos:

```python
PAUSE_BETWEEN_FILES = True  # Aguarda fechar janela antes de continuar
```

## 📝 Formatos Suportados

- `.flac`
- `.mp3`
- `.wav`

## 🐛 Troubleshooting

### "Pasta de entrada não encontrada":
- Verifique o caminho em `INPUT_DIR`
- Use caminho absoluto se relativo não funcionar

### "Nenhum arquivo encontrado":
- Certifique-se que os áudios estão na **raiz** da pasta (não em subpastas)
- Verifique se as extensões são `.flac`, `.mp3` ou `.wav`

### Gráficos não aparecem:
- Configure `SHOW_PLOTS = True`
- Em ambientes sem interface gráfica (servidor), use apenas `SAVE_PLOTS = True`

## 📈 Exemplo de Saída no Terminal

```
==============================================================
ANALISADOR ESPECTRAL DE ÁUDIO
Detecção de Upsampling Artificial
==============================================================

Configuração:
  Pasta de entrada: /home/user/audios
  Pasta de saída: /home/user/analise_espectral
  Threshold de energia: -60 dB
  Formatos suportados: ['.flac', '.mp3', '.wav']

Encontrados 3 arquivos de áudio

[1/3] Processando: audio1.flac
============================================================
Analisando: audio1.flac
============================================================
Áudio carregado: audio1.flac
  Sample rate: 48000 Hz
  Canais: 2
  Duração: 351.23s
Detectando sample rate efetivo...
Gerando visualizações...
  Espectrograma salvo: espectrograma.png
  Espectro salvo: espectro_frequencia.png
  Análise de bandas salva: analise_bandas.png
  Resultado JSON salvo: resultado.json

Resultado:
  Sample Rate Declarado: 48000 Hz
  Sample Rate Efetivo: 24000 Hz
  Frequência de Corte: 12000 Hz
  Status: UPSAMPLED
  Qualidade Real: 24kHz

============================================================
RESUMO GERAL
============================================================
Total processado: 3 áudios
Tempo total: 45.32s

Status:
  Real: 1
  Upsampled: 2

Resumo geral salvo: resumo_geral.json
Resultados em: /home/user/analise_espectral

============================================================
Processamento concluído!
============================================================
```
