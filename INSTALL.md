# Instalação — ambiente do servidor (GPU)

Este documento reproduz o **ambiente de referência**: o env conda
`katube_final`, montado do zero e provado por comportamento (pipeline rodada
de ponta a ponta, 6 de 6 áudios processados, 0 erros).

Todas as versões abaixo são as do ambiente provado. Não troque nenhuma delas:
a pipeline depende de combinações específicas (`torch` 2.5.1 + `numpy` 1.26.4
+ `pyannote.audio` 3.3.2, entre outras).

---

## 1. Pré-requisitos de sistema

`ffmpeg`, `ffprobe` e `sox` vêm do **sistema operacional (apt)**, **não do
conda**. No ambiente de referência estão em `/usr/bin`. Sem eles a pipeline
quebra no M02 (conversão para WAV) e no M13 (normalização de áudio).

| Item | Versão no ambiente de referência |
|------|----------------------------------|
| Sistema operacional | Ubuntu 24.04.3 LTS |
| Python | 3.10.19 |
| ffmpeg | 6.1.1-3ubuntu5 (`/usr/bin/ffmpeg`) |
| ffprobe | 6.1.1-3ubuntu5 (`/usr/bin/ffprobe`) |
| sox | SoX v14.4.2 (`/usr/bin/sox`) |
| Driver NVIDIA | 580.95.05 |
| GPU | NVIDIA A10G |

Confira o que está instalado antes de seguir:

```bash
which ffmpeg ffprobe sox
ffmpeg -version | head -1
sox --version
nvidia-smi --query-gpu=driver_version,name --format=csv
```

## 2. Conta HuggingFace

O M07 (detecção de overlap) usa `pyannote/speaker-diarization-3.1`, que é um
modelo **restrito**. São necessárias duas coisas, e a falta de qualquer uma
delas resulta em erro:

1. Um **token de acesso Read-Only**, criado em
   https://huggingface.co/settings/tokens
2. O **aceite dos termos de uso** do modelo de diarização e dos modelos que
   ele carrega, feito na página de cada modelo enquanto logado.

Sem o aceite, o download é recusado com **HTTP 401 mesmo com token válido**.

## 3. Criar o ambiente conda

Do conda vem **somente o Python**. Nenhuma biblioteca do projeto é instalada
por ele.

```bash
conda create -n katube_final python=3.10.19
conda activate katube_final
```

## 4. Instalar as dependências

Uma **única** chamada, para que o resolvedor decida tudo de uma vez:

```bash
pip install -r requirements-servidor.txt
```

Não instale em partes, não reordene o arquivo e não desafixe versão.

O arquivo inclui `setuptools==80.9.0` e `wheel==0.45.1` no topo. Eles não
aparecem em `pip freeze`, mas são necessários: sem eles o `lightning_fabric`
falha com `ModuleNotFoundError: No module named 'pkg_resources'`.

## 5. Configurar o `.env`

O `.env` fica na raiz do projeto, **nunca é versionado** e é lido por
`main.py`, `src/m01_load_models.py` e `src/m07_overlap1.py`.

```bash
cp .env.example .env
```

Edite e preencha o `HF_TOKEN` com o token da seção 2.

**Armadilha real, encontrada no servidor:** o valor **não pode estar entre
aspas**. O `python-dotenv` entrega as aspas junto com o token, e o
HuggingFace recusa a autenticação.

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx      # correto
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"    # errado - o token vai com aspas
```

## 6. Verificação

Os três comandos abaixo são os que de fato foram usados para provar o
ambiente.

**6.1 — Consistência das dependências.** Esperado: `No broken requirements
found.`

```bash
pip check
```

**6.2 — Comparação com o congelamento.** Esperado: **diff vazio** (nenhuma
saída, código de retorno 0).

```bash
pip freeze | sort -f > /tmp/freeze_atual.txt
grep -v '^#' requirements-servidor.txt | grep -viE '^(setuptools|wheel)==' | sort -f > /tmp/freeze_esperado.txt
diff /tmp/freeze_atual.txt /tmp/freeze_esperado.txt
```

`setuptools` e `wheel` são excluídos da comparação porque o `pip freeze` não
os lista.

**6.3 — Versões carregadas e acesso à GPU.**

```bash
python -c "import torch, numpy, transformers, pyannote.audio; print('torch', torch.__version__); print('cuda disponivel', torch.cuda.is_available()); print('cuda torch', torch.version.cuda); print('numpy', numpy.__version__); print('transformers', transformers.__version__); print('pyannote.audio', pyannote.audio.__version__)"
```

Esperado no ambiente de referência:

```
torch 2.5.1
cuda disponivel True
cuda torch 12.4
numpy 1.26.4
transformers 4.44.2
pyannote.audio 3.3.2
```

O `cuda torch` deve bater com as bibliotecas `nvidia-*-cu12==12.4.*` fixadas
no `requirements-servidor.txt`. O `cuda disponivel False` no servidor indica
driver ou GPU indisponível — a pipeline até roda, mas em CPU e muito mais
lenta.

## 7. Problemas comuns

Apenas os que ocorreram de verdade durante a montagem do ambiente.

| Sintoma | Causa | Correção |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'pkg_resources'` (no `lightning_fabric`) | `setuptools`/`wheel` ausentes; o `pip freeze` os omite | instalar as duas linhas do topo do `requirements-servidor.txt` |
| Token recusado, mesmo estando correto | valor do `HF_TOKEN` entre aspas no `.env` | remover as aspas |
| `401` ao baixar o modelo de diarização | termos de uso do pyannote não aceitos na conta | aceitar os termos na página do modelo, logado |
| Falha no M02 ou no M13 | `ffmpeg`, `ffprobe` ou `sox` ausentes no sistema | instalar pelo apt (seção 1) — não pelo conda |

## 8. Variação CPU

Existe um ambiente local, **só para desenvolvimento em CPU**, com versões
**divergentes** das listadas aqui.

Ele **não é reproduzido por este documento** e **a validação local não prova
o comportamento do servidor**. Qualquer alteração validada em CPU precisa ser
revalidada no ambiente de referência antes de ser considerada boa.
