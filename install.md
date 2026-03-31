# Instalação do Ambiente

## Requisitos

- Miniconda ou Anaconda
- CUDA 12.1 compatível com o driver da GPU
- Token HuggingFace com acesso ao modelo `pyannote/speaker-diarization-3.1`

---

## Passo a Passo

### 1. Criar o ambiente conda

```bash
conda env create -f environment.yml
```

Este comando cria o ambiente `katube-2026` com Python 3.10, PyTorch (CUDA 12.1), e as dependências científicas base.

> **Importante:** o PyTorch é instalado via canal `pytorch/nvidia`. Não instale torch via pip posteriormente — isso pode quebrar a compatibilidade com CUDA.

### 2. Ativar o ambiente

```bash
conda activate katube-2026
```

### 3. Instalar dependências pip

```bash
pip install -r requirements.txt
```

> O `requirements.txt` está organizado em blocos com ordem definida. Não reorganize as instalações manualmente.

### 4. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
cp .env.example .env
```

Edite o `.env` e preencha:

```
HF_TOKEN=seu_token_huggingface_aqui
```

O token é necessário para baixar o modelo `pyannote/speaker-diarization-3.1`.

---

## Verificação

Após a instalação, verifique se o ambiente está correto:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Saída esperada:
```
2.8.0+cu121
True
```

---

## Problemas Comuns

**torch.cuda.is_available() retorna False**
Verifique se o driver NVIDIA é compatível com CUDA 12.1:
```bash
nvidia-smi
```

**Erro ao carregar pyannote (401 Unauthorized)**
O token HuggingFace não está configurado ou não tem acesso ao modelo. Verifique o `.env` e confirme o aceite dos termos de uso do modelo em huggingface.co.

**Conflito de versão do numpy**
Certifique-se de que o numpy foi instalado via conda antes do pip (`numpy=1.26.4`). Não atualize o numpy isoladamente — numba e librosa dependem desta versão específica.

**DeepFilterNet falha ao inicializar**
Confirme que torch e torchaudio estão instalados e funcionais antes de instalar o deepfilternet. Reinstale na ordem correta se necessário:
```bash
pip uninstall deepfilternet deepfilterlib -y
pip install deepfilternet==0.5.6
```