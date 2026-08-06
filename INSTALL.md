# Installation — server environment (GPU)

This document reproduces the **reference environment**: the `katube_final`
conda env, set up from scratch and proven by behavior (pipeline run
end-to-end, 6 out of 6 audio files processed, 0 errors).

All the versions below are the ones from the proven environment. Do not
swap any of them: the pipeline depends on specific combinations (`torch`
2.5.1 + `numpy` 1.26.4 + `pyannote.audio` 3.3.2, among others).

---

## 1. System prerequisites

`ffmpeg`, `ffprobe` and `sox` come from the **operating system (apt)**,
**not from conda**. In the reference environment they are in `/usr/bin`.
Without them the pipeline breaks in M02 (WAV conversion) and in M13 (audio
normalization).

| Item | Version in the reference environment |
|------|----------------------------------|
| Operating system | Ubuntu 24.04.3 LTS |
| Python | 3.10.19 |
| ffmpeg | 6.1.1-3ubuntu5 (`/usr/bin/ffmpeg`) |
| ffprobe | 6.1.1-3ubuntu5 (`/usr/bin/ffprobe`) |
| sox | SoX v14.4.2 (`/usr/bin/sox`) |
| NVIDIA driver | 580.95.05 |
| GPU | NVIDIA A10G |

Check what is installed before proceeding:

```bash
which ffmpeg ffprobe sox
ffmpeg -version | head -1
sox --version
nvidia-smi --query-gpu=driver_version,name --format=csv
```

## 2. HuggingFace account

M07 (overlap detection) uses `pyannote/speaker-diarization-3.1`, which is
a **gated** model. Two things are required, and missing either results in
an error:

1. A **Read-Only access token**, created at
   https://huggingface.co/settings/tokens
2. **Acceptance of the terms of use** of the diarization model and of the
   models it loads, done on each model's page while logged in.

Without acceptance, the download is refused with **HTTP 401 even with a
valid token**.

## 3. Create the conda environment

Conda provides **only Python**. No project library is installed by it.

```bash
conda create -n katube_final python=3.10.19
conda activate katube_final
```

## 4. Install the dependencies

A **single** call, so the resolver decides everything at once:

```bash
pip install -r requirements-servidor.txt
```

Do not install in parts, do not reorder the file and do not unpin a
version.

The file includes `setuptools==80.9.0` and `wheel==0.45.1` at the top.
They do not appear in `pip freeze`, but they are required: without them
`lightning_fabric` fails with `ModuleNotFoundError: No module named
'pkg_resources'`.

## 5. Configure the `.env`

The `.env` sits at the project root, **is never versioned** and is read
by `main.py`, `src/m01_load_models.py` and `src/m07_overlap1.py`.

```bash
cp .env.example .env
```

Edit it and fill in `HF_TOKEN` with the token from section 2.

**Real pitfall, found on the server:** the value **cannot be in
quotes**. `python-dotenv` passes the quotes along with the token, and
HuggingFace refuses authentication.

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx      # correct
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"    # wrong - the token goes with quotes
```

## 6. Verification

The three commands below are the ones actually used to prove the
environment.

**6.1 — Dependency consistency.** Expected: `No broken requirements
found.`

```bash
pip check
```

**6.2 — Comparison against the freeze.** Expected: **empty diff** (no
output, exit code 0).

```bash
pip freeze | sort -f > /tmp/freeze_atual.txt
grep -v '^#' requirements-servidor.txt | grep -viE '^(setuptools|wheel)==' | sort -f > /tmp/freeze_esperado.txt
diff /tmp/freeze_atual.txt /tmp/freeze_esperado.txt
```

`setuptools` and `wheel` are excluded from the comparison because `pip
freeze` does not list them.

**6.3 — Loaded versions and GPU access.**

```bash
python -c "import torch, numpy, transformers, pyannote.audio; print('torch', torch.__version__); print('cuda disponivel', torch.cuda.is_available()); print('cuda torch', torch.version.cuda); print('numpy', numpy.__version__); print('transformers', transformers.__version__); print('pyannote.audio', pyannote.audio.__version__)"
```

Expected in the reference environment:

```
torch 2.5.1
cuda disponivel True
cuda torch 12.4
numpy 1.26.4
transformers 4.44.2
pyannote.audio 3.3.2
```

`cuda torch` must match the `nvidia-*-cu12==12.4.*` libraries pinned in
`requirements-servidor.txt`. `cuda disponivel False` on the server
indicates an unavailable driver or GPU — the pipeline still runs, but on
CPU and much slower.

## 7. Common problems

Only the ones that actually occurred while setting up the environment.

| Symptom | Cause | Fix |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'pkg_resources'` (in `lightning_fabric`) | `setuptools`/`wheel` missing; `pip freeze` omits them | install the two lines at the top of `requirements-servidor.txt` |
| Token refused, even though it's correct | `HF_TOKEN` value in quotes in the `.env` | remove the quotes |
| `401` when downloading the diarization model | pyannote terms of use not accepted on the account | accept the terms on the model's page, while logged in |
| Failure in M02 or M13 | `ffmpeg`, `ffprobe` or `sox` missing on the system | install via apt (section 1) — not via conda |

## 8. CPU variant

There is a local environment, **for CPU development only**, with
versions that **differ** from the ones listed here.

It **is not reproduced by this document** and **local validation does
not prove server behavior**. Any change validated on CPU needs to be
revalidated in the reference environment before being considered good.
