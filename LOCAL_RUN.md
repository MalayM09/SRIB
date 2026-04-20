# LOCAL_RUN.md — Ground Truth for Local Setup & Pilot Smoke-Test

**Machine:** MacBook Air M1, 20–30 GB free disk budget
**Goal:** Stand up the repo, install deps, pull minimum data, scaffold the model + training loop, run a **2-epoch CPU/MPS smoke-test** on Speech Commands V2, then push to GitHub so Kaggle can clone. No full training here — real runs happen on Kaggle (see KAGGLE_RUN.md).

This file is **authoritative**. Any deviation from it must be reflected back here before the code is changed.

---

## 0. Storage Budget (hard cap 25 GB)

| Artifact                              | Size   | Where                                  |
|---------------------------------------|--------|----------------------------------------|
| Speech Commands V2                    | 2.3 GB | `data/speech_commands_v2/`             |
| MUSAN subset (noise + music, 1 h)     | ~1 GB  | `data/musan_small/`                    |
| OpenSLR Small Room RIRs (subset)      | ~0.4 GB| `data/rirs_small/`                     |
| Python venv + PyTorch wheels          | ~4 GB  | `.venv/`                               |
| Code + checkpoints (smoke-test only)  | ~0.5 GB| repo                                   |
| **Total**                             | ~8 GB  | buffer: 12+ GB                         |

Full MUSAN (11 GB), full RIRs (25 GB), VoxCeleb1 (30 GB), teacher weights → **Kaggle only**. Do not download them locally.

---

## 1. System Prerequisites

```bash
# Xcode CLT (for compiling soundfile/sox bindings if needed)
xcode-select --install 2>/dev/null || true

# Homebrew deps
brew install python@3.11 ffmpeg sox git-lfs
git lfs install
```

Verify:
```bash
python3.11 --version   # 3.11.x
ffmpeg -version | head -1
sox --version
```

---

## 2. Repository Scaffold

```bash
cd /Users/malaymishra/Desktop/srib
mkdir -p code && cd code
git init
```

Create the following tree (empty files are fine for now — we fill them in Section 5):

```
code/
├── README.md
├── pyproject.toml
├── .gitignore
├── .python-version                  # "3.11"
├── configs/
│   ├── pilot_smoke.yaml             # 2-epoch local verification
│   ├── pilot_p0_baseline.yaml       # Kaggle P0
│   ├── pilot_p1_pcen_ablation.yaml  # Kaggle P1
│   └── pilot_p2_film_toy.yaml       # Kaggle P2 (stretch)
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── bc_resnet.py             # BC-ResNet-8 trunk
│   │   ├── pcen.py                  # Learnable PCEN
│   │   └── heads.py                 # KWS + SV heads (stub until Kaggle)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── speech_commands.py       # SC V2 loader + splits
│   │   └── augment.py               # SpecAugment + RIR + MUSAN mixer
│   ├── losses/
│   │   ├── __init__.py
│   │   └── focal.py                 # Focal CE (KWS)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   └── logging.py
│   └── train.py                     # single-loop trainer (no KD yet)
├── scripts/
│   ├── download_sc_v2.sh
│   ├── download_musan_small.sh
│   └── download_rirs_small.sh
├── notebooks/
│   └── 00_local_smoke.ipynb         # sanity-check forward pass
└── tests/
    └── test_shapes.py               # pytest: model in/out shapes
```

### `.gitignore` (must include)
```
.venv/
data/
__pycache__/
*.pt
*.ckpt
runs/
wandb/
.DS_Store
.ipynb_checkpoints/
```

### `.python-version`
```
3.11
```

---

## 3. Python Environment

```bash
cd /Users/malaymishra/Desktop/srib/code
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
```

### `pyproject.toml` (minimal local subset — full deps on Kaggle)
```toml
[project]
name = "srib-edge-kws-sv"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "torch==2.3.1",
  "torchaudio==2.3.1",
  "numpy==1.26.4",
  "soundfile==0.12.1",
  "librosa==0.10.2.post1",
  "pyyaml==6.0.2",
  "tqdm==4.66.4",
  "einops==0.8.0",
  "pytest==8.3.2",
  "matplotlib==3.9.2",
  "tensorboard==2.17.1",
]
```

Install:
```bash
pip install -e .
```

Verify MPS:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expect: MPS available: True
```

> **Note:** PyTorch MPS has partial op coverage. We use MPS when ops are supported; fall back to CPU otherwise via `PYTORCH_ENABLE_MPS_FALLBACK=1`.

---

## 4. Data Downloads

### 4.1 Speech Commands V2 (2.3 GB)
`scripts/download_sc_v2.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
DEST=../data/speech_commands_v2
mkdir -p "$DEST" && cd "$DEST"
curl -L -O http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz
rm speech_commands_v0.02.tar.gz
echo "SC V2 unpacked → $(pwd)"
```

Official splits live in `validation_list.txt` and `testing_list.txt` inside the tarball — we re-use those. `src/data/speech_commands.py` reads them.

### 4.2 MUSAN subset (~1 GB)
Full MUSAN is 11 GB. For local, pull only `noise/` + 1 hour of `music/`:
```bash
#!/usr/bin/env bash
set -euo pipefail
DEST=../data/musan_small
mkdir -p "$DEST" && cd "$DEST"
curl -L -O https://www.openslr.org/resources/17/musan.tar.gz
tar -xzf musan.tar.gz --strip-components=1 musan/noise
# Grab first ~60 music files for local smoke test
mkdir -p music
tar -xzf musan.tar.gz --wildcards 'musan/music/*' -C .
find music -type f -name '*.wav' | tail -n +61 | xargs rm -f
rm musan.tar.gz
```

### 4.3 RIRs subset (~400 MB)
```bash
#!/usr/bin/env bash
set -euo pipefail
DEST=../data/rirs_small
mkdir -p "$DEST" && cd "$DEST"
curl -L -O https://www.openslr.org/resources/28/rirs_noises.zip
unzip -q rirs_noises.zip 'RIRS_NOISES/simulated_rirs/smallroom/*'
rm rirs_noises.zip
```

Run all three:
```bash
chmod +x scripts/*.sh
bash scripts/download_sc_v2.sh
bash scripts/download_musan_small.sh
bash scripts/download_rirs_small.sh
du -sh ../data/*
```

Expected: ~3.7 GB total under `data/`.

---

## 5. Source Files — What Each Must Contain

> Keep each file under ~200 LOC for the pilot. Full-featured versions (KD, FiLM, TRM, MQMHA, AAM-Softmax, FAR-in-loop) are added in Phase 2 on Kaggle.

### 5.1 `src/model/pcen.py`
- Implements **learnable PCEN** (Wang et al. 2017) as an `nn.Module` with per-channel parameters `α, δ, r, s` (all trainable, softplus-reparameterised).
- Input: mel-spectrogram `(B, F, T)`. Output: same shape, PCEN-compressed.
- Must expose a `stateful_step()` path for streaming (not exercised locally, stubbed).

### 5.2 `src/model/bc_resnet.py`
- BC-ResNet-8 per Kim et al. 2021 (Broadcasted Residual Learning), Qualcomm reference layout.
- Inverted-residual blocks (2× expansion: 1×1 expand → 3×1 DW freq + SubSpectralNorm → freq-pool → 1×3 DW time + BN/SiLU → 1×1 contract).
- Depth config: `[2, 2, 4, 4]` blocks, channels `[40, 60, 80, 100]`, freq strides `[2,2,2,1]`, time dilations `[1,2,4,8]`, SubSpectralNorm groups = 5.
- Forward: `(B, 1, F=40, T=98)` → trunk feature map `(B, 100, F', T)`.
- Parameter count: **322.6K trunk, 324.0K total** — matches paper's τ=8 target (~321K). Our inverted-residual block is ~5× cheaper per block than the paper's BC block, so channels are widened from the paper's `[16,24,32,40]` to keep the same param budget. Test asserts 250K ≤ trunk ≤ 400K and total < 3M (competition cap).

### 5.3 `src/model/heads.py` (stub for local)
- `KWSHead`: global avg-pool → linear → `num_classes=12` (10 commands + `_unknown_` + `_silence_`).
- `SVHead`: stub returning zeros of shape `(B, 192)` — real AAM-Softmax head is added on Kaggle.

### 5.4 `src/data/speech_commands.py`
- Reads the official `validation_list.txt` / `testing_list.txt`.
- 10-word V2 subset (yes, no, up, down, left, right, on, off, stop, go) + `_unknown_` + `_silence_`.
- `__getitem__` returns `(waveform_16k_1sec, label)`.
- Pads/truncates to 16000 samples.

### 5.5 `src/data/augment.py`
- `SpecAugment(F=10, T=20, num_f=1, num_t=2)`.
- `RIRConvolver`: samples a RIR from `data/rirs_small`, FFT convolves, normalises.
- `MUSANMixer`: samples noise, scales to target SNR ∈ U(-5, 30) dB.
- Waveform-domain pipeline for local (we add log-Mel + PCEN inside the model).

### 5.6 `src/losses/focal.py`
- Standard focal CE (γ=2.0) for KWS class imbalance (`_silence_` / `_unknown_`).

### 5.7 `src/train.py`
- Reads YAML config.
- Builds model = `MelFrontend` (torchaudio MelSpec → optional learnable PCEN) + `BCResNet8` + `KWSHead`. SV path idle locally.
- Optim: AdamW (lr from config), cosine schedule, weight_decay=1e-4.
- **Local smoke-test config (`configs/pilot_smoke.yaml`):** 4 epochs, batch 128, 1 worker, `device=mps`, subset = 5000 training clips, full val set, RIR/MUSAN off (kept off locally — they slow dataloading on M1 with negligible benefit at this scale; full augmentation is enabled in the Kaggle configs).
- Logs: stdout + `runs/smoke/metrics.json`. Saves `best.pt` (highest val acc) and `final.pt`.

### 5.8 `tests/test_shapes.py`
- `test_pcen_shape`: PCEN preserves `(B, F, T)`.
- `test_pcen_gradients`: backward through PCEN produces finite grads on every parameter.
- `test_bcresnet_forward_and_param_count`: dummy `(2, 1, 40, 98)` → asserts trunk output shape and `250_000 ≤ params ≤ 400_000` (paper τ=8 band).
- `test_kws_head_end_to_end`: trunk + KWS head → `(B, 12)`.
- `test_sv_head_stub_shape`: trunk + SV stub → `(B, 192)`.
- `test_total_param_budget_under_cap`: trunk + head < 3M params (competition cap).

Run:
```bash
pytest -q
```

All six must pass before continuing.

---

## 6. Config — `configs/pilot_smoke.yaml`

```yaml
run:
  name: smoke
  seed: 1337
  device: mps           # auto-falls back to cpu for unsupported ops
  out_dir: runs/smoke

data:
  root: ../data/speech_commands_v2
  subset_train: 5000     # smoke only
  batch_size: 128
  num_workers: 1
  sample_rate: 16000

model:
  trunk: bc_resnet8
  use_pcen: true         # learnable PCEN
  n_mels: 40
  num_classes: 12

optim:
  lr: 4.0e-3
  weight_decay: 1.0e-4
  epochs: 4
  scheduler: cosine

augment:
  specaugment: true
  rir_prob: 0.0          # off for smoke — speeds up dataloading on M1
  musan_prob: 0.0
  snr_range: [0, 20]
```

> Earlier draft used 2 epochs / batch 64 / lr 1e-3 / RIR+MUSAN on. That version learned but plateaued near chance in 2 epochs on M1 — the gates below were unreachable at that compute budget. The values above are empirically calibrated on this machine: `subset_train=5000`, `batch_size=128`, `lr=4e-3`, `epochs=4`, augmentation off — total wall ~3 min on MPS, comfortably clears the gate.

---

## 7. Verification Protocol

Exactly these commands, in order. **Do not** skip or reorder.

```bash
cd /Users/malaymishra/Desktop/srib/code
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

# (a) unit tests
pytest -q
# PASS: 6 passed

# (b) smoke training (expect ~6-13 minutes on M1 MPS, depends on bg load)
python -m src.train --config configs/pilot_smoke.yaml
# PASS criteria:
#   - epoch 1 val acc ≥ 60%
#   - epoch 2 val acc ≥ 65%
#   - epoch 4 val acc ≥ 80%
#   - monotonically-decreasing train loss, no NaNs, no crashes
#   - checkpoints written to runs/smoke/{best.pt, final.pt} + metrics.json
# Reference run on M1 (2026-04, 322.6K trunk): 0.675 → 0.709 → 0.859 → 0.863
# (best 0.8634).

# (c) quick inference sanity
python -c "
import torch
from src.model.bc_resnet import BCResNet8
from src.model.heads import KWSHead
m = BCResNet8(); h = KWSHead(40, 12)
x = torch.randn(1, 1, 40, 98)
print('logits:', h(m(x)).shape)    # expect (1, 12)
"
```

If (b) fails the accuracy gate, **stop and debug** before pushing. Common causes: wrong SpecAugment axis, PCEN NaN (check softplus init), label mapping bug.

---

## 8. GitHub Push (for Kaggle clone)

```bash
cd /Users/malaymishra/Desktop/srib/code
git add -A
git commit -m "pilot scaffold: BC-ResNet-8 + PCEN + SC V2 loader, smoke-test green"

# Create the repo on github.com first, then:
git remote add origin git@github.com:<your-user>/srib-edge-kws-sv.git
git branch -M main
git push -u origin main
```

Kaggle will clone from this repo (see KAGGLE_RUN.md §3).

---

## 9. Exit Criteria (Local Phase)

All of the following **must** be true before moving to Kaggle:

- [ ] `pytest -q` → 6 passed
- [ ] `pilot_smoke.yaml` run: val acc epoch-2 ≥ 65%, epoch-4 ≥ 80% (clean SC V2, 5k-sample subset)
- [ ] `runs/smoke/{best.pt, final.pt, metrics.json}` exist; `final.pt` loads and forward pass OK
- [ ] `du -sh ../data ../code/.venv` shows < 12 GB total
- [ ] GitHub remote has `main` branch with the scaffold pushed
- [ ] No `__pycache__/` / `*.pt` / `data/` tracked in git

When every box is ticked, open KAGGLE_RUN.md.

---

## 10. Known Local Limitations (document these in the blueprint)

- MPS lacks a stable `torch.fft` for long RIRs → we FFT-convolve on CPU in the DataLoader worker.
- Full MUSAN / VoxCeleb1 / teacher weights are **not** reachable locally; KD experiments cannot run on M1.
- Full 30-epoch SC V2 training takes ~8 h on M1 CPU+MPS vs. ~45 min on Kaggle T4 ×2 — this is the reason training is deferred to Kaggle.
