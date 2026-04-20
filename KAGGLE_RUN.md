# KAGGLE_RUN.md — Ground Truth for Kaggle Pilot Training

**Platform:** Kaggle Notebooks, **Accelerator: GPU T4 ×2**, Python 3.10, 30 h/week quota
**Goal:** Produce three blueprint-ready result artifacts — **P0 baseline, P1 PCEN ablation, P2 FiLM toy** — by training the scaffolded code from `LOCAL_RUN.md` on real datasets with full augmentation. Outputs feed directly into `blueprint.tex` (§Results/Plots).

This file is **authoritative**. No Kaggle experiment runs without a config listed here.

---

## 0. Why Kaggle, Not Local

| Constraint             | Local (M1)                 | Kaggle (T4 ×2)                         |
|------------------------|----------------------------|----------------------------------------|
| Wall-clock for 30 ep SC| ~8 h                       | ~45 min                                |
| Disk                   | 20–30 GB cap               | 73 GB working + 20 GB /kaggle/input   |
| VRAM                   | unified 8 GB, partial ops  | 16 GB × 2, full PyTorch kernel support|
| Big datasets (MUSAN full, VoxCeleb1, WavLM, ECAPA) | infeasible | hosted as public Kaggle datasets |
| Session limit          | n/a                        | 12 h per notebook run                 |

We therefore run **no full training on M1**. M1 is only for smoke-tests and code iteration.

---

## 1. Prerequisites

- Kaggle account, phone-verified (required for GPU).
- Confirmed **GPU quota > 25 hours** remaining this week (Kaggle → Settings → GPU usage).
- GitHub repo from `LOCAL_RUN.md §8` pushed to `main`.
- Weights & Biases account (free tier) — we log metrics there for reproducibility; API key set as Kaggle Secret `WANDB_API_KEY`.

---

## 2. Kaggle Datasets to Attach (all public, all zero-cost)

| Purpose              | Kaggle dataset / URL                                                                     | Size    | Kind     |
|----------------------|------------------------------------------------------------------------------------------|---------|----------|
| Speech Commands V2   | `sylkaladin/speech-commands-v2` (folder-per-class layout, matches our loader; fallbacks: `yashdogra/speech-commands`, `bahraleloom/tensorflow-speech-commands` — verify file tree before attaching) | 2.3 GB  | public   |
| MUSAN noise          | `nhattruongdev/musan-noise` (noise subset only, ~5 GB — sufficient for KWS aug)          | ~5 GB   | public   |
| RIRs smallroom       | upload `rirs_small.tar.gz` as a **private** dataset (~100 MB compressed, smallroom only; no suitable public Kaggle mirror found) | ~0.1 GB | private |
| VoxCeleb1 test       | `parampreetsethi/voxceleb1-test` (P2 stretch only)                                       | 1.3 GB  | public   |
| WavLM-Base+ weights  | `pytorch/fairseq-wavlm-base-plus` HuggingFace mirror via `transformers` download         | 380 MB  | HF       |
| ECAPA-TDNN weights   | `speechbrain/spkrec-ecapa-voxceleb` via `speechbrain` (downloads in notebook)            | 80 MB   | HF       |

> **RIRs upload** (one-time, from the Mac):
> ```bash
> cd /Users/malaymishra/Desktop/srib/data && tar czf rirs_small.tar.gz rirs_small
> # Kaggle → "+ Create Dataset" → upload rirs_small.tar.gz → name it "rirs-small"
> ```
> Cell 4 below untars it into the working dir (Kaggle `/kaggle/input/` is read-only).

> **MUSAN layout check:** our [MUSANMixer](code/src/data/augment.py) globs `musan_dir/noise/*.wav` + `musan_dir/music/*.wav`. If the `musan-noise` dataset has a flat layout (no `noise/` subfolder), either adjust the symlink target or tweak `augment.py`'s patterns. Music subset missing is fine (graceful empty-list fallback).

> If any public mirror is unavailable on the day, fall back to uploading as a private Kaggle dataset. Log SHA256 in the notebook cell header.

Attach via **Add data → Datasets** on the right panel. They mount at `/kaggle/input/<slug>/`.

---

## 3. Notebook Bootstrap (common cell for every run)

Create a notebook per run: `P0_baseline.ipynb`, `P1_pcen_ablation.ipynb`, `P2_film_toy.ipynb`. Each starts with:

```python
# Cell 1 — environment
import os, sys, subprocess, pathlib, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")

# Cell 2 — clone repo at a pinned SHA (reproducibility)
REPO_SHA = "7b66680"
!git clone https://github.com/MalayM09/SRIB.git /kaggle/working/srib
%cd /kaggle/working/srib/code
!git -C /kaggle/working/srib checkout $REPO_SHA

# Cell 3 — install deps (Kaggle has torch; we add the rest)
!pip install -q einops==0.8.0 soundfile==0.12.1 librosa==0.10.2.post1 \
               pyyaml==6.0.2 tqdm==4.66.4 wandb==0.17.7 \
               speechbrain==1.0.0 transformers==4.44.2

# Cell 4 — attach datasets into ../data (repo root is /kaggle/working/srib,
# configs reference ../data from code/, matching the local layout)
!mkdir -p /kaggle/working/srib/data

# SC V2 (public, folder-per-class). Adjust the source path to match the actual
# slug you attached (e.g. /kaggle/input/speech-commands-v2 for sylkaladin's mirror).
!ln -s /kaggle/input/speech-commands-v2 /kaggle/working/srib/data/speech_commands_v2

# MUSAN noise subset (public) — symlink directly; confirm the dataset has a `noise/`
# subfolder, else change the target or adjust augment.py's glob pattern.
!ln -s /kaggle/input/musan-noise /kaggle/working/srib/data/musan

# RIRs smallroom (private, uploaded as tar.gz since /kaggle/input/ is read-only)
!tar xzf /kaggle/input/rirs-small/rirs_small.tar.gz -C /kaggle/working/srib/data/
# → creates /kaggle/working/srib/data/rirs_small

# Cell 5 — verify
import torch
print("CUDA:", torch.cuda.is_available(), "| n_gpu:", torch.cuda.device_count())
print("device 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

---

## 4. Run P0 — Baseline (MUST-HAVE for blueprint)

**Question answered:** Does our BC-ResNet-8 + PCEN scaffold reproduce near-SOTA on Speech Commands V2 clean?
**Why it matters for the blueprint:** Section "Evidence of feasibility" — a number we can cite.

### 4.1 Config — `configs/pilot_p0_baseline.yaml`
```yaml
run:
  name: p0_baseline
  seed: 1337
  device: cuda
  out_dir: /kaggle/working/runs/p0
  wandb_project: srib-edge-kws-sv

data:
  root: data/speech_commands_v2
  subset_train: null       # full set
  batch_size: 256
  num_workers: 4
  sample_rate: 16000

model:
  trunk: bc_resnet8
  use_pcen: true
  n_mels: 40
  num_classes: 12

optim:
  lr: 4.0e-3
  weight_decay: 1.0e-4
  epochs: 30
  scheduler: cosine
  warmup_epochs: 2

augment:
  specaugment: true
  rir_prob: 0.0            # clean baseline — RIRs OFF
  musan_prob: 0.0          # clean baseline — MUSAN OFF
```

### 4.2 Run
```python
!python -m src.train --config configs/pilot_p0_baseline.yaml --ddp 2
```

### 4.3 Expected
- Wall-clock: **45–60 min** on T4 ×2
- Top-1 val acc: **97.5–98.5%**
- Top-1 test acc: within ±0.3% of val
- Param count logged: **420K trunk + ~6K head ≈ 426K total** (far below the 3M cap)

### 4.4 Artifacts to save back into the blueprint
- `runs/p0/final.pt`
- `runs/p0/metrics.json` (val/test curves)
- `runs/p0/confusion_matrix.png`
- WandB run URL (paste into blueprint.tex under "Phase-1 evidence")

### 4.5 Fail gates
If val acc < 96.5%, **stop**. Most likely: PCEN init bad, label smoothing missing, or class weights off. Do not proceed to P1.

---

## 5. Run P1 — PCEN Ablation (MUST-HAVE for blueprint)

**Question answered:** Does learnable PCEN beat log-Mel under noise (the core claim in §Technical Details)?
**Why it matters:** This is the single plot that justifies PCEN's 0.2K extra params.

### 5.1 Config — `configs/pilot_p1_pcen_ablation.yaml`
Two runs, identical except `use_pcen`:

```yaml
run:
  name: p1_pcen_on            # second run: p1_logmel
  seed: 1337
  device: cuda
  out_dir: /kaggle/working/runs/p1_pcen_on
  wandb_project: srib-edge-kws-sv

data:
  root: data/speech_commands_v2
  batch_size: 256
  num_workers: 4
  sample_rate: 16000

model:
  trunk: bc_resnet8
  use_pcen: true              # flip to false for the log-Mel run
  n_mels: 40
  num_classes: 12

optim:
  lr: 4.0e-3
  weight_decay: 1.0e-4
  epochs: 25
  scheduler: cosine
  warmup_epochs: 2

augment:
  specaugment: true
  rir_prob: 0.5
  musan_prob: 0.7
  snr_range: [-5, 20]          # noisy regime — where PCEN should win
```

### 5.2 Run
```python
# Run 1: PCEN on
!python -m src.train --config configs/pilot_p1_pcen_ablation.yaml --ddp 2

# Run 2: log-Mel (edit config: use_pcen: false, name: p1_logmel, out_dir: ...p1_logmel)
!python -m src.train --config configs/pilot_p1_pcen_ablation.yaml \
        --override model.use_pcen=false run.name=p1_logmel run.out_dir=/kaggle/working/runs/p1_logmel \
        --ddp 2
```

### 5.3 Evaluation — slice by SNR band
Separate test-set eval at fixed SNR ∈ {-5, 0, 5, 10, 20, clean}. Produce one CSV:

| SNR (dB) | log-Mel acc | PCEN acc | Δ (pp) |
|----------|-------------|----------|--------|
| clean    | ~98.2       | ~98.0    | ~−0.2  |
| 20       | ~96         | ~96.5    | +0.5   |
| 10       | ~89         | ~92      | +3     |
| 5        | ~78         | ~84      | +6     |
| 0        | ~62         | ~72      | +10    |
| −5       | ~44         | ~58      | +14    |

Numbers above are **target bands** from Wang et al. 2017 + BC-ResNet literature — treat as sanity floor, not hard pass.

### 5.4 Wall-clock
- 25 epochs × 2 runs × ~30 min/run ≈ **2 h total**. Comfortably inside a 12 h Kaggle session.

### 5.5 Artifacts
- `runs/p1_pcen_on/metrics.json`, `runs/p1_logmel/metrics.json`
- `runs/p1_snr_sweep.png` — line plot, both curves
- Feeds directly into blueprint slide 7 (Results / Evidence).

### 5.6 Fail gates
If ΔPCEN − log-Mel at −5 dB is **< +5 pp**, PCEN is not pulling its weight — check: (a) α/δ/r/s actually trainable (grad norms > 0), (b) MUSAN mixer hitting the target SNR (log a histogram), (c) batch-norm epsilon not swallowing dynamic range.

---

## 6. Run P2 — FiLM Toy (STRETCH, only if P0 + P1 pass cleanly)

**Question answered:** Does injecting a speaker vector via FiLM into the trunk improve trigger-word detection for that specific speaker without retraining?
**Why it matters:** Plants a flag for the FiLM claim in the blueprint novelty section. We are **not** chasing final numbers — one convincing qualitative plot is enough.

### 6.1 Setup
- Use Speech Commands V2 speaker IDs (the filename prefix) to pseudo-enroll 5 speakers, 5 utterances each.
- Freeze P0 trunk. Add a tiny `nn.Linear(192, 2·C)` FiLM generator consuming a mean-pooled PCEN-trunk embedding from enrollment clips. Fine-tune FiLM params + head for 5 epochs on a 1-word subset ("yes").
- Compare EER/FRR on the target speaker vs. an un-conditioned (P0) baseline.

### 6.2 Config — `configs/pilot_p2_film_toy.yaml`
```yaml
run:
  name: p2_film_toy
  seed: 1337
  device: cuda
  out_dir: /kaggle/working/runs/p2
  wandb_project: srib-edge-kws-sv

data:
  root: data/speech_commands_v2
  trigger_word: yes
  enrolled_speakers: 5
  utterances_per_speaker: 5
  batch_size: 128
  num_workers: 4

model:
  trunk: bc_resnet8
  use_pcen: true
  load_from: /kaggle/working/runs/p0/final.pt
  freeze_trunk: true
  film: true
  film_dim: 192
  n_mels: 40

optim:
  lr: 1.0e-3
  weight_decay: 1.0e-4
  epochs: 5
  scheduler: cosine
```

### 6.3 Wall-clock
- ~1.5 h on T4 ×2.

### 6.4 Artifact
- `runs/p2/per_speaker_det.png` — DET curve per enrolled speaker vs. baseline.
- A single one-liner number for blueprint: "FiLM conditioning reduces FRR@1FA/hr from X% → Y% for enrolled speaker."

### 6.5 Stretch-only rule
If **GPU quota < 6 hours** remaining after P0+P1, **skip P2**. A missing P2 does not block the blueprint; a missing P0 or P1 does.

---

## 7. Checkpoint / Output Management

Kaggle notebooks expose `/kaggle/working/` as downloadable. After each run:

```python
# Tar the run dir for download
!tar -czf /kaggle/working/p0_bundle.tar.gz -C /kaggle/working/runs p0
```

Download via the right-panel Output tab, then move to local:
```bash
mv ~/Downloads/p0_bundle.tar.gz /Users/malaymishra/Desktop/srib/results/
```

Create `/Users/malaymishra/Desktop/srib/results/` to collect all three bundles.

---

## 8. Blueprint Integration Checklist

Once the runs finish, wire results into `blueprint.tex`:

- [ ] **Slide 4 (Technical Details)** — add a footnote citing P0 val acc
- [ ] **Slide 5 (Novelty)** — add P1 SNR-sweep plot under the FAR-in-Loop diagram
- [ ] **Slide 7 (Evidence)** — add P1 table (SNR × method) and, if available, P2 per-speaker DET curve
- [ ] **Slide 8 (Param budget)** — annotate actual measured param count from P0 logs (overrides the 930.2K estimate in §6_param_budget.py if different)
- [ ] Regenerate `illustrations/6_param_budget.png` if the total shifts by > 3%

---

## 9. Reproducibility Invariants (every notebook must print these)

```python
import torch, numpy as np, random, hashlib, subprocess, json
print("git_sha:", subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip())
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("n_gpu:", torch.cuda.device_count())
print("seed: 1337")
# print config SHA
cfg_bytes = open(f"configs/{RUN_CFG}").read().encode()
print("config_sha:", hashlib.sha256(cfg_bytes).hexdigest()[:12])
```

Paste this header output into the top of every WandB run note and every result figure caption in the blueprint.

---

## 10. Exit Criteria (Kaggle Phase)

All of the following **must** be true before we declare Phase-1 evidence complete and resume blueprint polishing:

- [ ] P0 val acc ≥ 97.5% on SC V2 clean
- [ ] P1 PCEN vs. log-Mel sweep CSV generated, PCEN wins by ≥ 5 pp at ≤ 0 dB SNR
- [ ] (stretch) P2 per-speaker DET plot generated
- [ ] Three `.tar.gz` bundles in `/Users/malaymishra/Desktop/srib/results/`
- [ ] Three WandB run URLs pasted into `blueprint.tex`
- [ ] `illustrations/6_param_budget.png` reflects measured param count

When every box is ticked, the Phase-1 blueprint is submission-ready.

---

## 11. Time & Quota Budget

| Run               | Wall-clock | Cumulative | GPU-hours used |
|-------------------|------------|------------|----------------|
| P0 baseline       | ~1 h       | 1 h        | 2              |
| P1 PCEN ablation  | ~2 h       | 3 h        | 4              |
| P2 FiLM toy       | ~1.5 h     | 4.5 h      | 3              |
| Reruns/debug pad  | ~3 h       | 7.5 h      | 6              |
| **Total**         | ~7.5 h     | —          | **~15**        |

Comfortably inside Kaggle's 30 h/week quota with ~15 h of slack.
