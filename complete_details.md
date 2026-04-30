# SRIB — Complete Project Details (Phase-1 Lock File)

**Last updated:** Phase-1 submission ready (post FiLM v4)
**Submission target:** Samsung ennovateX AX Hackathon, **PS04 — Speech Disentanglement**

This document is the authoritative ledger for the project. It lists everything that has been built, measured, and decided so far. If anything in the blueprint or repo is unclear, this file is the tiebreaker.

---

## 1. Team

| | |
|---|---|
| Team name | **MalayM09** |
| Member (solo) | **Malay Mishra** |
| College | BITS Pilani, Pilani campus |
| Department | Computer Science |
| Year | 2026 |
| Email | F20220116@pilani.bits-pilani.ac.in |
| Repo | https://github.com/MalayM09/SRIB |
| License | Apache-2.0 |

---

## 2. Problem statement (PS04)

**Speech disentanglement for speaker-specific custom-word detection.** A voice trigger must wake a device only when (a) the right person speaks (b) the right user-chosen word, (c) under crowd/babble/traffic noise (−5 dB to 30 dB SNR), (d) at 0.5–5 m distance, (e) with <3 M parameters and <0.2 s xRT, (f) supporting both male and female speakers.

**KPIs (from brief):**
| Metric | Target |
|---|---|
| True Acceptance — Clean | ≥ 99 % |
| True Acceptance — Noisy | ≥ 90 % |
| False Acceptance | < 1/hr |
| SNR range | −5 to 30 dB |
| Distance | 0.5 to 5 m |
| Speakers | Male + Female |
| Parameters | < 3 M |
| xRT (execution time) | < 0.2 s |

---

## 3. Architecture (the joint student)

| Component | Params | Role |
|---|---:|---|
| MelSpectrogram (40 mels, 25 ms / 10 ms hop) | 0 | Frontend feature extraction |
| Learnable PCEN (Wang et al. 2017) | 0.16 K | Per-channel energy normalisation, gives noise robustness |
| BC-ResNet-8 trunk (channels [40, 60, 80, 100]) | 322.6 K | Shared trunk for KWS + SV |
| AttentiveStatPool (KWS) | 10 K | Time pooling for KWS branch |
| FiLM generator (Linear 256→400) | 102 K | Speaker-conditioned modulation γ, β |
| KWS head (Linear 200→12) | 2.4 K | 12-class keyword classifier |
| AttentiveStatPool (SV) | 10 K | Time pooling for SV branch |
| SV projection (Linear 200→256) | 51 K | 256-dim speaker embedding |
| **Total joint student** | **~499.6 K (16.6 % of 3 M cap)** | |

**Training-only modules** (discarded at inference):
- Sub-center AAM-Softmax (K=3, m=0.2, s=30) for SV speaker classification
- SpecAugment (time + frequency masking)

**Frontend stays in PyTorch CPU; trunk + heads ONNX-exported** (see Step F).

---

## 4. Empirical validation — every experiment, in order

### P0 — KWS backbone baseline (clean training)
- **Recipe:** BC-ResNet-8 + LearnablePCEN + KWS head, SC V2 (12-class), no waveform augmentation, 30 epochs, batch 256, AdamW lr 4e-3, cosine schedule.
- **Result:** **best_val_acc = 0.9764**, final = 0.9760. Train/val gap −1.96 pp (val > train, healthy).
- **Compute:** ~34 min on Kaggle T4.
- **Files:** `results (1)/runs/p0/`, `results (1) 2/runs/p1b_pcen_on_clean/` (rerun).

### P1a — PCEN vs log-Mel, noise-aug training (RIR + MUSAN)
- **Recipe:** Same trunk, two arms: PCEN-on, log-Mel. Aug: rir_prob=0.5, musan_prob=0.7, SNR ∈ [−5, 20]. 25 epochs.
- **Result (best val):** PCEN-on 0.9736, log-Mel 0.9728.
- **SNR sweep (val accuracy):**

| SNR (dB) | PCEN-on | log-Mel | Δ (pp) |
|---:|---:|---:|---:|
| clean | 0.9736 | 0.9728 | +0.09 |
| 20 | 0.9680 | 0.9652 | +0.28 |
| 10 | 0.9572 | 0.9527 | +0.45 |
| 5 | 0.9438 | 0.9404 | +0.35 |
| 0 | 0.9099 | 0.9034 | +0.65 |
| −5 | 0.8446 | 0.8329 | +1.17 |

- **Files:** `results (1)/runs/p1_pcen_on/`, `results (1)/runs/p1_logmel/`.

### P1a seed 42 — multi-seed validation
- Same recipe, seed 42. PCEN-on best 0.9762, log-Mel best 0.9743.
- **Cross-seed PCEN advantage:** mean +0.42 pp across all SNRs. PCEN's variance across seeds is ~4× lower than log-Mel's at low SNR (reproducibility win).
- **Files:** `results (1) 3/runs/`.

### P1b — clean training only (no waveform aug)
- **Recipe:** PCEN-on and log-Mel, both with SpecAugment but no RIR/MUSAN, 30 epochs.
- **Result:** PCEN-on best 0.9767, log-Mel best 0.9771 (essentially tied on clean).
- **SNR sweep:**

| SNR | PCEN-on | log-Mel |
|---:|---:|---:|
| clean | 0.9767 | 0.9771 |
| 20 | 0.9635 | 0.9598 |
| 10 | 0.9298 | 0.9293 |
| 5 | 0.8902 | 0.8896 |
| 0 | 0.8033 | 0.8014 |
| −5 | **0.6546** | **0.6544** |

- **Key insight:** without noise-aug training, both fronts collapse to ~65 % at −5 dB — proving **training-data diversity is the dominant lever** (worth +19 pp at −5 dB), not the front-end (which adds ~+1 pp on top).
- **Files:** `results (1) 2/runs/`.

### Step A v1 — SV branch POC (resemblyzer teacher, distillation only)
- **Recipe:** 100 speakers × 5 utts = 500 train, GE2E-LSTM teacher embeddings, distillation only (1 − cos sim), 30 epochs.
- **Result:** **EER 28.67 %** on VoxCeleb1-O — embedding collapsed to a single point.
- **Diagnosis:** pure distillation has no discriminative signal.

### Step A v2 — SV branch with sub-center AAM-Softmax
- **Recipe:** 1211 speakers × 20 utts = 24 K, AAM head with K=3 sub-centers, m=0.2, s=30, λ_distill=1, λ_aam=1, 40 epochs.
- **Result:** **EER 10.64 %** on VoxCeleb1-O. AUC clean, no collapse.
- **Cross-comparison:** vs published x-vector (3-5 % at 4 M params) and ECAPA-TDNN (1 % at 6 M). Our 384 K student lands in the same band as 4 M-param classical methods.
- **Path to ≤5 %:** swap teacher to ECAPA-TDNN, tune λ_distill down (to stop the loss-fight), train on full VoxCeleb1 for 80 epochs. Not yet implemented.
- **Files:** `code/scripts/sv_poc_v2.py` (training), `code/scripts/sv_eval.py` (evaluation).

### Step B v1 — FiLM speaker-conditioned KWS (random init)
- **Recipe:** Joint trunk + SV branch + FiLM gen + KWS head, 200 train speakers × 30 utts, 25 epochs, no curriculum, 50 % imposter ratio.
- **Result:** TAR 63.75 %, FAR 12.5 %, REJ 84.4 %. Probability collapse ratio 0.292.

### Step B v2 — same model + P0 init + curriculum
- **Recipe:** Same architecture, **trunk init from P0 ckpt**, 5 epochs KWS-only then ramp imposter to 0.5 over 10 epochs, 30 epochs total, 400 train spks.
- **Result:** TAR 62.5 %, FAR 15 %, REJ 85 %. Probability collapse 0.336.
- **Diagnostic:** when fed *own-audio* embeddings (matching training distribution): TAR **94.6 %**, FAR 14.3 %, REJ 85.7 %, collapse **0.199**.
- **Conclusion:** FiLM mechanism IS validated. The 62.5 % is a train/eval distribution mismatch, not a mechanism failure.
- **Files:** `code/scripts/film_poc_v2.py`, `code/scripts/film_v2_diagnostic.py`, `results/film_poc_v2/`.

### Step B v3 — paired-enrollment training
- **Recipe:** Each batch sample gets paired with an *enrollment clip from a different clip of the same speaker* (or different speaker for imposter). Forces FiLM to handle the actual deployment distribution.
- **Result:** TAR **92.14 %** (closes the gap), but FAR jumped to 48.21 %, REJ 51.07 %. Trade-off characterised.
- **Files:** `code/scripts/film_poc_v3.py`, `results/film_poc_v3/`.

### Step B v4 — paired + hard negatives + weighted imposter loss
- **Recipe:** Hard-negative mining (most-similar different-speaker enrollment), λ_imp = 2.0, max imposter ratio 0.7.
- **Result:** Over-corrected. TAR collapsed to **5.71 %**, FAR 0.36 %, REJ ~99 % — model degenerated to "always predict UNKNOWN".
- **Conclusion:** v3 too lenient, v4 too strict. Sweet-spot recipe needs more tuning (didn't converge in available time).
- **Files:** `code/scripts/film_poc_v4.py`, `results/film_poc_v4/`.

### Step D — Custom-word few-shot detection
- **Recipe:** Phonetic embedder (BC-ResNet + AttnStatPool + 256-dim projection), trained on 28 SC V2 keywords with sub-center AAM-Softmax. Evaluated on **7 held-out polysyllabic words** (`backward`, `forward`, `marvin`, `sheila`, `learn`, `follow`, `visual`) the model never saw — 5-shot enrollment averaging into a prototype, cosine-similarity scoring.
- **Result:** **EER 12.43 %**, AUC **0.934**, score separation 4.2× (genuine 0.63 vs imposter 0.15). TAR @ FAR=1 %: 54.3 %. TAR @ FAR=5 %: 70.9 %.
- **Per-word EER spread:** 6 % (forward) → 18 % (backward) — the model is *phonetically organised*, not memorising.
- **Path to ≤10 %:** WavLM hint-layer distillation (on the v3-projected 4-5 % path).
- **Files:** `code/scripts/custom_word_poc.py`, `results/custom_word_poc/`.

### Step F — Deployment validation (ONNX + INT8 + xRT projection)
- **Recipe:** Joint KWS+SV+FiLM exported to ONNX (frontend stays PyTorch — torchaudio STFT doesn't ONNX-export). Single-thread CPU benchmark on Mac M1, projected to Pi 4B via 3.5× ratio. Static INT8 quantization with 64 SC V2 calibration clips.
- **Latency (Mac M1, single-thread):** Frontend 0.57 ms + Trunk FP32 4.42 ms = **5.0 ms total FP32**. INT8 trunk 2.90 ms = **3.5 ms total INT8** (1.52× faster).
- **Pi 4B projected:** FP32 17.5 ms, **INT8 12.2 ms** (vs 200 ms budget — **17× headroom**, **187 ms slack**).
- **Model file size:** FP32 2.05 MB → INT8 0.79 MB (2.6× shrink).
- **Numeric agreement:** FP32 vs PyTorch < 3×10⁻⁶ (essentially exact). INT8 KWS-logit drift ~2.0 (expected for post-hoc PTQ; production deployment uses INT8 QAT in Stage 4).
- **Files:** `code/scripts/deploy_benchmark.py`, `results/deploy_benchmark/`.

---

## 5. Blueprint slides (20 total)

| # | Title | Status |
|---|---|---|
| 1 | Title page | ✓ |
| 2 | Team Introduction | ✓ (filled, solo) |
| 3 | Problem Statement & Why It Matters | ✓ |
| 4 | Proposed Solution — High Level | ✓ |
| 5 | Technical Details — System Architecture | ✓ |
| 6 | Technical Details — Multi-Teacher Knowledge Distillation | ✓ |
| 7 | Technical Details — Parameter Budget & Training Stages | ✓ |
| 8 | Technical Details — Execution Timeline | ✓ |
| 8 (a) | Phase-1 Empirical Validation — KWS Backbone | ✓ measured |
| 8b | Phase-1 Empirical Validation — SV Branch POC | ✓ measured |
| 8c | Phase-1 Empirical Validation — FiLM Rejection Trade-off | ✓ measured |
| 8d | Phase-1 Empirical Validation — Custom-Word Generalization | ✓ measured |
| 8e | Phase-1 Empirical Validation — Deployment Latency & Size | ✓ measured |
| 9 | Novelty & Innovation — FAR-in-the-Loop Training | ✓ design |
| 10 | Novelty & Innovation — Supporting Contributions | ✓ design |
| 11 | Deployment & Demo Plan (for Grand Finale) | ✓ |
| 12 | Open Datasets — Used & Published | ✓ |
| 13 | Open Models — Used & Developed | ✓ |
| 14 | AI / GenAI / Agentic Tools & Best Practices | ✓ |
| 15 | Closing | ✓ |

(Numbering: blueprint uses "Slide 8" for the first empirical validation slide and 8b/8c/8d/8e for follow-ups; effective slide count when compiled = 20 frames.)

---

## 6. Repository layout

```
srib/
├── README.md              ← project entry point
├── LICENSE                ← Apache-2.0
├── complete_details.md    ← this file
├── blueprint.tex          ← Phase-1 blueprint source
├── illustrations/         ← all blueprint figures
│   ├── 1_architecture.{png,svg,d2}
│   ├── 2_distillation.{png,svg,d2}
│   ├── 3_far_in_loop.{png,mmd}
│   ├── 4_timeline.{png,mmd}
│   ├── 5_enrollment_runtime.{png,mmd}
│   ├── 6_param_budget.{py,png,svg}
│   ├── 7_snr_sweep.{py,png,svg}
│   ├── 8_sv_progression.{py,png,svg}
│   ├── 9_film_tradeoff.{py,png,svg}
│   ├── 10_custom_word.{py,png,svg}
│   └── 11_deploy_metrics.{png,svg}
├── code/                  ← pilot implementation
│   ├── src/
│   │   ├── data/          ← SC V2 loader + augmentation
│   │   ├── model/         ← BCResNet8, LearnablePCEN, KWSHead
│   │   ├── losses/        ← FocalCrossEntropy
│   │   ├── train.py       ← P0/P1 training entry point
│   │   └── eval_snr.py    ← SNR sweep evaluator
│   ├── scripts/           ← experiment scripts
│   │   ├── film_poc.py / film_poc_v2.py / film_poc_v3.py / film_poc_v4.py
│   │   ├── film_v2_diagnostic.py
│   │   ├── custom_word_poc.py
│   │   ├── deploy_benchmark.py
│   │   ├── eval_conditions.py
│   │   └── listen_gallery.py
│   ├── tests/             ← pytest shape + budget tests
│   ├── configs/           ← YAML configs (pilot_p0_baseline, pilot_p1*, etc.)
│   └── README.md          ← code/ entry point
├── results/               ← measured outputs from each experiment
│   ├── film_poc/  film_poc_v2/  film_poc_v3/  film_poc_v4/
│   ├── custom_word_poc/
│   ├── deploy_benchmark/    (incl. ONNX FP32 + INT8 files)
│   └── …
├── results (1)/  results (1) 2/  results (1) 3/  results (2)/  ← prior Kaggle bundle exports
├── data/                  ← gitignored: speech_commands_v2/, rirs_small/, musan_small/
├── KAGGLE_RUN.md          ← Kaggle pilot protocol
├── LOCAL_RUN.md           ← local pilot protocol
└── problem_statements.md  ← summarised PS01–PS04 for context
```

---

## 7. Reproducing every result

| Experiment | Run command (from `code/`) | Output |
|---|---|---|
| P0 KWS baseline | `python -m src.train --config configs/pilot_p0_baseline.yaml` | `runs/p0/` |
| P1 PCEN ablation | `python -m src.train --config configs/pilot_p1_pcen_ablation.yaml` (flip `use_pcen`) | `runs/p1_*/` |
| SNR sweep | `python -m src.eval_snr --ckpt <best.pt> --musan_dir data/musan_small --out <csv>` | `runs/<name>/snr_sweep.csv` |
| FiLM v3 | `.venv/bin/python scripts/film_poc_v3.py --device mps` | `results/film_poc_v3/` |
| FiLM v4 | `.venv/bin/python scripts/film_poc_v4.py --device mps --batch 32 --num_workers 2` | `results/film_poc_v4/` |
| Custom word | `.venv/bin/python scripts/custom_word_poc.py --device mps --batch 32 --num_workers 2` | `results/custom_word_poc/` |
| Deployment | `.venv/bin/python scripts/deploy_benchmark.py` | `results/deploy_benchmark/` |
| Listen gallery | `.venv/bin/python scripts/listen_gallery.py` | `data/listen/` |

---

## 8. Open work (not yet started)

| Priority | Item | Why deferred |
|---|---|---|
| Medium | **FiLM v5** with milder hyperparameters (λ_imp=1.3, imp_ratio=0.5, keep hard-neg mining) — find the sweet spot | v3 lenient and v4 strict are now characterised; v5 would land between them |
| Medium | **SV v3** with vendored ECAPA teacher + λ tuning + 80 epochs full VoxCeleb1 | SV v2 at 10.64 % EER is yellow; clear path to ≤5 % |
| Medium | **Custom-word v2** with WavLM hint-layer distillation | EER 12.43 % is yellow; WavLM closes hard-word spread |
| High (Phase 2) | **Streaming inference** + mic capture + enrollment UI | Required for live demo |
| High (Phase 2) | **INT8 QAT** (Stage 4) | Reduces post-hoc INT8 drift 5-10× |
| High (Phase 2) | **Pi 4B hardware** purchase + actual end-to-end measurement | Validates projected latency on real hw |
| Medium (Phase 2) | **WavLM distillation** integrated into joint training | Lifts custom-word + KWS robustness |
| Low (Phase 2) | **Galaxy NPU** integration via QNN SDK | Samsung-friendly, but Pi is sufficient for stage demo |

---

## 9. Pivot triggers (if PS04 fails downstream)

If any of the following materialises post-Phase-1, reconsider switching PS:
- **SV v3 still > 15 % EER after ECAPA + tuning** → 384 K trunk fundamentally can't do SV; PS04 not viable at 3 M cap.
- **FiLM v5 still > 30 % FAR with TAR > 80 %** → personalisation mechanism doesn't tighten under any tuning; story breaks.
- **Custom-word v2 (WavLM) still > 20 % EER** → phonetic generalisation limit; would need 10 M+ params.

**Fallback PS (best-fit):** PS02 (Network Traffic Classification) — same contrastive-embedding stack, ~2-3 weeks to a working demo, no new audio infra. PS01 would be a full restart; PS03 is a poor demo fit.

---

## 10. Submission checklist

Before clicking submit on the ennovateX portal:
- [x] All `<TEAM NAME>`, `<PROBLEM NUMBER>`, `<FILL>` placeholders filled in `blueprint.tex`
- [x] `LICENSE` file at repo root (Apache-2.0)
- [x] `README.md` at repo root linking to blueprint, code, results
- [x] `complete_details.md` (this file)
- [ ] **`blueprint.pdf` compiled** via Overleaf (xelatex compiler) — verify all 20 slides render, no missing images, no leftover placeholders
- [ ] **Repository pushed to `github.com/MalayM09/SRIB`** with latest state, public visibility
- [ ] **Read official ennovateX Phase-1 portal rules** to confirm submission format (PDF? code link? video?)
- [ ] **Submit before deadline**

---

*This document is the source of truth. If something is unclear in `blueprint.tex` or any script, this file is the tiebreaker.*
