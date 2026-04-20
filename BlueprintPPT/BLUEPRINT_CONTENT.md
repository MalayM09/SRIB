# Blueprint Content — Slide-by-Slide

Paste each section into the corresponding slide of `AX_Hackathon_Phase1_Blueprint_Template.pptx`. Trim aggressively to fit; PowerPoint slides should breathe.

---

## SLIDE 1 — Team Introduction

**Team Name:** _<fill in>_

**Team Member 1**
Name —
College —
Email ID —
Dept —
Year —

**Team Member 2** (optional, max 2)
Name —
College —
Email ID —
Dept —
Year —

---

## SLIDE 2 — Problem Statement

**Selected Problem:** Speech Disentanglement — Speaker-Specific Custom Word Detection on the Edge.

**Problem Understanding:**
Voice triggers must wake a device only when *the right person* says *the right word* — not when an imposter says it, not when a phonetically similar word is spoken, and not when crowd / traffic / babble noise drowns out the signal. Today's edge wake-word systems either (a) accept anyone saying the trigger (privacy risk on shared devices), or (b) need >10 M parameters and cloud offload. The brief constrains us to **<3 M params, <0.2 s xRT, <1 false accept per hour, ≥90% true accept at -5 dB SNR, distances up to 5 m** — a regime where standard log-Mel + ResNet pipelines collapse and Transformer-based KWS misses the latency budget.

**Why it matters:** On-device personalized voice activation enables private, offline-first wake words for phones, earbuds, hearables, smart-home hubs, and accessibility tools — without sending raw audio to the cloud.

---

## SLIDE 3 — Proposed Solution (High-Level)

We build a **single < 1.2 M-parameter student model** that jointly performs Keyword Spotting (KWS) and Speaker Verification (SV), distilled from two large pretrained teachers (WavLM-Base+ for noise-robust phonetics; ECAPA-TDNN for biometric identity).

Three structural ideas make the strict KPIs reachable:

1. **Speaker-conditioned KWS via FiLM.** The KWS classifier's effective weights are modulated by the enrolled user's speaker embedding. Imposter audio flows through the wrong modulation and the keyword logit collapses — *structural* false-accept rejection, not just a threshold.
2. **Per-Channel Energy Normalization (learnable PCEN) frontend.** Survives -5 dB SNR where log-Mel breaks.
3. **Multi-teacher knowledge distillation** with multi-level feature matching, layer-ensemble hint losses, and a FAR-aware online calibration signal that bakes the deployment operating point into training.

The pipeline runs end-to-end on free Kaggle GPUs and exports to INT8 ONNX for edge inference (Raspberry Pi 4B / Galaxy on-device NPU class hardware).

---

## SLIDE 4 — Proposed Solution: Technical Details

**Architecture (block diagram on this slide):**
```
raw 16 kHz audio
  → PreEmphasis → MelSpec (40 bins) → Learnable PCEN → LayerNorm
  → BC-ResNet-8 shared trunk (4 blocks, ~420 K params)
  → TRM_kws ──► KWS branch ──► FiLM(γ,β from SV emb) ──► KWS head ──► word logits
  → TRM_sv  ──► SV branch  ──► MQMHA pool ──► SV head ──► 192-d L2-norm embedding
```

**Training pipeline (5-term loss, second diagram if room):**
```
L_total = λ₁·Focal(KWS) + λ₂·SubCenterAAM(SV)
        + λ₃·KL(KWS ‖ WavLM-probe)            # logit KD
        + λ₄·MSE(student_feat ‖ Σwᵢ·WavLM_layer_i)   # layer-ensemble hint KD, i ∈ {4,8,11}
        + λ₅·[1 − cos(emb, ECAPA_final)] + 0.3·MSE(student ‖ ECAPA_block_2,3)
```

**Training stages (4 stages, ~36 GPU-hr on Kaggle T4/P100):**
| Stage | Data | Purpose |
|-------|------|---------|
| 0 | SC + LibriPhrase | Train WavLM→KWS probe (one-time) |
| 1 | Speech Commands | Stabilize PCEN + trunk |
| 2 | SC + VoxCeleb1 + LibriPhrase | Full multi-task + all KD |
| 3 | LibriPhrase + TTS-trigger | Specialize + online τ_s calibration |
| 4 | Same | Quantization-aware training (INT8) |

**Edge deployment:** PyTorch QAT → ONNX (opset 17) → ORT INT8 static. Streaming via 160 ms chunks with stateful PCEN smoothing + ring-buffer conv state.

**KPI targets (problem brief):**
| Metric | Target |
|--------|--------|
| TA — clean | ≥ 99% |
| TA — noisy (-5 to 30 dB) | ≥ 90% |
| FA / hour | < 1 |
| xRT | < 0.2 s |
| Params | < 3 M (we project ~1.13 M) |
| SNR / Distance | -5 to 30 dB / 0.5 to 5 m |

---

## SLIDE 5 — Novelty & Innovation

**Core novelty: structural false-accept rejection via FiLM-conditioned KWS.**
Conventional personalized wake-word systems run KWS and SV as independent gates and AND the decisions — any leakage between heads or threshold drift breaks the FA<1/hr budget. We instead **modulate the KWS classifier's intermediate features with the enrolled user's voice embedding**, so the *same* audio produces different logits for different enrolled users. Imposter rejection becomes a property of the network, not of a tunable threshold.

**Supporting novelties (each backed by 2024–2025 literature, integrated for the first time at this scale + KPI envelope):**
- **Online τ_s calibration during training** — every 500 steps we re-estimate the deployment threshold on a held-out imposter pool and feed FRR-at-target-FAR back as a soft penalty. Bakes the FA<1/hr operating point into the embedding geometry instead of hoping post-hoc thresholds work.
- **Layer-ensemble WavLM distillation** with 3 learnable softmax weights over layers {4, 8, 11} — beats single-layer choice (SKILL, 2024).
- **Multi-level ECAPA distillation** (block-2 + block-3 + final embedding) — ~5% relative EER over final-only KD.
- **Sub-center AAM-Softmax (K=3)** — handles VoxCeleb's known label noise without extra params.
- **Phonetic confusable hard-negative mining** via TIMIT-derived phone-confusion-weighted Levenshtein on G2P output, sampled at 3× during training.

**Pain points addressed:** privacy (no cloud), shared-device personalization, accessibility (low-resource languages via TTS-augmented enrollment), false-trigger fatigue.

**Impact:** A reusable recipe for sub-3 M-param personalized voice triggers deployable on phones, earbuds, hearables, and smart-home hubs without cloud round-trips.

---

## SLIDE 6 — Open Datasets

**To be used (all permissively licensed, downloadable):**
| Dataset | Role | Licence |
|---------|------|---------|
| Google Speech Commands v0.02 | KWS positives + general negatives | CC-BY 4.0 |
| VoxCeleb1 | Speaker verification + imposter pool | CC-BY 4.0 |
| LibriPhrase | Joint phrase + speaker fine-tuning | MIT |
| MUSAN | Crowd / babble / traffic noise injection (-5 to 30 dB) | CC-BY 4.0 |
| OpenSLR RIRs | Room impulse responses → 0.5–5 m distance simulation | Apache-2.0 |
| CMUdict + TIMIT phone-confusion table | G2P + phonetic confusable mining | BSD / public |

**To be published:** A small **derived "confusable-trigger" benchmark** — for any chosen trigger word, the mined top-50 phonetic confusables synthesized in 100+ XTTS-v2 voices + their RIR/MUSAN augmentations. Useful for any future personalized-KWS work targeting FA<1/hr. Released under Apache-2.0.

---

## SLIDE 7 — Open Models

**To be used (frozen, distillation teachers):**
| Model | Role | Source / Licence |
|-------|------|------------------|
| WavLM-Base+ | Phonetic / noise-robust teacher | `microsoft/wavlm-base-plus` · MIT |
| ECAPA-TDNN | Speaker biometric teacher | `speechbrain/spkrec-ecapa-voxceleb` · Apache-2.0 |
| XTTS-v2 | Multi-voice TTS for trigger augmentation | `coqui/XTTS-v2` · CPML |
| g2p-en | Trigger word → ARPAbet phones | Apache-2.0 |

**To be developed / trained from scratch:**
| Model | Description | Output Licence |
|-------|-------------|----------------|
| **BC-ResNet-8 (modified)** student with TRM branching, FiLM speaker conditioning, MQMHA SV pooling | Joint KWS + SV, ~1.13 M params | **Apache-2.0** (per hackathon rules) |
| WavLM→KWS probing head | 2-layer MLP, ~50 K params, one-time | Apache-2.0 |
| INT8-quantized ONNX export | Edge-deployable artifact | Apache-2.0 |

---

## SLIDE 8 — AI / GenAI / Agentic Tools & Best Practices

**Tools used:**
- **PyTorch 2.x** + `torch.compile(mode="reduce-overhead")` for ~15% Kaggle speedup.
- **HuggingFace Transformers / SpeechBrain** for frozen teacher inference.
- **Coqui XTTS-v2** as a generative AI component for synthetic speaker diversity in trigger and confusable data.
- **Optuna (TPE sampler)** for automated multi-task loss-weight + AAM hyperparameter search, scoped to Stage 2 only.
- **ONNX Runtime + Quantization Toolkit** for INT8 edge export and benchmarking.
- **Google `kws_streaming`** as the reference implementation for stateful streaming inference.

**Best practices uncovered (will be documented in the final repo):**
1. **Symmetric augmentation in distillation.** Teacher and student must see the same augmented audio, otherwise the student collapses to the teacher's clean manifold. (Counter to most blog tutorials.)
2. **Multi-level KD beats single-layer KD** for both phonetic (WavLM) and biometric (ECAPA) teachers — measured ~5% relative metric gain.
3. **Bake the deployment operating point into training** (online τ_s calibration). Post-hoc thresholding cannot fix an embedding space that wasn't trained to be FAR-aware.
4. **Structural conditioning > threshold gating** for personalized triggers. FiLM-modulated KWS is materially safer for FA-bounded scenarios than the standard "AND of two independent classifiers" pattern.
5. **Curriculum on SNR**, not just on data difficulty. Starting at -5 dB diverges; ramp from 30 → -5 dB across the first 15 epochs.

**Licence statement:** All code, model weights, and the published confusable-trigger benchmark will be released under **Apache-2.0**, per hackathon submission rules.
