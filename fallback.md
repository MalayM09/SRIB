# Fallback Problem Statements — Strategic Context

**Purpose:** if PS04 hits a wall, this file is the lookup table for which alternate problem statement to pivot to and why.

**Current PS:** PS04 (Speech Disentanglement) — Phase-1 blueprint submitted with full empirical validation. See [`complete_details.md`](complete_details.md).

---

## TL;DR ranking

| Rank | PS | Why | Pivot cost | Code reuse |
|---|---|---|---|---|
| 🥇 | **PS11 — Multi-User Smart Assistant** | PS04 + speech separation; same domain | ~2-3 weeks | ~70 % |
| 🥈 | **PS08 — Immersive Spatial Voice Calls** | Audio domain, edge-AI focus, sidesteps speaker-ID | ~3-4 weeks | ~50 % |
| 🥉 | **PS02 — Network Traffic Classification** | Same ML pattern (contrastive + classifier), different data | ~2-3 weeks | ~30 % |
| ❌ | PS01, PS03, PS05, PS06, PS07, PS09, PS10 | Wrong domain / wrong infra / wrong skill profile | full restart | ~0 % |

---

## When to pivot — trigger conditions

Stay on PS04 *unless* one of these fires (defined in `complete_details.md` §9):

1. **SV v3 still > 15 % EER** after ECAPA teacher + λ tuning + 80 epochs full VoxCeleb1 → core SV claim broken
2. **FiLM v5 still > 30 % FAR with TAR > 80 %** → personalization mechanism doesn't tighten under any tuning
3. **Custom-word v2 (WavLM) still > 20 % EER** → phonetic generalization limit; would need 10 M+ params

Action by trigger:

| Trigger fires | Recommended pivot |
|---|---|
| Only SV breaks (FiLM and custom-word OK) | Stay on PS04 — drop SV branch, use FiLM-only personalization |
| Only FiLM breaks (SV and custom-word OK) | **Pivot to PS11** — use SV-embedding-similarity for speaker ID, add separation |
| Only custom-word breaks | Stay on PS04 with **fixed keyword list** (SC V2 12-class) — claim custom-word as Phase-2 |
| Multiple audio capabilities break | **Pivot to PS08** — audio domain but no speaker-ID dependency |
| Whole audio approach unworkable | **Pivot to PS02** — same ML pattern, different domain |

---

## 🥇 PS11 — Real-Time Multi-User Smart Assistant

**One-liner:** Smart assistant that handles simultaneous multi-user voice queries in noisy environments. Identify and separate users, distinguish commands from noise, recognise overlapping speech, support multiple simultaneous keywords, deliver per-user personalised responses.

### Required KPIs

| KPI | Target |
|---|---|
| SI-SNR (Scale-Invariant SNR) — clean ≤ 2 spk | > 25 dB |
| SI-SNR — clean > 2 spk | > 15 dB |
| SI-SNR — noisy/RIR ≤ 2 spk | > 18 dB |
| SI-SNR — noisy/RIR > 2 spk | > 10 dB |
| WER — clean ≤ 2 spk | < 5 % |
| WER — clean > 2 spk | < 10 % |
| WER — noisy/RIR ≤ 2 spk | < 10 % |
| WER — noisy/RIR > 2 spk | < 20 % |
| Model parameters | < 5 M |
| Real-Time Factor (xRT) | < 0.5 |

### Why PS11 is our best fallback

PS11's KPI set is essentially **PS04 + a multi-speaker separation module**. Our existing infrastructure maps directly:

| What we have | Maps to PS11 KPI |
|---|---|
| BC-ResNet-8 trunk (322.6 K params, validated) | Feature extractor for separation network |
| Sub-center AAM-Softmax SV branch (10.64 % EER) | "Identify and separate different users' voices" |
| PCEN + RIR + MUSAN noise-aug recipe | "Distinguish between commands and background noise" |
| KWS head (97.6 % clean / 84 % at −5 dB) | "Recognizing multiple keywords simultaneously" |
| Joint model: 499.6 K params | Already < 5 M cap (10× headroom) |
| ONNX + INT8 deployment (12 ms Pi 4B projected) | "Real-Time Factor xRT < 0.5" — easily met (xRT ~0.06) |
| Custom-word few-shot embedding (12.4 % EER) | Multi-keyword detection extension |

### What we'd need to build

1. **Speech separation module.** TasNet or DPRNN from the [Asteroid](https://github.com/asteroid-team/asteroid) toolkit. Can use our BC-ResNet trunk as a feature backbone or use Asteroid's reference architectures.
2. **LibriMix dataset integration.** 2-speaker and 3-speaker mixtures derived from LibriSpeech. New dataloader, similar pattern to our SC V2 loader.
3. **Multi-keyword head.** Extend our KWS head to multi-label classification (multiple simultaneous active keywords).
4. **Per-speaker response routing.** Logic, not ML — match SV embedding of separated source to enrolled user, route to per-user response.

### Pivot cost: ~2-3 weeks

- Week 1: Wire up Asteroid + LibriMix, train separation network on top of our trunk.
- Week 2: Multi-label KWS head, per-speaker routing logic, integration tests.
- Week 3: Demo polish, evaluation against PS11 KPIs.

### Demo

Live mic in a noisy room. 2-3 people speaking simultaneously, each says a different command (one user says "play music", another says "set timer"). System separates voices, identifies speakers from enrolment, recognises each command, routes responses. **Visually compelling and audibly impressive.** Better stage demo than PS04's quieter trigger-detection scenario.

### Risk

Most of the risk is in the separation module quality (DPRNN/TasNet are well-known but need tuning for our trunk). The downstream pieces (SV, KWS, deployment) are already de-risked.

---

## 🥈 PS08 — Immersive Spatial Voice Calls

**One-liner:** Transform monaural voice calls into 3D spatially-aware audio using HRTF-based binaural rendering. Real-time inference < 20 ms, model < 50 MB, support up to 3 concurrent users.

### Required KPIs

| KPI | Target |
|---|---|
| Audio Latency (Inference) | < 20 ms |
| Audio Latency (Input) | 40-60 ms |
| Model Size | < 50 MB |
| Spatial Audio Accuracy | ≥ 95 % |
| User Engagement Score | ≥ 85 % |
| Concurrent users supported | up to 3 |
| UI Responsiveness | < 100 ms |

### Why PS08 is the second-best fallback

Same audio domain, same edge-AI deployment story, sidesteps the personalization (SV/FiLM) capability entirely. Useful as a "deeper" fallback if multiple PS04 capabilities break.

| What we have | Maps to PS08 |
|---|---|
| Mel + PCEN audio frontend | Audio preprocessing for spatial system |
| ONNX + INT8 deployment (0.79 MB INT8) | "Model Size < 50 MB" — 60× headroom |
| Sub-200 ms xRT measurement | "Inference < 20 ms" — needs careful measurement but in our reach |
| BC-ResNet trunk (validated) | Backbone for HRTF-aware encoder |
| Audio dataset infrastructure (SC V2, MUSAN, RIRs) | Adds SONICOM HRTF + MRSAudio datasets |
| Real-time stream processing patterns | Reusable |

### What we'd need to build

1. **HRTF convolution module.** Math-heavy but well-documented. Convolves a source signal with the HRTF for a target spatial position to produce binaural (left+right) output simulating that source coming from that direction.
2. **Per-source spatial positioning logic.** Each call participant gets a virtual position in 3D space; HRTFs are interpolated based on relative position to the listener.
3. **Multi-party mixing.** Sum of binaural-rendered sources, accounting for distance attenuation and reverb (RIRs!).
4. **Head-tracking simulation** (optional, but raises engagement score).

### Pivot cost: ~3-4 weeks

Higher than PS11 because the *task* is fundamentally different — synthesizing audio rather than detecting/classifying. Need to learn spatial audio fundamentals (HRTF math, binaural rendering, source attenuation models). Most ML papers in this space use simpler architectures than ours — we'd be over-equipped on the encoder side and under-equipped on the rendering side.

### Demo

Headphones-required demo: a 3-way phone call where participants are positioned at virtual locations (left, right, behind). Listener turns their head; positions stay fixed in world coordinates. **Compelling for the wearer but harder to convey to a panel watching from the audience** — judges would need to put on headphones in turns.

### Risk

We have no spatial audio experience. Reading an HRTF dataset and convolving correctly is straightforward, but achieving 95 % spatial accuracy (a perceptual metric) needs careful implementation. Lower KPI bar overall but novel territory for us.

---

## 🥉 PS02 — Network Traffic Classification

**One-liner:** Build a packet-flow encoder that produces context-aware embeddings combining flow features (5-tuple, packet rate, inter-arrival times) and network characteristics (RTT, jitter). Train via contrastive learning. Classify encrypted traffic types with kNN/SVM.

### Required KPIs

| KPI | Target |
|---|---|
| Cosine sim — intra-class | > 0.7 |
| Cosine sim — inter-class | < 0.3 |
| Classification accuracy | ≥ 90 % |
| Generalization to unseen traffic | ≥ 85 % |
| Inference time per flow | < 100 ms |

### Why PS02 is a viable fallback

Our SV branch is **exactly this recipe** with audio swapped for packets:
- Contrastive metric learning (we use sub-center AAM-Softmax; PS02 wants similar embedding pull-together / push-apart)
- Cosine similarity at inference for classification
- Real-time inference target

| What we have | Maps to PS02 |
|---|---|
| Sub-center AAM-Softmax loss | Same loss, different classes (traffic types instead of speakers) |
| Cosine similarity scoring + EER computation | Same eval pattern, different labels |
| Train/val/eval scaffolding | Reusable with new dataset wrappers |
| ONNX deployment pipeline | "Inference < 100 ms per flow" — easily met |

### What we'd need to build

1. **Packet-flow encoder.** Replace BC-ResNet (audio-specific) with LSTM or Transformer over packet-time-series features. Reference: PacketCLIP architecture.
2. **Dataset pipeline.** New: CESNET-QUIC22, MAWI, optionally manual packet captures (running YouTube/Netflix under varied network conditions). All TCP/IP packet processing — totally new code.
3. **Network-feature extraction.** 5-tuple, packet rate, inter-arrival times, RTT, jitter — needs `scapy` or similar packet-parsing library.
4. **Classifier integration.** kNN or SVM over the learned embeddings.

### Pivot cost: ~2-3 weeks

The ML training loop is reusable (~30 % code reuse) but the entire signal-processing layer changes. Audio domain expertise becomes useless; we'd be learning packet semantics, traffic patterns, network protocols. New dataset onboarding is significant (CESNET-QUIC22 is several GB of packet captures).

### Demo

Live packet capture from the laptop's network interface, classify each flow in real time as YouTube / Netflix / gaming / web / etc. Shows on a dashboard. **Less flashy than the audio demos** — networking is invisible to non-experts. Reviewers might not get a "wow" moment.

### Risk

Lower technical risk than PS04 (the problem is well-understood) but lower demo impact. Best as the "completely-different-problem with familiar tooling" fallback if PS04's audio domain itself proves unworkable.

---

## Why the other 7 PSes are not viable fallbacks

| PS | Disqualifier |
|---|---|
| **PS01 — VLA Robotic** | Different domain (vision + robotics + language). Needs PyBullet/ROS simulation infra we don't have. Fine-tuning a 7B-param OpenVLA needs serious GPU compute. Demo is video of sim tasks, not edge hardware. Full restart with no transferable assets. |
| **PS03 — Mobile Adaptive Memory** | Systems-heavy (memory manager, RL for allocation, app pre-loading). Needs Android emulator/device. Hard to demo visually — KPIs are numerical baseline-improvements that don't make for a stage moment. RL specialty we don't have. |
| **PS05 — Empathetic UX** | Primarily a *user-research* and *product-design* problem. Required deliverables include 8-10 qualitative test users + 30-50 quantitative survey. Solo team can't realistically do user testing at this scale. Wrong skill profile (we're ML engineers, not UX designers). |
| **PS06 — SLM Reasoning RL** | "Recommended Infra: 2× NVIDIA GPU with ≥ 24 GB VRAM, 12+ CPU cores, 64 GB system RAM" — solo team without cloud budget can't meet hardware bar. Different domain (NLP reasoning, not audio). RL specialty different from ours. |
| **PS07 — Ethical Deepfake Generation** | Diffusion/GAN training for video generation. Needs 200 unique identities with demographic balance, FFHQ/CelebDF/MEAD datasets are huge. Video generation isn't our wheelhouse. Ethical-safeguards layer adds scope creep. |
| **PS09 — Occlusion-Aware 3D Reconstruction** | RGB-D / sparse depth, mesh reconstruction, navigation. Computer vision / robotics domain. ScanNet dataset, SLAM-style infrastructure. F1 > 0.95 target is brutally tight. Zero overlap with our skills or assets. |
| **PS10 — Telecom RAG Assistant** | Tractable in principle (RAG over 3GPP specs is a well-defined task). But zero code reuse — entirely new domain knowledge required (telecom RAN, ORAN, 5G specs). Vector store + LLM + retrieval pipeline = different stack. Reasonable fallback for a different team profile, not ours. |

---

## Decision flowchart

```
PS04 trigger fires?
│
├── No  → stay on PS04, continue Phase-2 build
│
└── Yes → which trigger?
        │
        ├── SV v3 > 15% EER     → PS11 (use SV-similarity for speaker ID, add separation)
        │
        ├── FiLM v5 > 30% FAR   → PS11 (drop FiLM, use SV-embedding similarity)
        │
        ├── Custom-word > 20%   → stay on PS04 with fixed keywords (SC V2 12-class)
        │
        ├── Multiple breaks     → PS08 (sidesteps speaker-ID + custom-word entirely)
        │
        └── Audio totally fails → PS02 (different domain, familiar ML pattern)
```

---

## Maintenance

If the triggers change or new pivot evidence emerges, update this file *and* `complete_details.md` §9 together so the two stay consistent.
