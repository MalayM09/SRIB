# Samsung ennovateX AX Hackathon — Problem Statements

Reference for strategic planning. Our current PS is **PS04 (Speech Disentanglement)**.

---

## PS01 — Vision-Language-Action Robotic Manipulation

**One-liner:** Build a VLA system that parses natural-language object-manipulation commands (e.g., "move the blue block to the right of the green cube"), detects objects, and generates robotic actions.

**KPIs:**
- Task Success Rate ≥ 80–85 % (SOTA: 70–75 %)
- Goal Condition Accuracy ≥ 90 %
- Command Interpretation Accuracy ≥ 85 %
- Task Completion Rate ≥ 80 %

**Stack:** OpenVLA 7B, Detectron2, HuggingFace Transformers (BERT/GPT), PyBullet or ROS for simulation.

**Datasets:** Open X-Embodiment, COCO, Something-Something V2, RoboNet, ALFRED, VLA-3D, LIBERO.

**Notes:** Very different domain. Needs simulation infrastructure (PyBullet/ROS). 7B-param models are compute-heavy even for fine-tuning. Demo = video of sim tasks, not edge hardware.

---

## PS02 — Context-Aware Flow Embeddings for Network Traffic Classification

**One-liner:** Build a neural packet-flow encoder that produces context-aware embeddings from flow features (5-tuple, packet rate, inter-arrival times) + network characteristics (RTT, jitter). Classify encrypted traffic types (video streaming vs gaming vs XR) via contrastive learning.

**KPIs:**
- Intra-class cosine similarity > 0.7, inter-class < 0.3
- Classification accuracy ≥ 90 %
- Generalization to unseen traffic ≥ 85 %
- Inference < 100 ms per flow

**Stack:** LSTM / Transformer / CRNN encoder + contrastive loss + kNN/SVM classifier.

**Datasets:** 5G Traffic Datasets (Kaggle), CESNET-QUIC22, MAWI, PacketCLIP. Optional: manual packet capture (YouTube/Netflix under varied network conditions).

**Notes:** Tractable in 2–3 weeks. Smaller datasets, standard contrastive-learning setup. Less "AI-flashy" for a panel demo (no speech/vision/robotics wow factor) but a cleaner engineering story.

---

## PS03 — Context-Aware Adaptive Memory for Mobile Agentic Systems

**One-liner:** On-device memory manager that dynamically allocates RAM based on predicted user context, pre-loads likely next apps, and adaptively caches — for smartphones/edge devices running multi-app agentic workloads.

**KPIs:**
- App launch time −10 %+, load time −20 %
- Memory thrashing −50 %+
- Next-context prediction ≥ 75 %, cache hit rate ≥ 85 %
- Zero stability issues

**Stack:** Transformer/LSTM/ARIMA for context prediction; TD3/DDPG RL for allocation decisions.

**Datasets:** Context Query Logs (Melbourne parking), Android Usage Patterns, KV Cache Workloads.

**Notes:** Systems-heavy. Needs Android emulator or real device for realistic measurement. Hard to demo visually; KPIs are all numerical improvements over a baseline. Not a good fit for a live stage demo.

---

## PS04 — Speech Disentanglement (OUR CURRENT PROBLEM)

**One-liner:** Real-time speaker-specific custom-word detection under −5 to 30 dB noise, 0.5–5 m distance, <3 M params, <0.2 s xRT. Must detect a user-chosen trigger spoken by the enrolled speaker, reject imposters, reject phonetic confusables.

**KPIs:**
- True Accept clean ≥ 99 %
- True Accept noisy ≥ 90 %
- False Accept < 1/hr
- <3 M params, <0.2 s xRT

**Stack:** LibriPhrase, Google Speech Commands, VoxCeleb1 + VoxCeleb2, MUSAN, OpenSLR RIRs.

**Our current state:** KWS backbone validated at 324 K params (97.6 % clean, 84 % at −5 dB, 77 % worst case reverb+noise). SV branch, FiLM speaker conditioning, custom-word detection, QAT, and deployment are all not-yet-built.

---

## Pivot-trigger thresholds (when to switch to an alternate PS)

If any of the following fail with engineering effort, PS04 becomes too risky for our timeline:

- **Step A fails** → SV EER > 15 % at <500 K params → core SV claim broken → consider PS02.
- **Step B fails** → FiLM-conditioned KWS doesn't reject imposters → no speaker-specific rejection path → consider PS02.
- **Step D fails** → Custom-word TAR < 70 % on TTS-synthesized data → can't generalize to user-chosen words without 10 M+ params → consider PS02.

PS02 is the most similar-difficulty alternative (still contrastive embeddings + classification). PS01 would be a full restart. PS03 is a poor demo fit.
