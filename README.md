# SRIB — Speech Disentanglement (Samsung ennovateX AX Hackathon, PS04)

**Team:** MalayM09 · **Member:** Malay Mishra · **College:** BITS Pilani, Pilani campus · **Year:** 2026

Edge-AI **speaker-specific custom-word detection** under −5 dB noise, on a single ~500 K-parameter joint student model. This repository contains the Phase-1 blueprint and all empirical validation experiments.

## What's in this repo

| Path | Contents |
|---|---|
| [`blueprint.tex`](blueprint.tex) | Phase-1 blueprint (20 slides, beamer + metropolis theme). Compile with `xelatex` for PDF. |
| [`illustrations/`](illustrations/) | All figures referenced in the blueprint (architecture, distillation, FAR loop, parameter budget, SNR sweeps, SV progression, FiLM trade-off, custom-word ROC, deployment latency). |
| [`code/`](code/) | Pilot implementation: BC-ResNet trunk, learnable PCEN, joint KWS+SV+FiLM, sub-center AAM-Softmax, training/eval scripts. See [`code/README.md`](code/README.md). |
| [`code/scripts/`](code/scripts/) | Standalone experiment scripts (`film_poc_*.py`, `custom_word_poc.py`, `deploy_benchmark.py`, etc.). |
| [`results/`](results/) | All measured outputs from each experiment (metrics JSON, score CSVs, plots, ONNX exports). |
| [`LOCAL_RUN.md`](LOCAL_RUN.md) | Local pilot setup + smoke-test protocol. |
| [`KAGGLE_RUN.md`](KAGGLE_RUN.md) | Kaggle pilot run instructions. |
| [`complete_details.md`](complete_details.md) | Full project ledger — every experiment, measurement, and design decision. |

## Phase-1 empirical validation summary

| KPI | Target | Measured | Slide |
|---|---|---|---|
| Clean TAR | ≥ 99 % | 97.6 % (KWS-only baseline; full pipeline projects to target) | 8 |
| Noisy TAR (−5 dB) | ≥ 90 % | 84 % (noise-aug-trained, full SNR sweep) | 8 |
| FAR (FiLM rejection mechanism) | < 1/hr | 86 % imposter rejection at matched embeddings; trade-off characterised | 8c |
| Param budget | < 3 M | **499.6 K (16.6 % of cap)** | 6, 8e |
| xRT (Pi 4B projected) | < 0.2 s | **12 ms INT8** (17× headroom) | 8e |
| Custom-word generalisation | (PS goal) | EER 12.4 %, AUC 0.934 on novel words | 8d |
| SV branch | (PS goal) | EER 10.64 % on VoxCeleb1-O | 8b |

## Compiling the blueprint

```bash
# Overleaf (recommended) — upload blueprint.tex + illustrations/ folder
# Compiler setting: XeLaTeX

# Or local TeX with the metropolis beamer theme installed:
xelatex blueprint.tex
```

## Reproducing the experiments

See [`code/README.md`](code/README.md) for setup, then [`code/scripts/`](code/scripts/) for individual run scripts. Each script writes its outputs under `results/<experiment_name>/`.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). All code, weights, and benchmarks released openly per hackathon rules.
