# srib-edge-kws-sv

Pilot scaffold for Samsung ennovateX AX Hackathon Phase-1 — edge-AI speaker-specific custom word detection.

See the project-root `LOCAL_RUN.md` for setup + smoke-test protocol and `KAGGLE_RUN.md` for Kaggle pilot runs. Both are the authoritative ground truth for this repo.

## Quickstart (local smoke-test)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
bash scripts/download_sc_v2.sh
bash scripts/download_musan_small.sh
bash scripts/download_rirs_small.sh
pytest -q
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config configs/pilot_smoke.yaml
```
