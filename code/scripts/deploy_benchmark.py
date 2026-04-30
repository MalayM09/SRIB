"""Step F — Deployment validation: ONNX export + CPU benchmark + INT8 quantization.

Validates the blueprint claims:
  - Joint model fits in <3 M params
  - xRT < 0.2 s on edge hardware (measured on Mac M1 CPU as Pi 4B proxy)

Pipeline:
  1. Load joint KWS+SV+FiLM checkpoint (FiLM v3, 500 K params)
  2. Export to ONNX (FP32)
  3. Benchmark FP32 single-threaded
  4. Calibrate + apply static INT8 quantization with SC V2 audio
  5. Benchmark INT8 single-threaded
  6. Verify numeric agreement: PyTorch vs ONNX FP32 vs ONNX INT8
  7. Project Pi 4B latency from Mac CPU number (~3.5x slower per core)
  8. Render slide-quality plot + summary JSON

Run from code/:
    .venv/bin/python scripts/deploy_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from film_poc_v3 import JointKWSSV  # the joint model we benchmark

REPO_ROOT = HERE.parent.parent
SC_ROOT   = REPO_ROOT / "data" / "speech_commands_v2"
CKPT_PATH = REPO_ROOT / "results" / "film_poc_v3" / "joint_kws_sv_film_v3.pt"
OUT_DIR   = REPO_ROOT / "results" / "deploy_benchmark"
SAMPLE_RATE = 16000
CLIP_LEN = 16000

# Mac M1 single-core → Pi 4B Cortex-A72 single-core ratio.
# Sources: M1 Firestorm ~3.2 GHz, A72 ~1.5 GHz, plus IPC differences.
# Conservative estimate: Pi is ~3.5x slower per core on this kind of workload.
M1_TO_PI_RATIO = 3.5


def load_clip(path: str) -> np.ndarray:
    wav, _ = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    if len(wav) < CLIP_LEN:
        wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
    return wav[:CLIP_LEN]


# ---------------------------------------------------------------- model wrapper for ONNX
# torchaudio.MelSpectrogram uses torch.stft → complex tensors → won't ONNX-export.
# Solution: keep frontend (mel + PCEN) in PyTorch, ONNX-export trunk + heads only.
# Total xRT = mel_pcen_time + onnx_trunk_time.
class DeployableTrunkHeads(nn.Module):
    """ONNX-exportable wrapper: takes pre-computed PCEN spectrogram and SV modulator,
    runs trunk + SV head + FiLM-modulated KWS head."""
    def __init__(self, joint: JointKWSSV):
        super().__init__()
        self.trunk = joint.trunk
        self.sv_pool = joint.sv_pool
        self.sv_proj = joint.sv_proj
        self.kws_pool = joint.kws_pool
        self.film_gen = joint.film_gen
        self.kws_head = joint.kws_head

    def forward(self, spec, sv_emb_modulator):
        # spec: (B, F, T) PCEN'd mel-spectrogram (frontend output)
        # sv_emb_modulator: (B, 256) enrolled speaker embedding
        feats = self.trunk(spec.unsqueeze(1))   # (B, C, F'', T'')
        feats = feats.mean(dim=2)               # (B, C, T'')
        sv_pooled = self.sv_pool(feats)
        import torch.nn.functional as F
        sv_emb = F.normalize(self.sv_proj(sv_pooled), dim=-1)
        kws_pooled = self.kws_pool(feats)
        film  = self.film_gen(sv_emb_modulator)
        gamma, beta = film.chunk(2, dim=-1)
        modulated = kws_pooled * (1 + gamma) + beta
        kws_logits = self.kws_head(modulated)
        return kws_logits, sv_emb


# ---------------------------------------------------------------- INT8 calibration
class WavCalibReader:
    """ONNX Runtime CalibrationDataReader interface using SC V2 wavs."""
    def __init__(self, samples, input_names):
        self.samples = iter(samples)
        self.input_names = input_names

    def get_next(self):
        try:
            wav, dummy_emb = next(self.samples)
            return {
                self.input_names[0]: wav,
                self.input_names[1]: dummy_emb,
            }
        except StopIteration:
            return None


def collect_calibration_data(n_samples, frontend, seed=1337):
    """Pull n random wav clips from SC V2, run frontend (mel+PCEN), build (spec, dummy_emb) pairs."""
    rng = random.Random(seed)
    all_wavs = []
    for word in ["yes", "no", "up", "down", "stop", "go", "left", "right"]:
        d = SC_ROOT / word
        all_wavs.extend(list(d.glob("*.wav")))
    rng.shuffle(all_wavs)
    samples = []
    with torch.no_grad():
        for p in all_wavs[:n_samples]:
            wav = load_clip(str(p))
            wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, 16000)
            spec = frontend(wav_t).numpy().astype(np.float32)  # (1, F, T)
            emb_vec = np.array([rng.gauss(0, 1) for _ in range(256)], dtype=np.float32)
            emb_vec = emb_vec / (np.linalg.norm(emb_vec) + 1e-9)
            samples.append((spec, emb_vec[None].astype(np.float32)))
    return samples


# ---------------------------------------------------------------- benchmark
def benchmark_onnx(onnx_path, sample_spec, n_iters=200, n_warmup=20, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), so,
                                   providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    dummy_emb = np.random.randn(1, 256).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb, axis=-1, keepdims=True) + 1e-9

    feed = {input_names[0]: sample_spec.astype(np.float32),
            input_names[1]: dummy_emb}
    for _ in range(n_warmup):
        session.run(None, feed)
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0
    return {
        "n_iters": n_iters, "threads": threads,
        "mean_ms": float(times.mean()), "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(times.min()), "max_ms": float(times.max()),
    }


def benchmark_frontend(frontend, n_iters=200, n_warmup=20):
    """Measure mel + PCEN time on raw waveform input, single-thread CPU."""
    torch.set_num_threads(1)
    frontend.eval()
    dummy = torch.randn(1, CLIP_LEN)
    with torch.no_grad():
        for _ in range(n_warmup):
            frontend(dummy)
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            frontend(dummy)
            times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0
    return {
        "n_iters": n_iters, "threads": 1,
        "mean_ms": float(times.mean()), "std_ms": float(times.std()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(times.min()), "max_ms": float(times.max()),
    }


def verify_numerics(pytorch_model, fp32_path, int8_path, samples, device="cpu"):
    """Compare PyTorch vs ONNX FP32 vs ONNX INT8 outputs on the same spec input."""
    import onnxruntime as ort
    pytorch_model.eval()

    sess_fp32 = ort.InferenceSession(str(fp32_path),
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path),
                                     providers=["CPUExecutionProvider"])
    in_fp32 = [i.name for i in sess_fp32.get_inputs()]
    in_int8 = [i.name for i in sess_int8.get_inputs()]

    diffs_kws_fp32, diffs_kws_int8 = [], []
    diffs_sv_fp32,  diffs_sv_int8  = [], []
    with torch.no_grad():
        for spec, emb in samples[:16]:
            spec_t = torch.from_numpy(spec).to(device)
            emb_t  = torch.from_numpy(emb).to(device)
            pyt_kws, pyt_sv = pytorch_model(spec_t, emb_t)
            pyt_kws = pyt_kws.cpu().numpy(); pyt_sv = pyt_sv.cpu().numpy()

            fp32_out = sess_fp32.run(None, {in_fp32[0]: spec, in_fp32[1]: emb})
            int8_out = sess_int8.run(None, {in_int8[0]: spec, in_int8[1]: emb})

            diffs_kws_fp32.append(np.max(np.abs(pyt_kws - fp32_out[0])))
            diffs_kws_int8.append(np.max(np.abs(pyt_kws - int8_out[0])))
            diffs_sv_fp32.append(np.max(np.abs(pyt_sv - fp32_out[1])))
            diffs_sv_int8.append(np.max(np.abs(pyt_sv - int8_out[1])))

    return {
        "kws_logits_max_abs_diff_fp32": float(np.mean(diffs_kws_fp32)),
        "kws_logits_max_abs_diff_int8": float(np.mean(diffs_kws_int8)),
        "sv_emb_max_abs_diff_fp32":     float(np.mean(diffs_sv_fp32)),
        "sv_emb_max_abs_diff_int8":     float(np.mean(diffs_sv_int8)),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_PATH))
    ap.add_argument("--n_iters", type=int, default=200)
    ap.add_argument("--n_warmup", type=int, default=20)
    ap.add_argument("--n_calib", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1, help="single core like Pi")
    args = ap.parse_args()

    print(f"output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model on CPU
    print("\n[1/8] Loading joint model on CPU")
    joint = JointKWSSV(num_keywords=12, sv_dim=256).to("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    joint.load_state_dict(ckpt["model"])
    joint.eval()
    deployable = DeployableTrunkHeads(joint).to("cpu").eval()
    n_params = sum(p.numel() for p in joint.parameters())
    n_params_trunk_heads = sum(p.numel() for p in deployable.parameters())
    print(f"  loaded {Path(args.ckpt).name}")
    print(f"  joint params total : {n_params/1e3:.1f}K")
    print(f"  trunk+heads (ONNX) : {n_params_trunk_heads/1e3:.1f}K")
    print(f"  frontend (PyTorch) : {(n_params - n_params_trunk_heads)/1e3:.1f}K")

    # 2. Benchmark frontend (mel + PCEN, stays in PyTorch CPU)
    print("\n[2/8] Benchmarking frontend (mel + PCEN, single-thread PyTorch)")
    frontend_perf = benchmark_frontend(joint.frontend,
                                       n_iters=args.n_iters,
                                       n_warmup=args.n_warmup)
    print(f"  mean={frontend_perf['mean_ms']:.2f} ms  "
          f"p50={frontend_perf['p50_ms']:.2f}  p95={frontend_perf['p95_ms']:.2f}")

    # 3. Build calibration data + sample spec for benchmarking
    print("\n[3/8] Computing calibration data (frontend → spectrograms)")
    samples = collect_calibration_data(args.n_calib, joint.frontend)
    print(f"  calibration samples: {len(samples)}  spec shape: {samples[0][0].shape}")
    sample_spec = samples[0][0]   # (1, F, T)

    # 4. ONNX export (FP32) of trunk + heads
    print("\n[4/8] Exporting trunk+heads → ONNX FP32")
    fp32_path = OUT_DIR / "trunk_heads_fp32.onnx"
    dummy_spec = torch.from_numpy(sample_spec)
    dummy_emb = torch.randn(1, 256)
    dummy_emb = dummy_emb / dummy_emb.norm(dim=-1, keepdim=True)
    torch.onnx.export(
        deployable,
        (dummy_spec, dummy_emb),
        str(fp32_path),
        input_names=["spec", "sv_emb_modulator"],
        output_names=["kws_logits", "sv_emb"],
        dynamic_axes={
            "spec": {0: "batch"},
            "sv_emb_modulator": {0: "batch"},
            "kws_logits": {0: "batch"},
            "sv_emb": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    fp32_size_mb = fp32_path.stat().st_size / 1e6
    print(f"  FP32 ONNX: {fp32_path.name} ({fp32_size_mb:.2f} MB)")

    # 5. Benchmark FP32 trunk+heads single-threaded
    print("\n[5/8] Benchmarking FP32 ONNX trunk+heads (single-thread)")
    fp32_perf = benchmark_onnx(fp32_path, sample_spec,
                               n_iters=args.n_iters,
                               n_warmup=args.n_warmup, threads=args.threads)
    print(f"  mean={fp32_perf['mean_ms']:.2f} ms  "
          f"p50={fp32_perf['p50_ms']:.2f}  p95={fp32_perf['p95_ms']:.2f}")

    # 6. Static INT8 quantization
    print("\n[6/8] Calibrating + quantizing trunk+heads to INT8")

    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    )

    class CalibReader(CalibrationDataReader):
        def __init__(self, samples, in_names):
            self.iter = iter(samples)
            self.in_names = in_names
        def get_next(self):
            try:
                w, e = next(self.iter)
                return {self.in_names[0]: w, self.in_names[1]: e}
            except StopIteration:
                return None

    int8_path = OUT_DIR / "trunk_heads_int8.onnx"
    import onnxruntime as ort
    tmp_session = ort.InferenceSession(str(fp32_path),
                                       providers=["CPUExecutionProvider"])
    in_names = [i.name for i in tmp_session.get_inputs()]
    quantize_static(
        str(fp32_path),
        str(int8_path),
        calibration_data_reader=CalibReader(samples, in_names),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    int8_size_mb = int8_path.stat().st_size / 1e6
    print(f"  INT8 ONNX: {int8_path.name} ({int8_size_mb:.2f} MB)")

    # 7. Benchmark INT8 single-threaded
    print("\n[7/8] Benchmarking INT8 ONNX trunk+heads (single-thread)")
    int8_perf = benchmark_onnx(int8_path, sample_spec,
                               n_iters=args.n_iters,
                               n_warmup=args.n_warmup, threads=args.threads)
    print(f"  mean={int8_perf['mean_ms']:.2f} ms  "
          f"p50={int8_perf['p50_ms']:.2f}  p95={int8_perf['p95_ms']:.2f}")

    # 8. Numeric agreement check
    print("\n[8/8] Verifying numeric agreement (PyTorch vs ONNX FP32 vs INT8)")
    accuracy = verify_numerics(deployable, fp32_path, int8_path, samples)
    print(f"  KWS logits  | FP32 max-abs-diff: {accuracy['kws_logits_max_abs_diff_fp32']:.6f}")
    print(f"  KWS logits  | INT8 max-abs-diff: {accuracy['kws_logits_max_abs_diff_int8']:.6f}")
    print(f"  SV embedding| FP32 max-abs-diff: {accuracy['sv_emb_max_abs_diff_fp32']:.6f}")
    print(f"  SV embedding| INT8 max-abs-diff: {accuracy['sv_emb_max_abs_diff_int8']:.6f}")

    # Pi 4B projection + summary
    print("\nProjecting Pi 4B latency + writing summary")
    # Total xRT = frontend (mel+PCEN) + trunk+heads (ONNX)
    fp32_total_m1 = frontend_perf["mean_ms"] + fp32_perf["mean_ms"]
    int8_total_m1 = frontend_perf["mean_ms"] + int8_perf["mean_ms"]
    fp32_total_pi = fp32_total_m1 * M1_TO_PI_RATIO
    int8_total_pi = int8_total_m1 * M1_TO_PI_RATIO

    summary = {
        "ckpt": Path(args.ckpt).name,
        "params_total": int(n_params),
        "params_3M_cap_pct": round(100 * n_params / 3_000_000, 2),
        "fp32_onnx_size_mb": round(fp32_size_mb, 3),
        "int8_onnx_size_mb": round(int8_size_mb, 3),
        "size_compression": round(fp32_size_mb / max(int8_size_mb, 1e-9), 2),
        "mac_m1_cpu_single_thread_ms": {
            "frontend_mel_pcen": frontend_perf,
            "trunk_heads_fp32":  fp32_perf,
            "trunk_heads_int8":  int8_perf,
            "total_fp32_mean":   round(fp32_total_m1, 2),
            "total_int8_mean":   round(int8_total_m1, 2),
            "trunk_speedup_int8": round(fp32_perf["mean_ms"] / max(int8_perf["mean_ms"], 1e-9), 2),
        },
        "pi4b_projection_single_core_ms": {
            "ratio_used":         M1_TO_PI_RATIO,
            "total_fp32_mean":    round(fp32_total_pi, 2),
            "total_int8_mean":    round(int8_total_pi, 2),
            "xRT_target_ms":      200,
            "fp32_meets_xRT":     fp32_total_pi < 200,
            "int8_meets_xRT":     int8_total_pi < 200,
            "int8_xRT_headroom":  round(200 - int8_total_pi, 2),
        },
        "numeric_agreement": accuracy,
    }
    json.dump(summary, open(OUT_DIR / "deploy_summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) stacked latency: frontend + trunk(FP32 or INT8), M1 vs Pi
    fe_m1 = frontend_perf["mean_ms"]
    fe_pi = fe_m1 * M1_TO_PI_RATIO
    bar_labels = ["M1 FP32", "M1 INT8", "Pi 4B FP32\n(projected)", "Pi 4B INT8\n(projected)"]
    fe_vals    = [fe_m1, fe_m1, fe_pi, fe_pi]
    trunk_vals = [fp32_perf["mean_ms"], int8_perf["mean_ms"],
                  fp32_perf["mean_ms"] * M1_TO_PI_RATIO,
                  int8_perf["mean_ms"] * M1_TO_PI_RATIO]
    xs = np.arange(len(bar_labels))
    b1 = ax1.bar(xs, fe_vals,    color="#1428A0", edgecolor="white",
                 label="Frontend (mel + PCEN)")
    b2 = ax1.bar(xs, trunk_vals, bottom=fe_vals, color="#65A30D", edgecolor="white",
                 label="Trunk + heads (ONNX)")
    for i, (fe, tr) in enumerate(zip(fe_vals, trunk_vals)):
        total = fe + tr
        ax1.text(i, total + 5, f"{total:.0f} ms", ha="center",
                 fontsize=10, fontweight="bold")
    ax1.axhline(200, color="#B91C1C", linestyle="--", lw=1.4,
                label="xRT budget (200 ms)")
    ax1.set_xticks(xs); ax1.set_xticklabels(bar_labels, fontsize=9)
    ax1.set_ylabel("end-to-end latency (ms)")
    ax1.set_title(f"(a) Joint KWS+SV+FiLM end-to-end latency  ({n_params/1e3:.1f}K params)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(max(b + t for b, t in zip(fe_vals, trunk_vals)), 200) * 1.3)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)

    # (b) model file size
    ax2.bar(["FP32 ONNX\n(trunk+heads)", "INT8 ONNX\n(trunk+heads)"],
            [fp32_size_mb, int8_size_mb],
            color=["#1428A0", "#65A30D"], edgecolor="white")
    for i, v in enumerate([fp32_size_mb, int8_size_mb]):
        ax2.text(i, v + 0.05, f"{v:.2f} MB", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("model file size (MB)")
    ax2.set_title(f"(b) Compressed model size  ({summary['size_compression']:.1f}× shrink)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(fp32_size_mb, int8_size_mb) * 1.4)
    ax2.grid(axis="y", linestyle="--", alpha=0.4); ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "deploy_metrics.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUT_DIR / "deploy_metrics.svg", bbox_inches="tight")
    print(f"\nwrote {OUT_DIR}/deploy_metrics.png")

    print()
    print("=" * 70)
    print("DEPLOYMENT VALIDATION VERDICT")
    print("=" * 70)
    print(f"  params           : {n_params/1e3:.1f}K  ({summary['params_3M_cap_pct']:.2f}% of 3M cap)")
    print(f"  M1 frontend      : {fe_m1:.2f} ms  (mel + PCEN)")
    print(f"  M1 trunk FP32    : {fp32_perf['mean_ms']:.2f} ms  (ONNX)")
    print(f"  M1 trunk INT8    : {int8_perf['mean_ms']:.2f} ms  ({summary['mac_m1_cpu_single_thread_ms']['trunk_speedup_int8']}x faster than FP32)")
    print(f"  M1 TOTAL FP32    : {fp32_total_m1:.2f} ms")
    print(f"  M1 TOTAL INT8    : {int8_total_m1:.2f} ms")
    print(f"  Pi 4B TOTAL FP32 : {fp32_total_pi:.2f} ms  "
          f"{'PASS xRT' if fp32_total_pi < 200 else 'OVER xRT'}")
    print(f"  Pi 4B TOTAL INT8 : {int8_total_pi:.2f} ms  "
          f"{'PASS xRT' if int8_total_pi < 200 else 'OVER xRT'}")
    print(f"  ONNX size        : FP32 {fp32_size_mb:.2f} MB → INT8 {int8_size_mb:.2f} MB ({summary['size_compression']:.1f}x)")


if __name__ == "__main__":
    main()
