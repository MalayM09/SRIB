"""Pilot trainer — single loop, KWS only.

Reads a YAML config, builds mel+PCEN front-end + BC-ResNet-8 + KWS head,
trains for N epochs on Speech Commands V2, logs top-1 accuracy each epoch.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train --config configs/pilot_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (MUSANMixer, RIRConvolver, SpeechCommandsV2, SpecAugment,
                      WaveformAugment)
from src.losses import FocalCrossEntropy
from src.model import BCResNet8, KWSHead, LearnablePCEN
from src.utils import get_logger, set_seed

log = get_logger("train")


def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MelFrontend(nn.Module):
    """waveform (B, T) -> (B, n_mels, frames) log-mel or PCEN spectrogram."""

    def __init__(self, n_mels: int = 40, sample_rate: int = 16000,
                 use_pcen: bool = True):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,   # 25 ms
            hop_length=160,   # 10 ms
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.use_pcen = use_pcen
        if use_pcen:
            self.pcen = LearnablePCEN(n_mels)
        else:
            self.eps = 1e-6

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T)
        mel = self.melspec(wav)  # (B, F, frames)
        if self.use_pcen:
            return self.pcen(mel)
        return torch.log(mel + self.eps)


class KWSModel(nn.Module):
    def __init__(
        self,
        n_mels: int = 40,
        num_classes: int = 12,
        use_pcen: bool = True,
        specaug: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.frontend = MelFrontend(n_mels=n_mels, sample_rate=sample_rate, use_pcen=use_pcen)
        self.specaug = SpecAugment() if specaug else nn.Identity()
        self.trunk = BCResNet8(n_mels=n_mels)
        self.head = KWSHead(self.trunk.out_channels, num_classes=num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.frontend(wav)           # (B, F, T')
        spec = self.specaug(spec)           # (B, F, T')
        feats = self.trunk(spec.unsqueeze(1))  # (B, C, F'', T')
        return self.head(feats)


def build_augment(cfg: dict) -> WaveformAugment | None:
    aug_cfg = cfg.get("augment", {})
    data_cfg = cfg["data"]
    rir = None
    musan = None
    rir_dir = Path(data_cfg["root"]).parent / "rirs_small"
    musan_dir = Path(data_cfg["root"]).parent / "musan_small"
    if aug_cfg.get("rir_prob", 0) > 0 and rir_dir.exists():
        rir = RIRConvolver(rir_dir, prob=aug_cfg["rir_prob"])
    if aug_cfg.get("musan_prob", 0) > 0 and musan_dir.exists():
        musan = MUSANMixer(
            musan_dir,
            prob=aug_cfg["musan_prob"],
            snr_range=tuple(aug_cfg.get("snr_range", (0.0, 20.0))),
        )
    if rir is None and musan is None:
        return None
    return WaveformAugment(rir=rir, musan=musan)


def train_one_epoch(model, loader, augment, loss_fn, optimizer, device, epoch, scheduler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"train ep{epoch}", leave=False)
    for wav, label in pbar:
        if augment is not None:
            wav = torch.stack([augment(w) for w in wav])
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(wav)
        loss = loss_fn(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        preds = logits.argmax(dim=-1)
        total += label.size(0)
        correct += (preds == label).sum().item()
        loss_sum += loss.item() * label.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{correct/total:.3f}")
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, split: str):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for wav, label in tqdm(loader, desc=f"{split}", leave=False):
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(wav)
        loss = loss_fn(logits, label)
        total += label.size(0)
        correct += (logits.argmax(-1) == label).sum().item()
        loss_sum += loss.item() * label.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_cfg = cfg["run"]
    out_dir = Path(run_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(run_cfg["seed"])
    device = pick_device(run_cfg["device"])
    log.info(f"device: {device} | config: {args.config}")

    # ---- datasets
    data_cfg = cfg["data"]
    train_ds = SpeechCommandsV2(
        data_cfg["root"], split="train",
        subset_size=data_cfg.get("subset_train"),
        seed=run_cfg["seed"],
    )
    val_ds = SpeechCommandsV2(
        data_cfg["root"], split="val", seed=run_cfg["seed"],
    )
    log.info(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"], shuffle=True,
        num_workers=data_cfg.get("num_workers", 1), drop_last=True,
        persistent_workers=data_cfg.get("num_workers", 1) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["batch_size"], shuffle=False,
        num_workers=data_cfg.get("num_workers", 1),
    )

    # ---- augmentation (waveform-domain, per-sample in main process for simplicity)
    augment = build_augment(cfg)
    if augment is not None:
        log.info(f"augment: rir={augment.rir is not None} musan={augment.musan is not None}")

    # ---- model
    model_cfg = cfg["model"]
    model = KWSModel(
        n_mels=model_cfg["n_mels"],
        num_classes=model_cfg["num_classes"],
        use_pcen=model_cfg.get("use_pcen", True),
        specaug=cfg.get("augment", {}).get("specaugment", True),
        sample_rate=data_cfg["sample_rate"],
    ).to(device)

    n_trunk = model.trunk.num_parameters()
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"params: trunk={n_trunk/1e3:.1f}K total={n_total/1e3:.1f}K")

    # ---- optim
    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * opt_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    loss_fn = FocalCrossEntropy()

    # ---- train
    history = []
    best_val = 0.0
    for epoch in range(1, opt_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, augment, loss_fn, optimizer, device, epoch, scheduler,
        )
        vl_loss, vl_acc = evaluate(model, val_loader, loss_fn, device, "val")
        dt = time.time() - t0
        log.info(
            f"ep{epoch} | train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | "
            f"val_loss={vl_loss:.3f} val_acc={vl_acc:.3f} | {dt:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc": vl_acc, "wall_s": dt,
        })
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                       out_dir / "best.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": opt_cfg["epochs"]},
               out_dir / "final.pt")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"history": history, "best_val_acc": best_val,
         "trunk_params": n_trunk, "total_params": n_total},
        indent=2,
    ))
    log.info(f"done. best_val={best_val:.3f} | wrote {out_dir}/final.pt")


if __name__ == "__main__":
    main()
