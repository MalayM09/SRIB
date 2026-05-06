"""WavLM distillation for KWS — closes the gap to ≥ 99% TA Clean.

Adds a final-layer cosine distillation loss to the standard KWS training:

  Total loss = CE(label-smoothed)  +  λ · (1 - cos_sim(student_proj, teacher_emb))

Teacher  : microsoft/wavlm-base-plus  (frozen, ~95M params)
Student  : existing KWSModel  +  one Linear projection head (trunk_C → 768)

Validates the WavLM-distillation claim already cited on Slide 5 of the blueprint.

Usage from code/:
    .venv/bin/python scripts/wavlm_distill.py --config configs/pilot_p3_wavlm_distill.yaml --smoke
    .venv/bin/python scripts/wavlm_distill.py --config configs/pilot_p3_wavlm_distill.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.data import SpeechCommandsV2
from src.train import KWSModel, build_augment, evaluate, pick_device
from src.utils import get_logger, set_seed

log = get_logger("wavlm_distill")


class DistillStudent(nn.Module):
    """KWSModel + an extra Linear projection (trunk_C → 768) for distillation.
    Both heads share trunk features."""

    def __init__(self, base_model: KWSModel, distill_dim: int = 768):
        super().__init__()
        self.base = base_model
        trunk_C = base_model.trunk.out_channels
        self.distill_proj = nn.Linear(trunk_C, distill_dim)

    def forward(self, wav: torch.Tensor):
        spec = self.base.frontend(wav)
        spec = self.base.specaug(spec)
        feats = self.base.trunk(spec.unsqueeze(1))           # (B, C, F'', T'')
        kws_logits = self.base.head(feats)
        # Global avg-pool over (F'', T'') → (B, C), then project to teacher dim
        distill_pooled = feats.mean(dim=[2, 3])              # (B, C)
        distill_emb = self.distill_proj(distill_pooled)       # (B, 768)
        return kws_logits, distill_emb


def load_teacher(name: str, device: torch.device) -> nn.Module:
    """Load a frozen WavLM teacher (use safetensors to avoid torch.load gating)."""
    from transformers import WavLMModel
    log.info(f"loading teacher: {name}")
    try:
        teacher = WavLMModel.from_pretrained(name, use_safetensors=True).to(device)
    except (TypeError, ValueError):
        teacher = WavLMModel.from_pretrained(name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in teacher.parameters())
    log.info(f"teacher loaded: {n/1e6:.1f}M params (frozen)")
    return teacher


@torch.no_grad()
def teacher_embed(teacher, wav: torch.Tensor) -> torch.Tensor:
    """WavLM teacher → mean-pooled hidden state at the final layer.
    wav: (B, T) raw 16 kHz audio. Returns (B, 768) L2-normalised."""
    out = teacher(wav)                                       # last_hidden_state: (B, T', 768)
    emb = out.last_hidden_state.mean(dim=1)                  # (B, 768)
    return F.normalize(emb, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 2 epochs, for local validation")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        cfg["data"]["subset_train"] = 800
        cfg["optim"]["epochs"] = 2
        cfg["data"]["batch_size"] = 16
        cfg["data"]["num_workers"] = 0
        log.info("[smoke mode] 800 utts / 2 epochs / batch 16")

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
    val_ds = SpeechCommandsV2(data_cfg["root"], split="val", seed=run_cfg["seed"])
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

    # ---- augmentation (waveform-domain)
    augment = build_augment(cfg)
    if augment is not None:
        log.info(f"augment: rir={augment.rir is not None} "
                 f"musan={augment.musan is not None} "
                 f"babble={augment.babble is not None}")

    # ---- student model
    model_cfg = cfg["model"]
    base = KWSModel(
        n_mels=model_cfg["n_mels"],
        num_classes=model_cfg["num_classes"],
        use_pcen=model_cfg.get("use_pcen", True),
        specaug=cfg.get("augment", {}).get("specaugment", True),
        sample_rate=data_cfg["sample_rate"],
        trunk_channels=model_cfg.get("trunk_channels"),
    )
    distill_dim = model_cfg.get("distill_dim", 768)
    model = DistillStudent(base, distill_dim=distill_dim).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_trunk = base.trunk.num_parameters()
    n_distill = sum(p.numel() for p in model.distill_proj.parameters())
    log.info(f"student params: trunk={n_trunk/1e3:.1f}K total={n_total/1e3:.1f}K "
             f"(distill_proj={n_distill/1e3:.1f}K, dropped at inference)")

    # ---- teacher
    teacher = load_teacher(
        model_cfg.get("teacher", "microsoft/wavlm-base-plus"), device,
    )

    # ---- optim + losses
    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * opt_cfg["epochs"],
    )
    ls = float(opt_cfg.get("label_smoothing", 0.1))
    kws_loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)
    lam = float(opt_cfg.get("lambda_distill", 1.0))
    log.info(f"loss: CE(label_smoothing={ls}) + {lam}·(1−cos_sim) on WavLM final-layer mean")

    # ---- train
    history = []
    best_val = 0.0
    for epoch in range(1, opt_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        # Teacher always eval
        teacher.eval()
        l_total_sum = l_kws_sum = l_dis_sum = 0.0
        correct = total = 0

        for wav, label in tqdm(train_loader, desc=f"train ep{epoch}", leave=False):
            wav = wav.to(device); label = label.to(device)
            if augment is not None:
                wav = torch.stack([augment(w.cpu()) for w in wav]).to(device)

            t_emb = teacher_embed(teacher, wav)              # (B, 768) frozen, normalised
            kws_logits, s_emb = model(wav)
            s_emb_n = F.normalize(s_emb, dim=-1)

            l_kws = kws_loss_fn(kws_logits, label)
            l_dis = (1.0 - F.cosine_similarity(s_emb_n, t_emb, dim=-1)).mean()
            loss = l_kws + lam * l_dis

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            l_total_sum += loss.item() * wav.size(0)
            l_kws_sum   += l_kws.item() * wav.size(0)
            l_dis_sum   += l_dis.item() * wav.size(0)
            preds = kws_logits.argmax(-1)
            total += int(label.size(0))
            correct += int((preds == label).sum().item())

        train_acc = correct / max(total, 1)

        # ---- validation (KWS only — no teacher needed)
        model.eval()
        v_correct = v_total = 0
        v_loss_sum = 0.0
        with torch.no_grad():
            for wav, label in val_loader:
                wav = wav.to(device); label = label.to(device)
                kws_logits, _ = model(wav)
                vl = kws_loss_fn(kws_logits, label)
                v_loss_sum += vl.item() * label.size(0)
                v_correct += int((kws_logits.argmax(-1) == label).sum().item())
                v_total   += int(label.size(0))
        val_acc = v_correct / max(v_total, 1)
        val_loss = v_loss_sum / max(v_total, 1)

        dt = time.time() - t0
        log.info(
            f"ep{epoch:>2} | l_total={l_total_sum/total:.3f} "
            f"l_kws={l_kws_sum/total:.3f} l_dis={l_dis_sum/total:.3f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"val_loss={val_loss:.4f} | {dt:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "l_total": l_total_sum / total,
            "l_kws":   l_kws_sum / total,
            "l_dis":   l_dis_sum / total,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "wall_s": dt,
        })

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                out_dir / "best.pt",
            )

    torch.save(
        {"model": model.state_dict(), "cfg": cfg, "epoch": opt_cfg["epochs"]},
        out_dir / "final.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps(
        {"history": history, "best_val_acc": best_val,
         "trunk_params": n_trunk, "total_params": n_total},
        indent=2,
    ))
    log.info(f"done. best_val={best_val:.4f} | wrote {out_dir}/final.pt")


if __name__ == "__main__":
    main()
