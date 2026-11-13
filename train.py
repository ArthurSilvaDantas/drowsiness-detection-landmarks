"""Train the facial landmark model on the WFLW TRAIN split only (the test split is never read).

Reproducible with a single command; seed and hyper-parameters come from the config file:

    python train.py --data data --config configs/001_baseline.yaml --out runs/001_baseline

Outputs in --out: best.pt (best validation epoch), last.pt, log.csv, curves.png, config.yaml.
The test split is never read here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import (Cache, CoordinateLoss, FaceLandmarks, build_cache, build_model,
                   count_parameters, ear, evaluate_arrays, read_faces, read_mirror,
                   split_train_val)

DEFAULT_CONFIG = Path("configs/013_final_strong_lowres.yaml")


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, gts, idxs = [], [], []
    for x, y, i in loader:
        coords, _ = model(x.to(device))
        preds.append(coords.float().cpu().numpy())
        gts.append(y.numpy())
        idxs.append(i.numpy())
    return np.concatenate(preds), np.concatenate(gts), np.concatenate(idxs)


def make_sampler(cache: Cache, indices, cfg, seed):
    s = cfg.get("sampler", {})
    w_closed = float(s.get("closed_eye_weight", 1.0))
    if w_closed == 1.0:
        return None
    e = ear(cache.landmarks_orig[indices].astype(np.float64))
    weights = np.where(e < s.get("ear_threshold", 0.15), w_closed, 1.0)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(indices),
                                 replacement=True, generator=g)


def save_checkpoint(path, model, cfg, epoch, val):
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val,
                "input_size": cfg["data"]["input_size"], "margin": cfg["data"]["margin"]},
               path)


def plot_curves(log_rows, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    ep = [r["epoch"] for r in log_rows]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    axs[0].plot(ep, [r["train_loss"] for r in log_rows], label="treino")
    axs[0].set_title("perda (L1 normalizada)")
    axs[1].plot(ep, [100 * r["val_nme"] for r in log_rows], label="NME")
    axs[1].plot(ep, [100 * r["val_eye_nme"] for r in log_rows], label="NME olhos")
    axs[1].set_title("validação: NME (%)")
    axs[1].legend()
    axs[2].plot(ep, [r["val_ear_corr"] for r in log_rows])
    axs[2].set_title("validação: correlação do EAR")
    for ax in axs:
        ax.set_xlabel("época")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data"), help="WFLW folder")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=None, help="output folder (runs/<name>)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
    p.add_argument("--epochs", type=int, default=None, help="override (quick tests)")
    p.add_argument("--limit", type=int, default=None, help="use only N train faces (tests)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    out = args.out or Path("runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed_everything(cfg["seed"])
    device = pick_device(args.device)
    tc = cfg["train"]

    cache = Cache(build_cache(args.data, "train"))
    faces = read_faces(args.data, "train")
    # a divisão usa uma semente própria (42 por padrão): trocar a semente do treino não muda a validação
    train_idx, val_idx = split_train_val(faces, cfg["data"]["val_fraction"], cfg["data"].get("split_seed", 42))
    if args.limit:
        train_idx = train_idx[:args.limit]
    mirror = read_mirror(args.data)
    common = {"input_size": cfg["data"]["input_size"], "margin": cfg["data"]["margin"]}
    train_ds = FaceLandmarks(cache, train_idx, augment=cfg["augment"], mirror=mirror,
                             seed=cfg["seed"], **common)
    val_ds = FaceLandmarks(cache, val_idx, **common)
    sampler = make_sampler(cache, train_idx, cfg, cfg["seed"])
    loader_kw = {"num_workers": tc["num_workers"], "persistent_workers": tc["num_workers"] > 0,
                 "pin_memory": device.type == "cuda"}
    train_dl = DataLoader(train_ds, tc["batch_size"], shuffle=sampler is None,
                          sampler=sampler, drop_last=True,
                          generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kw)
    val_dl = DataLoader(val_ds, 64, shuffle=False, **loader_kw)

    model = build_model(cfg).to(device)
    params = count_parameters(model)
    print(f"device={device} | train={len(train_ds)} val={len(val_ds)} | params={params}")
    lc = cfg["loss"]
    loss_fn = CoordinateLoss(lc["kind"], cfg["data"]["input_size"], lc.get("eye_weight", 1.0),
                             opening_weight=lc.get("opening_weight", 0.0),
                             heatmap_weight=lc.get("heatmap_weight", 0.0),
                             heatmap_sigma=lc.get("heatmap_sigma", 1.5)).to(device)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": tc["backbone_lr"]},
        {"params": model.head.parameters(), "lr": tc["lr"]},
    ], weight_decay=tc["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    steps_per_epoch = len(train_dl)
    total_steps = tc["epochs"] * steps_per_epoch
    warmup = tc.get("warmup_epochs", 0) * steps_per_epoch

    def set_lr(step):
        if step < warmup:
            f = (step + 1) / warmup
        else:
            f = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f

    select = tc.get("select_metric", "val_nme")
    best, log_rows, step = math.inf, [], 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, _ in train_dl:
            set_lr(step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            coords, logits = model(x)
            loss = loss_fn(coords, y, logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            step += 1
        pred, gt, _ = predict_loader(model, val_dl, device)
        val = evaluate_arrays(pred, gt)
        row = {"epoch": epoch, "train_loss": running / steps_per_epoch,
               **{f"val_{k}": v for k, v in val.items()},
               "lr": opt.param_groups[1]["lr"], "seconds": time.time() - t0}
        log_rows.append(row)
        print(f"ep {epoch:3d} | loss {row['train_loss']:.4f} | NME {100 * val['nme']:.2f}% "
              f"| eye {100 * val['eye_nme']:.2f}% | FR {100 * val['fr10']:.1f}% "
              f"| EARcorr {val['ear_corr']:.3f} | slope {val['ear_slope']:.2f} "
              f"| fechados {val['closed_ear_pred']:.3f} ({100 * val['closed_recall']:.0f}%) "
              f"| {row['seconds']:.0f}s", flush=True)
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(log_rows)
        score = row[select] if select != "val_ear_corr" else -row[select]
        if score < best:
            best = score
            save_checkpoint(out / "best.pt", model, cfg, epoch, val)
        save_checkpoint(out / "last.pt", model, cfg, epoch, val)
        plot_curves(log_rows, out / "curves.png")

    best_ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    summary = {"best_epoch": best_ckpt["epoch"], "val": best_ckpt["val"], "params": params,
               "device": str(device), "train_faces": len(train_ds), "val_faces": len(val_ds),
               "minutes": sum(r["seconds"] for r in log_rows) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    shutil.copy(args.config, out / "config_source.yaml")


if __name__ == "__main__":
    main()
