from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pair_dataset import (
    ChainPairDataset,
    collate_chains,
    compute_pos_weights,
    files_from_manifest,
)
from src.models.pair_mlp import PairMLP


def run_epoch(model, loader, device, pos_w, lambdas, opt=None):
    train = opt is not None
    model.train(train)
    tot = 0.0
    preds_t, tgts_t, preds_c, tgts_c = [], [], [], []
    for X, yc, yt, yd, dmask in loader:
        X, yc, yt, yd, dmask = [t.to(device) for t in (X, yc, yt, yd, dmask)]
        with torch.set_grad_enabled(train):
            lc, lt, ld = model(X)
            loss_c = F.binary_cross_entropy_with_logits(lc, yc)
            loss_t = F.binary_cross_entropy_with_logits(lt, yt, pos_weight=pos_w)
            if dmask.sum() > 0:
                loss_d = (F.mse_loss(ld, yd, reduction="none") * dmask).sum() / dmask.sum()
            else:
                loss_d = torch.zeros((), device=device)
            loss = (lambdas[0] * loss_c + lambdas[1] * loss_t + lambdas[2] * loss_d)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        tot += float(loss) * X.shape[0]
        preds_t.append(torch.sigmoid(lt).detach().cpu().numpy())
        tgts_t.append(yt.cpu().numpy())
        preds_c.append(torch.sigmoid(lc).detach().cpu().numpy())
        tgts_c.append(yc.cpu().numpy())
    return (
        tot / len(loader.dataset),
        np.concatenate(preds_c), np.concatenate(tgts_c),
        np.concatenate(preds_t), np.concatenate(tgts_t),
    )


def auprc_per_type(pred, tgt, names):
    out = {}
    for k, n in enumerate(names):
        if tgt[:, k].sum() > 0:
            out[n] = float(average_precision_score(tgt[:, k], pred[:, k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/baseline")
    ap.add_argument("--no-resume", action="store_true", help="ignora last.pt e comeca do zero")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = cfg["train"]
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train_f = files_from_manifest(cfg, "train")
    val_f = files_from_manifest(cfg, "val")
    print(f">> chains (cluster split from the manifest): train={len(train_f)} val={len(val_f)}")

    tr = ChainPairDataset(cfg, train_f, rng)
    va = ChainPairDataset(cfg, val_f, rng)
    names = tr.type_names
    print(f">> feat_dim={tr.feat_dim} n_types={tr.n_types} tipos={names}")
    print(f">> cadeias c/ embedding: treino={len(tr)} val={len(va)}")

    bc = tcfg.get("batch_chains", 8)
    tl = DataLoader(tr, batch_size=bc, shuffle=True, collate_fn=collate_chains,
                    num_workers=tcfg.get("num_workers", 2))
    vl = DataLoader(va, batch_size=bc, collate_fn=collate_chains,
                    num_workers=tcfg.get("num_workers", 2))

    model = PairMLP(tr.feat_dim, tr.n_types, tcfg["hidden"], tcfg["layers"], tcfg["dropout"]).to(device)
    pos_w = compute_pos_weights(train_f, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    lambdas = (tcfg["lambda_contact"], tcfg["lambda_types"], tcfg["lambda_dist"])

    best = -1.0
    start_ep = 1
    ckpt_path = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        best = ck["best"]
        print(f">> retomando de last.pt: proxima epoca={start_ep} best={best:.3f}")

    for ep in range(start_ep, tcfg["epochs"] + 1):
        trl, *_ = run_epoch(model, tl, device, pos_w, lambdas, opt)
        vll, pc, tc, pt, tt = run_epoch(model, vl, device, pos_w, lambdas)
        ap_types = auprc_per_type(pt, tt, names)
        ap_contact = average_precision_score(tc, pc) if tc.sum() > 0 else float("nan")
        macro = float(np.mean(list(ap_types.values()))) if ap_types else 0.0
        if ep % 5 == 0 or ep == 1:
            det = " ".join(f"{n}:{v:.2f}" for n, v in ap_types.items())
            print(f"ep{ep:3d} trL={trl:.3f} vaL={vll:.3f} AUPRC_contact={ap_contact:.2f} "
                  f"AUPRC_types_macro={macro:.2f} | {det}")
        if macro > best:
            best = macro
            torch.save({"model": model.state_dict(), "cfg": cfg, "types": names},
                       os.path.join(args.out, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": best, "cfg": cfg, "types": names}, ckpt_path)
    print(f">> melhor AUPRC_types_macro={best:.3f} salvo em {args.out}/best.pt")


if __name__ == "__main__":
    main()
