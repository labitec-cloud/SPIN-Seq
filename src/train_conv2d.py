"""Training of the 2D head - on w x w crops.

Reuses the manifest's cluster split and the baseline's per-type AUPRC metric; the difference is
that the model is PairConv2D and the metrics/loss aggregate only over the VALID PIXELS of each
crop (mask `lm`). Checkpoint resumable per epoch, as in train.py.

Uso:
    python src/train_conv2d.py                      # configs/default.yaml -> outputs/conv2d
    python src/train_conv2d.py --no-resume
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.crop_dataset import ChainCropDataset
from src.data.pair_dataset import compute_class_weights, files_from_manifest
from src.losses import multitask_loss
from src.models.pair_conv2d import PairConv2D
from src.train import auprc_per_type


def ema_init(model):
    """Shadow (float) copy of the model tensors - exponential moving average of the weights."""
    return {k: v.detach().clone().float()
            for k, v in model.state_dict().items() if v.dtype.is_floating_point}


@torch.no_grad()
def ema_update(ema, model, decay):
    msd = model.state_dict()
    for k, v in ema.items():
        v.mul_(decay).add_(msd[k].detach().float(), alpha=1.0 - decay)


def ema_swap(model, ema):
    """Installs the EMA weights into the model and returns a backup of the originals (to restore)."""
    msd = model.state_dict()
    backup = {k: msd[k].detach().clone() for k in ema}
    model.load_state_dict({k: (ema[k].to(msd[k].dtype) if k in ema else msd[k])
                           for k in msd}, strict=True)
    return backup


def ema_restore(model, backup):
    msd = model.state_dict()
    model.load_state_dict({k: backup.get(k, msd[k]) for k in msd}, strict=True)


def save_atomic(obj, path):
    """Grava em .tmp e renomeia: queda de energia no meio do save nunca corrompe o checkpoint."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_epoch(model, loader, device, lambdas, gamma, alpha, opt=None, class_weight=None,
              dist_edges=None, asl=None, lambda_ss=0.0, ema=None, ema_decay=0.0):
    """Training: loss only (does not accumulate predictions - that would cost GBs of RAM per epoch).
    Validation (opt=None): collects predictions on the valid pixels for AUPRC."""
    train = opt is not None
    model.train(train)
    tot, nb = 0.0, 0
    pc, tc, pt, tt = [], [], [], []
    for batch in loader:
        batch = [b.to(device) for b in batch]
        er, ec, sep, cf, yc, yt, yd, dm, lm, ar, ac, sr, sc = batch
        with torch.set_grad_enabled(train):
            out = model(er, ec, sep, cf, aa_row=ar, aa_col=ac, ss_row=sr, ss_col=sc)
            loss, _ = multitask_loss(out, batch, lambdas, gamma, alpha, class_weight,
                                     dist_edges, asl)
            if lambda_ss > 0 and getattr(model, "use_ss", False):
                ssl = model.ss_predict(er)          # B x w x 3
                mss = sr != 3                        # valid residues (not padding)
                if mss.any():
                    aux = torch.nn.functional.cross_entropy(ssl[mss], sr[mss])
                    loss = loss + lambda_ss * aux
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ema is not None:
                    ema_update(ema, model, ema_decay)
        tot += float(loss)
        nb += 1
        if not train:
            m = lm > 0
            lc, lt, _ = out
            pc.append(torch.sigmoid(lc)[m].cpu().numpy())
            tc.append(yc[m].cpu().numpy())
            pt.append(torch.sigmoid(lt).permute(0, 2, 3, 1)[m].cpu().numpy())
            tt.append(yt.permute(0, 2, 3, 1)[m].cpu().numpy())
    if train:
        return tot / max(nb, 1), None, None, None, None
    return (
        tot / max(nb, 1),
        np.concatenate(pc), np.concatenate(tc),
        np.concatenate(pt), np.concatenate(tt),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/conv2d")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="fraction of the TRAINING split (learning curve); val untouched")
    ap.add_argument("--epochs", type=int, default=None, help="overrides conv2d.epochs")
    ap.add_argument("--cpu", action="store_true",
                    help="allows training on CPU; without it, a missing CUDA aborts the run")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides cfg.seed: changes initialisation and data order, NOT the "
                         "split (which comes from the manifest) nor the --frac subset")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cc = cfg["conv2d"]
    if args.epochs is not None:
        cc["epochs"] = args.epochs
    seed = cfg["seed"] if args.seed is None else args.seed
    # cfg["seed"] is left intact on purpose: it is what fixes the --frac subset, which must
    # stay nested across fractions even when the training seed changes.
    cfg["run_seed"] = seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if args.seed is not None:
        print(f">> seed={seed} (overrides cfg.seed={cfg['seed']})")
    # Fail fast instead of degrading silently: a run that falls back to CPU without warning
    # burns the whole night and never finishes. It happens in practice - suspending the
    # quebra o contexto CUDA (`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` resolve),
    # machine, with nvidia-smi still looking healthy. Use --cpu to train on CPU on purpose.
    if not torch.cuda.is_available():
        if not args.cpu:
            raise SystemExit(
                "CUDA unavailable and the config asks for a GPU. If the machine was suspended:\n"
                "    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm\n"
                "To train on CPU anyway (slow), pass --cpu.")
        print(">> AVISO: treinando em CPU (--cpu)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train_f = files_from_manifest(cfg, "train")
    val_f = files_from_manifest(cfg, "val")
    if args.frac < 1.0:
        # its own rng: the permutation must not depend on the dataset's consumption of `rng`,
        # otherwise the subsets stop being NESTED across fractions (25% in 50% in 75%).
        perm = np.random.default_rng(cfg["seed"]).permutation(len(train_f))
        k = max(1, int(round(args.frac * len(train_f))))
        train_f = [train_f[i] for i in perm[:k]]
        print(f">> curva de aprendizado: frac={args.frac} -> {k}/{len(perm)} cadeias de treino "
              f"(val e test intactos)")
    tr = ChainCropDataset(cfg, train_f, rng)
    va = ChainCropDataset(cfg, val_f, rng)
    names = tr.type_names
    print(f">> cadeias: treino={len(tr.pairs)} val={len(va.pairs)} | crop={cc['crop_size']} "
          f"n_types={tr.n_types} tipos={names}")

    bc, nw = cc.get("batch_crops", 8), cc.get("num_workers", 2)
    tl = DataLoader(tr, batch_size=bc, shuffle=True, num_workers=nw, drop_last=True)
    vl = DataLoader(va, batch_size=bc, num_workers=nw)

    db = cc.get("dist_bins")
    n_dist_bins = int(db["n_bins"]) if db else 1
    dist_edges = None
    if db:
        # fronteiras internas (n_bins-1) p/ bucketize → bins uniformes em [d_min, d_max]
        dist_edges = torch.linspace(db["d_min"], db["d_max"], n_dist_bins + 1)[1:-1].to(device)
        print(f">> dist em {n_dist_bins} bins sobre [{db['d_min']}, {db['d_max']}] A")

    model = PairConv2D(
        emb_dim=tr._load(0)[0].shape[1], n_types=tr.n_types,
        proj_dim=cc["proj_dim"], channels=cc["channels"], n_blocks=cc["n_blocks"],
        dilations=tuple(cc["dilations"]), dropout=cc["dropout"],
        use_esm_contact=bool(cfg["train"].get("use_esm_contact_feat", True)),
        n_dist_bins=n_dist_bins, aa_dim=int(cc.get("aa_dim", 0)),
        use_ss=bool(cc.get("use_ss", False)), ss_pair=cc.get("ss_pair"),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cc["lr"], weight_decay=cc["weight_decay"])
    lambdas = (cc["lambda_contact"], cc["lambda_types"], cc["lambda_dist"])
    gamma, alpha = cc["focal_gamma"], cc["focal_alpha"]
    asl = cc.get("asl")
    if asl:
        print(f">> perda de tipos: ASL gamma_pos={asl['gamma_pos']} "
              f"gamma_neg={asl['gamma_neg']} clip={asl['clip']}")
    lambda_ss = float(cc.get("lambda_ss", 0.0))
    if cc.get("use_ss", False):
        print(f">> SS: head auxiliar (alvo, limpo) lambda_ss={lambda_ss} | "
              f"canal de par (ENTRADA, exige estrutura)={model.ss_pair}"
              + ("  <-- ABLACAO-TETO, nao e so-sequencia" if model.ss_pair else ""))

    class_weight = None
    if cc.get("class_balanced", False):
        class_weight = compute_class_weights(train_f, cfg, cc.get("class_weight_cap", 3.0)).to(device)
        print(">> class_weight: " +
              " ".join(f"{n}:{w:.2f}" for n, w in zip(names, class_weight.tolist())))

    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cc["epochs"])
             if cc.get("cosine", False) else None)
    patience = int(cc.get("early_stop_patience", 0))

    # EMA is a PASSIVE copy (it does not alter the optimisation) -> both arms (raw and EMA)
    # come from the SAME run: a paired comparison, same seed/data order, half the cost of 2 runs.
    ema_decay = float(cc.get("ema_decay", 0.0))
    ema = ema_init(model) if ema_decay > 0 else None
    best_ema = -1.0
    if ema is not None:
        print(f">> EMA ativo (decay={ema_decay}) — salva best_ema.pt em paralelo ao best.pt")

    best, start_ep, since_best = -1.0, 1, 0
    ckpt_path = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        since_best = ck.get("since_best", 0)
        if ema is not None and ck.get("ema") is not None:
            ema = {k: v.to(device) for k, v in ck["ema"].items()}
            best_ema = ck.get("best_ema", -1.0)
        if sched is not None:
            for _ in range(start_ep - 1):
                sched.step()
        print(f">> retomando de last.pt: proxima epoca={start_ep} best={best:.3f}")

    for ep in range(start_ep, cc["epochs"] + 1):
        trl, *_ = run_epoch(model, tl, device, lambdas, gamma, alpha, opt, class_weight,
                            dist_edges, asl, lambda_ss, ema, ema_decay)
        vll, pcv, tcv, ptv, ttv = run_epoch(model, vl, device, lambdas, gamma, alpha,
                                            dist_edges=dist_edges, asl=asl, lambda_ss=lambda_ss)
        if sched is not None:
            sched.step()
        ap_types = auprc_per_type(ptv, ttv, names)
        ap_contact = average_precision_score(tcv, pcv) if tcv.sum() > 0 else float("nan")
        macro = float(np.mean(list(ap_types.values()))) if ap_types else 0.0

        macro_ema = None
        if ema is not None:  # validate the EMA arm on the same data
            bk = ema_swap(model, ema)
            _, pce, tce, pte, tte = run_epoch(model, vl, device, lambdas, gamma, alpha,
                                              dist_edges=dist_edges, asl=asl, lambda_ss=lambda_ss)
            ema_restore(model, bk)
            ap_e = auprc_per_type(pte, tte, names)
            macro_ema = float(np.mean(list(ap_e.values()))) if ap_e else 0.0
            if macro_ema > best_ema:
                best_ema = macro_ema
                bk = ema_swap(model, ema)
                save_atomic({"model": model.state_dict(), "cfg": cfg, "types": names},
                            os.path.join(args.out, "best_ema.pt"))
                ema_restore(model, bk)

        if ep % 5 == 0 or ep == 1:
            det = " ".join(f"{n}:{v:.2f}" for n, v in ap_types.items())
            extra = f" EMA_macro={macro_ema:.2f}" if macro_ema is not None else ""
            print(f"ep{ep:3d} trL={trl:.3f} vaL={vll:.3f} AUPRC_contact={ap_contact:.2f} "
                  f"AUPRC_types_macro={macro:.2f}{extra} | {det}", flush=True)
        if macro > best:
            best, since_best = macro, 0
            save_atomic({"model": model.state_dict(), "cfg": cfg, "types": names},
                        os.path.join(args.out, "best.pt"))
        else:
            since_best += 1
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                     "epoch": ep, "best": best, "since_best": since_best,
                     "best_ema": best_ema, "ema": ema,
                     "cfg": cfg, "types": names}, ckpt_path)
        if patience and since_best >= patience:
            print(f">> early stopping at epoch {ep} (no improvement for {since_best} epochs)")
            break
    print(f">> melhor AUPRC_types_macro={best:.3f} salvo em {args.out}/best.pt")
    if ema is not None:
        print(f">> melhor EMA AUPRC_types_macro={best_ema:.3f} salvo em {args.out}/best_ema.pt")


if __name__ == "__main__":
    main()
