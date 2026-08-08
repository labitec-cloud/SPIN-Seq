"""Per-class learning curve - how much each interaction type improves with MORE data.

Trains the baseline on increasing fractions of the training split (e.g. 25/50/75/100%),
ALWAYS evaluating on the same validation set. It serves two diagnoses:
  1) where more data still pays (AUPRC rising) vs. where it has saturated (noise/model);
  2) a comparison baseline for tuning hyperparameters (rerun with another config
     e sobrepor as curvas).

Saídas em outputs/learning_curve/:
  - curve.csv  (frac, n_chains, contact, types_macro, <auprc por tipo>)
  - curve.png  (uma linha por classe: AUPRC vs. nº de cadeias)

Uso:
    python scripts/learning_curve.py                          # 0.25,0.5,0.75,1.0 x 15 ep
    python scripts/learning_curve.py --fracs 0.1,0.25,0.5,1.0 --epochs 20
    python scripts/learning_curve.py --out outputs/lc_lr3e4   # comparar hiperparametros
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
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
from src.train import auprc_per_type, run_epoch


def train_eval(cfg, train_f, val_f, epochs, device, rng, frac_tag=""):
    """Treina do zero em train_f e devolve as métricas de val da MELHOR época (por macro)."""
    tcfg = cfg["train"]
    tr = ChainPairDataset(cfg, train_f, rng)
    va = ChainPairDataset(cfg, val_f, rng)
    names = tr.type_names
    bc = tcfg.get("batch_chains", 8)
    nw = tcfg.get("num_workers", 2)
    tl = DataLoader(tr, batch_size=bc, shuffle=True, collate_fn=collate_chains, num_workers=nw)
    vl = DataLoader(va, batch_size=bc, collate_fn=collate_chains, num_workers=nw)

    model = PairMLP(tr.feat_dim, tr.n_types, tcfg["hidden"], tcfg["layers"], tcfg["dropout"]).to(device)
    pos_w = compute_pos_weights(train_f, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    lambdas = (tcfg["lambda_contact"], tcfg["lambda_types"], tcfg["lambda_dist"])

    best_macro, best = -1.0, None
    for ep in range(1, epochs + 1):
        run_epoch(model, tl, device, pos_w, lambdas, opt)
        _, pc, tc, pt, tt = run_epoch(model, vl, device, pos_w, lambdas)
        ap_types = auprc_per_type(pt, tt, names)
        ap_contact = float(average_precision_score(tc, pc)) if tc.sum() > 0 else float("nan")
        macro = float(np.mean(list(ap_types.values()))) if ap_types else 0.0
        if macro > best_macro:
            best_macro = macro
            best = {"contact": ap_contact, "types_macro": macro, **ap_types}
        if ep % 5 == 0 or ep == 1:
            det = " ".join(f"{n}:{ap_types.get(n, float('nan')):.2f}" for n in names)
            print(f"   [{frac_tag}] ep{ep:3d} contact={ap_contact:.2f} "
                  f"macro={macro:.2f} | {det}", flush=True)
    return names, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="outputs/learning_curve")
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--epochs", type=int, default=15, help="épocas por ponto (menos que o treino final)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    fracs = [float(x) for x in args.fracs.split(",")]

    all_train = files_from_manifest(cfg, "train")
    val_f = files_from_manifest(cfg, "val")
    print(f">> train total={len(all_train)} val={len(val_f)} device={device} epochs/ponto={args.epochs}")

    rows, type_names = [], None
    for f in fracs:
        rng = np.random.default_rng(cfg["seed"])  # same seed -> nested/stable subsets
        idx = rng.permutation(len(all_train))
        k = max(1, int(round(f * len(all_train))))
        sub = [all_train[i] for i in idx[:k]]
        names, m = train_eval(cfg, sub, val_f, args.epochs, device, rng, frac_tag=f"frac={f:.2f}")
        type_names = names
        m2 = {"frac": f, "n_chains": k, **m}
        rows.append(m2)
        det = " ".join(f"{n}:{m.get(n, float('nan')):.2f}" for n in names)
        print(f">> frac={f:.2f} n={k:4d} contact={m['contact']:.2f} "
              f"types_macro={m['types_macro']:.2f} | {det}")

    # CSV
    cols = ["frac", "n_chains", "contact", "types_macro"] + type_names
    csv_path = os.path.join(args.out, "curve.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f">> salvo {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["n_chains"] for r in rows]
        plt.figure(figsize=(8, 5))
        for key in ["contact", "types_macro"] + type_names:
            ys = [r.get(key, np.nan) for r in rows]
            style = "--" if key in ("contact", "types_macro") else "-"
            lw = 2.5 if key in ("contact", "types_macro") else 1.3
            plt.plot(xs, ys, style, marker="o", linewidth=lw, label=key)
        plt.xlabel("cadeias de treino")
        plt.ylabel("AUPRC (validation)")
        plt.title(f"Curva de aprendizado por classe ({args.epochs} ep/ponto)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        png = os.path.join(args.out, "curve.png")
        plt.savefig(png, dpi=130)
        print(f">> salvo {png}")
    except Exception as exc:  # noqa: BLE001
        print(f">> (plot pulado: {exc})")


if __name__ == "__main__":
    main()
