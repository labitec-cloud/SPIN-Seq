"""Evaluation on the TEST split (never-seen chains) - the official baseline number.

Loads a checkpoint (best.pt), runs on the manifest's `test` split and reports per-class +
macro AUPRC and contact AUPRC - the SAME metrics as training/curve, for direct comparison.

The checkpoint may have been trained with a type vocabulary different from the current one
(e.g. before removing xbond/metal). That is why the model is sized by the types SAVED in the
checkpoint and the columns are aligned to the current dataset BY TYPE NAME - never by raw index.

Uso:
    python src/eval.py                                  # outputs/baseline/best.pt no split test
    python src/eval.py --ckpt outputs/baseline/best.pt --split test
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
from src.data.pair_dataset import (
    ChainPairDataset,
    collate_chains,
    files_from_manifest,
)
from src.models.pair_mlp import PairMLP
from src.train import auprc_per_type


def drop_inconsistent(files, emb_dir):
    """Drops chains whose embedding does not match the label length (a data bug)."""
    good, bad = [], 0
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        l = np.load(lf, allow_pickle=True)
        e = np.load(ef, allow_pickle=True)
        if e["emb"].shape[0] == int(l["length"]):
            good.append(lf)
        else:
            bad += 1
    if bad:
        print(f">> dropped {bad} chains with inconsistent emb/label")
    return good


@torch.no_grad()
def infer(model, loader, device, sel):
    """Roda o modelo e devolve (pred_contact, tgt_contact, pred_types[:, sel], tgt_types)."""
    model.eval()
    pc, tc, pt, tt = [], [], [], []
    for X, yc, yt, yd, dmask in loader:
        X = X.to(device)
        lc, lt, _ = model(X)
        pc.append(torch.sigmoid(lc).cpu().numpy())
        tc.append(yc.numpy())
        pt.append(torch.sigmoid(lt[:, sel]).cpu().numpy())
        tt.append(yt.numpy())
    return (np.concatenate(pc), np.concatenate(tc),
            np.concatenate(pt), np.concatenate(tt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="outputs/baseline/best.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tcfg = cfg["train"]
    rng = np.random.default_rng(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device)
    ckpt_types = [str(t) for t in ck["types"]]
    print(f">> ckpt={args.ckpt} tipos_no_ckpt={len(ckpt_types)}: {ckpt_types}")

    files = files_from_manifest(cfg, args.split)
    files = drop_inconsistent(files, cfg["paths"]["embeddings"])
    ds = ChainPairDataset(cfg, files, rng)
    names = ds.type_names  # current vocabulary (may be a subset of the ckpt)
    print(f">> split={args.split} cadeias={len(ds)} feat_dim={ds.feat_dim} "
          f"tipos_avaliados={len(names)}: {names}")

    # map each evaluated type to the matching column in the model output (ckpt order)
    missing = [n for n in names if n not in ckpt_types]
    if missing:
        raise RuntimeError(f"tipos do dataset ausentes no ckpt: {missing}")
    sel = [ckpt_types.index(n) for n in names]

    model = PairMLP(ds.feat_dim, len(ckpt_types), tcfg["hidden"], tcfg["layers"],
                    tcfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])

    loader = DataLoader(ds, batch_size=tcfg.get("batch_chains", 8),
                        collate_fn=collate_chains, num_workers=tcfg.get("num_workers", 2))

    pc, tc, pt, tt = infer(model, loader, device, sel)

    ap_types = auprc_per_type(pt, tt, names)
    ap_contact = float(average_precision_score(tc, pc)) if tc.sum() > 0 else float("nan")
    macro = float(np.mean(list(ap_types.values()))) if ap_types else 0.0

    print(f"\n== EVALUATION ({args.split}) ==")
    print(f"AUPRC_contact     = {ap_contact:.3f}")
    print(f"AUPRC_types_macro = {macro:.3f}")
    for n in names:
        v = ap_types.get(n, float("nan"))
        pos = int(tt[:, names.index(n)].sum())
        print(f"  {n:12s} AUPRC={v:.3f}  (positivos={pos})")


if __name__ == "__main__":
    main()
