"""hbond|polar diagnosis - is hbond label noise or recoverable signal?

Already measured on the labels: 96.3% of hbond edges are also polar. Hypothesis: the
model does not isolate the hbond subset inside polar because the difference depends on a
proton placed almost arbitrarily by OpenBabel.

Teste: restringe aos pares POLAR-positivos do test e mede a AUPRC do head de
the model's hbond head ON THAT SUBSET vs the prevalence of hbond there. If the
conditional AUPRC ~ prevalence, the head does not tell hbond from polar -> the label is
pure noise (which becomes a result of the paper). If it lands above, there is recoverable
signal.

Uso:
    python src/diag_hbond_polar.py --config configs/esm650m_aa.yaml \
        --ckpt outputs/conv2d_650m_aa/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pair_dataset import files_from_manifest
from src.eval_dense import load_chain, predict_conv2d
from src.eval_ensemble import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/esm650m_aa.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
    emb_dir = cfg["paths"]["embeddings"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    ckpt_types = [str(t) for t in ck["types"]]

    files = files_from_manifest(cfg, args.split)
    chains, names = [], None
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        ch = load_chain(ef, lf, seq_sep)
        if ch is not None:
            chains.append(ch)
            names = ch["names"]
    sel = [ckpt_types.index(n) for n in names]
    ih, ip = names.index("hbond"), names.index("polar")
    model, _ = build_model("conv2d", cfg, ck, chains[0]["emb"].shape[1], use_esm, device)

    P_h, P_p, Y_h = [], [], []       # score hbond, score polar, alvo hbond
    Yp = []                          # alvo polar
    for ch in chains:
        _, pt = predict_conv2d(model, ch, device, sel, use_esm)
        vi, vj = ch["vi"], ch["vj"]
        P_h.append(pt[:, ih]); P_p.append(pt[:, ip])
        Y_h.append(ch["tgt_t"][ih, vi, vj]); Yp.append(ch["tgt_t"][ip, vi, vj])
    P_h = np.concatenate(P_h); P_p = np.concatenate(P_p)
    Y_h = np.concatenate(Y_h); Yp = np.concatenate(Yp)

    # (a) reference: hbond AUPRC over ALL valid pairs (= the gate)
    ap_all = average_precision_score(Y_h, P_h)
    # (b) conditional: polar-positive pairs only
    m = Yp > 0
    prev = float(Y_h[m].mean())
    ap_cond_h = average_precision_score(Y_h[m], P_h[m]) if Y_h[m].sum() > 0 else float("nan")
    # (c) does the polar score predict hbond inside polar? (confusion baseline)
    ap_cond_polar = average_precision_score(Y_h[m], P_p[m]) if Y_h[m].sum() > 0 else float("nan")

    print(f">> valid pairs={len(Y_h):,} | polar-positives={int(m.sum()):,} | "
          f"hbond-positivos={int(Y_h.sum()):,}")
    print(f"\n== hbond ==")
    print(f"AUPRC(hbond | all pairs)             = {ap_all:.4f}   (the gate number)")
    print(f"prevalence of hbond INSIDE polar     = {prev:.4f}   (chance floor)")
    print(f"AUPRC(hbond | polar), head de hbond  = {ap_cond_h:.4f}")
    print(f"AUPRC(hbond | polar), head de polar  = {ap_cond_polar:.4f}   (score de polar)")
    lift = ap_cond_h - prev
    print(f"\nlift sobre o acaso dentro de polar   = {lift:+.4f}")
    print("VEREDITO: " + (
        "~ prevalence -> hbond is label noise (the open challenge of the paper)."
        if lift < 0.03 else
        "above chance -> there is recoverable signal (reduce/DSSP justified)."))


if __name__ == "__main__":
    main()
