"""Ensemble no portao denso — media de probabilidades de dois modelos.

Cada modelo tem seu proprio config (embeddings/vocabulario). As cadeias do split sao as
mesmas (mesma chave de nome); as probabilidades sao alinhadas por NOME de classe e mediadas.
Serve de sanity-check da complementaridade (ex.: MLP-650M + conv2d-v2-150M) antes de apostar
no treino conjunto. Mesmo protocolo do eval_dense (todos os pares validos, AUPRC + Top-L).

Uso:
    python src/eval_ensemble.py \
        --a mlp    --a-config configs/esm650m.yaml  --a-ckpt outputs/baseline_650m/best.pt \
        --b conv2d --b-config configs/default.yaml  --b-ckpt outputs/conv2d_v2/best.pt
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
from src.eval_dense import load_chain, predict_conv2d, predict_mlp
from src.metrics import DIVISORS, RANGES, TopLAccumulator
from src.models.pair_conv2d import PairConv2D
from src.models.pair_mlp import PairMLP


def build_model(kind, cfg, ck, emb_dim, use_esm, device):
    ckpt_types = [str(t) for t in ck["types"]]
    if kind == "mlp":
        tcfg = cfg["train"]
        feat_dim = 3 * emb_dim + 1 + (1 if use_esm else 0)
        m = PairMLP(feat_dim, len(ckpt_types), tcfg["hidden"], tcfg["layers"],
                    tcfg["dropout"]).to(device)
    else:
        cc = cfg["conv2d"]
        n_dist_bins = ck["model"]["head_dist.weight"].shape[0]
        aa_dim = ck["model"]["aa_emb.weight"].shape[1] if "aa_emb.weight" in ck["model"] else 0
        use_ss = "ss_head.weight" in ck["model"]
        c_in_ck = ck["model"]["stem.weight"].shape[1]
        base = 3 * cc["proj_dim"] + 1 + (1 if use_esm else 0) + 3 * aa_dim
        ss_pair = (c_in_ck - base) >= 2
        m = PairConv2D(emb_dim, len(ckpt_types), cc["proj_dim"], cc["channels"],
                       cc["n_blocks"], tuple(cc["dilations"]), cc["dropout"],
                       use_esm, n_dist_bins, aa_dim, use_ss, ss_pair).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ckpt_types


def main():
    ap = argparse.ArgumentParser()
    for tag in ("a", "b"):
        ap.add_argument(f"--{tag}", choices=["mlp", "conv2d"], required=True)
        ap.add_argument(f"--{tag}-config", required=True)
        ap.add_argument(f"--{tag}-ckpt", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    legs = []
    for tag in ("a", "b"):
        cfg = yaml.safe_load(open(getattr(args, f"{tag}_config")))
        ck = torch.load(getattr(args, f"{tag}_ckpt"), map_location=device)
        legs.append(dict(
            kind=getattr(args, tag),
            cfg=cfg,
            ck=ck,
            emb_dir=cfg["paths"]["embeddings"],
            use_esm=bool(cfg["train"].get("use_esm_contact_feat", True)),
        ))
        print(f">> {tag}: {getattr(args, tag)} cfg={getattr(args, f'{tag}_config')} "
              f"ckpt={getattr(args, f'{tag}_ckpt')}")

    seq_sep = int(legs[0]["cfg"]["arpeggio"]["seq_sep_min"])
    label_dir = legs[0]["cfg"]["paths"]["labels"]
    files = files_from_manifest(legs[0]["cfg"], args.split)

    # nomes de classe: os do primeiro leg (ambos compartilham o vocabulario)
    names = None
    for leg in legs:
        leg["types"] = [str(t) for t in leg["ck"]["types"]]

    gc_p, gc_t, gt_p, gt_t = [], [], None, None
    acc_c = None
    acc_t = None
    used = 0

    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        # cada leg carrega a cadeia com SEU proprio embedding
        chs = []
        ok = True
        for leg in legs:
            ef = os.path.join(leg["emb_dir"], f"{name}.npz")
            if not os.path.exists(ef):
                ok = False
                break
            ch = load_chain(ef, lf, seq_sep)
            if ch is None:
                ok = False
                break
            chs.append(ch)
        if not ok:
            continue

        if names is None:
            names = chs[0]["names"]
            T = len(names)
            gt_p = [[] for _ in range(T)]
            gt_t = [[] for _ in range(T)]
            acc_c = TopLAccumulator()
            acc_t = [TopLAccumulator() for _ in range(T)]

        pcs, pts = [], []
        for leg, ch in zip(legs, chs):
            sel = [leg["types"].index(n) for n in names]
            if leg["kind"] == "mlp":
                pc, pt = predict_mlp(build_leg_model(leg, ch, device), ch, device,
                                     sel, leg["use_esm"])
            else:
                pc, pt = predict_conv2d(build_leg_model(leg, ch, device), ch, device,
                                        sel, leg["use_esm"])
            pcs.append(pc)
            pts.append(pt)

        pc = np.mean(pcs, axis=0)
        pt = np.mean(pts, axis=0)
        ch0 = chs[0]
        vi, vj = ch0["vi"], ch0["vj"]
        sep = np.abs(vi - vj)
        yc = ch0["tgt_c"][vi, vj]
        gc_p.append(pc); gc_t.append(yc)
        acc_c.add(pc, yc, sep, ch0["L"])
        for k in range(len(names)):
            yt = ch0["tgt_t"][k, vi, vj]
            gt_p[k].append(pt[:, k]); gt_t[k].append(yt)
            acc_t[k].add(pt[:, k], yt, sep, ch0["L"])
        used += 1

    print(f">> split={args.split} cadeias={used} tipos={names}")
    gc_p, gc_t = np.concatenate(gc_p), np.concatenate(gc_t)
    ap_c = float(average_precision_score(gc_t, gc_p)) if gc_t.sum() > 0 else float("nan")
    ap_t = {}
    for k, n in enumerate(names):
        yt = np.concatenate(gt_t[k])
        ap_t[n] = (float(average_precision_score(yt, np.concatenate(gt_p[k])))
                   if yt.sum() > 0 else float("nan"))
    macro = float(np.nanmean(list(ap_t.values())))

    print(f"\n== ENSEMBLE DENSO ({args.split}) — todos os pares validos ==")
    print(f"AUPRC_contact     = {ap_c:.3f}")
    print(f"AUPRC_types_macro = {macro:.3f}")
    print(f"{'classe':12s} {'AUPRC':>6s}  " + "  ".join(f"L/{d}" if d > 1 else "L"
                                                        for d in DIVISORS) + "  (long-range)")
    tl_c = acc_c.mean()
    print(f"{'contact':12s} {ap_c:6.3f}  " +
          "  ".join(f"{tl_c[f'long/L{d}' if d > 1 else 'long/L']:.3f}" for d in DIVISORS))
    for k, n in enumerate(names):
        tl = acc_t[k].mean()
        print(f"{n:12s} {ap_t[n]:6.3f}  " +
              "  ".join(f"{tl[f'long/L{d}' if d > 1 else 'long/L']:.3f}" for d in DIVISORS))


_MODEL_CACHE = {}


def build_leg_model(leg, ch, device):
    key = id(leg)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key], _ = build_model(leg["kind"], leg["cfg"], leg["ck"],
                                           ch["emb"].shape[1], leg["use_esm"], device)
    return _MODEL_CACHE[key]


if __name__ == "__main__":
    main()
