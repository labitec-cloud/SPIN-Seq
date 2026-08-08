"""Avaliação DENSA no split test — protocolo justo para comparar MLP vs Conv2D.

Ambos os modelos são pontuados sobre EXATAMENTE o mesmo conjunto de pares: TODOS os pares
válidos (i<j, |i-j| >= seq_sep_min) de cada cadeia do test, SEM amostragem de negativos. É o
único protocolo em que 0.481 (MLP, negativos amostrados) e 0.379 (conv2d, pixels densos) viram
comparáveis. Reporta AUPRC por classe/contato + Top-L por classe e por faixa de separação.

O modelo é dimensionado pelos tipos SALVOS no checkpoint; as colunas são alinhadas ao
vocabulário atual POR NOME (permite avaliar ckpt de vocabulário antigo).

Uso:
    python src/eval_dense.py --model mlp    --ckpt outputs/baseline/best.pt
    python src/eval_dense.py --model conv2d --ckpt outputs/conv2d/best.pt
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
from src.data.pair_dataset import (AA_UNK, EXCLUDED_TYPES, files_from_manifest,
                                    seq_to_idx, symmetric_pair_feat)
from src.metrics import DIVISORS, RANGES, TopLAccumulator
from src.models.pair_conv2d import PairConv2D
from src.models.pair_mlp import PairMLP


def load_chain(ef, lf, seq_sep, ss_dir="data/ss", keep_emb=True):
    """keep_emb=False descarta o embedding APÓS a checagem de comprimento: rankers que não
    usam ESM (propensão, oráculo) avaliam as mesmas 494 cadeias sem os ~900 MB de emb."""
    e = np.load(ef, allow_pickle=True)
    l = np.load(lf, allow_pickle=True)
    emb = e["emb"].astype(np.float32)
    L = int(l["length"])
    if emb.shape[0] != L:
        return None
    if not keep_emb:
        emb = np.zeros((L, 0), np.float32)
    contacts = (np.nan_to_num(e["contacts"].astype(np.float32)) if "contacts" in e
                else np.zeros((L, L), np.float32))
    types = [str(t) for t in l["types"]]
    tc = [k for k, t in enumerate(types) if t not in EXCLUDED_TYPES]
    names = [types[k] for k in tc]
    idx_i, idx_j = l["idx_i"].astype(np.int64), l["idx_j"].astype(np.int64)
    labels = l["labels"][:, tc].astype(np.float32)

    tgt_c = np.zeros((L, L), np.float32)
    tgt_t = np.zeros((len(tc), L, L), np.float32)
    tgt_c[idx_i, idx_j] = 1.0
    tgt_t[:, idx_i, idx_j] = labels.T
    aa = seq_to_idx(str(l["seq"]))[:L] if "seq" in l else np.full(L, AA_UNK, np.int64)
    name = os.path.splitext(os.path.basename(lf))[0]
    sf = os.path.join(ss_dir, f"{name}.npz")
    ss = (np.load(sf)["ss"].astype(np.int64)[:L] if os.path.exists(sf)
          else np.full(L, 3, np.int64))
    vi, vj = np.triu_indices(L, k=seq_sep)  # todos os pares válidos i<j, j-i>=seq_sep
    return dict(emb=emb, contacts=contacts, L=L, names=names, aa=aa, ss=ss,
                vi=vi, vj=vj, tgt_c=tgt_c, tgt_t=tgt_t)


@torch.no_grad()
def predict_mlp(model, ch, device, sel, use_esm, chunk=200_000):
    emb, vi, vj = ch["emb"], ch["vi"], ch["vj"]
    sep_feat = np.log1p(np.abs(vi - vj)).astype(np.float32)[:, None]
    hi, hj = emb[vi], emb[vj]
    z = symmetric_pair_feat(hi, hj)
    extra = [sep_feat]
    if use_esm:
        extra.append(ch["contacts"][vi, vj].astype(np.float32)[:, None])
    feats = np.concatenate([z] + extra, axis=1)
    pc, pt = [], []
    for a in range(0, feats.shape[0], chunk):
        X = torch.from_numpy(feats[a:a + chunk]).to(device)
        lc, lt, _ = model(X)
        pc.append(torch.sigmoid(lc).cpu().numpy())
        pt.append(torch.sigmoid(lt[:, sel]).cpu().numpy())
    return np.concatenate(pc), np.concatenate(pt)


@torch.no_grad()
def predict_conv2d(model, ch, device, sel, use_esm):
    emb, L, vi, vj = ch["emb"], ch["L"], ch["vi"], ch["vj"]
    er = torch.from_numpy(emb).unsqueeze(0).to(device)
    sep = torch.from_numpy(np.log1p(np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]))
                           .astype(np.float32)).unsqueeze(0).to(device)
    cf = torch.from_numpy(ch["contacts"]).unsqueeze(0).to(device) if use_esm else None
    aa = torch.from_numpy(ch["aa"]).unsqueeze(0).to(device)
    ss = torch.from_numpy(ch["ss"]).unsqueeze(0).to(device)
    lc, lt, _ = model(er, er, sep, cf, symmetrize=True, aa_row=aa, aa_col=aa,
                      ss_row=ss, ss_col=ss)
    pc = torch.sigmoid(lc)[0].cpu().numpy()[vi, vj]
    pt = torch.sigmoid(lt)[0, sel].cpu().numpy()[:, vi, vj].T  # (n_pairs, n_types)
    return pc, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["mlp", "conv2d"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir, label_dir = cfg["paths"]["embeddings"], cfg["paths"]["labels"]

    ck = torch.load(args.ckpt, map_location=device)
    ckpt_types = [str(t) for t in ck["types"]]
    print(f">> model={args.model} ckpt={args.ckpt} tipos_no_ckpt={ckpt_types}")

    files = files_from_manifest(cfg, args.split)
    chains, names = [], None
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        ch = load_chain(ef, lf, seq_sep, cfg["paths"].get("ss", "data/ss"))
        if ch is None:
            continue
        chains.append(ch)
        names = ch["names"]
    sel = [ckpt_types.index(n) for n in names]
    print(f">> split={args.split} cadeias={len(chains)} tipos_avaliados={names}")

    if args.model == "mlp":
        tcfg = cfg["train"]
        feat_dim = 3 * chains[0]["emb"].shape[1] + 1 + (1 if use_esm else 0)
        model = PairMLP(feat_dim, len(ckpt_types), tcfg["hidden"], tcfg["layers"],
                        tcfg["dropout"]).to(device)
    else:
        cc = cfg["conv2d"]
        n_dist_bins = ck["model"]["head_dist.weight"].shape[0]  # casa v1/v2 (1) e v3 (bins)
        aa_dim = ck["model"]["aa_emb.weight"].shape[1] if "aa_emb.weight" in ck["model"] else 0
        use_ss = "ss_head.weight" in ck["model"]
        # canal de par de SS existe? deduz pela largura do stem (ss_pair soma 2 canais)
        c_in_ck = ck["model"]["stem.weight"].shape[1]
        base = 3 * cc["proj_dim"] + 1 + (1 if use_esm else 0) + 3 * aa_dim
        ss_pair = (c_in_ck - base) >= 2
        model = PairConv2D(chains[0]["emb"].shape[1], len(ckpt_types), cc["proj_dim"],
                           cc["channels"], cc["n_blocks"], tuple(cc["dilations"]),
                           cc["dropout"], use_esm, n_dist_bins, aa_dim, use_ss,
                           ss_pair).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    T = len(names)
    gc_p, gc_t = [], []                       # contato global (para AUPRC)
    gt_p = [[] for _ in range(T)]             # tipos global por classe
    gt_t = [[] for _ in range(T)]
    acc_c = TopLAccumulator()
    acc_t = [TopLAccumulator() for _ in range(T)]

    predict = predict_mlp if args.model == "mlp" else predict_conv2d
    kw = dict(chunk=200_000) if args.model == "mlp" else {}
    for ch in chains:
        pc, pt = predict(model, ch, device, sel, use_esm, **kw)
        vi, vj = ch["vi"], ch["vj"]
        sep = np.abs(vi - vj)
        yc = ch["tgt_c"][vi, vj]
        gc_p.append(pc); gc_t.append(yc)
        acc_c.add(pc, yc, sep, ch["L"])
        for k in range(T):
            yt = ch["tgt_t"][k, vi, vj]
            gt_p[k].append(pt[:, k]); gt_t[k].append(yt)
            acc_t[k].add(pt[:, k], yt, sep, ch["L"])

    gc_p, gc_t = np.concatenate(gc_p), np.concatenate(gc_t)
    ap_c = float(average_precision_score(gc_t, gc_p)) if gc_t.sum() > 0 else float("nan")
    ap_t = {}
    for k, n in enumerate(names):
        yt = np.concatenate(gt_t[k])
        ap_t[n] = (float(average_precision_score(yt, np.concatenate(gt_p[k])))
                   if yt.sum() > 0 else float("nan"))
    macro = float(np.nanmean(list(ap_t.values())))

    print(f"\n== AVALIAÇÃO DENSA ({args.split}, {args.model}) — todos os pares válidos ==")
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

    print("\n== Top-L por faixa de separação (contato) ==")
    print(f"{'faixa':8s} " + "  ".join(f"L/{d}" if d > 1 else "L" for d in DIVISORS))
    for rng in list(RANGES) + ["all"]:
        print(f"{rng:8s} " +
              "  ".join(f"{tl_c[f'{rng}/L{d}' if d > 1 else f'{rng}/L']:.3f}" for d in DIVISORS))

    # §0.3: Top-L por CLASSE x faixa de separação (mostra que não é só contato local)
    print("\n== Top-L (L) por classe x faixa de separação ==")
    print(f"{'classe':12s} " + "  ".join(f"{r:>7s}" for r in list(RANGES) + ["all"]))
    for k, n in enumerate(names):
        tl = acc_t[k].mean()
        print(f"{n:12s} " +
              "  ".join(f"{tl[f'{rng}/L']:7.3f}" for rng in list(RANGES) + ["all"]))


if __name__ == "__main__":
    main()
