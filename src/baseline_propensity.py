"""Amino-acid pair propensity baseline - the obvious reviewer objection.

`ionic` is Arg-Asp, `aromatic` is His-His, `covalent` is Cys-Cys. How much of SPIN-Seq's
AUPRC in those classes is ESM-2 and how much is a 21x21 lookup table? This baseline
responde: estima P(tipo | aa_i, aa_j, faixa de |i-j|) contando o split de TREINO e
scores the test set at the SAME dense gate, without seeing any embedding.

This is deliberately the STRONG version of the objection: the table captures the full
interaction between the two amino acids (a logistic regression with additive one-hot would
be weaker, and beating a weak baseline proves nothing). Smoothing towards the class marginal
is mandatory - `covalent` has 1,628 positives spread over thousands of cells, and without
smoothing the table becomes noise and the champion "wins" by artefact.

Uso:
    python src/baseline_propensity.py --config configs/esm650m_aa.yaml --fit
    python src/baseline_propensity.py --config configs/esm650m_aa.yaml   # avalia
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pair_dataset import (AA_UNK, EXCLUDED_TYPES, files_from_manifest,
                                    seq_to_idx)
from src.eval_dense import load_chain
from src.metrics import DIVISORS, RANGES, TopLAccumulator

N_AA = AA_UNK + 1                       # 21 symbols (20 AAs + unknown)
SEP_EDGES = np.array([3, 4, 5, 6, 12, 24, 50, 100])  # aligned with the ranges in metrics.RANGES
N_BINS = len(SEP_EDGES)
N_CELLS = N_AA * N_AA * N_BINS
DEFAULT_TABLE = "outputs/propensity_table.npz"


def pair_cells(aa: np.ndarray, vi: np.ndarray, vj: np.ndarray) -> np.ndarray:
    """Cell (unordered AA pair x separation range) of each pair (i,j)."""
    a, b = aa[vi], aa[vj]
    code = np.minimum(a, b) * N_AA + np.maximum(a, b)
    bins = np.searchsorted(SEP_EDGES, np.abs(vi - vj), side="right") - 1
    return code * N_BINS + bins


def fit(cfg: dict, split: str = "train") -> dict:
    """Counts positives and valid pairs per cell in the given split. Reads labels only."""
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    emb_dir = cfg["paths"]["embeddings"]
    files = files_from_manifest(cfg, split)

    names = None
    den = np.zeros(N_CELLS, np.float64)
    num_c = np.zeros(N_CELLS, np.float64)
    num_t = None
    n_used = 0
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        # same filter as training: a chain without an embedding never entered the champion fit
        if not os.path.exists(os.path.join(emb_dir, f"{name}.npz")):
            continue
        l = np.load(lf, allow_pickle=True)
        L = int(l["length"])
        seq = str(l["seq"])
        if len(seq) != L:
            continue
        types = [str(t) for t in l["types"]]
        tc = [k for k, t in enumerate(types) if t not in EXCLUDED_TYPES]
        if names is None:
            names = [types[k] for k in tc]
            num_t = np.zeros((N_CELLS, len(tc)), np.float64)

        aa = seq_to_idx(seq)[:L]
        vi, vj = np.triu_indices(L, k=seq_sep)
        den += np.bincount(pair_cells(aa, vi, vj), minlength=N_CELLS)

        idx_i, idx_j = l["idx_i"].astype(np.int64), l["idx_j"].astype(np.int64)
        keep = (idx_j - idx_i) >= seq_sep
        idx_i, idx_j = idx_i[keep], idx_j[keep]
        if idx_i.size:
            pc = pair_cells(aa, idx_i, idx_j)
            num_c += np.bincount(pc, minlength=N_CELLS)
            lab = l["labels"][keep][:, tc].astype(np.float64)
            for k in range(len(tc)):
                num_t[:, k] += np.bincount(pc, weights=lab[:, k], minlength=N_CELLS)
        n_used += 1

    return dict(den=den, num_c=num_c, num_t=num_t, names=names, n_chains=n_used)


def smooth(counts: dict, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooths towards the class marginal: p = (num + a*prior) / (den + a).

    Without this, low-count cells (the `covalent` case) would return 0 or 1 by sampling
    accident and the baseline would be artificially weak.
    """
    den, num_c, num_t = counts["den"], counts["num_c"], counts["num_t"]
    tot = max(den.sum(), 1.0)
    prior_c = num_c.sum() / tot
    prior_t = num_t.sum(0) / tot
    p_c = (num_c + alpha * prior_c) / (den + alpha)
    p_t = (num_t + alpha * prior_t[None, :]) / (den + alpha)[:, None]
    return p_c.astype(np.float32), p_t.astype(np.float32)


def load_table(path: str = DEFAULT_TABLE) -> dict:
    z = np.load(path, allow_pickle=True)
    return dict(p_c=z["p_c"], p_t=z["p_t"], names=[str(t) for t in z["names"]],
                alpha=float(z["alpha"]), n_chains=int(z["n_chains"]))


def predict_propensity(table, ch, device=None, sel=None, use_esm=None):
    """Signature compatible with predict_mlp/predict_conv2d: returns (pc, pt).

    `sel` is ignored (the table is already indexed by name); it is kept in the signature so
    the baseline can stand in for a model in eval/bootstrap with no special case.
    """
    cells = pair_cells(ch["aa"], ch["vi"], ch["vj"])
    order = [table["names"].index(n) for n in ch["names"]]
    return table["p_c"][cells], table["p_t"][cells][:, order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/esm650m_aa.yaml")
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--fit", action="store_true", help="reajusta a tabela no treino")
    ap.add_argument("--alpha", type=float, default=50.0, help="smoothing strength")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    emb_dir = cfg["paths"]["embeddings"]

    if args.fit or not os.path.exists(args.table):
        print(">> fitting the table on the TRAIN split (labels only, no embedding)...")
        counts = fit(cfg, "train")
        p_c, p_t = smooth(counts, args.alpha)
        os.makedirs(os.path.dirname(args.table), exist_ok=True)
        np.savez_compressed(args.table, p_c=p_c, p_t=p_t, names=counts["names"],
                            alpha=args.alpha, n_chains=counts["n_chains"],
                            den=counts["den"])
        occ = int((counts["den"] > 0).sum())
        print(f"   cadeias de treino={counts['n_chains']} pares={counts['den'].sum():,.0f}")
        print(f"   occupied cells={occ}/{N_CELLS} alpha={args.alpha}")
        print(f"   tabela salva em {args.table}")

    table = load_table(args.table)
    print(f">> baseline=propensity aa_i x aa_j x range(|i-j|) "
          f"treino={table['n_chains']} cadeias alpha={table['alpha']:g}")

    files = files_from_manifest(cfg, args.split)
    names = None
    gc_p, gc_t = [], []
    gt_p, gt_t, acc_t = None, None, None
    acc_c = TopLAccumulator()
    used = 0
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        # load_chain (rather than reading the label directly) so as to evaluate EXACTLY the
        # same chains as the champion: it is the emb.shape[0]==L test that drops 5hbl_A and
        # 9qlx_A.
        ch = load_chain(ef, lf, seq_sep, keep_emb=False)
        if ch is None:
            continue
        if names is None:
            names = ch["names"]
            T = len(names)
            gt_p = [[] for _ in range(T)]
            gt_t = [[] for _ in range(T)]
            acc_t = [TopLAccumulator() for _ in range(T)]
        pc, pt = predict_propensity(table, ch)
        vi, vj = ch["vi"], ch["vj"]
        sep = np.abs(vi - vj)
        yc = ch["tgt_c"][vi, vj]
        # float16/uint8 no acumulado: 5,18 M pares x 8 classes cabem em ~120 MB em vez de
        # ~480 MB. AUPRC is rank-based, so the reduced precision does not change the number.
        gc_p.append(pc.astype(np.float16)); gc_t.append(yc.astype(np.uint8))
        acc_c.add(pc, yc, sep, ch["L"])
        for k in range(len(names)):
            yt = ch["tgt_t"][k, vi, vj]
            gt_p[k].append(pt[:, k].astype(np.float16))
            gt_t[k].append(yt.astype(np.uint8))
            acc_t[k].add(pt[:, k], yt, sep, ch["L"])
        used += 1
        if used % 100 == 0:
            print(f"   {used} cadeias pontuadas", flush=True)

    print(f">> split={args.split} cadeias={used} tipos_avaliados={names}")
    gc_p, gc_t = np.concatenate(gc_p), np.concatenate(gc_t)
    ap_c = float(average_precision_score(gc_t, gc_p))
    ap_t = {}
    for k, n in enumerate(names):
        yt = np.concatenate(gt_t[k])
        pk = np.concatenate(gt_p[k])
        ap_t[n] = float(average_precision_score(yt, pk)) if yt.sum() > 0 else float("nan")
        gt_t[k], gt_p[k] = None, None  # free before the next class
        del yt, pk
    macro = float(np.nanmean(list(ap_t.values())))

    print(f"\n== DENSE EVALUATION ({args.split}, propensity) - all valid pairs ==")
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

    print("\n== Top-L per separation range (contact) ==")
    print(f"{'faixa':8s} " + "  ".join(f"L/{d}" if d > 1 else "L" for d in DIVISORS))
    for rng in list(RANGES) + ["all"]:
        print(f"{rng:8s} " +
              "  ".join(f"{tl_c[f'{rng}/L{d}' if d > 1 else f'{rng}/L']:.3f}" for d in DIVISORS))

    print("\n== Top-L (L) per class x separation range ==")
    print(f"{'classe':12s} " + "  ".join(f"{r:>7s}" for r in list(RANGES) + ["all"]))
    for k, n in enumerate(names):
        tl = acc_t[k].mean()
        print(f"{n:12s} " +
              "  ".join(f"{tl[f'{rng}/L']:7.3f}" for rng in list(RANGES) + ["all"]))


if __name__ == "__main__":
    main()
