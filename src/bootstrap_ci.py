"""Bootstrap confidence intervals over the test chains.

Generates the model predictions ONCE (same protocol as the dense gate), stores them
per chain, and resamples the chains WITH replacement B times, recomputing contact and
per-class AUPRC + macro over the pooled pairs of each resample. Reports the mean and a
95% CI (percentiles 2.5/97.5). Without this, small deltas between models are
indefensible in a referee report.

With --ckpt-b it enters PAIRED mode: it resamples the SAME chains for both models and
reports the CI of the DELTA (b - a). This is the correct test for ablation - the variance
between proteins (a hard chain is hard for both) cancels out, which separate CIs do not
achieve. Overlapping CIs do NOT imply a non-significant delta.

Uso:
    python src/bootstrap_ci.py --config configs/esm650m_aa.yaml \
        --model conv2d --ckpt outputs/conv2d_650m_aa/best.pt --B 1000

    python src/bootstrap_ci.py --config configs/esm650m_aa.yaml \
        --ckpt outputs/conv2d_650m_aa/best.pt \
        --config-b configs/esm650m_aa_ssaux.yaml --ckpt-b outputs/conv2d_ssaux/best.pt

    # paired delta against the AA-pair propensity baseline
    python src/bootstrap_ci.py --config configs/esm650m_aa.yaml \
        --model propensity --ckpt outputs/propensity_table.npz \
        --model-b conv2d    --ckpt-b outputs/conv2d_650m_aa/best.pt
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
from src.baseline_propensity import load_table, predict_propensity
from src.data.pair_dataset import files_from_manifest
from src.eval_dense import load_chain, predict_conv2d, predict_mlp
from src.eval_ensemble import build_model


def ap(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


# Descending order of the 65,536 representable float16 values, precomputed once.
# Non-finite values go to the end (and are barred by an assert in ap_hist).
_CODE_VALS = np.arange(65536, dtype=np.uint16).view(np.float16)
_CODE_ORDER = np.argsort(np.where(np.isfinite(_CODE_VALS), _CODE_VALS,
                                  -np.inf))[::-1].copy()


def ap_hist(y, p):
    """EXACT AUPRC without argsort: groups by distinct float16 value via bincount.

    AUPRC depends only on grouping the scores by distinct value - which is what sklearn's
    `distinct_value_indices` does after sorting. Since the scores are already float16
    (~15k possible values), the argsort over 5.18M elements can be skipped and the counts
    taken directly: O(N) instead of O(N log N), 16x faster and bit-for-bit identical.
    """
    if y.sum() == 0:
        return float("nan")
    codes = np.ascontiguousarray(p).view(np.uint16)
    n = np.bincount(codes, minlength=65536)[_CODE_ORDER]
    tp = np.bincount(codes, weights=y, minlength=65536)[_CODE_ORDER]
    keep = n > 0
    ctp = np.cumsum(tp[keep])
    prec = ctp / np.cumsum(n[keep])
    rec = ctp / ctp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def predict_all(kind, cfg, ckpt_path, chains, names, device, tag):
    """Per-chain predictions of ONE ranker. float16 for scores (AUPRC is rank-based) and
    uint8 for targets -> ~4x less RAM (7.5 GB machine, aggressive OOM killer).

    kind='propensity' comes in through the AA-pair table instead of a checkpoint, which
    enables the PAIRED delta against the baseline - two separate CIs would not do."""
    if kind == "propensity":
        model, sel, use_esm = load_table(ckpt_path), None, None
        predict = predict_propensity
    else:
        use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
        ck = torch.load(ckpt_path, map_location=device)
        sel = [[str(t) for t in ck["types"]].index(n) for n in names]
        model, _ = build_model(kind, cfg, ck, chains[0]["emb"].shape[1], use_esm, device)
        predict = predict_mlp if kind == "mlp" else predict_conv2d
    per = []  # (pc, yc, pt[T], yt[T])
    for i, ch in enumerate(chains):
        pc, pt = predict(model, ch, device, sel, use_esm)
        vi, vj = ch["vi"], ch["vj"]
        per.append((pc.astype(np.float16), ch["tgt_c"][vi, vj].astype(np.uint8),
                    pt.astype(np.float16), ch["tgt_t"][:, vi, vj].astype(np.uint8)))
        if (i + 1) % 100 == 0:
            print(f"   [{tag}] prediction {i+1}/{len(chains)}", flush=True)
    del model
    if device == "cuda" and kind != "propensity":
        torch.cuda.empty_cache()
    return per


def make_metrics(per, T, exact=False):
    """exact=True uses sklearn (the reference); the default uses ap_hist, which gives the
    SAME number and makes the 1000 resamples take minutes instead of hours."""
    fn = ap if exact else ap_hist

    def metrics(idx):
        pc = np.concatenate([per[i][0] for i in idx])
        yc = np.concatenate([per[i][1] for i in idx])
        row = [fn(yc, pc)]
        for k in range(T):
            pk = np.concatenate([per[i][2][:, k] for i in idx])
            yk = np.concatenate([per[i][3][k] for i in idx])
            row.append(fn(yk, pk))
        row.append(float(np.nanmean(row[1:])))  # macro
        return row  # [contact, *types, macro]
    return metrics


def cache_path(cache_dir, ckpt, split):
    tag = os.path.splitext(os.path.basename(ckpt))[0]
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    return os.path.join(cache_dir, f"pred_{parent}_{tag}_{split}.npz")


def save_cache(path, per):
    lens = np.array([len(x[0]) for x in per], np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, lens=lens,
             pc=np.concatenate([x[0] for x in per]),
             yc=np.concatenate([x[1] for x in per]),
             pt=np.concatenate([x[2] for x in per]),
             yt=np.concatenate([x[3] for x in per], axis=1))


def load_cache(path, n_chains):
    z = np.load(path)
    lens = z["lens"]
    if len(lens) != n_chains:
        return None
    cut = np.cumsum(lens)[:-1]
    return list(zip(np.split(z["pc"], cut), np.split(z["yc"], cut),
                    np.split(z["pt"], cut), np.split(z["yt"], cut, axis=1)))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--config", default="configs/esm650m_aa.yaml")
    ap_.add_argument("--model", choices=["mlp", "conv2d", "propensity"], default="conv2d")
    ap_.add_argument("--ckpt", required=True)
    ap_.add_argument("--config-b")
    ap_.add_argument("--model-b", choices=["mlp", "conv2d", "propensity"], default="conv2d")
    ap_.add_argument("--ckpt-b", help="se dado, modo PAREADO: IC do delta (b - a)")
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--B", type=int, default=1000)
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--cache-dir", default="outputs/pred_cache",
                     help="reuse already computed predictions (the expensive step)")
    ap_.add_argument("--no-cache", action="store_true")
    args = ap_.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    emb_dir = cfg["paths"]["embeddings"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_b = None
    if args.ckpt_b:
        cfg_b = yaml.safe_load(open(args.config_b or args.config))
        if (cfg_b["paths"]["embeddings"] != emb_dir
                and "propensity" not in (args.model, args.model_b)):
            raise SystemExit("modo pareado exige o MESMO embedding nos dois modelos")

    # the embedding is only needed if one arm is a neural model
    keep_emb = not all(k == "propensity"
                       for k in ([args.model] + ([args.model_b] if args.ckpt_b else [])))

    files = files_from_manifest(cfg, args.split)
    chains, names = [], None
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        ch = load_chain(ef, lf, seq_sep, keep_emb=keep_emb)
        if ch is not None:
            chains.append(ch)
            names = ch["names"]
    T = len(names)

    def get_preds(kind, cfg_, ckpt, tag):
        cp = cache_path(args.cache_dir, ckpt, args.split)
        if not args.no_cache and os.path.exists(cp):
            per = load_cache(cp, len(chains))
            if per is not None:
                print(f"   [{tag}] predictions from cache: {cp}", flush=True)
                return per
        per = predict_all(kind, cfg_, ckpt, chains, names, device, tag)
        if not args.no_cache:
            save_cache(cp, per)
            print(f"   [{tag}] predictions saved to {cp}", flush=True)
        return per

    per_a = get_preds(args.model, cfg, args.ckpt, "A")
    per_b = (get_preds(args.model_b, cfg_b, args.ckpt_b, "B") if args.ckpt_b else None)
    n = len(per_a)
    print(f">> A={args.ckpt}" + (f" B={args.ckpt_b}" if per_b else "") +
          f" cadeias={n} B={args.B}", flush=True)

    chains.clear()  # the dense targets (~1.9 GB) are already distilled into per_a/per_b

    met_a = make_metrics(per_a, T)
    met_b = make_metrics(per_b, T) if per_b else None
    cols = ["contact"] + names + ["MACRO"]
    full = np.arange(n)
    rng = np.random.default_rng(args.seed)

    # ap_hist is exact, but this is the difference between publishing and taking it on
    # faith: check against sklearn at the point estimate (1x) before spending the B
    # resamples on the fast path.
    d = np.nanmax(np.abs(np.array(met_a(full))
                         - np.array(make_metrics(per_a, T, exact=True)(full))))
    print(f">> ap_hist vs sklearn (estimativa pontual): max |Δ| = {d:.2e}", flush=True)
    if d > 1e-9:
        raise SystemExit("ap_hist divergiu do sklearn — abortado")

    if met_b is None:
        point = met_a(full)
        boot = np.empty((args.B, T + 2))
        for b in range(args.B):
            boot[b] = met_a(rng.integers(0, n, n))
            if (b + 1) % 100 == 0:
                print(f"   bootstrap {b+1}/{args.B}", flush=True)
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        print(f"\n{'métrica':12s} {'ponto':>7s} {'IC95 low':>9s} {'IC95 high':>10s} {'±':>7s}")
        for j, c in enumerate(cols):
            print(f"{c:12s} {point[j]:7.3f} {lo[j]:9.3f} {hi[j]:10.3f} "
                  f"{(hi[j] - lo[j]) / 2:7.3f}")
        return

    # pareado: a MESMA reamostra de cadeias alimenta os dois modelos
    pa, pb = np.array(met_a(full)), np.array(met_b(full))
    boot = np.empty((args.B, T + 2))
    for b in range(args.B):
        idx = rng.integers(0, n, n)
        boot[b] = np.array(met_b(idx)) - np.array(met_a(idx))
        if (b + 1) % 100 == 0:
            print(f"   bootstrap pareado {b+1}/{args.B}", flush=True)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    # two-sided p: fraction of resamples crossing zero (minimum 1/B, never 0)
    pv = 2 * np.minimum((boot <= 0).mean(axis=0), (boot >= 0).mean(axis=0))
    pv = np.maximum(pv, 1.0 / args.B)

    print(f"\n== DELTA PAREADO (B - A), {args.split}, B={args.B} ==")
    print(f"{'métrica':12s} {'A':>7s} {'B':>7s} {'delta':>7s} "
          f"{'IC95 low':>9s} {'IC95 high':>10s} {'p':>7s} {'sig':>4s}")
    for j, c in enumerate(cols):
        sig = "*" if lo[j] > 0 or hi[j] < 0 else ""
        print(f"{c:12s} {pa[j]:7.3f} {pb[j]:7.3f} {pb[j] - pa[j]:+7.3f} "
              f"{lo[j]:9.3f} {hi[j]:10.3f} {pv[j]:7.3f} {sig:>4s}")
    print("\n* = the 95% CI of the delta does not cross zero.")


if __name__ == "__main__":
    main()
