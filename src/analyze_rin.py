"""Tese RIN vs contact map (§9.7/§9.9) — quanto se perde ao só detectar proximidade.

Duas partes, ambas no portão denso (test, todos os pares válidos):

A) SÓ RÓTULOS (sem modelo): dado que o par é contato, quanto resta de incerteza sobre o TIPO?
   Reporta multiplicidade de tipos por aresta e a entropia condicional H(tipo|contato) por classe,
   em bits. Se fosse ~0, o contact map determinaria o RIN e o artigo não teria tese.

B) COM MODELO (opcional, --ckpt): AUPRC por classe de quatro rankers no MESMO conjunto de pares:
     prevalence  — ranker aleatório (piso).
     oracle      — contato VERDADEIRO como score do tipo. É o TETO da abordagem contact-map:
                   o melhor que um contact map perfeito consegue sem tipar a aresta.
     p_contact   — head de contato do nosso modelo usado como score do tipo (contact map real).
     RIN         — head de tipos do nosso modelo.
   Ganho da tese = RIN - oracle. Positivo ⇒ tipar a aresta usa informação que proximidade não dá.

Uso:
    python src/analyze_rin.py --config configs/esm650m.yaml                     # só parte A
    python src/analyze_rin.py --config configs/esm650m.yaml \
        --model conv2d --ckpt outputs/conv2d_650m/best.pt --device cpu
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
from src.baseline_propensity import (DEFAULT_TABLE, load_table, predict_propensity)
from src.data.pair_dataset import files_from_manifest
from src.eval_dense import load_chain, predict_conv2d, predict_mlp
from src.eval_ensemble import build_model


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def report_labels(names, y_types, y_contact):
    """Parte A: incerteza do tipo condicionada a contato."""
    n_pairs = y_contact.shape[0]
    n_contact = int(y_contact.sum())
    on_contact = y_types[y_contact > 0]
    mult = on_contact.sum(1)

    print(f"\n== A) SÓ RÓTULOS — pares={n_pairs:,} contatos={n_contact:,} "
          f"({100 * n_contact / n_pairs:.2f}% dos pares) ==")
    print(f"tipos por aresta de contato: media={mult.mean():.2f} "
          f"mediana={np.median(mult):.0f} max={int(mult.max())}")
    print("  distribuicao (# tipos: % das arestas): " + "  ".join(
        f"{k}:{100 * float((mult == k).mean()):.1f}%" for k in range(0, int(mult.max()) + 1)))

    print(f"\n{'classe':12s} {'p(tipo|contato)':>15s} {'H(tipo|contato) bits':>21s}")
    h_tot = 0.0
    for k, n in enumerate(names):
        p = float(on_contact[:, k].mean())
        h = binary_entropy(p)
        h_tot += h
        print(f"{n:12s} {p:15.4f} {h:21.4f}")
    print(f"{'SOMA':12s} {'':15s} {h_tot:21.4f}  "
          f"(bits que o contact map NAO resolve por aresta)")


def report_models(names, y_types, y_contact, p_contact=None, p_types=None, p_prop=None):
    """Parte B: AUPRC por classe dos rankers (piso, teto-do-contato, propensão, modelo).

    `propensity` é a tabela de par de AA (src/baseline_propensity.py): responde quanto do
    resultado é identidade de aminoácido e não o ESM-2. As colunas de ganho isolam as duas
    objeções — RIN-oracle (proximidade não basta) e RIN-prop (identidade não basta).
    """
    cols = ["prevalence", "oracle"]
    if p_prop is not None:
        cols.append("propensity")
    if p_types is not None:
        cols += ["p_contact", "RIN"]
    gains = ([("RIN-oracle", "oracle")] +
             ([("RIN-prop", "propensity")] if p_prop is not None else [])
             ) if p_types is not None else []

    print(f"\n== B) RANKERS — AUPRC por classe (test denso) ==")
    print(f"{'classe':12s} " + " ".join(f"{c:>10s}" for c in cols) +
          "".join(f" {g:>11s}" for g, _ in gains))
    rows = {}
    for k, n in enumerate(names):
        y = y_types[:, k]
        if y.sum() == 0:
            continue
        vals = {"prevalence": float(y.mean()),
                "oracle": float(average_precision_score(y, y_contact))}
        if p_prop is not None:
            vals["propensity"] = float(average_precision_score(y, p_prop[:, k]))
        if p_types is not None:
            vals["p_contact"] = float(average_precision_score(y, p_contact))
            vals["RIN"] = float(average_precision_score(y, p_types[:, k]))
        rows[n] = vals
        print(f"{n:12s} " + " ".join(f"{vals[c]:10.3f}" for c in cols) +
              "".join(f" {vals['RIN'] - vals[b]:+11.3f}" for _, b in gains))

    macro = {c: float(np.mean([r[c] for r in rows.values()])) for c in cols}
    print(f"{'MACRO':12s} " + " ".join(f"{macro[c]:10.3f}" for c in cols) +
          "".join(f" {macro['RIN'] - macro[b]:+11.3f}" for _, b in gains))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/esm650m.yaml")
    ap.add_argument("--model", choices=["mlp", "conv2d"])
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--propensity", nargs="?", const=DEFAULT_TABLE,
                    help="adiciona a coluna da baseline de propensão de par de AA")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
    emb_dir = cfg["paths"]["embeddings"]
    device = args.device

    model, ckpt_types = None, None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        ckpt_types = [str(t) for t in ck["types"]]

    table = load_table(args.propensity) if args.propensity else None

    names = None
    yc_all, yt_all, pc_all, pt_all, pp_all = [], [], [], [], []
    for lf in files_from_manifest(cfg, args.split):
        name = os.path.splitext(os.path.basename(lf))[0]
        ef = os.path.join(emb_dir, f"{name}.npz")
        if not os.path.exists(ef):
            continue
        ch = load_chain(ef, lf, seq_sep, keep_emb=bool(args.ckpt))
        if ch is None:
            continue
        if names is None:
            names = ch["names"]
        vi, vj = ch["vi"], ch["vj"]
        yc_all.append(ch["tgt_c"][vi, vj].astype(np.uint8))
        yt_all.append(ch["tgt_t"][:, vi, vj].T.astype(np.uint8))
        if table is not None:
            pp_all.append(predict_propensity(table, ch)[1].astype(np.float16))
        if args.ckpt:
            if model is None:
                model, _ = build_model(args.model, cfg, ck, ch["emb"].shape[1], use_esm, device)
            sel = [ckpt_types.index(n) for n in names]
            fn = predict_mlp if args.model == "mlp" else predict_conv2d
            pc, pt = fn(model, ch, device, sel, use_esm)
            pc_all.append(pc)
            pt_all.append(pt)

    y_contact = np.concatenate(yc_all)
    y_types = np.concatenate(yt_all)
    print(f">> split={args.split} cadeias={len(yc_all)} tipos={names}")

    p_prop = np.concatenate(pp_all) if pp_all else None
    report_labels(names, y_types, y_contact)
    if args.ckpt:
        report_models(names, y_types, y_contact, np.concatenate(pc_all),
                      np.concatenate(pt_all), p_prop)
    else:
        report_models(names, y_types, y_contact, p_prop=p_prop)


if __name__ == "__main__":
    main()
