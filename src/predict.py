"""SPIN-Seq inference - a predicted RIN from ONE SEQUENCE, with no structure.

Runs the full path: sequence -> ESM-2 (frozen) -> PairConv2D -> probability of
contact and of the 8 interaction types for each pair (i,j) with i<j and |i-j| >= seq_sep_min.

Nothing here reads PDB, DSSP or Arpeggio: this is the real usage mode. The optional `--pdb` is
only for COMPARING against labels that already exist in the dataset (verification mode, not inference).

Uso:
    python src/predict.py --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
    python src/predict.py --fasta minha.fasta --top 30
    python src/predict.py --seq ... --csv saida.csv --npz saida.npz
    python src/predict.py --pdb 1ubq_A          # check against the dataset labels
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pair_dataset import AA_UNK, seq_to_idx
from src.models.pair_conv2d import PairConv2D

AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str) -> tuple[str, str]:
    """First FASTA record -> (name, uppercase sequence)."""
    name, parts = os.path.basename(path), []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:            
                    break
                name = line[1:].split()[0] or name
            else:
                parts.append(line)
    return name, "".join(parts).upper()


def build_model(ck, cfg, emb_dim, use_esm, device):
    """Rebuilds the PairConv2D from the checkpoint ITSELF.

    aa_dim, use_ss, the number of distance bins and the presence of the SS pair channel are deduced
    from the state_dict, so the config only needs to match proj_dim/channels/n_blocks/dilations.
    """
    cc = cfg["conv2d"]
    sd = ck["model"]
    n_dist_bins = sd["head_dist.weight"].shape[0]
    aa_dim = sd["aa_emb.weight"].shape[1] if "aa_emb.weight" in sd else 0
    use_ss = "ss_head.weight" in sd
    base = 3 * cc["proj_dim"] + 1 + (1 if use_esm else 0) + 3 * aa_dim
    ss_pair = (sd["stem.weight"].shape[1] - base) >= 2
    m = PairConv2D(emb_dim, len(ck["types"]), cc["proj_dim"], cc["channels"], cc["n_blocks"],
                   tuple(cc["dilations"]), cc["dropout"], use_esm, n_dist_bins, aa_dim,
                   use_ss, ss_pair).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m, ss_pair


@torch.no_grad()
def predict(model, emb, contacts, seq, device, use_esm):
    """Runs the whole chain at once (L<=350 fits comfortably) and symmetrises the output."""
    L = len(seq)
    er = torch.from_numpy(emb).unsqueeze(0).to(device)
    sep = torch.from_numpy(
        np.log1p(np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])).astype(np.float32)
    ).unsqueeze(0).to(device)
    cf = torch.from_numpy(contacts).unsqueeze(0).to(device) if use_esm else None
    aa = torch.from_numpy(seq_to_idx(seq)).unsqueeze(0).to(device)
    ss = torch.full((1, L), 3, dtype=torch.long, device=device)  # SS unknown: sequence-only
    lc, lt, _ = model(er, er, sep, cf, symmetrize=True,
                      aa_row=aa, aa_col=aa, ss_row=ss, ss_col=ss)
    return torch.sigmoid(lc)[0].cpu().numpy(), torch.sigmoid(lt)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="RIN inference from the sequence")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="amino-acid sequence (one-letter)")
    src.add_argument("--fasta", help="arquivo FASTA (usa o primeiro registro)")
    src.add_argument("--pdb", help="dataset name (e.g. 1ubq_A): uses cached embeddings and labels")
    ap.add_argument("--config", default="configs/esm650m_aa.yaml")
    ap.add_argument("--ckpt", default="outputs/conv2d_650m_aa/best.pt")
    ap.add_argument("--top", type=int, default=20, help="how many pairs to list per class")
    ap.add_argument("--csv", help="save ALL valid pairs to CSV")
    ap.add_argument("--npz", help="salva os mapas L×L completos")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda se houver)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
    use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 1. sequence + ESM-2 features -------------------------------------------------
    labels = None
    if args.pdb:
        name = args.pdb
        ef = os.path.join(cfg["paths"]["embeddings"], f"{name}.npz")
        if not os.path.exists(ef):
            ap.error(f"no cached embedding: {ef}")
        e = np.load(ef, allow_pickle=True)
        emb = e["emb"].astype(np.float32)
        contacts = np.nan_to_num(e["contacts"].astype(np.float32))
        seq = str(e["seq"])
        lf = os.path.join(cfg["paths"]["labels"], f"{name}.npz")
        if os.path.exists(lf):
            labels = np.load(lf, allow_pickle=True)
        print(f">> cache: {name} L={len(seq)}" + ("  (with labels for verification)" if labels
                                                  is not None else ""))
    else:
        name, seq = (("seq", args.seq.strip().upper()) if args.seq else read_fasta(args.fasta))
        if not seq:
            ap.error("empty sequence")
        bad = sorted(set(seq) - AA_VALID)
        if bad:
            # ESM-2 tolerates X/B/Z; the AA embedding sends all of those to AA_UNK
            print(f">> warning: non-standard residues {bad} -> treated as unknown (AA_UNK)")
        dmin, dmax = cfg["dataset"]["min_len"], cfg["dataset"]["max_len"]
        if not (dmin <= len(seq) <= dmax):
            print(f">> AVISO: L={len(seq)} fora da faixa de TREINO [{dmin}, {dmax}] — "
                  "the model never saw this regime; treat the result as extrapolation")
        print(f">> ESM-2 ({cfg['esm']['model']}) em {device}...")
        from src.features.esm_embeddings import ESMExtractor

        ecfg = dict(cfg)
        ecfg["esm"] = dict(cfg["esm"], device=device)
        r = ESMExtractor(ecfg).extract(seq, want_contacts=True)
        emb, contacts = r["emb"], np.nan_to_num(r["contacts"])

    L = len(seq)
    if L <= seq_sep:
        ap.error(f"sequence too short: L={L} <= seq_sep_min={seq_sep}")

    # ---- 2. modelo ------------------------------------------------------------------------
    ck = torch.load(args.ckpt, map_location=device)
    types = [str(t) for t in ck["types"]]
    model, ss_pair = build_model(ck, cfg, emb.shape[1], use_esm, device)
    if ss_pair:
        ap.error(f"{args.ckpt} uses SS (DSSP) as an INPUT - it depends on the 3D structure and cannot "
                 "be used for sequence-only inference. Use the sequence-only checkpoint.")
    n_par = sum(p.numel() for p in model.parameters())
    print(f">> ckpt={args.ckpt} params={n_par/1e6:.2f}M tipos={types}")

    # ---- 3. inference --------------------------------------------------------------------
    pc, pt = predict(model, emb, contacts, seq, device, use_esm)
    vi, vj = np.triu_indices(L, k=seq_sep)
    print(f">> L={L} valid pairs={len(vi)} (i<j, |i-j|>={seq_sep})")

    # ---- 4. report ---------------------------------------------------------------------
    pc_v = pc[vi, vj]
    print(f"\n== TOP-{args.top} CONTATOS ==")
    print(f"{'i':>5s} {'j':>5s} {'aa':>5s} {'|i-j|':>6s} {'contato':>8s}  " +
          "  ".join(f"{t[:6]:>6s}" for t in types))
    for k in np.argsort(-pc_v)[:args.top]:
        i, j = int(vi[k]), int(vj[k])
        print(f"{i+1:5d} {j+1:5d} {seq[i]+'-'+seq[j]:>5s} {j-i:6d} {pc_v[k]:8.3f}  " +
              "  ".join(f"{pt[t, i, j]:6.3f}" for t in range(len(types))))

    print(f"\n== TOP-{min(args.top, 5)} PER TYPE ==")
    for t, tname in enumerate(types):
        v = pt[t][vi, vj]
        top = np.argsort(-v)[:min(args.top, 5)]
        s = "  ".join(f"{int(vi[k])+1}{seq[vi[k]]}-{int(vj[k])+1}{seq[vj[k]]}:{v[k]:.2f}"
                      for k in top)
        print(f"{tname:12s} {s}")

    # Top-L: os L pares mais confiantes, a leitura usada no artigo.
    nl = np.argsort(-pc_v)[:L]
    print(f"\n>> predicted density: Top-L (L={L}) has mean contact "
          f"{pc_v[nl].mean():.3f}; pares acima de 0.5 = {int((pc_v > 0.5).sum())}")

    if labels is not None:
        ltypes = [str(t) for t in labels["types"]]
        tgt = np.zeros((L, L), np.float32)
        tgt[labels["idx_i"].astype(int), labels["idx_j"].astype(int)] = 1.0
        y = tgt[vi, vj]
        order = np.argsort(-pc_v)[:L]
        print(f">> CHECK against labels: true positives={int(y.sum())} | "
              f"Top-L precision={y[order].mean():.3f}")
        del ltypes

    # ---- 5. outputs ------------------------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["i", "j", "aa_i", "aa_j", "sep", "contact"] + types)
            for k in range(len(vi)):
                i, j = int(vi[k]), int(vj[k])
                w.writerow([i + 1, j + 1, seq[i], seq[j], j - i, round(float(pc_v[k]), 4)] +
                           [round(float(pt[t, i, j]), 4) for t in range(len(types))])
        print(f">> salvo {args.csv} ({len(vi)} linhas)")
    if args.npz:
        np.savez_compressed(args.npz, contact=pc.astype(np.float32),
                            types=pt.astype(np.float32), type_names=np.array(types),
                            seq=np.array(seq), seq_sep_min=seq_sep)
        print(f">> salvo {args.npz} (mapas {L}×{L})")


if __name__ == "__main__":
    main()
