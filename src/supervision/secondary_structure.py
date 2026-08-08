"""Estrutura secundária (SS) 3-estados por resíduo, alinhada aos rótulos/embeddings.

§1.2 do plano de execução. Reusa `index_chains` (mesma ordem de polímero que os
rótulos e o embedding) para garantir alinhamento resíduo-a-resíduo. A SS é
computada com pydssp (DSSP em python puro) a partir do backbone N/CA/C/O; sem
mkdssp/sudo. Saída: índice int8 por resíduo — 0=coil(-), 1=hélice(H), 2=folha(E).

Uso:
    python src/supervision/secondary_structure.py --config configs/esm650m_aa.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import gemmi
import numpy as np
import pydssp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.supervision.arpeggio_labels import index_chains

SS3 = {"-": 0, "H": 1, "E": 2}
BACKBONE = ("N", "CA", "C", "O")


def ss_for_chain(cif_path: str, chain: str) -> np.ndarray | None:
    """SS 3-estados (int8, len=L) da cadeia, na ordem do polímero (= ordem dos rótulos)."""
    idx = index_chains(cif_path)
    if chain not in idx:
        return None
    ci = idx[chain]
    L = ci.length
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    poly = st[0][chain].get_polymer()

    coord = np.full((L, 4, 3), np.nan, np.float32)
    for k, res in enumerate(poly):
        if k >= L:
            break
        for a, name in enumerate(BACKBONE):
            at = res.find_atom(name, "*")
            if at is not None:
                coord[k, a] = [at.pos.x, at.pos.y, at.pos.z]
    ok = ~np.isnan(coord).any(axis=(1, 2))  # resíduos com backbone completo
    ss = np.zeros(L, np.int8)  # default coil
    if ok.sum() >= 4:
        filled = coord.copy()
        filled[~ok] = 0.0  # pydssp exige geometria finita; corrigido depois
        c3 = pydssp.assign(filled, out_type="c3")
        ss = np.array([SS3.get(str(s), 0) for s in c3], np.int8)
        ss[~ok] = 0  # resíduos sem backbone completo → coil (não confiar)
    return ss


def main():
    import csv
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/esm650m_aa.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/ss")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    label_dir = cfg["paths"]["labels"]
    os.makedirs(args.out, exist_ok=True)
    manifest = cfg["dataset"]["manifest"]
    names = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            names.append(r["name"])

    done = miss_cif = miss_align = 0
    hist = np.zeros(3, np.int64)
    for name in names:
        outp = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(outp):
            done += 1
            continue
        pdb, _, chain = name.rpartition("_")
        cif = os.path.join(args.raw, f"{pdb}.cif")
        if not os.path.exists(cif):
            miss_cif += 1
            continue
        lf = os.path.join(label_dir, f"{name}.npz")
        if not os.path.exists(lf):
            continue
        L = int(np.load(lf, allow_pickle=True)["length"])
        try:
            ss = ss_for_chain(cif, chain)
        except Exception as e:
            print(f"  {name}: erro {type(e).__name__}: {e}", flush=True)
            continue
        if ss is None or len(ss) != L:
            miss_align += 1
            print(f"  {name}: desalinhado (ss={None if ss is None else len(ss)} vs L={L})",
                  flush=True)
            continue
        for c in range(3):
            hist[c] += int((ss == c).sum())
        np.savez_compressed(outp, ss=ss)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(names)}", flush=True)
    tot = max(hist.sum(), 1)
    print(f">> ok={done} sem_cif={miss_cif} desalinhado={miss_align}")
    print(f">> SS coil={hist[0]/tot:.2%} helice={hist[1]/tot:.2%} folha={hist[2]/tot:.2%}")


if __name__ == "__main__":
    main()
