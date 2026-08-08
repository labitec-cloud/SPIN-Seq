"""Fase 1 — gera rótulos RIN (L×L×T) para uma lista de PDBs.

Para cada PDB: baixa o mmCIF (se faltar), roda o pdbe-arpeggio (se faltar o JSON),
constrói os rótulos por cadeia e salva em data/labels/<pdb>_<chain>.npz.
Ao final imprime estatísticas globais por tipo de interação.

Uso:
    python scripts/build_labels.py 1ubq 1crn 4hhb
    python scripts/build_labels.py --pdb-list configs/pdb_ids.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.supervision.arpeggio_labels import build_labels, save_labels


def ensure_cif(pdb: str, raw_dir: str) -> str:
    path = os.path.join(raw_dir, f"{pdb}.cif")
    if not os.path.exists(path):
        os.makedirs(raw_dir, exist_ok=True)
        url = f"https://files.rcsb.org/download/{pdb}.cif"
        print(f"  baixando {url}")
        urllib.request.urlretrieve(url, path)
    return path


def ensure_arpeggio(pdb: str, cif: str, arp_dir: str) -> str:
    out_dir = os.path.join(arp_dir, pdb)
    json_path = os.path.join(out_dir, f"{pdb}.json")
    if not os.path.exists(json_path):
        os.makedirs(out_dir, exist_ok=True)
        print(f"  rodando pdbe-arpeggio ({pdb})")
        subprocess.run(
            ["pdbe-arpeggio", "-o", out_dir, cif],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return json_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdbs", nargs="*", help="IDs de PDB")
    ap.add_argument("--pdb-list", help="arquivo com um PDB ID por linha")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]

    pdbs = list(args.pdbs)
    if args.pdb_list:
        pdbs += [l.strip() for l in open(args.pdb_list) if l.strip() and not l.startswith("#")]
    if not pdbs:
        ap.error("informe PDBs ou --pdb-list")

    os.makedirs(paths["labels"], exist_ok=True)
    global_counts: Counter = Counter()
    total_pairs = 0
    total_residues = 0
    n_chains = 0
    types_ref: list[str] | None = None

    for pdb in pdbs:
        pdb = pdb.lower()
        print(f"[{pdb}]")
        try:
            cif = ensure_cif(pdb, paths["raw"])
            js = ensure_arpeggio(pdb, cif, paths["arpeggio"])
            results = build_labels(cif, js, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERRO: {exc}")
            continue

        for cname, r in results.items():
            if r.length == 0:
                continue
            types_ref = r.types
            out = os.path.join(paths["labels"], f"{pdb}_{cname}.npz")
            save_labels(r, out)
            n_chains += 1
            total_pairs += len(r.idx_i)
            total_residues += r.length
            for t, c in r.stats().items():
                global_counts[t] += c
            print(f"  cadeia {cname}: L={r.length} pares={len(r.idx_i)}")

    print("\n===== ESTATÍSTICAS GLOBAIS =====")
    print(f"cadeias: {n_chains} | resíduos: {total_residues} | pares positivos: {total_pairs}")
    if types_ref:
        print("por tipo de interação:")
        for t in types_ref:
            c = global_counts[t]
            pct = 100 * c / max(total_pairs, 1)
            print(f"  {t:12s} {c:8d}  ({pct:5.1f}% dos pares)")


if __name__ == "__main__":
    main()
