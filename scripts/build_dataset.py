"""Fase 2.5 — monta um dataset não-redundante ponta a ponta.

Dedup: 1 representative per 30% sequence-identity cluster (RCSB file) -> no
leakage between splits. For each representative: downloads the mmCIF, filters (resolution,
tamanho, proteína), roda Arpeggio, gera rótulos e embeddings ESM-2, e registra
in the manifest with the split. Resumable: skips what is already done.

Uso:
    python scripts/build_dataset.py --target 300
    python scripts/build_dataset.py --target 50 --no-embeddings
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import urllib.request

import gemmi
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.supervision.arpeggio_labels import build_labels, save_labels


def ensure_cluster_file(url: str, path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f">> baixando clusters: {url}")
        urllib.request.urlretrieve(url, path)


import json


def pdb_to_cluster(cluster_file: str) -> dict[str, int]:
    m: dict[str, int] = {}
    with open(cluster_file) as fh:
        for cid, line in enumerate(fh):
            for tok in line.split():
                pdb = tok.split("_")[0].lower()
                m.setdefault(pdb, cid)
    return m


def search_candidates(dcfg: dict, want: int) -> list[str]:
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal",
            "value": dcfg["max_resolution"]}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein", "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_polymer_monomer_count", "operator": "range",
            "value": {"from": dcfg["min_len"], "to": dcfg["max_len"]}}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.deposited_model_count", "operator": "equals", "value": 1}},
    ]
    out: list[str] = []
    page = 10000
    for start in range(0, want, page):
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
             "return_type": "entry",
             "request_options": {"paginate": {"start": start, "rows": min(page, want - start)}}}
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(q).encode(), headers={"Content-Type": "application/json"})
        rs = json.loads(urllib.request.urlopen(req, timeout=90).read()).get("result_set", [])
        if not rs:
            break
        out += [r["identifier"].lower() for r in rs]
    return out


def representative_pdbs(dcfg: dict, want: int, seen: set[str],
                        rng: np.random.Generator) -> list[str]:
    raw = search_candidates(dcfg, min(max(want * 15, 800), 60000))
    rng.shuffle(raw)
    p2c = pdb_to_cluster(dcfg["cluster_file"])
    used_clusters = {p2c[p] for p in seen if p in p2c}
    out: list[str] = []
    for pdb in raw:
        if pdb in seen:
            continue
        cid = p2c.get(pdb)
        if cid is not None and cid in used_clusters:
            continue  # dedup by sequence cluster (30%)
        if cid is not None:
            used_clusters.add(cid)
        out.append(pdb)
        if len(out) >= want * 5:  # margin for rejections (pick_chain/arpeggio/0 pairs)
            break
    return out


def pick_chain(cif: str, dcfg: dict):
    st = gemmi.read_structure(cif)
    st.setup_entities()
    res = st.resolution
    if res <= 0 or res > dcfg["max_resolution"]:
        return None
    # Arpeggio processa a estrutura INTEIRA → descarta assemblies grandes.
    total_atoms = sum(len(r) for ch in st[0] for r in ch)
    if total_atoms > dcfg.get("max_total_atoms", 8000):
        return None
    for chain in st[0]:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        if poly.check_polymer_type() not in (
            gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        L = len(poly)
        if dcfg["min_len"] <= L <= dcfg["max_len"]:
            return chain.name, L, res
    return None


def download_cif(pdb: str, raw_dir: str, max_mb: float) -> str | None:
    path = os.path.join(raw_dir, f"{pdb}.cif")
    if os.path.exists(path):
        return path
    url = f"https://files.rcsb.org/download/{pdb}.cif"
    cap = int(max_mb * 1e6)
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as fh:
            size = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:  # aborta downloads grandes sem HEAD
                    fh.close()
                    os.remove(tmp)
                    return None
                fh.write(chunk)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    return path


def run_arpeggio(pdb: str, cif: str, arp_dir: str, timeout: int = 60) -> str | None:
    out_dir = os.path.join(arp_dir, pdb)
    js = os.path.join(out_dir, f"{pdb}.json")
    if os.path.exists(js):
        return js
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(["pdbe-arpeggio", "-o", out_dir, cif], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
    except Exception:
        return None
    return js if os.path.exists(js) else None


def process_pdb(pdb: str, cfg: dict) -> dict | None:
    """Worker CPU: download → filtra → Arpeggio → rótulos → salva. Sem GPU."""
    dcfg, paths = cfg["dataset"], cfg["paths"]
    cif = download_cif(pdb, paths["raw"], dcfg.get("max_cif_mb", 3.0))
    if not cif:
        return None
    try:
        picked = pick_chain(cif, dcfg)
    except Exception:
        return None
    if not picked:
        return None
    chain, L, res = picked
    js = run_arpeggio(pdb, cif, paths["arpeggio"], dcfg.get("arpeggio_timeout", 60))
    if not js:
        return None
    try:
        r = build_labels(cif, js, cfg, chain=chain)[chain]
    except Exception:
        return None
    if len(r.idx_i) == 0:
        return None
    name = f"{pdb}_{chain}"
    save_labels(r, os.path.join(paths["labels"], f"{name}.npz"))
    return {"pdb": pdb, "chain": chain, "name": name, "L": L,
            "resolution": f"{res:.2f}", "n_pairs": len(r.idx_i), "split": ""}


def _worker(args):
    return process_pdb(*args)


SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ["pdb", "chain", "name", "L", "resolution", "n_pairs", "split"]


def write_manifest(path: str, rows: list[dict], split_fr, seed: int,
                   reshuffle: bool = False) -> None:
    """Draws the split ONLY for new rows; already assigned ones stay frozen.

    The previous version redid `permutation(len(rows))` on every call - and since
    build calls this for every chain added, any change in the row count
    reembaralhava TODOS os splits. Cadeias do test migravam para o train, o que
    invalidates every number already measured and contaminates the model-selection split.

    `reshuffle=True` redoes the split from scratch. Only with explicit intent: it discards the
    histórico experimental inteiro.
    """
    if reshuffle:
        for r in rows:
            r["split"] = ""

    fr = dict(zip(SPLITS, split_fr))
    new = [r for r in rows if not r.get("split")]
    if new:
        counts = {s: sum(1 for r in rows if r.get("split") == s) for s in SPLITS}
        total = len(rows)
        # ordem embaralhada (determinística pelo seed) + preenchimento do split mais
        # that is furthest behind: converges to the target fractions even with incremental additions.
        for k in np.random.default_rng(seed).permutation(len(new)):
            s = max(SPLITS, key=lambda x: fr[x] * total - counts[x])
            new[k]["split"] = s
            counts[s] += 1

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target", type=int)
    ap.add_argument("--no-embeddings", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="processos paralelos p/ rótulos (CPU). >1 desliga embeddings inline.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dcfg = cfg["dataset"]
    paths = cfg["paths"]
    target = args.target or dcfg["target"]
    rng = np.random.default_rng(cfg["seed"])
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    ensure_cluster_file(dcfg["cluster_url"], dcfg["cluster_file"])

    manifest = dcfg["manifest"]
    rows: list[dict] = []
    seen: set[str] = set()
    if os.path.exists(manifest):
        with open(manifest) as fh:
            rows = list(csv.DictReader(fh))
        seen = {r["pdb"] for r in rows}
        print(f">> manifest existente: {len(rows)} cadeias")
    # also resumable from labels already generated (e.g. after a timeout)
    import glob
    for lf in glob.glob(os.path.join(paths["labels"], "*.npz")):
        nm = os.path.splitext(os.path.basename(lf))[0]
        pdb = nm.rsplit("_", 1)[0]
        if pdb not in seen:
            seen.add(pdb)
            if not any(r["name"] == nm for r in rows):
                z = np.load(lf, allow_pickle=True)
                rows.append({"pdb": pdb, "chain": nm.rsplit("_", 1)[1],
                             "name": nm, "L": int(z["length"]),
                             "resolution": "", "n_pairs": len(z["idx_i"]), "split": ""})

    ext = None
    if not args.no_embeddings and args.workers == 1:
        from src.features.esm_embeddings import ESMExtractor
        print(f">> carregando ESM-2 ({cfg['esm']['model']})")
        ext = ESMExtractor(cfg)

    need = target - len(rows)
    candidates = representative_pdbs(dcfg, need, seen, rng) if need > 0 else []
    print(f">> alvo={target} faltam={need} candidatos={len(candidates)} workers={args.workers}", flush=True)

    added = 0

    def commit(r: dict):
        nonlocal added
        rows.append(r)
        seen.add(r["pdb"])
        added += 1
        write_manifest(manifest, rows, dcfg["split"], cfg["seed"])
        print(f"  +{added} total={len(rows)} {r['name']} L={r['L']} pares={r['n_pairs']}", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            for r in pool.imap_unordered(_worker, ((p, cfg) for p in candidates)):
                if r is not None:
                    commit(r)
                if len(rows) >= target:
                    pool.terminate()
                    break
    else:
        for pdb in candidates:
            if len(rows) >= target:
                break
            r = process_pdb(pdb, cfg)
            if r is None:
                continue
            if ext is not None:
                try:
                    z = np.load(os.path.join(paths["labels"], f"{r['name']}.npz"), allow_pickle=True)
                    ext.extract(str(z["seq"]),
                                out_path=os.path.join(paths["embeddings"], f"{r['name']}.npz"),
                                want_contacts=True)
                except Exception:
                    continue
            commit(r)

    write_manifest(manifest, rows, dcfg["split"], cfg["seed"])
    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"\n>> dataset: {len(rows)} cadeias | +{added} novas | splits={counts}")
    print(f">> manifest: {manifest}")


if __name__ == "__main__":
    main()
