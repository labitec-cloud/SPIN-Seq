"""Converts the pdbe-arpeggio JSON output into residue-residue labels (a typed RIN).

Pipeline (Fase 1 do PLANO):
  1. read the structure (mmCIF) with gemmi -> sequential residue index per chain;
  2. read the Arpeggio JSON -> typed atomic contacts;
  3. filter out water and keep only intra-chain interactions between protein residues;
  4. map raw types -> vocabulary T (config);
  5. aggregate atomic contacts per residue pair (OR of the types, minimum distance);
  6. aplica filtro |i-j| >= seq_sep_min;
  7. return sparse L x L x T labels + metadata.

Programmatic use:
    from src.supervision.arpeggio_labels import build_labels
    result = build_labels("data/raw/1ubq.cif", "data/arpeggio/1ubq/1ubq.json", cfg)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gemmi
import numpy as np


# Unique key of a residue within the structure.
ResKey = tuple[str, int, str]  # (auth_asym_id, auth_seq_id, ins_code)


@dataclass
class ChainIndex:
    """Map (chain, seq_id, icode) -> index 0..L-1 and sequence of a chain."""

    chain: str
    seq: str
    reskeys: list[ResKey]
    key_to_idx: dict[ResKey, int] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.reskeys)


def index_chains(cif_path: str) -> dict[str, ChainIndex]:
    """Indexes the polymer residues of every protein chain of the first model."""
    st = gemmi.read_structure(cif_path)
    st.setup_entities()
    model = st[0]
    chains: dict[str, ChainIndex] = {}
    for chain in model:
        poly = chain.get_polymer()
        if len(poly) == 0:
            continue
        # protein (peptide) chains only
        ptype = poly.check_polymer_type()
        if ptype not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        seq = gemmi.one_letter_code(poly.extract_sequence())
        reskeys: list[ResKey] = []
        key_to_idx: dict[ResKey, int] = {}
        for res in poly:
            key: ResKey = (chain.name, res.seqid.num, res.seqid.icode.strip())
            key_to_idx[key] = len(reskeys)
            reskeys.append(key)
        chains[chain.name] = ChainIndex(chain.name, seq, reskeys, key_to_idx)
    return chains


def _atom_reskey(atom: dict[str, Any]) -> ResKey:
    return (
        atom["auth_asym_id"],
        int(atom["auth_seq_id"]),
        (atom.get("pdbx_PDB_ins_code") or "").strip(),
    )


@dataclass
class LabelResult:
    chain: str
    seq: str
    length: int
    types: list[str]
    idx_i: np.ndarray          # (n_pos,) int32
    idx_j: np.ndarray          # (n_pos,) int32  (i < j)
    labels: np.ndarray         # (n_pos, T) uint8 multi-label
    min_dist: np.ndarray       # (n_pos,) float32 minimum distance of the pair
    reskeys: list[ResKey]

    def stats(self) -> dict[str, int]:
        counts = self.labels.sum(axis=0)
        return {t: int(c) for t, c in zip(self.types, counts)}


def build_labels(
    cif_path: str,
    arpeggio_json: list[dict] | str,
    cfg: dict,
    chain: str | None = None,
) -> dict[str, LabelResult]:
    """Builds labels per chain. If `chain` is None, processes every chain.

    Retorna {chain_name: LabelResult}.
    """
    import json

    if isinstance(arpeggio_json, str):
        with open(arpeggio_json) as fh:
            records = json.load(fh)
    else:
        records = arpeggio_json

    acfg = cfg["arpeggio"]
    types: list[str] = list(acfg["interaction_types"])
    type_idx = {t: k for k, t in enumerate(types)}
    type_map: dict[str, str] = acfg["type_map"]
    seq_sep_min: int = int(acfg["seq_sep_min"])
    want_entities = acfg.get("interacting_entities", "INTRA_SELECTION")
    drop_water = acfg.get("drop_water", True)

    chains = index_chains(cif_path)
    if chain is not None:
        chains = {chain: chains[chain]}

    # Per-chain accumulators: (i,j)->set of channels and minimum distance.
    per_chain_edges: dict[str, dict[tuple[int, int], np.ndarray]] = {
        c: {} for c in chains
    }
    per_chain_dist: dict[str, dict[tuple[int, int], float]] = {c: {} for c in chains}

    for rec in records:
        if drop_water and rec.get("interacting_entities") == "WATER_WATER":
            continue
        if want_entities and rec.get("interacting_entities") != want_entities:
            continue

        bgn, end = _atom_reskey(rec["bgn"]), _atom_reskey(rec["end"])
        if bgn[0] != end[0]:
            continue  # intra-cadeia apenas
        cname = bgn[0]
        cidx = chains.get(cname)
        if cidx is None:
            continue
        if bgn not in cidx.key_to_idx or end not in cidx.key_to_idx:
            continue  # water/ligand/hetero atom outside the polymer

        i = cidx.key_to_idx[bgn]
        j = cidx.key_to_idx[end]
        if i == j:
            continue
        if abs(i - j) < seq_sep_min:
            continue
        if i > j:
            i, j = j, i

        # mapeia tipos brutos -> canais
        vec = per_chain_edges[cname].get((i, j))
        if vec is None:
            vec = np.zeros(len(types), dtype=np.uint8)
            per_chain_edges[cname][(i, j)] = vec
        for raw in rec.get("contact", []):
            cls = type_map.get(raw)
            if cls is not None:
                vec[type_idx[cls]] = 1

        d = float(rec.get("distance", np.inf))
        prev = per_chain_dist[cname].get((i, j), np.inf)
        if d < prev:
            per_chain_dist[cname][(i, j)] = d

    results: dict[str, LabelResult] = {}
    for cname, cidx in chains.items():
        edges = per_chain_edges[cname]
        if edges:
            pairs = sorted(edges.keys())
            idx_i = np.array([p[0] for p in pairs], dtype=np.int32)
            idx_j = np.array([p[1] for p in pairs], dtype=np.int32)
            labels = np.stack([edges[p] for p in pairs]).astype(np.uint8)
            min_dist = np.array(
                [per_chain_dist[cname][p] for p in pairs], dtype=np.float32
            )
        else:
            idx_i = np.zeros(0, dtype=np.int32)
            idx_j = np.zeros(0, dtype=np.int32)
            labels = np.zeros((0, len(types)), dtype=np.uint8)
            min_dist = np.zeros(0, dtype=np.float32)

        results[cname] = LabelResult(
            chain=cname,
            seq=cidx.seq,
            length=cidx.length,
            types=types,
            idx_i=idx_i,
            idx_j=idx_j,
            labels=labels,
            min_dist=min_dist,
            reskeys=cidx.reskeys,
        )
    return results


def save_labels(result: LabelResult, out_path: str) -> None:
    """Salva um LabelResult como .npz esparso."""
    np.savez_compressed(
        out_path,
        chain=result.chain,
        seq=result.seq,
        length=result.length,
        types=np.array(result.types),
        idx_i=result.idx_i,
        idx_j=result.idx_j,
        labels=result.labels,
        min_dist=result.min_dist,
    )
