"""Sanity tests for label construction.

Run (with the venv active, from the repo root):
    python -m pytest src/supervision/test_arpeggio_labels.py -q
or, without pytest:
    python src/supervision/test_arpeggio_labels.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# permite rodar tanto via pytest quanto direto (python src/.../test_*.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CFG = {
    "arpeggio": {
        "seq_sep_min": 3,
        "interacting_entities": "INTRA_SELECTION",
        "drop_water": True,
        "interaction_types": ["hbond", "polar", "vdw", "proximal"],
        "type_map": {
            "hbond": "hbond",
            "polar": "polar",
            "vdw": "vdw",
            "vdw_clash": "vdw",
            "proximal": "proximal",
        },
    }
}


def _index_stub(monkey_seq_len: int = 10):
    """Builds a synthetic ChainIndex of a chain 'A' with residues 1..N."""
    from src.supervision.arpeggio_labels import ChainIndex

    reskeys = [("A", n, "") for n in range(1, monkey_seq_len + 1)]
    key_to_idx = {k: i for i, k in enumerate(reskeys)}
    return {"A": ChainIndex("A", "A" * monkey_seq_len, reskeys, key_to_idx)}


def _atom(chain: str, seq: int) -> dict:
    return {
        "auth_asym_id": chain,
        "auth_atom_id": "CA",
        "auth_seq_id": seq,
        "label_comp_id": "ALA",
        "pdbx_PDB_ins_code": " ",
    }


def _rec(a: int, b: int, contacts: list[str], dist: float, ent="INTRA_SELECTION") -> dict:
    return {
        "bgn": _atom("A", a),
        "end": _atom("A", b),
        "contact": contacts,
        "distance": dist,
        "interacting_entities": ent,
        "type": "atom-atom",
    }


def test_aggregation_and_filters(monkeypatch):
    from src.supervision import arpeggio_labels as mod

    monkeypatch.setattr(mod, "index_chains", lambda _p: _index_stub(10))

    records = [
        _rec(1, 5, ["hbond", "proximal"], 3.0),   # par (0,4)
        _rec(1, 5, ["vdw", "proximal"], 2.8),      # mesmo par -> OR + dist menor
        _rec(2, 3, ["hbond", "proximal"], 2.9),    # |i-j|=1 < 3 -> descartado
        _rec(4, 8, ["polar", "proximal"], 3.5),    # par (3,7)
        _rec(1, 6, ["hbond"], 2.5, ent="SELECTION_WATER"),  # entidade errada -> fora
    ]

    res = mod.build_labels("dummy.cif", records, CFG)["A"]
    types = res.types

    # dois pares positivos: (0,4) e (3,7)
    pairs = set(zip(res.idx_i.tolist(), res.idx_j.tolist()))
    assert pairs == {(0, 4), (3, 7)}

    # pair (0,4): hbond, vdw, proximal on; minimum distance 2.8
    k = list(zip(res.idx_i, res.idx_j)).index((0, 4))
    lab = res.labels[k]
    assert lab[types.index("hbond")] == 1
    assert lab[types.index("vdw")] == 1
    assert lab[types.index("proximal")] == 1
    assert lab[types.index("polar")] == 0
    assert abs(res.min_dist[k] - 2.8) < 1e-6

    # i<j e |i-j|>=3 em todos
    assert (res.idx_i < res.idx_j).all()
    assert (np.abs(res.idx_i - res.idx_j) >= 3).all()


if __name__ == "__main__":
    # minimal runner without pytest
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)

    test_aggregation_and_filters(_MP())
    print("OK: test_aggregation_and_filters")
