"""Regression test for the manifest's incremental split (anti-leakage).

`write_manifest` used to draw the split for ALL rows on every commit. With a fixed row
count that was harmless, but when expanding the dataset the whole permutation changed: a
single new row migrated 174 chains of the real manifest, 23 of them moving from test into
train. Since build calls write_manifest for every chain added, expanding the dataset would
contaminate the model-selection split and invalidate every number already measured.

These tests run against the REAL `data/manifest.csv`, not a synthetic one: it is the
experimental history that must be preserved.

Run (with the venv active, from the repo root):
    python -m pytest scripts/test_manifest_split.py -q
ou, sem pytest:
    python scripts/test_manifest_split.py
"""
from __future__ import annotations

import collections
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.build_dataset import write_manifest

MANIFEST = os.path.join(os.path.dirname(__file__), "..", "data", "manifest.csv")
FR = [0.8, 0.1, 0.1]
SEED = 42


def _orig():
    with open(MANIFEST) as fh:
        return list(csv.DictReader(fh))


def _read(path):
    with open(path) as fh:
        return {r["name"]: r["split"] for r in csv.DictReader(fh)}


def _grow(rows, n):
    for i in range(n):
        rows.append({"pdb": f"zz{i:04d}", "chain": "A", "name": f"zz{i:04d}_A",
                     "L": 100, "resolution": "2.0", "n_pairs": 500, "split": ""})


def test_noop_preserva_splits():
    orig = _orig()
    before = {r["name"]: r["split"] for r in orig}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.csv")
        write_manifest(p, [dict(r) for r in orig], FR, SEED)
        assert _read(p) == before


def test_incremental_expansion_migrates_nobody():
    """The bug scenario: one call per chain added, as commit() does."""
    orig = _orig()
    before = {r["name"]: r["split"] for r in orig}
    rows = [dict(r) for r in orig]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.csv")
        for _ in range(300):
            _grow(rows, 1)
            write_manifest(p, rows, FR, SEED)
        after = _read(p)
    migrou = [n for n, s in before.items() if after[n] != s]
    assert not migrou, f"{len(migrou)} cadeias migraram de split"


def test_fracoes_alvo_sao_mantidas():
    orig = _orig()
    rows = [dict(r) for r in orig]
    _grow(rows, 1000)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.csv")
        write_manifest(p, rows, FR, SEED)
        c = collections.Counter(_read(p).values())
    n = sum(c.values())
    for s, alvo in zip(("train", "val", "test"), FR):
        assert abs(c[s] / n - alvo) < 0.02, f"{s}={c[s]/n:.3f} vs alvo {alvo}"


def test_determinismo():
    orig = _orig()
    out = []
    for _ in range(2):
        rows = [dict(r) for r in orig]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.csv")
            for _ in range(50):
                _grow(rows, 1)
                write_manifest(p, rows, FR, SEED)
            out.append(_read(p))
    assert out[0] == out[1]


def test_reshuffle_explicito_reembaralha():
    orig = _orig()
    before = {r["name"]: r["split"] for r in orig}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.csv")
        write_manifest(p, [dict(r) for r in orig], FR, SEED, reshuffle=True)
        after = _read(p)
    assert sum(1 for n, s in before.items() if after[n] != s) > 100


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK  {name}")
    print("\ntodos os testes passaram")
