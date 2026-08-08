"""Contact-prediction metrics: Top-L per class and per separation range.

Top-L (and Top-L/2, Top-L/5) = precision among the L most confident predictions of a chain,
where L is the chain length. It is the convention of the field and gives a more representative
number than the global AUPRC (it scores only the confident predictions, not every pair). Computed
PER CHAIN and then averaged over chains (ignoring those with no positives in that slice).

Sequence-separation ranges |i-j| (CASP-style convention):
- short  6–11 · medium 12–23 · long >=24 · all >= seq_sep_min do treino.
"""
from __future__ import annotations

import numpy as np

RANGES = {"short": (6, 12), "medium": (12, 24), "long": (24, np.inf)}
DIVISORS = (1, 2, 5)  # Top-L, Top-L/2, Top-L/5


def top_l_precision(prob: np.ndarray, is_pos: np.ndarray, L: int, divisor: int) -> float:
    """Precision among the top-(L/divisor) pairs by probability, for one chain/class."""
    n = prob.shape[0]
    if n == 0 or is_pos.sum() == 0:
        return float("nan")
    k = max(1, min(n, L // divisor))
    top = np.argpartition(-prob, k - 1)[:k]
    return float(is_pos[top].mean())


def chain_top_l(prob: np.ndarray, is_pos: np.ndarray, sep: np.ndarray, L: int) -> dict:
    """Top-L/2/5 for one chain/class, per separation range and overall (all)."""
    out = {}
    for rng, (lo, hi) in RANGES.items():
        m = (sep >= lo) & (sep < hi)
        for d in DIVISORS:
            out[f"{rng}/L{d}" if d > 1 else f"{rng}/L"] = top_l_precision(
                prob[m], is_pos[m], L, d)
    for d in DIVISORS:
        out[f"all/L{d}" if d > 1 else "all/L"] = top_l_precision(prob, is_pos, L, d)
    return out


class TopLAccumulator:
    """Accumulates Top-L per chain and returns the mean (nanmean) per slice."""

    def __init__(self):
        self._rows: list[dict] = []

    def add(self, prob, is_pos, sep, L):
        self._rows.append(chain_top_l(prob, is_pos, sep, L))

    def mean(self) -> dict:
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in self._rows])) for k in keys}
