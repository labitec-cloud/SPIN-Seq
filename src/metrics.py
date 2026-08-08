"""Métricas de contact prediction: Top-L por classe e por faixa de separação.

Top-L (e Top-L/2, Top-L/5) = precisão entre as L predições mais confiantes de uma cadeia,
onde L é o comprimento da cadeia. É a convenção da área e dá um número mais representativo que
o AUPRC global (avalia só as predições confiantes, não todos os pares). Calculado POR CADEIA e
depois promediado sobre as cadeias (ignorando as sem positivos naquele recorte).

Faixas de separação sequencial |i-j| (convenção estilo CASP):
- short  6–11 · medium 12–23 · long >=24 · all >= seq_sep_min do treino.
"""
from __future__ import annotations

import numpy as np

RANGES = {"short": (6, 12), "medium": (12, 24), "long": (24, np.inf)}
DIVISORS = (1, 2, 5)  # Top-L, Top-L/2, Top-L/5


def top_l_precision(prob: np.ndarray, is_pos: np.ndarray, L: int, divisor: int) -> float:
    """Precisão entre os top-(L/divisor) pares por probabilidade, para uma cadeia/classe."""
    n = prob.shape[0]
    if n == 0 or is_pos.sum() == 0:
        return float("nan")
    k = max(1, min(n, L // divisor))
    top = np.argpartition(-prob, k - 1)[:k]
    return float(is_pos[top].mean())


def chain_top_l(prob: np.ndarray, is_pos: np.ndarray, sep: np.ndarray, L: int) -> dict:
    """Top-L/2/5 de uma cadeia/classe, por faixa de separação e no total (all)."""
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
    """Acumula Top-L por cadeia e devolve a média (nanmean) por recorte."""

    def __init__(self):
        self._rows: list[dict] = []

    def add(self, prob, is_pos, sep, L):
        self._rows.append(chain_top_l(prob, is_pos, sep, L))

    def mean(self) -> dict:
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in self._rows])) for k in keys}
