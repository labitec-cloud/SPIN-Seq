"""Losses of the 2D head.

Everything operates on `w x w` maps and aggregates ONLY over valid pixels:
- `lm` (loss mask): 1 where the pair is valid (not padding and |i-j| >= seq_sep);
- `dm` (dist mask): 1 only where there is contact (distance exists only for contacting pairs).

`focal_bce` substitui o `pos_weight` do baseline: o fator `(1-p_t)^gamma` foca nos positivos
hard examples of the rare classes, and `alpha` rebalances positive/negative - better suited
to severe per-class imbalance.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def masked_bce(logits, targets, mask):
    """Contact BCE, averaged over the valid pixels (mask: B x H x W)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return _masked_mean(ce, mask)


def focal_bce(logits, targets, mask, gamma: float = 2.0, alpha: float = 0.25,
              class_weight=None):
    """Multi-label focal loss. logits/targets: B x T x H x W; mask: B x H x W (broadcast over T).

    class_weight (T,): per-class weight (rebalances rare classes smothered in the dense regime)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal * ce                      # B x T x H x W
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)
    m = mask.unsqueeze(1)                            # B x 1 x H x W
    return (loss * m).sum() / (m.sum() * targets.shape[1] + 1e-6)


def asymmetric_loss(logits, targets, mask, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                    clip: float = 0.05, class_weight=None, eps: float = 1e-8):
    """Multi-label Asymmetric Loss (Ben-Baruch 2021). Same shapes as `focal_bce`.

    Fixes the focal loss's long-tail defect: there `alpha`(=0.25) REDUCES the weight of the
    positive, the opposite of what is needed when the positive is already extremely rare. Here
    the focusing terms are DECOUPLED - `gamma_pos`=0 does not discount positives, `gamma_neg`>0
    discounts only the easy negatives - and `clip` (probability shifting) discards outright the
    negatives with p < clip, which dominate the dense regime and were consuming the gradient of
    the rare classes. Aggregation identical to `focal_bce`, to keep the comparison isolated.
    """
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    l_pos = targets * (1 - p).pow(gamma_pos) * torch.log(p.clamp(min=eps))
    l_neg = (1 - targets) * p_neg.pow(gamma_neg) * torch.log((1 - p_neg).clamp(min=eps))
    loss = -(l_pos + l_neg)                          # B x T x H x W
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)
    m = mask.unsqueeze(1)
    return (loss * m).sum() / (m.sum() * targets.shape[1] + 1e-6)


def masked_mse(pred, targets, dmask):
    """Distance MSE only where there is contact (dmask: B x H x W)."""
    se = (pred - targets) ** 2
    return _masked_mean(se, dmask)


def masked_dist_ce(logits, targets, dmask, edges):
    """Distance CE over BINS (trRosetta style), only where there is contact (dmask: B x H x W).

    logits: B x n_bins x H x W; targets: B x H x W (distance in A); edges: internal boundaries
    (n_bins-1,) for `bucketize` -> bin index in [0, n_bins-1]."""
    bins = torch.bucketize(targets, edges)            # B x H x W em [0, n_bins-1]
    ce = F.cross_entropy(logits, bins, reduction="none")  # B x H x W
    return _masked_mean(ce, dmask)


def multitask_loss(out, batch, lambdas, gamma: float = 2.0, alpha: float = 0.25,
                   class_weight=None, dist_edges=None, asl=None):
    """out = (logit_contact, logit_types, logit_dist); batch = output of ChainCropDataset.

    If `dist_edges` is given, distance is learned as bin classification (trRosetta);
    otherwise it falls back to the legacy MSE (1-channel head). If `asl` (dict) is given, the
    canais de tipo usam Asymmetric Loss no lugar do focal. Devolve (loss, parcelas).
    """
    lc, lt, ld = out
    yc, yt, yd, dm, lm = batch[4:9]
    loss_c = masked_bce(lc, yc, lm)
    if asl is not None:
        loss_t = asymmetric_loss(lt, yt, lm, asl["gamma_pos"], asl["gamma_neg"],
                                 asl["clip"], class_weight)
    else:
        loss_t = focal_bce(lt, yt, lm, gamma, alpha, class_weight)
    if dist_edges is not None:
        loss_d = masked_dist_ce(ld, yd, dm, dist_edges)
    else:
        loss_d = masked_mse(ld.squeeze(1), yd, dm)
    total = lambdas[0] * loss_c + lambdas[1] * loss_t + lambdas[2] * loss_d
    return total, {"contact": float(loss_c), "types": float(loss_t), "dist": float(loss_d)}
