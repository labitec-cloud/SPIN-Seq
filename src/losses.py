"""Perdas da cabeça 2D (Fase 4).

Tudo opera sobre mapas `w×w` e agrega SÓ nos pixels válidos:
- `lm` (loss mask): 1 onde o par é válido (não-preenchido e |i-j| >= seq_sep);
- `dm` (dist mask): 1 só onde há contato (a distância só existe para pares em contato).

`focal_bce` substitui o `pos_weight` do baseline: o fator `(1-p_t)^gamma` foca nos positivos
difíceis das classes raras, e `alpha` reequilibra positivo/negativo — melhor para o
desbalanceamento severo por classe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    return (x * mask).sum() / (mask.sum() + eps)


def masked_bce(logits, targets, mask):
    """BCE de contato, média sobre os pixels válidos (mask: B×H×W)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return _masked_mean(ce, mask)


def focal_bce(logits, targets, mask, gamma: float = 2.0, alpha: float = 0.25,
              class_weight=None):
    """Focal loss multi-rótulo. logits/targets: B×T×H×W; mask: B×H×W (broadcast em T).

    class_weight (T,): peso por classe (reequilibra as raras sufocadas no regime denso)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal * ce                      # B×T×H×W
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)
    m = mask.unsqueeze(1)                            # B×1×H×W
    return (loss * m).sum() / (m.sum() * targets.shape[1] + 1e-6)


def asymmetric_loss(logits, targets, mask, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                    clip: float = 0.05, class_weight=None, eps: float = 1e-8):
    """Asymmetric Loss (Ben-Baruch 2021) multi-rótulo. Mesmas formas do `focal_bce`.

    Corrige o defeito do focal para cauda longa: lá o `alpha`(=0.25) REDUZ o peso do positivo,
    que é o oposto do necessário quando o positivo já é raríssimo. Aqui os focos são
    DESACOPLADOS — `gamma_pos`=0 não desconta os positivos, `gamma_neg`>0 desconta só os
    negativos fáceis — e o `clip` (probability shifting) descarta de vez os negativos com
    p < clip, que dominam o regime denso e vinham consumindo o gradiente das raras.
    Agregação idêntica à do `focal_bce` para manter a comparação isolada.
    """
    p = torch.sigmoid(logits)
    p_neg = (p - clip).clamp(min=0) if clip > 0 else p
    l_pos = targets * (1 - p).pow(gamma_pos) * torch.log(p.clamp(min=eps))
    l_neg = (1 - targets) * p_neg.pow(gamma_neg) * torch.log((1 - p_neg).clamp(min=eps))
    loss = -(l_pos + l_neg)                          # B×T×H×W
    if class_weight is not None:
        loss = loss * class_weight.view(1, -1, 1, 1)
    m = mask.unsqueeze(1)
    return (loss * m).sum() / (m.sum() * targets.shape[1] + 1e-6)


def masked_mse(pred, targets, dmask):
    """MSE de distância só onde há contato (dmask: B×H×W)."""
    se = (pred - targets) ** 2
    return _masked_mean(se, dmask)


def masked_dist_ce(logits, targets, dmask, edges):
    """CE de distância em BINS (estilo trRosetta), só onde há contato (dmask: B×H×W).

    logits: B×n_bins×H×W; targets: B×H×W (distância em Å); edges: fronteiras internas
    (n_bins-1,) para `bucketize` → índice do bin em [0, n_bins-1]."""
    bins = torch.bucketize(targets, edges)            # B×H×W em [0, n_bins-1]
    ce = F.cross_entropy(logits, bins, reduction="none")  # B×H×W
    return _masked_mean(ce, dmask)


def multitask_loss(out, batch, lambdas, gamma: float = 2.0, alpha: float = 0.25,
                   class_weight=None, dist_edges=None, asl=None):
    """out = (logit_contact, logit_types, logit_dist); batch = saída do ChainCropDataset.

    Se `dist_edges` for dado, a distância é aprendida como classificação em bins
    (trRosetta); senão, cai no MSE legado (head de 1 canal). Se `asl` (dict) for dado, os
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
