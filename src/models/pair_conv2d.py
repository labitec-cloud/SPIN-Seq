"""2D convolutional head (trRosetta style) for the multimodal RIN.

Unlike PairMLP (which decides each pair in isolation), here the pair features form an
L x L map and pass through a dilated 2D ResNet, giving each output (i,j) a large receptive
field over the matrix neighbourhood - needed for the classes that saturated in the baseline.

Projects the ESM-2 embedding (640) down to a small `proj_dim` BEFORE forming the pairs
(VRAM control on the GTX 1650), builds symmetric per-pair channels, applies the dilated
residual blocks and symmetrises the output. Three 1x1 heads: contact, types, distance.

Supports rectangular crops: `forward` takes row and column embeddings separately, so a crop
[a:a+wr] x [b:b+wc] is built by passing emb[a:a+wr] and emb[b:b+wc].
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock2D(nn.Module):
    def __init__(self, c: int, dilation: int, dropout: float):
        super().__init__()
        p = dilation
        self.conv1 = nn.Conv2d(c, c, 3, padding=p, dilation=dilation)
        self.norm1 = nn.InstanceNorm2d(c, affine=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=p, dilation=dilation)
        self.norm2 = nn.InstanceNorm2d(c, affine=True)
        self.drop = nn.Dropout2d(dropout)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class PairConv2D(nn.Module):
    def __init__(self, emb_dim: int, n_types: int, proj_dim: int = 32,
                 channels: int = 64, n_blocks: int = 12,
                 dilations: tuple[int, ...] = (1, 2, 4, 8),
                 dropout: float = 0.15, use_esm_contact: bool = True,
                 n_dist_bins: int = 1, aa_dim: int = 0, use_ss: bool = False,
                 ss_pair: bool | None = None):
        super().__init__()
        self.use_esm_contact = use_esm_contact
        self.aa_dim = aa_dim
        # `use_ss` enables the AUXILIARY SS HEAD (SS only as a TARGET during training) - clean, it
        # does not require structure at inference. `ss_pair` enables the same-helix/strand PAIR
        # CHANNEL, which uses SS as an INPUT - that requires DSSP from the 3D structure and VIOLATES
        # the sequence-only premise. Keep ss_pair=True only for the CEILING ablation (SS-oracle).
        self.use_ss = use_ss
        self.ss_pair = use_ss if ss_pair is None else bool(ss_pair)
        self.proj = nn.Linear(emb_dim, proj_dim)
        c_in = 3 * proj_dim + 1 + (1 if use_esm_contact else 0)  # sum|absdiff|prod + sep + contato
        # Ladder step 3: per-pair AA identity (D/E<->K/R/H for ionic, F/W/Y for aromatic) injected
        # WITHOUT going through the 1280->32 projection that erases that signal. 21 = 20 AAs + unk/pad.
        if aa_dim > 0:
            self.aa_emb = nn.Embedding(21, aa_dim)
            c_in += 3 * aa_dim
        # Backbone SS. A "same helix"/"same strand" pair channel (hbond/carbonyl are
        # backbone-dependent) + an auxiliary SS head on the projection (pressures it to retain SS).
        if self.ss_pair:
            c_in += 2                              # pair channel (INPUT - ceiling ablation only)
        if use_ss:
            self.ss_head = nn.Linear(proj_dim, 3)  # coil/helix/strand per residue (training only)
        self.stem = nn.Conv2d(c_in, channels, 1)
        self.blocks = nn.ModuleList(
            [ResBlock2D(channels, dilations[k % len(dilations)], dropout) for k in range(n_blocks)]
        )
        self.head_contact = nn.Conv2d(channels, 1, 1)
        self.head_types = nn.Conv2d(channels, n_types, 1)
        self.head_dist = nn.Conv2d(channels, n_dist_bins, 1)  # bins (trRosetta) ou 1 (MSE legado)

    def _pair_sym(self, r, c):
        """Symmetric pair features [r+c, |r-c|, r*c] -> B x 3d x wr x wc from B x w x d."""
        a = r.unsqueeze(2)                          # B x wr x 1 x d
        b = c.unsqueeze(1)                          # B x 1 x wc x d
        return torch.cat([a + b, (a - b).abs(), a * b], dim=-1).permute(0, 3, 1, 2)

    def build_map(self, emb_row, emb_col, sep, contact=None, aa_row=None, aa_col=None,
                  ss_row=None, ss_col=None):
        """Builds the pair map B x C_in x wr x wc from the row and column embeddings.

        emb_row: B×wr×emb_dim   emb_col: B×wc×emb_dim
        sep:     B×wr×wc  (log1p|i-j|)   contact: B×wr×wc opcional (score de contato ESM)
        aa_row/aa_col: B x wr / B x wc AA indices (long), used only if aa_dim>0.
        ss_row/ss_col: B x wr / B x wc SS indices (0=coil,1=H,2=E), used only if use_ss.
        """
        m = self._pair_sym(self.proj(emb_row), self.proj(emb_col))  # B x 3d x wr x wc
        extra = [sep.unsqueeze(1)]
        if self.use_esm_contact:
            extra.append((contact if contact is not None else torch.zeros_like(sep)).unsqueeze(1))
        if self.aa_dim > 0 and aa_row is not None:
            extra.append(self._pair_sym(self.aa_emb(aa_row), self.aa_emb(aa_col)))
        if self.ss_pair:
            if ss_row is not None:
                r, c = ss_row.unsqueeze(2), ss_col.unsqueeze(1)  # B x wr x 1, B x 1 x wc
                same_h = ((r == 1) & (c == 1)).to(sep.dtype)     # both helix
                same_e = ((r == 2) & (c == 2)).to(sep.dtype)     # ambos folha
                extra.append(same_h.unsqueeze(1)); extra.append(same_e.unsqueeze(1))
            else:
                z = torch.zeros_like(sep)
                extra.append(z.unsqueeze(1)); extra.append(z.unsqueeze(1))
        return torch.cat([m] + extra, dim=1)       # B x C_in x wr x wc

    def ss_predict(self, emb_row):
        """Per-residue SS logits (B x w x 3) from the projection - auxiliary loss, training only."""
        return self.ss_head(self.proj(emb_row))

    def forward(self, emb_row, emb_col, sep, contact=None, symmetrize=False,
                aa_row=None, aa_col=None, ss_row=None, ss_col=None):
        x = self.stem(self.build_map(emb_row, emb_col, sep, contact, aa_row, aa_col,
                                     ss_row, ss_col))
        for blk in self.blocks:
            x = blk(x)
        if symmetrize:  # valid only for an aligned square crop (wr==wc, same interval)
            x = 0.5 * (x + x.transpose(-1, -2))
        return (
            self.head_contact(x).squeeze(1),   # B x wr x wc
            self.head_types(x),                # B x T x wr x wc
            self.head_dist(x),                 # B x n_bins x wr x wc (logits dos bins)
        )
