"""Cabeça 2D convolucional (estilo trRosetta) para o RIN multimodal — Fase 4.

Ao contrário do PairMLP (decide cada par isolado), aqui as features de par formam um mapa
L×L e passam por uma ResNet 2D dilatada, dando a cada saída (i,j) um campo receptivo grande
sobre a vizinhança da matriz — necessário para as classes que saturaram no baseline.

Projeta o embedding do ESM-2 (640) para uma dimensão pequena `proj_dim` ANTES de formar os
pares (controle de VRAM na GTX 1650), monta canais simétricos por par, aplica os blocos
residuais dilatados e simetriza a saída. Três cabeças 1×1: contato, tipos, distância.

Suporta crop retangular: `forward` recebe embeddings de linha e de coluna separados, então um
crop [a:a+wr] × [b:b+wc] é montado passando emb[a:a+wr] e emb[b:b+wc].
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
        # `use_ss` liga a CABEÇA AUXILIAR de SS (SS só como ALVO no treino) — limpo, não exige
        # estrutura na inferência. `ss_pair` liga o CANAL DE PAR mesma-hélice/folha, que usa SS
        # como ENTRADA — isso requer DSSP da estrutura 3D e VIOLA a premissa "só sequência".
        # Manter ss_pair=True apenas para a ablação de TETO (SS-oráculo).
        self.use_ss = use_ss
        self.ss_pair = use_ss if ss_pair is None else bool(ss_pair)
        self.proj = nn.Linear(emb_dim, proj_dim)
        c_in = 3 * proj_dim + 1 + (1 if use_esm_contact else 0)  # sum|absdiff|prod + sep + contato
        # §9.8 passo 3: identidade de AA por par (D/E↔K/R/H p/ ionic, F/W/Y p/ aromatic) injetada
        # SEM passar pela projeção 1280→32 que apaga esse sinal. 21 = 20 AAs + desconhecido/pad.
        if aa_dim > 0:
            self.aa_emb = nn.Embedding(21, aa_dim)
            c_in += 3 * aa_dim
        # §1.2: SS de backbone. Canal de par "mesma hélice"/"mesma folha" (hbond/carbonyl são
        # backbone-dependentes) + cabeça auxiliar de SS na projeção (pressiona a proj a reter SS).
        if self.ss_pair:
            c_in += 2                              # canal de par (ENTRADA — só p/ ablação-teto)
        if use_ss:
            self.ss_head = nn.Linear(proj_dim, 3)  # coil/hélice/folha por resíduo (treino-only)
        self.stem = nn.Conv2d(c_in, channels, 1)
        self.blocks = nn.ModuleList(
            [ResBlock2D(channels, dilations[k % len(dilations)], dropout) for k in range(n_blocks)]
        )
        self.head_contact = nn.Conv2d(channels, 1, 1)
        self.head_types = nn.Conv2d(channels, n_types, 1)
        self.head_dist = nn.Conv2d(channels, n_dist_bins, 1)  # bins (trRosetta) ou 1 (MSE legado)

    def _pair_sym(self, r, c):
        """Features de par simétricas [r+c, |r-c|, r*c] → B×3d×wr×wc a partir de B×w×d."""
        a = r.unsqueeze(2)                          # B×wr×1×d
        b = c.unsqueeze(1)                          # B×1×wc×d
        return torch.cat([a + b, (a - b).abs(), a * b], dim=-1).permute(0, 3, 1, 2)

    def build_map(self, emb_row, emb_col, sep, contact=None, aa_row=None, aa_col=None,
                  ss_row=None, ss_col=None):
        """Monta o mapa de pares B×C_in×wr×wc a partir dos embeddings de linha e coluna.

        emb_row: B×wr×emb_dim   emb_col: B×wc×emb_dim
        sep:     B×wr×wc  (log1p|i-j|)   contact: B×wr×wc opcional (score de contato ESM)
        aa_row/aa_col: B×wr / B×wc índices de AA (long), usados só se aa_dim>0.
        ss_row/ss_col: B×wr / B×wc índices de SS (0=coil,1=H,2=E), usados só se use_ss.
        """
        m = self._pair_sym(self.proj(emb_row), self.proj(emb_col))  # B×3d×wr×wc
        extra = [sep.unsqueeze(1)]
        if self.use_esm_contact:
            extra.append((contact if contact is not None else torch.zeros_like(sep)).unsqueeze(1))
        if self.aa_dim > 0 and aa_row is not None:
            extra.append(self._pair_sym(self.aa_emb(aa_row), self.aa_emb(aa_col)))
        if self.ss_pair:
            if ss_row is not None:
                r, c = ss_row.unsqueeze(2), ss_col.unsqueeze(1)  # B×wr×1, B×1×wc
                same_h = ((r == 1) & (c == 1)).to(sep.dtype)     # ambos hélice
                same_e = ((r == 2) & (c == 2)).to(sep.dtype)     # ambos folha
                extra.append(same_h.unsqueeze(1)); extra.append(same_e.unsqueeze(1))
            else:
                z = torch.zeros_like(sep)
                extra.append(z.unsqueeze(1)); extra.append(z.unsqueeze(1))
        return torch.cat([m] + extra, dim=1)       # B×C_in×wr×wc

    def ss_predict(self, emb_row):
        """Logits de SS por resíduo (B×w×3) a partir da projeção — perda auxiliar, treino-only."""
        return self.ss_head(self.proj(emb_row))

    def forward(self, emb_row, emb_col, sep, contact=None, symmetrize=False,
                aa_row=None, aa_col=None, ss_row=None, ss_col=None):
        x = self.stem(self.build_map(emb_row, emb_col, sep, contact, aa_row, aa_col,
                                     ss_row, ss_col))
        for blk in self.blocks:
            x = blk(x)
        if symmetrize:  # só válido para crop quadrado alinhado (wr==wc, mesmo intervalo)
            x = 0.5 * (x + x.transpose(-1, -2))
        return (
            self.head_contact(x).squeeze(1),   # B×wr×wc
            self.head_types(x),                # B×T×wr×wc
            self.head_dist(x),                 # B×n_bins×wr×wc (logits dos bins)
        )
