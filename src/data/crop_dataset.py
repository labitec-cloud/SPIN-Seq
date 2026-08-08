"""2D CROP dataset for the convolutional head.

PairConv2D consumes the L x L matrix, which does not fit in VRAM for large chains. Here each
item is a square `w x w` window of a chain: row/column embedding sub-tensors and the dense
targets (contact, types, distance) cut from that window out of the SPARSE labels
(`idx_i/idx_j/labels`, positive pairs with i<j only).

A fixed size `w` allows batch stacking: chains with L<w are padded and the padded region is
zeroed in the loss mask. The window can be centred on a positive pair (with probability
`pos_center_prob`) so as not to waste steps on empty crops.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.pair_dataset import AA_UNK, EXCLUDED_TYPES, seq_to_idx


class ChainCropDataset(Dataset):
    def __init__(self, cfg: dict, files: list[str], rng: np.random.Generator):
        self.rng = rng
        self.seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
        ccfg = cfg.get("conv2d", {})
        self.w = int(ccfg.get("crop_size", 64))
        self.crops_per_chain = int(ccfg.get("crops_per_chain", 4))
        self.pos_center_prob = float(ccfg.get("pos_center_prob", 0.7))
        self.rare_center_prob = float(ccfg.get("rare_center_prob", 0.0))
        self._rare_types = set(ccfg.get("rare_types", []))
        self.use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
        emb_dir = cfg["paths"]["embeddings"]
        self.ss_dir = cfg["paths"].get("ss", "data/ss")  # §1.2: SS de backbone (0=coil,1=H,2=E)

        self.pairs: list[tuple[str, str]] = []
        for lf in files:
            name = os.path.splitext(os.path.basename(lf))[0]
            ef = os.path.join(emb_dir, f"{name}.npz")
            if os.path.exists(ef):
                self.pairs.append((ef, lf))
        if not self.pairs:
            raise RuntimeError("no chain with both embedding and label was found")

        # tipos preditos (ordem/nomes) a partir da 1ª cadeia
        l0 = np.load(self.pairs[0][1], allow_pickle=True)
        types = [str(t) for t in l0["types"]]
        self.type_channels = [k for k, t in enumerate(types) if t not in EXCLUDED_TYPES]
        self.type_names = [types[k] for k in self.type_channels]
        self.rare_cols = [k for k, n in enumerate(self.type_names) if n in self._rare_types]

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def __len__(self):
        return len(self.pairs) * self.crops_per_chain

    def _load(self, k):
        ef, lf = self.pairs[k]
        e = np.load(ef, allow_pickle=True)
        l = np.load(lf, allow_pickle=True)
        emb = e["emb"].astype(np.float32)
        L = int(l["length"])
        # some ESM contact maps contain NaN/inf (16 chains) -> zero them so they do not propagate
        contacts = (np.nan_to_num(e["contacts"].astype(np.float32))
                    if self.use_esm and "contacts" in e else None)
        idx_i = l["idx_i"].astype(np.int64)
        idx_j = l["idx_j"].astype(np.int64)
        labels = l["labels"][:, self.type_channels].astype(np.float32)
        dist = l["min_dist"].astype(np.float32)
        aa = seq_to_idx(str(l["seq"]))[:L] if "seq" in l else np.full(L, AA_UNK, np.int64)
        name = os.path.splitext(os.path.basename(self.pairs[k][1]))[0]
        sf = os.path.join(self.ss_dir, f"{name}.npz")
        ss = (np.load(sf)["ss"].astype(np.int64)[:L] if os.path.exists(sf)
              else np.full(L, 3, np.int64))  # 3 = pad/desconhecido
        return emb, L, contacts, idx_i, idx_j, labels, dist, aa, ss

    def _pick_window(self, L, idx_i, idx_j, rare_mask=None):
        """Chooses (a,b) = row/column start of the w x w window."""
        w = self.w
        if L <= w:
            return 0, 0, L  # the whole chain, no slack to shift
        if len(idx_i) > 0 and self.rng.random() < self.pos_center_prob:
            # with rare_center_prob, centre on a rare-class positive (if any)
            if (rare_mask is not None and rare_mask.any()
                    and self.rng.random() < self.rare_center_prob):
                cand = np.where(rare_mask)[0]
            else:
                cand = np.arange(len(idx_i))
            p = int(cand[self.rng.integers(0, len(cand))])
            ci, cj = int(idx_i[p]), int(idx_j[p])
            a = int(np.clip(ci - w // 2, 0, L - w))
            b = int(np.clip(cj - w // 2, 0, L - w))
        else:
            a = int(self.rng.integers(0, L - w + 1))
            b = int(self.rng.integers(0, L - w + 1))
        return a, b, w

    def __getitem__(self, idx):
        emb, L, contacts, idx_i, idx_j, labels, dist, aa, ss = self._load(idx % len(self.pairs))
        w = self.w
        rare_mask = (labels[:, self.rare_cols].any(1) if self.rare_cols else None)
        a, b, wa = self._pick_window(L, idx_i, idx_j, rare_mask)

        er = np.zeros((w, emb.shape[1]), np.float32)
        ec = np.zeros((w, emb.shape[1]), np.float32)
        er[:wa] = emb[a:a + wa]
        ec[:wa] = emb[b:b + wa]

        ar = np.full(w, AA_UNK, np.int64)  # AA da linha/coluna; padding = desconhecido
        ac = np.full(w, AA_UNK, np.int64)
        ar[:wa] = aa[a:a + wa]
        ac[:wa] = aa[b:b + wa]

        sr = np.full(w, 3, np.int64)  # SS da linha/coluna; padding = 3 (desconhecido)
        sc = np.full(w, 3, np.int64)
        sr[:wa] = ss[a:a + wa]
        sc[:wa] = ss[b:b + wa]

        gi = np.arange(a, a + wa)  # global indices of the valid rows
        gj = np.arange(b, b + wa)  # and of the columns
        sep = np.zeros((w, w), np.float32)
        sep[:wa, :wa] = np.log1p(np.abs(gi[:, None] - gj[None, :]))

        cfeat = np.zeros((w, w), np.float32)
        if contacts is not None:
            cfeat[:wa, :wa] = contacts[a:a + wa][:, b:b + wa]

        yc = np.zeros((w, w), np.float32)
        yt = np.zeros((self.n_types, w, w), np.float32)
        yd = np.zeros((w, w), np.float32)
        dm = np.zeros((w, w), np.float32)

        # loss mask: valid in the non-padded region and with |gi-gj| >= seq_sep
        lm = np.zeros((w, w), np.float32)
        valid = np.abs(gi[:, None] - gj[None, :]) >= self.seq_sep
        lm[:wa, :wa] = valid.astype(np.float32)

        # densify the positives in both orientations (the label stores only i<j):
        # (row=i,col=j) and the mirror (row=j,col=i) - the matrix is symmetric
        for ri, ci in ((idx_i, idx_j), (idx_j, idx_i)):
            mr = (ri >= a) & (ri < a + wa)
            mc = (ci >= b) & (ci < b + wa)
            sel = mr & mc
            rr = ri[sel] - a
            cc = ci[sel] - b
            yc[rr, cc] = 1.0
            yt[:, rr, cc] = labels[sel].T
            yd[rr, cc] = dist[sel]
            dm[rr, cc] = 1.0

        return (
            torch.from_numpy(er), torch.from_numpy(ec),
            torch.from_numpy(sep), torch.from_numpy(cfeat),
            torch.from_numpy(yc), torch.from_numpy(yt),
            torch.from_numpy(yd), torch.from_numpy(dm), torch.from_numpy(lm),
            torch.from_numpy(ar), torch.from_numpy(ac),
            torch.from_numpy(sr), torch.from_numpy(sc),
        )
