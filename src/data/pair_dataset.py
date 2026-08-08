from __future__ import annotations

import csv
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# types outside the vocabulary: 'proximal' is a fallback for proximity with no type;
# 'xbond'/'metal' have 0 training examples -> unlearnable.
EXCLUDED_TYPES = {"proximal", "xbond", "metal"}

# Alphabet of 20 AAs; index 20 = unknown/padding (X, gaps, non-standard residues).
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}
AA_UNK = len(AA_ALPHABET)  # 20


def seq_to_idx(seq: str) -> np.ndarray:
    """Sequence -> int64 vector of AA_ALPHABET indices (non-standard -> AA_UNK)."""
    return np.array([AA_TO_IDX.get(c, AA_UNK) for c in seq], dtype=np.int64)


def symmetric_pair_feat(hi: np.ndarray, hj: np.ndarray) -> np.ndarray:
    return np.concatenate([hi + hj, np.abs(hi - hj), hi * hj], axis=-1)


class ChainSample:
    def __init__(self, emb_path: str, label_path: str):
        e = np.load(emb_path, allow_pickle=True)
        l = np.load(label_path, allow_pickle=True)
        self.emb = e["emb"].astype(np.float32)
        self.contacts = (np.nan_to_num(e["contacts"].astype(np.float32))
                         if "contacts" in e else None)
        self.L = int(l["length"])
        self.idx_i = l["idx_i"].astype(np.int64)
        self.idx_j = l["idx_j"].astype(np.int64)
        self.labels = l["labels"].astype(np.float32)
        self.min_dist = l["min_dist"].astype(np.float32)
        self.types = [str(t) for t in l["types"]]
        assert self.emb.shape[0] == self.L
        self.prox_idx = self.types.index("proximal")
        # type channels = everything except the excluded ones (proximal + classes with 0 examples)
        self.type_channels = [k for k, t in enumerate(self.types) if t not in EXCLUDED_TYPES]
        self.type_names = [self.types[k] for k in self.type_channels]
        self.pos_set = set(zip(self.idx_i.tolist(), self.idx_j.tolist()))


class ChainPairDataset(Dataset):
    """Lazy dataset: one item = one whole chain (positives + sampled negatives).

    The pair features are generated on the fly in __getitem__ and are NEVER materialised
    in bulk, which keeps RAM low even with thousands of chains. The negatives are
    resampled on every access (a new epoch = new negatives).
    """

    def __init__(self, cfg: dict, files: list[str], rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.seq_sep = int(cfg["arpeggio"]["seq_sep_min"])
        self.neg_ratio = int(cfg["train"]["neg_ratio"])
        self.use_esm = bool(cfg["train"].get("use_esm_contact_feat", True))
        emb_dir = cfg["paths"]["embeddings"]
        label_dir = cfg["paths"]["labels"]

        self.pairs: list[tuple[str, str]] = []  # (emb_path, label_path)
        for lf in files:
            name = os.path.splitext(os.path.basename(lf))[0]
            ef = os.path.join(emb_dir, f"{name}.npz")
            if os.path.exists(ef):
                self.pairs.append((ef, lf))
        if not self.pairs:
            raise RuntimeError("no chain with both embedding and label was found")

        # metadados (feat_dim, tipos) a partir da 1ª cadeia
        s0 = ChainSample(*self.pairs[0])
        self.type_names = s0.type_names
        self._feat_dim = self._feat(s0, s0.idx_i[0], s0.idx_j[0]).shape[0]

    def _feat(self, s: ChainSample, i: int, j: int) -> np.ndarray:
        z = symmetric_pair_feat(s.emb[i], s.emb[j])
        extra = [np.log1p(abs(int(i) - int(j)))]
        if self.use_esm and s.contacts is not None:
            extra.append(float(s.contacts[i, j]))
        return np.concatenate([z, np.array(extra, np.float32)])

    def _sample_negatives(self, s: ChainSample, n_neg: int):
        out, tries, target = [], 0, min(n_neg, s.L * s.L)
        while len(out) < target and tries < target * 20:
            tries += 1
            i = int(self.rng.integers(0, s.L))
            j = int(self.rng.integers(0, s.L))
            if i > j:
                i, j = j, i
            if j - i < self.seq_sep or (i, j) in s.pos_set:
                continue
            out.append((i, j))
        return out

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def n_types(self) -> int:
        return len(self.type_names)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        s = ChainSample(*self.pairs[k])
        tc = s.type_channels
        feats, yc, yt, yd, dm = [], [], [], [], []
        for p in range(len(s.idx_i)):
            i, j = int(s.idx_i[p]), int(s.idx_j[p])
            feats.append(self._feat(s, i, j))
            yc.append(1.0)
            yt.append(s.labels[p, tc])
            yd.append(s.min_dist[p])
            dm.append(1.0)
        for i, j in self._sample_negatives(s, self.neg_ratio * len(s.idx_i)):
            feats.append(self._feat(s, i, j))
            yc.append(0.0)
            yt.append(np.zeros(len(tc), np.float32))
            yd.append(0.0)
            dm.append(0.0)
        return (
            torch.tensor(np.stack(feats), dtype=torch.float32),
            torch.tensor(yc, dtype=torch.float32),
            torch.tensor(np.stack(yt), dtype=torch.float32),
            torch.tensor(yd, dtype=torch.float32),
            torch.tensor(dm, dtype=torch.float32),
        )


def collate_chains(batch):
    """Concatenates the pairs of several chains into a single mini-batch."""
    xs, ycs, yts, yds, dms = zip(*batch)
    return (
        torch.cat(xs), torch.cat(ycs), torch.cat(yts),
        torch.cat(yds), torch.cat(dms),
    )


def compute_pos_weights(files: list[str], cfg: dict) -> torch.Tensor:
    """Per-type pos_weight from a cheap sweep over the labels (labels only)."""
    neg_ratio = int(cfg["train"]["neg_ratio"])
    label_dir = cfg["paths"]["labels"]
    emb_dir = cfg["paths"]["embeddings"]
    pos_per_type = None
    n_pos = 0
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        if not os.path.exists(os.path.join(emb_dir, f"{name}.npz")):
            continue
        l = np.load(lf, allow_pickle=True)
        types = list(l["types"])
        tc = [k for k, t in enumerate(types) if t not in EXCLUDED_TYPES]
        labels = l["labels"].astype(np.float32)
        if labels.size == 0:
            continue
        if pos_per_type is None:
            pos_per_type = np.zeros(len(tc), np.float64)
        pos_per_type += labels[:, tc].sum(0)
        n_pos += labels.shape[0]
    total = n_pos * (1 + neg_ratio)
    neg = total - pos_per_type
    w = np.clip(neg / np.clip(pos_per_type, 1, None), None, 100.0)
    return torch.tensor(w, dtype=torch.float32)


def compute_class_weights(files: list[str], cfg: dict, cap: float = 3.0) -> torch.Tensor:
    """Per-class weight: inverse square root of frequency, normalised by the MOST abundant class
    (=1) and capped at `cap`. Gives ~cap:1 to the rare classes without letting the rarest one
    (covalent) dominate. Order = type_channels (identical to compute_pos_weights)."""
    emb_dir = cfg["paths"]["embeddings"]
    pos = None
    for lf in files:
        name = os.path.splitext(os.path.basename(lf))[0]
        if not os.path.exists(os.path.join(emb_dir, f"{name}.npz")):
            continue
        l = np.load(lf, allow_pickle=True)
        types = list(l["types"])
        tc = [k for k, t in enumerate(types) if t not in EXCLUDED_TYPES]
        labels = l["labels"].astype(np.float32)
        if labels.size == 0:
            continue
        if pos is None:
            pos = np.zeros(len(tc), np.float64)
        pos += labels[:, tc].sum(0)
    freq = pos / max(pos.sum(), 1.0)
    w = 1.0 / np.sqrt(np.clip(freq, 1e-8, None))
    w = w / w.min()
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


def files_from_manifest(cfg: dict, split: str) -> list[str]:
    """Lists the labels of a split (train/val/test) according to data/manifest.csv."""
    manifest = cfg["dataset"]["manifest"]
    label_dir = cfg["paths"]["labels"]
    out = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                out.append(os.path.join(label_dir, f"{r['name']}.npz"))
    return sorted(p for p in out if os.path.exists(p))


def list_label_files(cfg: dict) -> list[str]:
    return sorted(glob.glob(os.path.join(cfg["paths"]["labels"], "*.npz")))
