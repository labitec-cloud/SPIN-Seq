"""Extraction and caching of ESM-2 features (the model is FROZEN).

ESM-2 is used only as an extractor: there is no training/fine-tuning here. For each
sequence we generate and cache on disk:
  - `emb`       : per-residue embeddings (L, d) from the chosen layer;
  - `attn`      : (optional) symmetric per-pair attention map (L, L, heads*layers)
                  - a 2D signal useful for the contact/type head;
  - `contacts`  : (optional) contact map predicted by ESM-2 itself (L, L).

The cache avoids reprocessing the pLM (the bottleneck) on every training epoch.

Uso:
    from src.features.esm_embeddings import ESMExtractor
    ext = ESMExtractor(cfg)
    ext.extract("MQIFV...", out_path="data/embeddings/1ubq_A.npz")
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ESMExtractor:
    cfg: dict

    def __post_init__(self) -> None:
        import esm

        ecfg = self.cfg["esm"]
        self.model_name: str = ecfg["model"]
        want = ecfg.get("device", "cuda")
        self.device = want if (want == "cpu" or torch.cuda.is_available()) else "cpu"
        self.fp16: bool = bool(ecfg.get("fp16", True)) and self.device == "cuda"
        self.max_len: int = int(ecfg.get("max_len", 1022))

        model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.model = model.eval().to(self.device)
        if self.fp16:
            self.model = self.model.half()
        for p in self.model.parameters():  # congelado
            p.requires_grad_(False)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        # representation layer: default = last
        self.repr_layer: int = int(ecfg.get("repr_layer", self.model.num_layers))

    @torch.no_grad()
    def extract(
        self,
        seq: str,
        out_path: str | None = None,
        want_attn: bool = False,
        want_contacts: bool = True,
    ) -> dict[str, np.ndarray]:
        if len(seq) > self.max_len:
            raise ValueError(f"seq len {len(seq)} > max_len {self.max_len}")

        _, _, tokens = self.batch_converter([("prot", seq)])
        tokens = tokens.to(self.device)

        out = self.model(
            tokens,
            repr_layers=[self.repr_layer],
            need_head_weights=want_attn,
            return_contacts=want_contacts,
        )

        L = len(seq)
        # drop the BOS/EOS tokens (positions 0 and L+1)
        emb = out["representations"][self.repr_layer][0, 1 : L + 1].float().cpu().numpy()
        result: dict[str, np.ndarray] = {"emb": emb.astype(np.float32), "seq": np.array(seq)}

        if want_contacts:
            result["contacts"] = out["contacts"][0].float().cpu().numpy().astype(np.float32)

        if want_attn:
            # attentions: (layers, batch, heads, L+2, L+2) -> (L, L, layers*heads)
            attn = out["attentions"][:, 0, :, 1 : L + 1, 1 : L + 1]
            attn = attn.permute(2, 3, 0, 1).reshape(L, L, -1)
            attn = 0.5 * (attn + attn.transpose(0, 1))  # simetriza
            result["attn"] = attn.float().cpu().numpy().astype(np.float16)

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, **result)
        return result

    @property
    def embed_dim(self) -> int:
        return int(self.model.embed_dim)
