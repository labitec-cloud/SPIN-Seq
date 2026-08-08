"""Fase 2 — cacheia embeddings ESM-2 para as cadeias já rotuladas (Fase 1).

Percorre data/labels/*.npz, extrai a sequência de cada cadeia e salva as features
do ESM-2 em data/embeddings/<mesmo_nome>.npz. O nome casa 1:1 com o rótulo, o que
garante alinhamento resíduo-a-resíduo entre features e supervisão.

Uso:
    python scripts/build_embeddings.py                 # todas as cadeias rotuladas
    python scripts/build_embeddings.py --attn          # inclui mapas de atenção
    python scripts/build_embeddings.py --glob '1ubq_*' # subconjunto
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.esm_embeddings import ESMExtractor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--glob", default="*", help="padrão dos rótulos em data/labels")
    ap.add_argument("--attn", action="store_true", help="salvar mapas de atenção (pesado)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    labels_dir = cfg["paths"]["labels"]
    emb_dir = cfg["paths"]["embeddings"]
    os.makedirs(emb_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(labels_dir, f"{args.glob}.npz")))
    if not label_files:
        ap.error(f"nenhum rótulo em {labels_dir} com padrão {args.glob}")

    print(f">> Carregando ESM-2 ({cfg['esm']['model']})...")
    ext = ESMExtractor(cfg)
    print(f">> device={ext.device} fp16={ext.fp16} embed_dim={ext.embed_dim}")

    done = skipped = failed = 0
    for lf in label_files:
        name = os.path.splitext(os.path.basename(lf))[0]
        out = os.path.join(emb_dir, f"{name}.npz")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        seq = str(np.load(lf, allow_pickle=True)["seq"])
        try:
            r = ext.extract(seq, out_path=out, want_attn=args.attn, want_contacts=True)
            done += 1
            extra = f" attn={r['attn'].shape}" if args.attn else ""
            print(f"  {name}: emb={r['emb'].shape}{extra}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {name}: ERRO {exc}")

    print(f"\n>> feito={done} pulados={skipped} falhas={failed}")


if __name__ == "__main__":
    main()
