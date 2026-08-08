"""ESM-2 smoke test: extracts per-residue embeddings of 1 sequence and the
de contatos prevista, confirmando que o modelo roda na GPU (com fp16 se pedido).

Uso:
    python scripts/smoke_esm.py
    python scripts/smoke_esm.py --model esm2_t12_35M_UR50D --device cpu
"""
import argparse

import torch
import esm


# Short example sequence (ubiquitin, 76 residues) - light for 4 GB of VRAM.
SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f">> Carregando {args.model} em {device} (fp16={args.fp16 and device=='cuda'})")

    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model = model.eval().to(device)
    if args.fp16 and device == "cuda":
        model = model.half()

    bc = alphabet.get_batch_converter()
    _, _, tokens = bc([("prot", SEQ)])
    tokens = tokens.to(device)

    repr_layer = model.num_layers
    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer], return_contacts=True)

    emb = out["representations"][repr_layer][0, 1 : len(SEQ) + 1]  # (L, d)
    contacts = out["contacts"][0]  # (L, L)

    print(f">> OK  | L={len(SEQ)}  embeddings={tuple(emb.shape)}  contatos={tuple(contacts.shape)}")
    if device == "cuda":
        print(f">> peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
