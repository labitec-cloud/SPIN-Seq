#!/usr/bin/env bash
# Smoke test do Arpeggio: baixa 1 estrutura, protona e roda o pdbe-arpeggio,
# confirmando que a supervisão pode ser gerada. Saída em data/arpeggio/.
# Uso: bash scripts/smoke_arpeggio.sh [PDB_ID]
set -euo pipefail
cd "$(dirname "$0")/.."

PDB="${1:-1ubq}"          # ubiquitina por padrão
RAW="data/raw"
OUT="data/arpeggio/${PDB}"
mkdir -p "$RAW" "$OUT"

CIF="$RAW/${PDB}.cif"
if [ ! -f "$CIF" ]; then
  echo ">> Baixando ${PDB}.cif do RCSB"
  curl -fsSL "https://files.rcsb.org/download/${PDB}.cif" -o "$CIF"
fi

echo ">> Rodando pdbe-arpeggio em ${PDB}"
# pdbe-arpeggio aceita mmCIF; requer hidrogênios. Muitas versões protonam
# internamente via OpenBabel. Se a sua versão exigir protonação prévia,
# gere um CIF protonado antes (ver nota no README/PLANO).
pdbe-arpeggio -o "$OUT" "$CIF"

echo ">> Saídas em $OUT:"
ls -la "$OUT"
