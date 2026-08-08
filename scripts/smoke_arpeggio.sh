#!/usr/bin/env bash
# Arpeggio smoke test: downloads 1 structure, protonates it and runs pdbe-arpeggio,
# confirming that the supervision can be generated. Output in data/arpeggio/.
# Usage: bash scripts/smoke_arpeggio.sh [PDB_ID]
set -euo pipefail
cd "$(dirname "$0")/.."

PDB="${1:-1ubq}"          # ubiquitin by default
RAW="data/raw"
OUT="data/arpeggio/${PDB}"
mkdir -p "$RAW" "$OUT"

CIF="$RAW/${PDB}.cif"
if [ ! -f "$CIF" ]; then
  echo ">> Downloading ${PDB}.cif from the RCSB"
  curl -fsSL "https://files.rcsb.org/download/${PDB}.cif" -o "$CIF"
fi

echo ">> Running pdbe-arpeggio on ${PDB}"
# pdbe-arpeggio accepts mmCIF; it requires hydrogens. Many versions protonate
# internally via OpenBabel. If your version requires prior protonation,
# generate a protonated CIF beforehand (see the note in the README).
pdbe-arpeggio -o "$OUT" "$CIF"

echo ">> Outputs in $OUT:"
ls -la "$OUT"
