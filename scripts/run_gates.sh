#!/usr/bin/env bash
# Portão denso (teste) nos checkpoints das seeds 43 e 44.
# As seeds 42 já têm gate: outputs/gate_passo3.txt e outputs/gate_ssaux.txt.
#
# Reentrante: pula gate cuja saída já tenha a linha de macro.
#
#   bash scripts/run_gates.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python

gate() {  # gate <config> <ckpt_dir> <saida>
    local cfg="$1" dir="$2" out="$3"
    if [[ -f "$out" ]] && grep -qi 'macro' "$out"; then
        echo ">>> PULANDO $out (já existe)"
        return 0
    fi
    echo ">>> $(date +%H:%M) portão de $dir -> $out"
    $PY -u src/eval_dense.py --config "$cfg" --model conv2d \
        --ckpt "$dir/best.pt" --split test > "$out" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! FALHOU $dir (rc=$rc) — ver $out"
        return $rc
    fi
    echo ">>> $(date +%H:%M) ok: $(grep -i macro "$out" | tail -1)"
}

AA=configs/esm650m_aa.yaml
SS=configs/esm650m_aa_ssaux.yaml

gate $AA outputs/conv2d_650m_aa_s43 outputs/gate_passo3_s43.txt || exit 1
gate $AA outputs/conv2d_650m_aa_s44 outputs/gate_passo3_s44.txt || exit 1
gate $SS outputs/conv2d_ssaux_s43   outputs/gate_ssaux_s43.txt  || exit 1
gate $SS outputs/conv2d_ssaux_s44   outputs/gate_ssaux_s44.txt  || exit 1

echo ">>> $(date +%H:%M) TODOS OS PORTÕES CONCLUÍDOS"
