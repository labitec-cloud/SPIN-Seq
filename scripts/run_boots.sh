#!/usr/bin/env bash
# Bootstrap pareado SS-aux (b) vs passo 3 (a), uma vez por semente.
# Delta positivo = SS-aux melhor. Reentrante: pula saída que já tenha o bloco de IC.
#
#   bash scripts/run_boots.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
AA=configs/esm650m_aa.yaml
SS=configs/esm650m_aa_ssaux.yaml

boot() {  # boot <ckpt_a> <ckpt_b> <saida>
    local a="$1" b="$2" out="$3"
    if [[ -f "$out" ]] && grep -qi 'IC95' "$out"; then
        echo ">>> PULANDO $out (já existe)"
        return 0
    fi
    echo ">>> $(date +%H:%M) bootstrap -> $out"
    $PY -u src/bootstrap_ci.py --config $AA --ckpt "$a/best.pt" \
        --config-b $SS --ckpt-b "$b/best.pt" --split test --B 1000 > "$out" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! FALHOU $out (rc=$rc)"
        return $rc
    fi
    echo ">>> $(date +%H:%M) ok $out"
}

boot outputs/conv2d_650m_aa     outputs/conv2d_ssaux     outputs/boot_ssaux_s42.txt || exit 1
boot outputs/conv2d_650m_aa_s43 outputs/conv2d_ssaux_s43 outputs/boot_ssaux_s43.txt || exit 1
boot outputs/conv2d_650m_aa_s44 outputs/conv2d_ssaux_s44 outputs/boot_ssaux_s44.txt || exit 1

echo ">>> $(date +%H:%M) TODOS OS BOOTSTRAPS CONCLUÍDOS"
