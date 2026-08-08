#!/usr/bin/env bash
# Initialisation variance + step 3 vs SS-aux tie-break.
#
# The existing runs (outputs/conv2d_650m_aa, outputs/conv2d_ssaux) are already seed 42,
# so only 43 and 44 of each arm are missing to complete 3 seeds per recipe.
#
# Sequential on purpose: the GTX 1650 (4 GB) cannot hold two training runs at once.
#
# Re-entrant: the machine can be switched off and resumed with the same command. A run is
# considered DONE only if the log has the final training line - the existence of best.pt is
# NOT enough, since it is rewritten on every validation improvement and therefore already
# exists in an interrupted run. An incomplete run is resumed from last.pt by train_conv2d.py
# (epoch, best and optimiser state), losing at most the epoch in progress.
#
#   bash scripts/run_seeds.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
FIM='^>> melhor AUPRC_types_macro'   # última linha que o treino imprime ao concluir

run() {  # run <config> <out> <seed>
    local cfg="$1" out="$2" seed="$3" log="${2}.log"
    if [[ -f "$log" ]] && grep -q "$FIM" "$log"; then
        echo ">>> SKIPPING $out (training already finished)"
        return 0
    fi
    if [[ -f "$out/last.pt" ]]; then
        echo ">>> $(date +%H:%M) RETOMANDO $out (seed=$seed) de last.pt"
    else
        echo ">>> $(date +%H:%M) iniciando $out (seed=$seed, $cfg)"
    fi
    # >> append: resuming does not erase the history of epochs already trained
    $PY -u src/train_conv2d.py --config "$cfg" --out "$out" --seed "$seed" >> "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!! FALHOU $out (rc=$rc) — ver $log"
        return $rc
    fi
    echo ">>> $(date +%H:%M) concluído $out: $(tail -1 "$log")"
}

AA=configs/esm650m_aa.yaml
SS=configs/esm650m_aa_ssaux.yaml

run $AA outputs/conv2d_650m_aa_s43 43 || exit 1
run $AA outputs/conv2d_650m_aa_s44 44 || exit 1
run $SS outputs/conv2d_ssaux_s43   43 || exit 1
run $SS outputs/conv2d_ssaux_s44   44 || exit 1

echo ">>> $(date +%H:%M) TODOS OS TREINOS CONCLUÍDOS"
