#!/usr/bin/env bash
# Variância de inicialização + desempate passo 3 vs SS-aux.
#
# Os runs existentes (outputs/conv2d_650m_aa, outputs/conv2d_ssaux) já são a seed 42,
# então aqui só faltam 43 e 44 de cada braço para fechar 3 sementes por receita.
#
# Sequencial de propósito: a GTX 1650 (4 GB) não comporta dois treinos ao mesmo tempo.
#
# Reentrante: dá para desligar a máquina e retomar com o mesmo comando. Um run só é
# considerado PRONTO se o log tiver a linha final do treino — NÃO basta existir best.pt,
# que é regravado a cada melhora de validação e portanto já existe em run interrompido.
# Run incompleto é retomado de last.pt pelo próprio train_conv2d.py (época, best e
# estado do otimizador), perdendo no máximo a época em andamento.
#
#   bash scripts/run_seeds.sh
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
FIM='^>> melhor AUPRC_types_macro'   # última linha que o treino imprime ao concluir

run() {  # run <config> <out> <seed>
    local cfg="$1" out="$2" seed="$3" log="${2}.log"
    if [[ -f "$log" ]] && grep -q "$FIM" "$log"; then
        echo ">>> PULANDO $out (treino já concluído)"
        return 0
    fi
    if [[ -f "$out/last.pt" ]]; then
        echo ">>> $(date +%H:%M) RETOMANDO $out (seed=$seed) de last.pt"
    else
        echo ">>> $(date +%H:%M) iniciando $out (seed=$seed, $cfg)"
    fi
    # >> append: retomada não apaga o histórico de épocas já treinadas
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
