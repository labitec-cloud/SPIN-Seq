#!/usr/bin/env bash
# Configura o ambiente do SPIN-Seq via venv + pip (máquina sem conda).
# Uso: bash scripts/setup_env.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ">> Criando virtualenv em .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

echo ">> Instalando dependências (requirements.txt)"
pip install -r requirements.txt

echo ">> Verificando torch/CUDA"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ">> Verificando pdbe-arpeggio"
python -c "import arpeggio" 2>/dev/null && echo 'arpeggio OK' || echo 'AVISO: arpeggio não importou — ver nota sobre OpenBabel no requirements.txt'

echo ">> Ambiente pronto. Ative com: source .venv/bin/activate"
