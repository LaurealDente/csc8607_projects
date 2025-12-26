#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

rm -rf runs results/* data/* artifacts/*

ENV_NAME="csc8607_env"

if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
    else
        echo "Error: conda/mamba not found."
        exit 1
    fi

    if ! $CONDA_CMD env list | grep -q "$ENV_NAME"; then
        $CONDA_CMD create -n "$ENV_NAME" python=3.10 -y
    fi
    
    source "$(dirname $(dirname $(which $CONDA_CMD)))/etc/profile.d/conda.sh"
    $CONDA_CMD activate "$ENV_NAME"
fi

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

python -m src.grid_search --config configs/config.yaml
python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets
python -m src.train --config configs/config.yaml --overfit_small
python -m src.lr_finder --config configs/config.yaml
python -m src.train --config configs/config.yaml
python -m src.train --config configs/config.yaml --final_run

if [ -f "artifacts/best_of_A.ckpt" ]; then
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A
fi

if [ -f "artifacts/best_of_B.ckpt" ]; then
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B
fi

if [ -f "artifacts/best_of_Special.ckpt" ]; then
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special
fi
