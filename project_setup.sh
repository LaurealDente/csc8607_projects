#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# ----------------------------------------------------------------------------
# AUTO-ALLOCATION SLURM
# Si on n'est pas déjà dans un job SLURM, on se relance via salloc
# ----------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    echo "🔒 Demande d'allocation SLURM (10h, 4 GPU, 32G)..."
    # On relance ce même script ($0) à l'intérieur de l'allocation
    exec salloc --time=10:00:00 --gres=gpu:4 --cpus-per-task=8 --mem=32G "$0" "$@"
fi
# ----------------------------------------------------------------------------

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

# Utilisation de srun pour exécuter les calculs sur les nœuds GPU alloués
srun python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets
srun python -m src.train --config configs/config.yaml --overfit_small --charge_datasets
srun python -m src.grid_search --config configs/config.yaml
srun python -m src.lr_finder --config configs/config.yaml
srun python -m src.train --config configs/config.yaml --charge_datasets
srun python -m src.train --config configs/config.yaml --final_run --charge_datasets


if [ -f "artifacts/best_of_A.ckpt" ]; then
    srun python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A
fi


if [ -f "artifacts/best_of_B.ckpt" ]; then
    srun python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B
fi


if [ -f "artifacts/best_of_Special.ckpt" ]; then
    srun python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special
fi
