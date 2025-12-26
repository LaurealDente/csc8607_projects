#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "🧹 Nettoyage des répertoires..."
for dir in runs results data artifacts; do
    mkdir -p "$dir"
    rm -rf "$dir"/*
done

ENV_NAME="csc8607_env"

if [ -n "$CONDA_EXE" ]; then
    CONDA_BASE="$(dirname $(dirname "$CONDA_EXE"))"
elif [ -d "$HOME/miniforge3" ]; then
    CONDA_BASE="$HOME/miniforge3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "❌ Impossible de trouver l'installation de Conda/Mamba."
    exit 1
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_SH"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "🆕 Création de l'environnement $ENV_NAME..."
    mamba create -n "$ENV_NAME" python=3.10 -y || conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

if [ -f "requirements.txt" ]; then
    echo "⬇️  Installation des dépendances..."
    pip install -r requirements.txt
fi


SLURM_OPTS="--time=10:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G"
ACTIVATE_CMD="source $CONDA_SH && conda activate $ENV_NAME"


echo "🚀 [1/6] Préparation des données..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets"


echo "🚀 [2/6] Lancement du Grid Search..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.grid_search --config configs/config.yaml"

echo "🚀 [3/6] Test Overfit..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --overfit_small --charge_datasets"

echo "🚀 [4/6] LR Finder..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.lr_finder --config configs/config.yaml"

echo "🚀 [5/6] Entraînement Standard (A & B)..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --charge_datasets"

echo "🚀 [6/6] Entraînement Final (Special)..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --final_run --charge_datasets"

echo "🚀 Évaluations..."

if [ -f "artifacts/best_of_A.ckpt" ]; then
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A"
fi

if [ -f "artifacts/best_of_B.ckpt" ]; then
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B"
fi

if [ -f "artifacts/best_of_Special.ckpt" ]; then
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special"
fi

echo "✅ Pipeline terminé."
