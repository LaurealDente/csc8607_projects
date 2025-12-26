#!/bin/bash

# --- NETTOYAGE DES VARIABLES SLURM FANTÔMES ---
# C'est cette partie qui corrige votre erreur "Invalid job id specified"
unset SLURM_JOB_ID
unset SLURM_JOBID
unset SLURM_SUBMIT_DIR
unset SLURM_TASK_PID
unset SLURM_JOB_NODELIST
unset SLURM_CLUSTER_NAME
# ----------------------------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

mkdir -p logs
rm -rf runs results/* data/* artifacts/*

ENV_NAME="csc8607_env"

# Configuration de l'environnement
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
    else
        echo "Error: conda/mamba not found."
        exit 1
    fi
    source "$(dirname $(dirname $(which $CONDA_CMD)))/etc/profile.d/conda.sh"
    $CONDA_CMD activate "$ENV_NAME"
fi

# Paramètres de vos jobs (10h, 4 GPU, 32G de RAM par job)
SLURM_ARGS="--time=10:00:00 --gres=gpu:4 --cpus-per-task=8 --mem=32G"

echo "--- Démarrage propre (Environnement nettoyé) ---"

# Chaque commande srun va maintenant créer son PROPRE job indépendant

echo "[1/9] Perte Initiale"
srun $SLURM_ARGS --job-name=init --output=logs/01_perte_initiale_%j.log \
    python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets

echo "[2/9] Overfit Small"
srun $SLURM_ARGS --job-name=overfit --output=logs/02_overfit_small_%j.log \
    python -m src.train --config configs/config.yaml --overfit_small --charge_datasets

echo "[3/9] Grid Search"
srun $SLURM_ARGS --job-name=grid --output=logs/03_grid_search_%j.log \
    python -m src.grid_search --config configs/config.yaml

echo "[4/9] LR Finder"
srun $SLURM_ARGS --job-name=lrfind --output=logs/04_lr_finder_%j.log \
    python -m src.lr_finder --config configs/config.yaml

echo "[5/9] Train Standard"
srun $SLURM_ARGS --job-name=train --output=logs/05_train_standard_%j.log \
    python -m src.train --config configs/config.yaml --charge_datasets

echo "[6/9] Train Final"
srun $SLURM_ARGS --job-name=final --output=logs/06_train_final_%j.log \
    python -m src.train --config configs/config.yaml --final_run --charge_datasets

# Evaluation conditionnelle
if [ -f "artifacts/best_of_A.ckpt" ]; then
    echo "[7/9] Evaluate A"
    srun $SLURM_ARGS --job-name=evalA --output=logs/07_evaluate_A_%j.log \
        python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A
fi

if [ -f "artifacts/best_of_B.ckpt" ]; then
    echo "[8/9] Evaluate B"
    srun $SLURM_ARGS --job-name=evalB --output=logs/08_evaluate_B_%j.log \
        python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B
fi

if [ -f "artifacts/best_of_Special.ckpt" ]; then
    echo "[9/9] Evaluate Special"
    srun $SLURM_ARGS --job-name=evalS --output=logs/09_evaluate_Special_%j.log \
        python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special
fi

echo "--- Séquence terminée ---"
