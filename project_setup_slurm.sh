#!/bin/bash
set -o pipefail  # Important : si srun échoue, le script s'arrête (grâce au pipe)

# --- NETTOYAGE OBLIGATOIRE ---
unset SLURM_JOB_ID SLURM_JOBID SLURM_SUBMIT_DIR SLURM_TASK_PID SLURM_JOB_NODELIST SLURM_CLUSTER_NAME
# -----------------------------

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

mkdir -p logs
rm -rf runs results/* data/* artifacts/*

ENV_NAME="csc8607_env"

# Activation Conda
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    if command -v mamba &> /dev/null; then CONDA_CMD="mamba"; elif command -v conda &> /dev/null; then CONDA_CMD="conda"; else echo "Error: conda not found."; exit 1; fi
    source "$(dirname $(dirname $(which $CONDA_CMD)))/etc/profile.d/conda.sh"
    $CONDA_CMD activate "$ENV_NAME"
fi

# Paramètres SLURM
SLURM_ARGS="--time=10:00:00 --gres=gpu:4 --cpus-per-task=8 --mem=32G"

echo "--- Démarrage avec affichage Terminal + Logs ---"

# Note : On utilise 'python -u' pour que l'affichage soit immédiat (unbuffered)
# Note : On utilise '| tee fichier.log' pour afficher à l'écran ET écrire dans le fichier

echo "[1/9] Perte Initiale"
srun $SLURM_ARGS --job-name=init python -u -m src.train --config configs/config.yaml --perte_initiale --charge_datasets | tee logs/01_perte_initiale.log

echo "[2/9] Overfit Small"
srun $SLURM_ARGS --job-name=overfit python -u -m src.train --config configs/config.yaml --overfit_small --charge_datasets | tee logs/02_overfit_small.log

echo "[3/9] Grid Search"
srun $SLURM_ARGS --job-name=grid python -u -m src.grid_search --config configs/config.yaml | tee logs/03_grid_search.log

echo "[4/9] LR Finder"
srun $SLURM_ARGS --job-name=lrfind python -u -m src.lr_finder --config configs/config.yaml | tee logs/04_lr_finder.log

echo "[5/9] Train Standard"
srun $SLURM_ARGS --job-name=train python -u -m src.train --config configs/config.yaml --charge_datasets | tee logs/05_train_standard.log

echo "[6/9] Train Final"
srun $SLURM_ARGS --job-name=final python -u -m src.train --config configs/config.yaml --final_run --charge_datasets | tee logs/06_train_final.log

# Evaluations
if [ -f "artifacts/best_of_A.ckpt" ]; then
    echo "[7/9] Evaluate A"
    srun $SLURM_ARGS --job-name=evalA python -u -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A | tee logs/07_evaluate_A.log
fi

if [ -f "artifacts/best_of_B.ckpt" ]; then
    echo "[8/9] Evaluate B"
    srun $SLURM_ARGS --job-name=evalB python -u -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B | tee logs/08_evaluate_B.log
fi

if [ -f "artifacts/best_of_Special.ckpt" ]; then
    echo "[9/9] Evaluate Special"
    srun $SLURM_ARGS --job-name=evalS python -u -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special | tee logs/09_evaluate_Special.log
fi

echo "--- Séquence terminée ---"
