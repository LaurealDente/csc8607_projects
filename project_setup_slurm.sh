#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

echo "Nettoyage des répertoires..."
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
    echo "Erreur : Conda non trouvé."
    exit 1
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_SH"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Création de l'environnement..."
    mamba create -n "$ENV_NAME" python=3.10 -y || conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
fi

SLURM_OPTS="--time=10:00:00 --gres=gpu:4 --cpus-per-task=8 --mem=32G"
ACTIVATE_CMD="source $CONDA_SH && conda activate $ENV_NAME"

echo ">>> [ETAPE 1] Génération des données (Bloquant)..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets"

if [ $? -ne 0 ]; then
    echo "Échec de la génération des données."
    exit 1
fi

echo ">>> Lancement des tâches parallèles..."

(
    echo "   [Grid Search] Démarré..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.grid_search --config configs/config.yaml"
    echo "   [Grid Search] Terminé."
) &

(
    echo "   [Overfit Test] Démarré..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --overfit_small"
    echo "   [Overfit Test] Terminé."
) &

(
    echo "   [LR Finder] Démarré..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.lr_finder --config configs/config.yaml"
    echo "   [LR Finder] Terminé."
) &

(
    echo "   [Train Standard A/B] Démarré..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml"
    
    echo "   [Eval Standard A/B] Démarrée..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && \
        python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_A.ckpt --model A && \
        python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_B.ckpt --model B"
    echo "   [Train/Eval Standard] Terminé."
) &

(
    echo "   [Train Final Special] Démarré..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --final_run"
    
    echo "   [Eval Final Special] Démarrée..."
    salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best_of_Special.ckpt --model Special"
    echo "   [Train/Eval Final] Terminé."
) &

wait
echo ">>> Toutes les tâches sont terminées."
