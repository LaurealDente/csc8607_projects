#!/bin/bash

# 1. Configuration du dossier
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Nettoyage
rm -rf runs results data artifacts

# 2. Configuration de l'environnement
ENV_NAME="csc8607_env"

# Détection robuste du chemin de base de conda
# On cherche le dossier qui contient 'etc/profile.d/conda.sh'
if [ -n "$CONDA_EXE" ]; then
    # Si conda est déjà activé ou dans le path, on utilise sa variable d'env
    CONDA_BASE="$(dirname $(dirname "$CONDA_EXE"))"
elif [ -d "$HOME/miniforge3" ]; then
    CONDA_BASE="$HOME/miniforge3"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "❌ Impossible de trouver l'installation de Conda/Mamba."
    echo "Veuillez définir CONDA_BASE manuellement dans le script."
    exit 1
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
echo "ℹ️  Conda détecté ici : $CONDA_BASE"

# Source pour le shell actuel (Controller)
source "$CONDA_SH"

# Création de l'environnement si nécessaire
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "🆕 Création de l'environnement $ENV_NAME..."
    mamba create -n "$ENV_NAME" python=3.10 -y || conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activation sur le controller pour installer les dépendances
conda activate "$ENV_NAME"

if [ -f "requirements.txt" ]; then
    echo "⬇️  Installation des dépendances..."
    pip install -r requirements.txt
fi

# 3. Lancement des jobs SLURM
# On passe la commande d'activation complète à chaque job
SLURM_OPTS="--time=10:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G"
ACTIVATE_CMD="source $CONDA_SH && conda activate $ENV_NAME"

echo "🚀 Lancement du Grid Search..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.grid_search --config configs/config.yaml"

echo "🚀 Préparation des données..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --perte_initiale --charge_datasets"

echo "🚀 Test Overfit..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --overfit_small"

echo "🚀 LR Finder..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.lr_finder --config configs/config.yaml"

echo "🚀 Entraînement Standard (A & B)..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml"

echo "🚀 Entraînement Final (Special)..."
salloc $SLURM_OPTS bash -c "$ACTIVATE_CMD && python -m src.train --config configs/config.yaml --final_run"

# Évaluations (Vérification des fichiers via python car le bash controller ne voit pas forcément les fichiers créés sur le noeud immédiatement ou si le path diffère)
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

echo "✅ Terminé."
