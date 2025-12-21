"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import yaml
import os
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Combien d'époques attendre après la dernière amélioration.
            min_delta (float): Amélioration minimale pour être considérée comme nouvelle meilleure.
            path (str): Où sauvegarder le meilleur modèle.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, model):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.save_checkpoint(val_acc, model)
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Sauvegarde le modèle quand la perte de validation diminue.'''
        torch.save(model.state_dict(), self.path)
        # print(f'Validation loss decreased ({self.best_acc:.6f} --> {val_acc:.6f}).  Saving model ...')


def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python). À implémenter."""
    raise NotImplementedError("set_seed doit être implémentée par l'étudiant·e.")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle. À implémenter."""
    raise NotImplementedError("count_parameters doit être implémentée par l'étudiant·e.")


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")


def visualize_data_debug(config_path="configs/config.yaml"):
    """Debug visuel - À appeler MANUELLEMENT."""
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = yaml.safe_load(open(os.path.join(script_dir, config_path)))
        
        train_data_path = os.path.join(script_dir, "data/preprocessed_dataset_train.pt")
        if not os.path.exists(train_data_path):
            print("❌ Fichiers .pt manquants. Lance preprocessing d'abord.")
            return
        
        train_data = torch.load(train_data_path, weights_only=False)
        train_images = train_data['image']
        
        print(f"✓ Debug OK - Shape: {train_images.shape}")
        print(f"  Min: {train_images.min():.4f}, Max: {train_images.max():.4f}")
        
        # Pas de plt.show() en SLURM → sauvegarde PNG
        plt.figure(figsize=(8, 4))
        plt.hist(train_images.flatten().numpy(), bins=50)
        plt.title("Distribution pixels normalisés")
        plt.savefig("debug_data_distrib.png")
        plt.close()
        print("📊 Histogramme sauvé: debug_data_distrib.png")
        
    except Exception as e:
        print(f"Debug skipped: {e}")