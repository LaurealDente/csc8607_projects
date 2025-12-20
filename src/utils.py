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


import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import yaml
import os
import sys

# ============ CHARGER LES DONNÉES ============
# Chemins
script_dir = os.path.dirname(os.path.abspath('configs/config.yaml'))
train_data_path = 'data/preprocessed_dataset_train.pt'

# Charger les données prétraitées
train_data = torch.load(train_data_path, weights_only=False)
train_images_preprocessed = train_data['image']  # Tenseur normalisé
train_labels = train_data['label']

print(f"✓ Données prétraitées chargées")
print(f"  Shape: {train_images_preprocessed.shape}")
print(f"  Min: {train_images_preprocessed.min():.4f}, Max: {train_images_preprocessed.max():.4f}")
print(f"  Mean: {train_images_preprocessed.mean():.4f}, Std: {train_images_preprocessed.std():.4f}")

# ============ CHARGER DONNÉES BRUTES (avant preprocessing) ============
# Charger les données BRUTES avant normalisation
raw_data_path = 'data/preprocessed_dataset_train.pt'  # C'est les données brutes sauvegardées
raw_data = torch.load(raw_data_path, weights_only=False)

# ============ DÉFINIR PIPELINE D'AUGMENTATION ============
# Reprendre ta configuration
augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

# ============ VISUALISATION AVANT/APRÈS ============
def visualize_preprocessing_augmentation(preprocessed_tensor, idx=0, augmentation_fn=None, num_augmentations=3):
    """
    Visualise:
    1. Image prétraitée (après preprocessing, avant augmentation)
    2. Plusieurs versions augmentées de la même image
    """
    
    # Image prétraitée (normalisée)
    img_preprocessed = preprocessed_tensor[idx]  # Shape: (3, 64, 64)
    
    # Rescale pour visualisation (les images normalisées sont centrées à 0)
    img_to_display = img_preprocessed.numpy()
    img_to_display = (img_to_display - img_to_display.min()) / (img_to_display.max() - img_to_display.min())
    img_to_display = np.transpose(img_to_display, (1, 2, 0))
    
    # Créer la figure
    n_cols = num_augmentations + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Afficher l'image prétraitée
    axes[0].imshow(img_to_display)
    axes[0].set_title('Après preprocessing\n(Normalisée)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Appliquer augmentation plusieurs fois
    if augmentation_fn:
        for aug_idx in range(1, num_augmentations + 1):
            # Appliquer augmentation
            img_aug = augmentation_fn(img_preprocessed)
            
            # Rescale pour visualisation
            img_aug_display = img_aug.numpy()
            img_aug_display = (img_aug_display - img_aug_display.min()) / (img_aug_display.max() - img_aug_display.min())
            img_aug_display = np.transpose(img_aug_display, (1, 2, 0))
            
            axes[aug_idx].imshow(img_aug_display)
            axes[aug_idx].set_title(f'Augmentation {aug_idx}', fontsize=12)
            axes[aug_idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Statistiques image {idx}:")
    print(f"  Prétraitée - Min: {img_preprocessed.min():.4f}, Max: {img_preprocessed.max():.4f}")
    print(f"              Mean: {img_preprocessed.mean():.4f}, Std: {img_preprocessed.std():.4f}")

# ============ AFFICHER MULTIPLE EXEMPLES ============
print("\n" + "="*60)
print("VISUALISATION: PREPROCESSING vs AUGMENTATION")
print("="*60)

# Exemple 1: Image 0
visualize_preprocessing_augmentation(
    train_images_preprocessed, 
    idx=0, 
    augmentation_fn=augmentation_pipeline,
    num_augmentations=3
)

# Exemple 2: Image 1
visualize_preprocessing_augmentation(
    train_images_preprocessed, 
    idx=1, 
    augmentation_fn=augmentation_pipeline,
    num_augmentations=3
)

# ============ COMPARAISON STATISTIQUES ============
print("\n" + "="*60)
print("IMPACT DU PREPROCESSING SUR LES STATISTIQUES")
print("="*60)

# Sélectionner un batch
batch_idx = slice(0, 32)
batch_preprocessed = train_images_preprocessed[batch_idx]

print(f"\nBatch (32 images) après preprocessing:")
print(f"  Min: {batch_preprocessed.min():.4f}")
print(f"  Max: {batch_preprocessed.max():.4f}")
print(f"  Mean: {batch_preprocessed.mean(dim=[0, 2, 3])}")  # Mean par canal
print(f"  Std:  {batch_preprocessed.std(dim=[0, 2, 3])}")   # Std par canal

# ============ GRID VISUALIZATION ============
print("\n" + "="*60)
print("GRID: 4 IMAGES PRÉTRAITÉES + AUGMENTATIONS")
print("="*60)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))

for row in range(4):
    for col in range(4):
        img_idx = row * 4 + col
        
        if col == 0:
            # Colonne 0: image prétraitée originale
            img = train_images_preprocessed[img_idx]
        else:
            # Colonnes 1-3: augmentations
            img = augmentation_pipeline(train_images_preprocessed[img_idx])
        
        # Rescale pour visualisation
        img_display = img.numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        img_display = np.transpose(img_display, (1, 2, 0))
        
        axes[row, col].imshow(img_display)
        if col == 0:
            axes[row, col].set_title(f'Image {row} (Original)', fontsize=10)
        else:
            axes[row, col].set_title(f'Aug {col}', fontsize=10)
        axes[row, col].axis('off')

plt.tight_layout()
plt.suptitle('Preprocessing (Col 0) vs Augmentations (Col 1-3)', fontsize=14, y=0.995)
plt.show()

print("\n✓ Visualisation complète!")
