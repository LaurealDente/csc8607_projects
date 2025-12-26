import random
import os
import numpy as np
import torch
import yaml

def set_seed(seed=42):
    """
    Fixe la graine aléatoire pour Python, NumPy et PyTorch (CPU et GPU)
    pour garantir la reproductibilité.
    """
    if seed is None:
        seed = 42
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Utils] Seed fixée à : {seed}")

def count_parameters(model):
    """
    Compte le nombre de paramètres entraînables (trainable=True) d'un modèle.
    Retourne un entier.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_config_snapshot(config, run_dir):
    """
    Sauvegarde une copie de la configuration actuelle dans le dossier de logs (run_dir).
    """
    os.makedirs(run_dir, exist_ok=True)
    snapshot_path = os.path.join(run_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[Utils] Snapshot de la config sauvegardé dans : {snapshot_path}")

def get_device(config=None):
    """
    Retourne l'objet torch.device approprié (cuda ou cpu).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class EarlyStopping:
    """
    Classe utilitaire pour arrêter l'entraînement si la métrique de validation
    ne s'améliore plus après 'patience' époques.
    Gère aussi la sauvegarde du meilleur modèle.
    """
    def __init__(self, patience=10, min_delta=0, path='checkpoint.ckpt', verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_metric, model):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Sauvegarde le modèle quand le score s'améliore.'''
        if self.verbose:
            print(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
