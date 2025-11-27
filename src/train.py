"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse
import yaml
import os
import src.model as model
import src.data_loading as data_loading
import src.augmentation as augmentation
import src.preprocessing as preprocessing
import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader, Dataset
import torch.optim as optim
import time
import numpy as np
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def overfitting_small(modele, config):
    optimizer = model.get_optimizer(modele, config)
    criterion = nn.CrossEntropyLoss()

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)
    modele.train()

    run_name = f"overfit_small_{time.strftime('%Y%m%d-%H%M%S')}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tensorboard_path = os.path.join(script_dir, "../runs/overfit_small/"+run_name)
    writer = SummaryWriter(log_dir=tensorboard_path)

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None
        )
        
    subset_size = config["train"]["batch_size"] * 2
    overfit_indices = list(range(subset_size))
    overfit_dataset = Subset(full_train_dataset, overfit_indices)

    overfit_loader = DataLoader(overfit_dataset, 
                                batch_size=config["train"]["batch_size"], 
                                shuffle=True)

    print(f"Taille du sous-ensemble : {subset_size} exemples.")
    print(f"Hyperparamètres modèle : B={config['model']['residual_blocks']}, dropout={config['model']['dropout']}")
    print(f"Optimisation : LR={config['train']['finder_start_lr']}, Weight Decay={config['train']['optimizer']['over_weight_decay']}")
    print(f"Nombre d'époques : {config['train']['overfit_epochs']}")

    for epoch in range(config['train']['overfit_epochs']):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in overfit_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = modele(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        avg_epoch_loss = epoch_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        writer.add_scalar('Overfit/Train_Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Overfit/Train_Accuracy', epoch_accuracy, epoch)

        print(f"Époque {epoch+1}/{config['train']['overfit_epochs']} | Perte: {avg_epoch_loss:.4f} | Précision: {epoch_accuracy:.2%}")

    print("\nEntraînement 'overfit' terminé.")
    print(f"Les logs sont disponibles dans le dossier : runs/{run_name}")
    
    writer.close()
    


def perte_premier_batch(modele, dataloader, config) :
    """
    Calcul de la perte sur le premier batch afin d'évaluer sa cohérence
    Vérification du fonctionnement de la backward propagation
    """
    writer = SummaryWriter(log_dir="runs/mon_experience_1") 

    criterion = nn.CrossEntropyLoss()

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)
    modele.eval()

    try:
        first_batch_images, first_batch_labels = next(iter(dataloader))
    except StopIteration:
        writer.close()
        print("Erreur : Le DataLoader est vide.")
        return


    first_batch_images = first_batch_images.to(device)
    first_batch_labels = first_batch_labels.to(device)

    outputs = modele(first_batch_images)
    loss = criterion(outputs, first_batch_labels)
    perte_observee = loss.item()

    print(f"Perte observée sur le premier batch : {perte_observee:.4f}")
    expected_loss = -math.log(1 / config["model"]["num_classes"])
    print(f"Perte théorique attendue : {expected_loss:.4f}")

    writer.add_scalar('Loss/initial_check', perte_observee, 0)
    writer.flush()
    writer.close()

    loss.backward()

    gradient_existe = False
    for name, param in modele.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            gradient_existe = True
            break

    if gradient_existe:
        print("Vérification : backward OK, gradients ≠ 0. Le modèle semble prêt à apprendre.")
    else:
        print("ERREUR DE VÉRIFICATION : Aucun gradient n'a été calculé. Le modèle n'apprendra pas.")

        
    return perte_observee, gradient_existe


def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "configs/config.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default=config_path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--perte_initiale", action="store_true")
    parser.add_argument("--lr_wd_finder", action="store_true")
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()    


    try :
        with open(os.path.join(os.getcwd(),args.config), "r") as f:
            config = yaml.safe_load(f)
    except:
        raise Exception("Mauvais chemin du fichier de configuration : " + os.path.join(os.getcwd(),args.config))

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -- Récupération des données + Enregistrement (data) + Augmentation + Preprocessing -- #
    preprocessing.get_preprocess_transforms(config)

    # -- Récupération du modèle -- #
    modele = model.build_model(config)

    # -- Récupération du pipeline d'augmentation -- #
    augmentation_pipeline = augmentation.get_augmentation_transforms(config)

    # -- Récupératon du dataloader -- #
    train_loader = data_loading.get_dataloaders("train", augmentation_pipeline, config)

    # -- Perte initiale sur le premier batch -- #
    if args.perte_initiale :
        perte_initiale = perte_premier_batch(modele, train_loader, config)

    # -- Small batch overfit -- #
    if args.overfit_small :
        overfit = overfitting_small(modele, config)


if __name__ == "__main__":
    main()
