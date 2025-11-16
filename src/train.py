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
from typing import Callable, List, Tuple
import copy
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def create_stratified_subset_loader_manual(
    dataset: Dataset, 
    subset_size: int, 
    batch_size: int, 
    num_workers: int = 0
    ) -> DataLoader:
    # 1. Vérifier que le dataset a bien une liste de labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError("Le Dataset doit avoir un attribut '.targets' ou '.labels' pour la stratification.")

    # S'assurer que la taille du sous-ensemble est réaliste
    subset_size = min(subset_size, len(dataset))

    # 2. Regrouper les indices de chaque image par label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # 3. Calculer combien d'échantillons prendre par classe
    final_indices = []
    total_size = len(dataset)
    
    for label, indices in label_to_indices.items():
        # Proportion de cette classe dans le dataset complet
        proportion = len(indices) / total_size
        # Nombre d'échantillons à prendre pour cette classe dans le sous-ensemble
        num_samples_for_label = int(subset_size * proportion)
        # Assurer qu'on prend au moins un échantillon si la classe est représentée
        num_samples_for_label = max(1, num_samples_for_label)
        
        # 4. Piocher aléatoirement les indices pour cette classe
        # `random.sample` pioche sans remise
        sampled_indices = random.sample(indices, min(len(indices), num_samples_for_label))
        final_indices.extend(sampled_indices)
        
    # 5. Créer le Subset et le DataLoader
    random.shuffle(final_indices) # Mélanger les indices de toutes les classes
    subset_dataset = Subset(dataset, final_indices)
    
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True, # Important de mélanger le sous-ensemble final
        num_workers=num_workers
    )
    
    print(f"Création d'un sous-ensemble stratifié manuel de {len(subset_dataset)} échantillons.")
    return subset_loader



def mini_grid_search(data_loader_train, data_loader_test, config, device, num_epochs, criterion):
    
    all_results = []

    # Hyperparamètres à tester
    dropout_options = [0.1, 0.3]
    block_config_options = [[2, 2, 2], [3, 3, 3]]

    # Paramètres fixes
    num_classes = 200
    use_residual = True
    use_batch_norm = True
    activation_fn = "relu"

    # Créer la liste de toutes les configurations
    configs_to_test = []
    for dropout in dropout_options:
        for block_config in block_config_options:
            config_adding = {
                "model": {
                    "residual_blocks": block_config,
                    "dropout": dropout,
                    "num_classes": num_classes,
                    "residual": use_residual,
                    "batch_norm": use_batch_norm,
                    "activation": activation_fn
                }
            }
            configs_to_test.append(config_adding)

    models_to_test = [model.build_model(cfg) for cfg in configs_to_test]

    # Créer le writer TensorBoard dans le dossier runs/grid_search
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_grid_search_path = os.path.join(script_dir, "runs/grid_search")

    writer = SummaryWriter(log_dir=runs_grid_search_path)

    for idx, (modele, cfg) in enumerate(zip(models_to_test, configs_to_test)):
        optimizer = get_optimizer(modele, config, config["grid_search"]["hparams"]["weight_decay"][0], config["grid_search"]["hparams"]["lr"][0])
        
        modele.to(device)
        epoch_iterator = tqdm(range(num_epochs))

        for epoch in epoch_iterator:
            modele.train()
            train_loss=0.0

            for image, label in data_loader_train:
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = modele(image)
                loss = criterion(outputs, label)
                if torch.isnan(loss):
                    print(f"\n[ERREUR] Loss est NaN. Arrêt de l'entraînement.")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            if torch.isnan(loss): break

            avg_train_loss = train_loss / len(data_loader_train)
            epoch_iterator.set_postfix(train_loss=f"{avg_train_loss:.4f}")

            if not torch.isnan(loss):
                modele.eval()
                val_loss, correct, total = 0, 0, 0
                with torch.no_grad():
                    for image, label in data_loader_test:
                        image, label = image.to(device), label.to(device)
                        outputs = modele(image)
                        val_loss += criterion(outputs, label).item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()
                
                final_val_loss = val_loss / len(data_loader_test)
                final_val_accuracy = 100 * correct / total
                notes = ""
            else:
                final_val_loss = float('nan')
                final_val_accuracy = 0.0
                notes = "Échoué (NaN)"

            print(f"Résultats -> Val Loss: {final_val_loss:.4f}, Val Accuracy: {final_val_accuracy:.2f}%")

            # Enregistrer dans TensorBoard (par modèle et epoch)
            writer.add_scalar(f"Model_{idx}/Val_Loss", final_val_loss, epoch)
            writer.add_scalar(f"Model_{idx}/Val_Accuracy", final_val_accuracy, epoch)

            all_results.append({
                "Modèle": idx,
                "Epoch": epoch,
                "Dropout": cfg["model"]["dropout"],
                "Block config": cfg["model"]["residual_blocks"],
                "Accuracy (%)": final_val_accuracy,
                "Loss": final_val_loss,
                "Notes": notes,
            })

        # Fermer le modèle avant de passer au suivant (optionnel)
        modele.cpu()

    writer.close()

    # Créer et afficher le tableau final
    results_df = pd.DataFrame(all_results).sort_values(by=["Modèle", "Epoch"])
    print("\n\n" + "="*50)
    print("        RÉCAPITULATIF FINAL DES PERFORMANCES")
    print("="*50)
    print(results_df.to_markdown(index=False))

    return results_df





def find_best_lr_wd(results: List[Tuple[float, float, float]]) -> Tuple[float, float, Tuple[float, float]]:
    """Analyse une liste de résultats pour trouver les meilleurs hyperparamètres."""
    if not results:
        raise ValueError("La liste de résultats ne peut pas être vide.")

    results_sorted = sorted(results, key=lambda x: x[2])
    best_lr, best_wd, min_loss = results_sorted[0]

    loss_threshold = min_loss * 1.05
    lr_candidates_for_best_wd = [res for res in results if res[1] == best_wd]
    stable_lrs = [lr for lr, wd, loss in lr_candidates_for_best_wd if loss <= loss_threshold]

    if stable_lrs:
        stable_window = (min(stable_lrs), max(stable_lrs))
    else:
        stable_window = (best_lr, best_lr)

    # --- Affichage des résultats ---
    print("\n" + "="*50)
    print("       Analyse des résultats de la recherche d'hyperparamètres")
    print("="*50)

    # Créer un DataFrame pour un affichage plus propre
    df = pd.DataFrame(results, columns=['Learning Rate', 'Weight Decay', 'Loss'])
    df_sorted = df.sort_values(by='Loss').reset_index(drop=True)

    print("\nClassement des combinaisons (de la meilleure à la moins bonne):\n")
    print(df_sorted.to_string())

    print("\n" + "-"*50)
    print("                 Synthèse")
    print("-"*50)
    print(f"-> Meilleure combinaison trouvée :")
    print(f"   - Learning Rate (LR) : {best_lr:g}")
    print(f"   - Weight Decay (WD)  : {best_wd:g}")
    print(f"   - Perte (Loss) Min   : {min_loss:.4f}\n")

    print(f"-> Fenêtre de Learning Rate stable (pour WD={best_wd:g}) :")
    print(f"   - Intervalle : [{stable_window[0]:g}, {stable_window[1]:g}]")
    print(f"   - Description: Plage de LR où la perte est proche du minimum (<= {loss_threshold:.4f})")
    print("="*50 + "\n")

    return best_lr, best_wd, stable_window



def lr_finder(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    criterion: Callable,
    lr_list: List[float],
    wd_list: List[float],
    config: dict, 
    epochs_per_trial: int = 1,
    device: str = 'cpu'
) -> Tuple[float, float, Tuple[float, float]]:
    
    results = []
    initial_state = copy.deepcopy(model.state_dict())
    model.to(device)

    print(f"Début de la recherche sur {len(lr_list) * len(wd_list)} combinaisons...")

    for wd in wd_list:
        for lr in lr_list:
            # Réinitialiser le modèle à son état initial pour chaque essai
            model.load_state_dict(initial_state)
            
            optimizer = get_optimizer(model, config, wd, lr)

            # Phase d'entraînement courte
            model.train()
            for epoch in range(epochs_per_trial):
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
            
            # Phase d'évaluation
            total_loss = 0.0
            count = 0
            model.eval()
            with torch.no_grad():
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    count += inputs.size(0)
            
            avg_loss = total_loss / count if count > 0 else float('inf')
            print(f"  Test -> LR: {lr:.1e}, WD: {wd:.1e}, Loss: {avg_loss:.4f}")
            results.append((lr, wd, avg_loss))

    print("\nAnalyse des résultats terminée.")
    return find_best_lr_wd(results)



def get_optimizer(model, config, weight_decay=None, lr=None):
    """
    Crée un optimiseur.
    Accepte weight_decay et lr comme arguments optionnels pour surcharger la config.
    """
    # Priorité à l'argument direct, sinon on prend la valeur dans le fichier config
    wd_to_use = weight_decay if weight_decay is not None else config["train"]["optimizer"].get("weight_decay", 0)
    lr_to_use = lr if lr is not None else config["train"]["optimizer"]["lr"]

    optimizer_name = config["train"]["optimizer"]["name"]
    
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr_to_use, weight_decay=wd_to_use)
    elif optimizer_name == 'SGD':
        # Assurez-vous d'avoir un momentum dans votre config si vous utilisez SGD
        momentum = config["train"]["optimizer"].get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr_to_use, weight_decay=wd_to_use, momentum=momentum)
    else:
        return torch.optim.Adam(model.parameters(), lr=float(lr_to_use), weight_decay=float(wd_to_use))



def overfitting_small(modele, config):
    optimizer = get_optimizer(modele, config)
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
    

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir_path = os.path.join(script_dir, "../runs/")

    # -- Récupération des données + Enregistrement (data) + Augmentation + Preprocessing -- #
    preprocessing.get_preprocess_transforms(config)

    # -- Récupération du modèle -- #
    modele = model.build_model(config)

    # -- Récupération du pipeline d'augmentation -- #
    augmentation_pipeline = augmentation.get_augmentation_transforms(config)

    # -- Récupératon du dataloader -- #
    train_loader = data_loading.get_dataloaders("train", augmentation_pipeline, config)
    test_loader = data_loading.get_dataloaders("test", augmentation_pipeline, config)

    # -- Perte initiale sur le premier batch -- #
    if args.perte_initiale :
        perte_initiale = perte_premier_batch(modele, train_loader, config)

    # -- Small batch overfit -- #
    if args.overfit_small :
        overfit = overfitting_small(modele, config)
    

    # -- DataSet de recherche -- #

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=augmentation_pipeline
    )

    full_test_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["test"]["chemin"])

    train_loader_subset = create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=10000,
        batch_size=config['train']['batch_size']
    )
    test_loader_subset = create_stratified_subset_loader_manual(
        dataset=full_test_dataset,
        subset_size=2000,
        batch_size=config['train']['batch_size']
    )

    # -- LR Finder -- #
    learning_rates_to_test = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays_to_test = [0, 1e-5, 1e-4, 1e-3]
    if args.lr_wd_finder :
        best_lr, best_wd, stable_window = lr_finder(modele, train_loader_subset, nn.CrossEntropyLoss(), learning_rates_to_test, weight_decays_to_test, config, 1, device)


    # -- Mini Grid Search -- #
    if args.grid_search:
        print("grid_search")
        mini_grid_search(train_loader_subset, test_loader_subset, config, device, 200, nn.CrossEntropyLoss())



if __name__ == "__main__":
    main()
