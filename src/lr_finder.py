"""
Recherche de taux d'apprentissage (LR finder).

Exécutable via :
    python -m src.lr_finder --config configs/config.yaml --lr_wd_finder

Exigences :
- lire la config YAML
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard
"""

import copy
from typing import Callable, List, Tuple
import argparse
import os
import time 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import yaml

import src.model as model
from src import augmentation
from src import data_loading


def find_best_lr_wd(results: List[Tuple[float, float, float]]) -> Tuple[float, float, Tuple[float, float]]:
    """
    Analyse une liste de résultats (lr, wd, loss) pour trouver les meilleurs hyperparamètres.

    Retourne :
        best_lr, best_wd, stable_window (lr_min, lr_max)
    """
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


    print("\n" + "=" * 50)
    print("       Analyse des résultats de la recherche d'hyperparamètres")
    print("=" * 50)

    df = pd.DataFrame(results, columns=['Learning Rate', 'Weight Decay', 'Loss'])
    df_sorted = df.sort_values(by='Loss').reset_index(drop=True)

    print("\nClassement des combinaisons (de la meilleure à la moins bonne):\n")
    print(df_sorted.to_string())

    print("\n" + "-" * 50)
    print("                 Synthèse")
    print("-" * 50)
    print(f"-> Meilleure combinaison trouvée :")
    print(f"   - Learning Rate (LR) : {best_lr:g}")
    print(f"   - Weight Decay (WD)  : {best_wd:g}")
    print(f"   - Perte (Loss) Min   : {min_loss:.4f}\n")

    print(f"-> Fenêtre de Learning Rate stable (pour WD={best_wd:g}) :")
    print(f"   - Intervalle : [{stable_window[0]:g}, {stable_window[1]:g}]")
    print(f"   - Description: Plage de LR où la perte est proche du minimum (<= {loss_threshold:.4f})")
    print("=" * 50 + "\n")

    return best_lr, best_wd, stable_window


def lr_finder(
    modele: torch.nn.Module,
    train_dataloader: DataLoader,
    criterion: Callable,
    lr_list: List[float],
    wd_list: List[float],
    config: dict,
    epochs_per_trial: int = 1,
    device: str = "cpu",
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Boucle sur toutes les combinaisons (lr, wd), réinitialise le modèle, entraîne brièvement,
    évalue la loss moyenne, logue dans TensorBoard, et renvoie la meilleure combinaison.
    """
    results: List[Tuple[float, float, float]] = []
    initial_state = copy.deepcopy(modele.state_dict())
    modele.to(device)
    
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)

    lr_wd_root = os.path.join(runs_dir, "lr_wd_finder")
    os.makedirs(lr_wd_root, exist_ok=True)
    
    run_name = f"lr_finder_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(lr_wd_root, run_name)
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Début de la recherche sur {len(lr_list) * len(wd_list)} combinaisons...")
    print(f"Logs TensorBoard dans : {os.path.join(runs_dir, run_name)}")

    trial_idx = 0


    for lr in lr_list:
        for wd in wd_list:
            modele.load_state_dict(initial_state)

            optimizer = model.get_optimizer(modele, config, weight_decay=wd, lr=lr)

            modele.train()
            for epoch in range(epochs_per_trial):
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = modele(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            total_loss = 0.0
            count = 0
            modele.eval()
            with torch.no_grad():
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = modele(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    count += inputs.size(0)

            avg_loss = total_loss / count if count > 0 else float("inf")
            print(f"  Test -> LR: {lr:.1e}, WD: {wd:.1e}, Loss: {avg_loss:.4f}")

            writer.add_scalar("lr_finder/loss", avg_loss, trial_idx)
            writer.add_scalar("lr_finder/lr", lr, trial_idx)
            writer.add_scalar("lr_finder/wd", wd, trial_idx)

            results.append((lr, wd, avg_loss))
            trial_idx += 1

    writer.close()
    print("\nAnalyse des résultats terminée.")
    return find_best_lr_wd(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()


    with open(os.path.join(os.getcwd(), args.config), "r") as f:
        config = yaml.safe_load(f)
        
    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None,
    )


    train_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=config["train"]["taille_finder"],
        batch_size=config["train"]["batch_size"],
    )

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")


    learning_rates_to_test = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays_to_test = [0.0, 1e-5, 1e-4, 1e-3]


    best_lr, best_wd, stable_window = lr_finder(
        modele=model.build_model(config["basic_model"]),
        train_dataloader=train_loader_subset,
        criterion=nn.CrossEntropyLoss(),
        lr_list=learning_rates_to_test,
        wd_list=weight_decays_to_test,
        config=config,
        epochs_per_trial=1,
        device=device,
    )

    print(f"Meilleur LR trouvé : {best_lr:g}")
    print(f"Meilleur WD trouvé : {best_wd:g}")
    print(f"Fenêtre LR stable : [{stable_window[0]:g}, {stable_window[1]:g}]")


if __name__ == "__main__":
    main()
