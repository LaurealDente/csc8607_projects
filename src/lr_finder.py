"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import copy
from typing import Callable, List, Tuple
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import src.model as model
from src import augmentation
from src import data_loading


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
    modele: torch.nn.Module,
    train_dataloader: DataLoader,
    criterion: Callable,
    lr_list: List[float],
    wd_list: List[float],
    config: dict, 
    epochs_per_trial: int = 1,
    device: str = 'cpu'
) -> Tuple[float, float, Tuple[float, float]]:
    
    results = []
    initial_state = copy.deepcopy(modele.state_dict())
    modele.to(device)

    print(f"Début de la recherche sur {len(lr_list) * len(wd_list)} combinaisons...")

    for wd in wd_list:
        for lr in lr_list:
            # Réinitialiser le modèle à son état initial pour chaque essai
            modele.load_state_dict(initial_state)
            
            optimizer = model.get_optimizer(modele, config, wd, lr)

            # Phase d'entraînement courte
            modele.train()
            for epoch in range(epochs_per_trial):
                for inputs, targets in train_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = modele(inputs)
                    loss = criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
            
            # Phase d'évaluation
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
            
            avg_loss = total_loss / count if count > 0 else float('inf')
            print(f"  Test -> LR: {lr:.1e}, WD: {wd:.1e}, Loss: {avg_loss:.4f}")
            results.append((lr, wd, avg_loss))

    print("\nAnalyse des résultats terminée.")
    return find_best_lr_wd(results)




import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
   

    # -- DataSet de recherche -- #

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=args.config["dataset"]["split"]["train"]["chemin"]
    )


    train_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=90000,
        batch_size=args.config['train']['batch_size']
    )


    device = torch.device(args.config["train"]["device"] if torch.cuda.is_available() else "cpu")

    # -- LR Finder -- #
    learning_rates_to_test = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decays_to_test = [0, 1e-5, 1e-4, 1e-3]
    if args.lr_wd_finder :
        best_lr, best_wd, stable_window = lr_finder(model.build_model(args.config), train_loader_subset, nn.CrossEntropyLoss(), learning_rates_to_test, weight_decays_to_test, args.config, 1, device)


if __name__ == "__main__":
    main()