"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse
import os
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import yaml

from src import model, augmentation, data_loading


def mini_grid_search(data_loader_train, data_loader_val, config, device, num_epochs, criterion):
    """
    Lance une mini grid search sur dropout et block_config,
    avec LR et WD fixés depuis config['grid_search']['hparams'].
    """
    all_results = []

    # Hyperparamètres à tester
    hparams_cfg = config["grid_search"]["hparams"]
    dropout_options = hparams_cfg["dropout_p"]
    block_config_options = hparams_cfg["block_config"]
    lr_options = hparams_cfg["lr"]
    wd_options = hparams_cfg["weight_decay"]

    # Paramètres fixes du modèle
    num_classes = config['model']['num_classes']
    use_residual = config['model']['residual']
    use_batch_norm = config['model']['batch_norm']
    activation_fn = config['model']['activation']

    # Toutes les configurations à tester (modèle + lr/wd)
    configs_to_test = []
    for dropout in dropout_options:
        for block_config in block_config_options:
            for lr in lr_options:
                for wd in wd_options:
                    cfg_model = {
                        "model": {
                            "residual_blocks": block_config,
                            "dropout": dropout,
                            "num_classes": num_classes,
                            "residual": use_residual,
                            "batch_norm": use_batch_norm,
                            "activation": activation_fn,
                        }
                    }
                    configs_to_test.append(
                        {
                            "model_cfg": cfg_model,
                            "lr": lr,
                            "wd": wd,
                        }
                    )

    print(f"Nombre de configurations à tester : {len(configs_to_test)}")

    # Créer le writer TensorBoard dans runs/grid_search/<run_name>
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = config["paths"]["runs_dir"]
    grid_root = os.path.join(script_dir, runs_dir, "grid_search")
    os.makedirs(grid_root, exist_ok=True)

    run_name = f"grid_search_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(grid_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logs TensorBoard de la grid search dans : {log_dir}")

    # Boucle sur toutes les configs
    for idx, cfg_pack in enumerate(configs_to_test):
        cfg_model = cfg_pack["model_cfg"]
        lr = cfg_pack["lr"]
        wd = cfg_pack["wd"]

        print(f"\n=== Modèle {idx} ===")
        print(f"  Blocks   : {cfg_model['model']['residual_blocks']}")
        print(f"  Dropout  : {cfg_model['model']['dropout']}")
        print(f"  LR       : {lr}")
        print(f"  WD       : {wd}")

        modele = model.build_model(cfg_model)
        modele.to(device)

        optimizer = model.get_optimizer(modele, config, weight_decay=wd, lr=lr)

        epoch_iterator = tqdm(range(num_epochs), desc=f"Model {idx}")

        for epoch in epoch_iterator:
            # ---- TRAIN ----
            modele.train()
            train_loss = 0.0

            for images, labels in data_loader_train:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = modele(images)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"\n[ERREUR] Loss est NaN. Arrêt de l'entraînement pour ce modèle.")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            if torch.isnan(loss):
                final_val_loss = float("nan")
                final_val_accuracy = 0.0
                notes = "Échoué (NaN)"
            else:
                avg_train_loss = train_loss / len(data_loader_train)
                epoch_iterator.set_postfix(train_loss=f"{avg_train_loss:.4f}")

                # ---- VAL ----
                modele.eval()
                val_loss, correct, total = 0.0, 0, 0
                with torch.no_grad():
                    for images, labels in data_loader_val:
                        images, labels = images.to(device), labels.to(device)
                        outputs = modele(images)
                        batch_loss = criterion(outputs, labels)
                        val_loss += batch_loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                final_val_loss = val_loss / len(data_loader_val)
                final_val_accuracy = 100.0 * correct / total if total > 0 else 0.0
                notes = ""

            print(
                f"Résultats -> Val Loss: {final_val_loss:.4f}, "
                f"Val Accuracy: {final_val_accuracy:.2f}%"
            )

            # Logs TensorBoard par modèle et epoch
            tag_prefix = f"Model_{idx}"
            writer.add_scalar(f"{tag_prefix}/Val_Loss", final_val_loss, epoch)
            writer.add_scalar(f"{tag_prefix}/Val_Accuracy", final_val_accuracy, epoch)

            writer.add_scalar(f"{tag_prefix}/LR", lr, epoch)
            writer.add_scalar(f"{tag_prefix}/WD", wd, epoch)
            writer.add_scalar(f"{tag_prefix}/Dropout", cfg_model["model"]["dropout"], epoch)

            # Garder les résultats dans un tableau
            all_results.append(
                {
                    "Model_id": idx,
                    "Epoch": epoch,
                    "Dropout": cfg_model["model"]["dropout"],
                    "Block_config": cfg_model["model"]["residual_blocks"],
                    "LR": lr,
                    "WD": wd,
                    "Val_Accuracy (%)": final_val_accuracy,
                    "Val_Loss": final_val_loss,
                    "Notes": notes,
                }
            )

        modele.cpu()

    writer.close()

    # Tableau final
    results_df = pd.DataFrame(all_results).sort_values(
        by=["Model_id", "Epoch"]
    )
    print("\n\n" + "=" * 50)
    print("        RÉCAPITULATIF FINAL DES PERFORMANCES")
    print("=" * 50)
    print(results_df.to_markdown(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de config YAML")
    args = parser.parse_args()

    # Charger la config YAML
    with open(os.path.join(os.getcwd(), args.config), "r") as f:
        config = yaml.safe_load(f)

    # Datasets (train / test) à partir des fichiers .pt
    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None,
    )
    full_test_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["test"]["chemin"],
        transform=None,
    )

    # Sous-ensembles pour grid search
    train_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=config["grid_search"]["subset_size"]["train"],
        batch_size=config["train"]["batch_size"],
    )
    test_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_test_dataset,
        subset_size=config["grid_search"]["subset_size"]["test"],
        batch_size=config["train"]["batch_size"],
    )

    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    print("Lancement de la mini grid search...")
    mini_grid_search(
        train_loader_subset,
        test_loader_subset,
        config,
        device,
        num_epochs=config["grid_search"]["epochs"],
        criterion=nn.CrossEntropyLoss(),
    )


if __name__ == "__main__":
    main()