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
    dropout_options = hparams_cfg["dropout"]
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


def mini_grid_final(data_loader_train, data_loader_val, config, device, num_epochs, criterion):
    """
    Grid finale : part du meilleur modèle (baseline) puis teste
    1) LR plus grande
    2) Weight decay plus grand
    3) Plus de blocs (4,4,4)
    4) Dropout plus grand
    → On ne change qu'un hyperparamètre à la fois.
    """
    from sklearn.metrics import f1_score

    all_results = []

    # Hyperparamètres 'config' pour la grid finale
    base_cfg = config["grid_final"]["base_model"]      # modèle de référence (A)
    lr_base = config["grid_final"]["lr"]
    wd_base = config["grid_final"]["weight_decay"]
    lr_high = config["grid_final"]["lr_high"]
    wd_high = config["grid_final"]["weight_decay_high"]
    blocks_high = config["grid_final"]["blocks_high"]
    dropout_high = config["grid_final"]["dropout_high"]

    num_classes = config["model"]["num_classes"]
    use_residual = config["model"]["residual"]
    use_batch_norm = config["model"]["batch_norm"]
    activation_fn = config["model"]["activation"]

    # Liste ORDonnée des expériences
    experiments = [
        {
            "name": "baseline",
            "residual_blocks": base_cfg["residual_blocks"],
            "dropout": base_cfg["dropout"],
            "lr": lr_base,
            "wd": wd_base,
        },
        {
            "name": "lr_high",
            "residual_blocks": base_cfg["residual_blocks"],
            "dropout": base_cfg["dropout"],
            "lr": lr_high,
            "wd": wd_base,
        },
        {
            "name": "wd_high",
            "residual_blocks": base_cfg["residual_blocks"],
            "dropout": base_cfg["dropout"],
            "lr": lr_base,
            "wd": wd_high,
        },
        {
            "name": "blocks_high",
            "residual_blocks": blocks_high,
            "dropout": base_cfg["dropout"],
            "lr": lr_base,
            "wd": wd_base,
        },
        {
            "name": "dropout_high",
            "residual_blocks": base_cfg["residual_blocks"],
            "dropout": dropout_high,
            "lr": lr_base,
            "wd": wd_base,
        },
    ]

    # Dossier TensorBoard : runs/grid_final/...
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = config["paths"]["runs_dir"]
    grid_root = os.path.join(script_dir, runs_dir, "grid_final")
    os.makedirs(grid_root, exist_ok=True)

    run_name = f"grid_final_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(grid_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logs TensorBoard de la grid finale dans : {log_dir}")

    for idx, exp in enumerate(experiments):
        print(f"\n=== Expérience {idx} : {exp['name']} ===")
        print(f"  Blocks   : {exp['residual_blocks']}")
        print(f"  Dropout  : {exp['dropout']}")
        print(f"  LR       : {exp['lr']}")
        print(f"  WD       : {exp['wd']}")

        cfg_model = {
            "model": {
                "residual_blocks": exp["residual_blocks"],
                "dropout": exp["dropout"],
                "num_classes": num_classes,
                "residual": use_residual,
                "batch_norm": use_batch_norm,
                "activation": activation_fn,
            }
        }

        modele = model.build_model(cfg_model)
        modele.to(device)

        optimizer = model.get_optimizer(
            modele, config,
            weight_decay=exp["wd"],
            lr=exp["lr"]
        )

        tag_prefix = f"{exp['name']}"

        epoch_iterator = tqdm(range(num_epochs), desc=f"{exp['name']}")

        for epoch in epoch_iterator:
            # ---- TRAIN ----
            modele.train()
            train_loss = 0.0
            correct = 0
            total = 0
            all_train_preds = []
            all_train_labels = []

            for images, labels in data_loader_train:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = modele(images)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"\n[ERREUR] Loss train NaN. Arrêt pour {exp['name']}.")
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                optimizer.step()

                batch_size = images.size(0)
                train_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total += batch_size
                correct += (predicted == labels).sum().item()

                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())

            if torch.isnan(loss):
                final_val_loss = float("nan")
                final_val_acc = 0.0
                final_val_f1 = 0.0
                notes = "Échoué (NaN)"
            else:
                avg_train_loss = train_loss / total
                train_acc = correct / total if total > 0 else 0.0
                train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

                epoch_iterator.set_postfix(train_loss=f"{avg_train_loss:.4f}")

                # ---- VAL ----
                modele.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_val_preds = []
                all_val_labels = []

                with torch.no_grad():
                    for images, labels in data_loader_val:
                        images, labels = images.to(device), labels.to(device)
                        outputs = modele(images)
                        batch_loss = criterion(outputs, labels)
                        if torch.isnan(batch_loss):
                            print(f"\n[ERREUR] Loss val NaN pour {exp['name']}.")
                            val_loss = float("nan")
                            break

                        val_loss += batch_loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        all_val_preds.extend(predicted.cpu().numpy())
                        all_val_labels.extend(labels.cpu().numpy())

                if np.isnan(val_loss):
                    final_val_loss = float("nan")
                    final_val_acc = 0.0
                    final_val_f1 = 0.0
                    notes = "Échoué (NaN val)"
                else:
                    final_val_loss = val_loss / val_total
                    final_val_acc = val_correct / val_total if val_total > 0 else 0.0
                    final_val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
                    notes = ""

                # Logs TRAIN
                writer.add_scalar(f"train/{tag_prefix}_loss", avg_train_loss, epoch)
                writer.add_scalar(f"train/{tag_prefix}_accuracy", train_acc, epoch)
                writer.add_scalar(f"train/{tag_prefix}_f1_macro", train_f1, epoch)

                # Logs VAL
                writer.add_scalar(f"val/{tag_prefix}_loss", final_val_loss, epoch)
                writer.add_scalar(f"val/{tag_prefix}_accuracy", final_val_acc, epoch)
                writer.add_scalar(f"val/{tag_prefix}_f1_macro", final_val_f1, epoch)

                # Logs hparams (constants)
                writer.add_scalar(f"hparams/{tag_prefix}_lr", exp["lr"], epoch)
                writer.add_scalar(f"hparams/{tag_prefix}_wd", exp["wd"], epoch)
                writer.add_scalar(f"hparams/{tag_prefix}_dropout", exp["dropout"], epoch)

                all_results.append(
                    {
                        "Exp_name": exp["name"],
                        "Epoch": epoch,
                        "Blocks": exp["residual_blocks"],
                        "Dropout": exp["dropout"],
                        "LR": exp["lr"],
                        "WD": exp["wd"],
                        "Train_Loss": avg_train_loss,
                        "Train_Acc": train_acc,
                        "Train_F1": train_f1,
                        "Val_Loss": final_val_loss,
                        "Val_Acc": final_val_acc,
                        "Val_F1": final_val_f1,
                        "Notes": notes,
                    }
                )

        modele.cpu()

    writer.close()

    results_df = pd.DataFrame(all_results).sort_values(
        by=["Exp_name", "Epoch"]
    )
    print("\n\n" + "=" * 50)
    print("        RÉCAPITULATIF FINAL DES PERFORMANCES (GRID FINALE)")
    print("=" * 50)
    print(results_df.to_markdown(index=False))

    return results_df




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Chemin vers le fichier de config YAML"
    )
    parser.add_argument(
        "--grid_final", action="store_true",
        help="Lance la grid finale (baseline + variations LR/WD/blocs/dropout)"
    )
    args = parser.parse_args()

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

    train_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=config["grid_search"]["subset_size"]["train"],
        batch_size=config["train"]["batch_size"],
    )
    val_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_test_dataset,
        subset_size=config["grid_search"]["subset_size"]["val"],
        batch_size=config["train"]["batch_size"],
    )

    device = torch.device(
        config["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    if args.grid_final:
        print("Lancement de la GRID FINALE (baseline + variations)...")
        mini_grid_final(
            train_loader_subset,
            val_loader_subset,
            config,
            device,
            num_epochs=config["grid_final"]["epochs"],
            criterion=nn.CrossEntropyLoss(),
        )
    else:
        print("Lancement de la MINI GRID SEARCH classique...")
        mini_grid_search(
            train_loader_subset,
            val_loader_subset,
            config,
            device,
            num_epochs=config["grid_search"]["epochs"],
            criterion=nn.CrossEntropyLoss(),
        )



if __name__ == "__main__":
    main()