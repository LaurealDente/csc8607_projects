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
from src import model, augmentation, data_loading
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd



def mini_grid_search(data_loader_train, data_loader_test, config, device, num_epochs, criterion):
    
    all_results = []

    # Hyperparamètres à tester
    dropout_options = [0.1, 0.3]
    block_config_options = [[2, 2, 2], [3, 3, 3]]

    # Paramètres fixes
    num_classes = config['model']['num_classes']
    use_residual = config['model']['residual']
    use_batch_norm = config['model']['batch_norm']
    activation_fn = config['model']['activation']

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
        optimizer = model.get_optimizer(modele, config, config["grid_search"]["hparams"]["weight_decay"][0], config["grid_search"]["hparams"]["lr"][0])
        
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





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=args.config["dataset"]["split"]["train"]["chemin"]
    )

    full_test_dataset = augmentation.AugmentationDataset(
        data_path=args.config["dataset"]["split"]["test"]["chemin"])

    train_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_train_dataset,
        subset_size=args.config['grid_search']['subset_size']['train'],
        batch_size=args.config['train']['batch_size']
    )
    test_loader_subset = data_loading.create_stratified_subset_loader_manual(
        dataset=full_test_dataset,
        subset_size=args.config['grid_search']['subset_size']['test'],
        batch_size=args.config['train']['batch_size']
    )

    device = torch.device(args.config["train"]["device"] if torch.cuda.is_available() else "cpu")

    # -- Mini Grid Search -- #
    if args.grid_search:
        print("grid_search")
        mini_grid_search(train_loader_subset, test_loader_subset, args.config, device, args.config['grid_search']['epochs'], nn.CrossEntropyLoss())


if __name__ == "__main__":
    main()