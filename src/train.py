import argparse
import yaml
import os
import src.model as model
import src.data_loading as data_loading
import src.augmentation as augmentation
import src.preprocessing as preprocessing
import src.utils as utils
import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, DataLoader
import time
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def overfitting_small(modele, config):
    optimizer = model.get_optimizer(modele, config)
    criterion = nn.CrossEntropyLoss()

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)
    modele.train()

    run_name = f"overfit_small_{time.strftime('%Y%m%d-%H%M%S')}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tensorboard_path = os.path.join(script_dir, "../runs/overfit_small/" + run_name)
    writer = SummaryWriter(log_dir=tensorboard_path)

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None
    )

    subset_size = config["train"]["batch_size"] * 4
    overfit_indices = list(range(subset_size))
    overfit_dataset = Subset(full_train_dataset, overfit_indices)

    overfit_loader = DataLoader(
        overfit_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )

    print(f"Taille du sous-ensemble : {subset_size} exemples.")
    print(f"Hyperparamètres modèle : B={config['model']['residual_blocks']}, dropout={config['model']['dropout']}")
    print(f"Optimisation : LR={config['train']['optimizer']['lr']}, Weight Decay={config['train']['optimizer']['weight_decay']}")
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


def perte_premier_batch(modele, dataloader, config):
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


def train(modele, train_loader, val_loader, config):
    """
    Entraînement final sur l'ensemble du dataset avec CosineAnnealingLR,
    early stopping, et sauvegarde du meilleur modèle dans artifacts/best.ckpt.
    """
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)

    optimizer = model.get_optimizer(modele, config)
    criterion = nn.CrossEntropyLoss()

    # Scheduler CosineAnnealing pour l'entraînement complet
    epochs = config["train"]["epochs"]
    base_lr = optimizer.param_groups[0]["lr"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=base_lr / 50.0
    )

    runs_dir = config["paths"]["runs_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    run_name = f"train_{time.strftime('%Y%m%d-%H%M%S')}"

    train_root = os.path.join(runs_dir, "train")
    os.makedirs(train_root, exist_ok=True)

    log_dir = os.path.join(train_root, run_name)
    writer = SummaryWriter(log_dir=log_dir)

    # Early stopping + checkpoint
    patience = config["train"].get("early_stopping_patience", 10)
    best_model_path = os.path.join(artifacts_dir, "best.ckpt")  # chemin demandé
    early_stopping = utils.EarlyStopping(patience=patience, path=best_model_path)

    max_steps = config["train"].get("max_steps", None)

    print(f"Début de l'entraînement sur {device} pour {epochs} époques (early stopping patience={patience}).")
    print(f"LR initiale = {base_lr}, scheduler = CosineAnnealingLR(T_max={epochs}, eta_min={base_lr/50.0})")

    global_step = 0

    for epoch in range(epochs):
        modele.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = modele(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = running_loss / total
        train_acc = correct / total

        modele.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = modele(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_size
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # Logging epoch
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Early stopping sur la val_loss
        early_stopping(val_loss, modele)
        if early_stopping.early_stop:
            print("Early stopping déclenché.")
            break

        # Step du scheduler après l'epoch
        scheduler.step()

    print("Chargement du meilleur modèle sauvegardé...")
    modele.load_state_dict(torch.load(best_model_path))
    writer.close()
    return modele


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
    args = parser.parse_args()

    try:
        with open(os.path.join(os.getcwd(), args.config), "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        raise Exception("Mauvais chemin du fichier de configuration : " + os.path.join(os.getcwd(), args.config))

    # Fixer ici la configuration finale pour le rapport
    config["train"]["optimizer"]["lr"] = 1e-4
    config["train"]["optimizer"]["weight_decay"] = 1e-4
    config["train"]["batch_size"] = 32
    config["train"]["epochs"] = 100

    # Choix du modèle A ou B en fonction d'un champ de config
    # Par exemple: config["model"]["version"] = "A" ou "B"
    version = config["model"].get("version", "A")
    if version == "A":
        config["model"]["dropout"] = 0.1
        config["model"]["residual_blocks"] = [3, 3, 3]
    else:
        config["model"]["dropout"] = 0.1
        config["model"]["residual_blocks"] = [2, 2, 2]

    preprocessing.get_preprocess_transforms(config)

    modele = model.build_model(config)

    augmentation_pipeline = augmentation.get_augmentation_transforms(config)

    train_loader = data_loading.get_dataloaders("train", augmentation_pipeline, config)
    val_loader = data_loading.get_dataloaders("val", None, config)

    if args.perte_initiale:
        _ = perte_premier_batch(modele, train_loader, config)

    if args.overfit_small:
        overfitting_small(modele, config)
    else:
        train(modele, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
