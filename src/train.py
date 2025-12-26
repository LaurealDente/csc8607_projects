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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from sklearn.metrics import f1_score

def overfitting_small(modele, config):
    print("\n>>> Démarrage du mode OVERFIT SMALL...")
    optimizer = model.get_optimizer(modele, config)
    criterion = nn.CrossEntropyLoss()

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)
    modele.train()

    run_name = f"overfit_small_{time.strftime('%Y%m%d-%H%M%S')}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Correction chemin logs
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["paths"]["runs_dir"])
    tensorboard_path = os.path.join(runs_dir, "overfit_small", run_name)
    writer = SummaryWriter(log_dir=tensorboard_path)

    full_train_dataset = augmentation.AugmentationDataset(
        data_path=config["dataset"]["split"]["train"]["chemin"],
        transform=None # Pas d'augmentation pour l'overfit test (plus stable)
    )

    # On prend un subset très petit
    subset_size = config["train"]["batch_size"] * 2  # ex: 64 images
    overfit_indices = list(range(subset_size))
    overfit_dataset = Subset(full_train_dataset, overfit_indices)

    overfit_loader = DataLoader(
        overfit_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )

    print(f"Taille du sous-ensemble : {subset_size} exemples.")
    print(f"Hyperparamètres : Blocks={config['model']['residual_blocks']}, Dropout={config['model']['dropout']}")
    
    # Overfit epochs
    n_epochs = config["train"].get("overfit_epochs", 50) 
    print(f"Nombre d'époques : {n_epochs}")

    for epoch in range(n_epochs):
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

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Époque {epoch+1}/{n_epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_accuracy:.2%}")

    print("\nEntraînement 'overfit' terminé.")
    print(f"Logs disponibles : {tensorboard_path}")
    writer.close()

def perte_premier_batch(modele, dataloader, config):
    print("\n>>> Vérification PERTE INITIALE...")
    run_name = f"sanity_check_{time.strftime('%Y%m%d-%H%M%S')}"
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["paths"]["runs_dir"])
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, "sanity_check", run_name))
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)
    modele.train() # Mode train pour checker les gradients

    try:
        first_batch_images, first_batch_labels = next(iter(dataloader))
    except StopIteration:
        print("Erreur : Le DataLoader est vide.")
        return

    first_batch_images = first_batch_images.to(device)
    first_batch_labels = first_batch_labels.to(device)

    outputs = modele(first_batch_images)
    loss = criterion(outputs, first_batch_labels)
    perte_observee = loss.item()

    num_classes = config["model"].get("num_classes", 200)
    expected_loss = -math.log(1 / num_classes)
    
    print(f"Perte observée : {perte_observee:.4f}")
    print(f"Perte théorique (-log(1/{num_classes})) : {expected_loss:.4f}")

    writer.add_scalar('Loss/initial_check', perte_observee, 0)
    writer.flush()
    writer.close()

    # Test Backward
    optimizer = model.get_optimizer(modele, config)
    optimizer.zero_grad()
    loss.backward()

    gradient_existe = False
    for name, param in modele.named_parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            gradient_existe = True
            break

    if gradient_existe:
        print("✅ BACKWARD OK : Des gradients non nuls ont été calculés.")
    else:
        print("❌ ERREUR : Aucun gradient calculé (param.grad est None ou 0).")

def train(modele, train_loader, val_loader, config, variant_name="default"):
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    modele.to(device)

    optimizer = model.get_optimizer(modele, config)
    criterion = nn.CrossEntropyLoss()

    epochs = config["train"]["epochs"]
    base_lr = float(config["train"]["optimizer"]["lr"]) # Assurer float

    # Paramètres Scheduler
    warmup_epochs = config["train"].get("warmup_epochs", 5)
    first_cycle_epochs = config["train"].get("first_cycle_epochs", 20)
    t_mult = config["train"].get("t_mult", 2.0)

    # Définition des schedulers
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Le cosine prend le relais après warmup
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_epochs,
        T_mult=int(t_mult)
    )

    # Chemins
    root_dir = os.path.dirname(os.path.dirname(__file__))
    runs_dir = os.path.join(root_dir, config["paths"]["runs_dir"])
    artifacts_dir = os.path.join(root_dir, config["paths"]["artifacts_dir"])
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Nom du run explicite
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = f"{variant_name}_lr{base_lr}_wd{config['train']['optimizer']['weight_decay']}_{timestamp}"
    
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, "train", run_name))

    # Early Stopping
    patience = config["train"].get("early_stopping_patience", 10)
    best_model_name = f"best_of_{variant_name}.ckpt"
    best_model_path = os.path.join(artifacts_dir, best_model_name)
    
    early_stopping = utils.EarlyStopping(
        patience=patience,
        path=best_model_path,
        verbose=True
    )

    max_steps = config["train"].get("max_steps", None)

    print(f"\n>>> Démarrage TRAIN : {run_name}")
    print(f"Device: {device} | Epochs: {epochs} | Patience: {patience}")

    global_step = 0

    for epoch in range(epochs):
        # --- TRAIN LOOP ---
        modele.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_train_preds = []
        all_train_labels = []

        # TQDM pour suivre l'époque
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs} [Train]", leave=False)
        
        for images, labels in pbar:
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
            
            # Stockage pour F1 (attention mémoire sur gros dataset)
            # Pour optimisation, on pourrait calculer F1 batch par batch ou ne pas le faire en train
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item()})

            if max_steps is not None and global_step >= max_steps:
                print("Max steps atteint.")
                break
        
        if max_steps is not None and global_step >= max_steps:
            break

        train_loss = running_loss / total
        train_acc = correct / total
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")

        # --- VAL LOOP ---
        modele.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = modele(images)
                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_size
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        # Affichage Console
        print(
            f"Ep {epoch+1}: "
            f"T.Loss={train_loss:.3f} T.Acc={train_acc:.1%} "
            f"V.Loss={val_loss:.3f} V.Acc={val_acc:.1%} V.F1={val_f1:.3f}"
        )

        # TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/f1_macro", train_f1, epoch)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/f1_macro", val_f1, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)

        # Update Scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step(epoch - warmup_epochs)

        # Early Stopping check
        early_stopping(val_f1, modele)
        if early_stopping.early_stop:
            print(f"Early stopping déclenché à l'époque {epoch+1}.")
            break

    writer.close()
    return modele

def main():
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "configs/config.yaml")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path, help="Chemin vers config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Force une seed aléatoire")
    parser.add_argument("--overfit_small", action="store_true", help="Lance mode overfit sur petit batch")
    parser.add_argument("--max_epochs", type=int, default=None, help="Surcharge le nb d'époques")
    parser.add_argument("--max_steps", type=int, default=None, help="Arrêt après X steps")
    parser.add_argument("--perte_initiale", action="store_true", help="Vérifie la loss avant de commencer")
    parser.add_argument("--final_run", action="store_true", help="Utilise les configs '_final' du yaml")
    
    args = parser.parse_args()

    # 1. Chargement YAML
    try:
        full_path = os.path.abspath(args.config)
        with open(full_path, "r") as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"Fichier de configuration introuvable : {full_path}")

    # 2. Gestion de la Seed (IMPORTANT: Le faire ici pour que ça s'applique partout)
    seed_to_use = args.seed if args.seed is not None else base_config["train"].get("seed", 42)
    utils.set_seed(seed_to_use) # Supposant que vous avez utils.set_seed
    base_config["train"]["seed"] = seed_to_use
    print(f"Seed fixée à : {seed_to_use}")

    # 3. Sélection de la configuration (Normale ou Finale)
    if args.final_run:
        print(">>> MODE : FINAL RUN (train_final, model_final)")
        train_cfg = base_config.get("train_final", base_config["train"])
        model_cfg = base_config.get("model_final", base_config["model"])
        augment_cfg = base_config.get("augment_final", base_config["augment"])
        
        # On remplace les sections principales par les sections finales
        base_config["train"].update(train_cfg)
        base_config["model"].update(model_cfg)
        base_config["augment"].update(augment_cfg)
        
        # Liste des variantes à exécuter (1 seule en mode final)
        variants = {"FinalModel": {}} 
    else:
        print(">>> MODE : CLASSIQUE (Grid Search manuelle via final_test)")
        # On récupère les variantes définies dans model.final_test
        # Si pas de final_test, on fait juste une variante par défaut
        if "final_test" in base_config["model"]:
            variants = base_config["model"]["final_test"]
        else:
            variants = {"Default": {}}

    # 4. Surcharge Arguments CLI (priorité sur le YAML)
    if args.max_epochs is not None:
        base_config["train"]["epochs"] = args.max_epochs
    if args.max_steps is not None:
        base_config["train"]["max_steps"] = args.max_steps
    
    # 5. Boucle d'exécution sur les variantes (A, B, Special...)
    for variant_name, hparams in variants.items():
        print(f"\n{'='*20} Traitement : {variant_name} {'='*20}")
        
        # Copie profonde pour ne pas polluer les itérations suivantes
        current_config = yaml.safe_load(yaml.dump(base_config))
        
        # Application des hyperparams spécifiques à la variante
        if hparams:
            for k, v in hparams.items():
                current_config["model"][k] = v
        current_config["model"]["version_name"] = variant_name
        
        
        # A. Preprocessing (Fixe)
        preprocessing.get_preprocess_transforms(current_config)

        # B. Modèle
        modele = model.build_model(current_config)

        # C. DataLoaders
        aug_pipeline = augmentation.get_augmentation_transforms(current_config)
        train_loader = data_loading.get_dataloaders("train", aug_pipeline, current_config)
        
        # Si on fait juste une perte initiale ou overfit, on n'a pas forcément besoin du val_loader tout de suite
        val_loader = data_loading.get_dataloaders("val", None, current_config)

        # D. Exécution des tâches demandées
        
        # Tâche 1: Sanity Check Loss
        if args.perte_initiale:
            perte_premier_batch(modele, train_loader, current_config)
            continue
        
        # Tâche 2: Overfit Small (Exclusif ou cumulatif selon besoin, ici exclusif souvent mieux)
        if args.overfit_small:
            overfitting_small(modele, current_config)
            continue # On passe à la variante suivante, pas de train complet en mode overfit

        # Tâche 3: Entraînement complet (si on n'est pas en overfit only)
        # Note: Si perte_initiale était True, on continue quand même vers le train
        if not args.overfit_small and not args.perte_initiale:
            train(modele, train_loader, val_loader, current_config, variant_name=variant_name)

if __name__ == "__main__":
    main()
