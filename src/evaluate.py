"""
Évaluation du modèle sur le jeu de test.

Exécution :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/bestof_Modele_A.ckpt --model A
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/bestof_Modele_B.ckpt --model B  
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/bestof_Special.ckpt --model Special
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import src.model as model
import src.data_loading as data_loading
import src.preprocessing as preprocessing


def load_config(config_path: str):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model_from_config(config, model_variant: str = "A"):
    """
    Reconstruit le modèle selon le variant spécifié (A, B, Special).
    """
    # Vérifier que le variant existe
    model_variants = config["model"]["final_test"]
    if model_variant not in model_variants:
        available = list(model_variants.keys())
        raise ValueError(f"Modèle '{model_variant}' inconnu. Disponibles : {available}")
    
    print(f">>> Construction du modèle '{model_variant}' : {model_variants[model_variant]}")
    
    # Copier la config et appliquer les hyperparams du variant
    config_variant = yaml.safe_load(yaml.dump(config))
    hparams = model_variants[model_variant]
    
    config_variant["model"]["dropout"] = hparams["dropout"]
    config_variant["model"]["residual_blocks"] = hparams["residual_blocks"]
    config_variant["model"]["version"] = model_variant
    
    net = model.build_model(config_variant)
    return net


def get_test_loader(config):
    """DataLoader test sans augmentation."""
    preprocessing.get_preprocess_transforms(config)
    test_loader = data_loading.get_dataloaders("test", None, config)
    return test_loader


def evaluate(model_net, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model_net.to(device)
    model_net.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model_net(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= total
    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    return test_loss, test_acc, test_f1, cm, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle sur test set")
    parser.add_argument("--config", type=str, required=True, help="Chemin config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin checkpoint")
    parser.add_argument("--model", type=str, default="A", 
                       choices=["A", "B", "Special"],
                       help="Variant modèle (A/B/Special). Défaut: A")
    args = parser.parse_args()

    # 1. Charger config
    config_path = os.path.join(os.getcwd(), args.config)
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation sur {device} - Modèle: {args.model}")

    # 2. Construire modèle selon variant
    net = build_model_from_config(config, args.model)

    # 3. Charger checkpoint
    checkpoint_path = os.path.join(os.getcwd(), args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    
    print(f"Chargement checkpoint : {args.checkpoint}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)

    # 4. Test loader
    test_loader = get_test_loader(config)

    # 5. Évaluer
    test_loss, test_acc, test_f1, cm, y_true, y_pred = evaluate(net, test_loader, device)

    # 6. Résultats
    print("\n" + "="*50)
    print("RÉSULTATS SUR LE JEU DE TEST")
    print("="*50)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.3%}")
    print(f"Test F1 Macro : {test_f1:.4f}")
    print(f"Test samples  : {len(y_true)}") 
    print("="*50)

    print(f"\nMatrice de confusion ({cm.shape[0]} classes):")
    print(cm[:10, :10], "..." if cm.shape[0] > 10 else "")  # Top-left corner

    print("\nRapport détaillé :")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
