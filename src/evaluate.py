"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

"""
Évaluation du modèle sur le jeu de test.

Exécution :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
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


def build_model_from_config(config):
    """
    Reconstruit le même modèle qu'en train.py.
    Si tu utilises train_final/model_final, la config doit déjà être cohérente.
    """
    # Ici on suppose que train.py a déjà fusionné train/train_final, model/model_final, etc.
    # Si ce n'est pas le cas, reproduis la logique de fusion de train.py ici.
    net = model.build_model(config)
    return net


def get_test_loader(config):
    """
    Récupère le DataLoader du jeu de test sans augmentation (préprocessing uniquement).
    """
    # S'assure que les transforms de préprocessing sont bien initialisées
    preprocessing.get_preprocess_transforms(config)

    # Pas d'augmentation en test
    test_loader = data_loading.get_dataloaders(
        split="test",
        augment_transforms=None,
        config=config
    )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Chemin vers le fichier de configuration YAML.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint (.ckpt) du meilleur modèle.")
    args = parser.parse_args()

    # 1. Charger la config
    config_path = os.path.join(os.getcwd(), args.config)
    config = load_config(config_path)

    # 2. Choix du device
    device = torch.device(config["train"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Évaluation sur device : {device}")

    # 3. Construire le modèle
    net = build_model_from_config(config)

    # 4. Charger le checkpoint
    checkpoint_path = os.path.join(os.getcwd(), args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    print(f"Chargement du checkpoint : {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)

    # 5. Dataloader de test
    test_loader = get_test_loader(config)

    # 6. Évaluation
    test_loss, test_acc, test_f1, cm, y_true, y_pred = evaluate(net, test_loader, device)

    # 7. Affichage des résultats
    print("\n===== Résultats sur le jeu de TEST =====")
    print(f"Test loss      : {test_loss:.4f}")
    print(f"Test accuracy  : {test_acc:.2%}")
    print(f"Test F1 macro  : {test_f1:.4f}")

    print("\nMatrice de confusion (shape = {}):".format(cm.shape))
    print(cm)

    # Optionnel : rapport détaillé si besoin
    try:
        print("\nRapport de classification :")
        print(classification_report(y_true, y_pred, digits=4))
    except Exception:
        pass


if __name__ == "__main__":
    main()
