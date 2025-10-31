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
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # train_dataset = ImageNet(root=args.config[""], split="train", transform=transform_features)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

if __name__ == "__main__":
    main()