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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class AugmentationDataset(Dataset):
    def __init__(self, data_path, column, transform=None):
        data = torch.load(data_path)
        self.images = data[column]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        
        if self.transform:
            image = self.transform(image)
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    train_augmente = AugmentationDataset(data_path=config["dataset"]["split"]["train"]["chemin"], 
                                         column=config["dataset"]["columns"]["image"], 
                                         transform=augmentation_pipeline)

    train_loader = DataLoader(dataset=train_augmente,
        batch_size=config["train"]["batch_size"],
        shuffle=train_augmente,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True
    )

    # train_dataset = ImageNet(root=args.config[""], split="train", transform=transform_features)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

if __name__ == "__main__":
    main()