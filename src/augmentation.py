"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torch
import yaml
import os

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation. À implémenter."""
    
     
    transformations = []
    if config["augment"]["random_flip"]:
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))

    if config["augment"]["random_crop"] is not None:
        transformations.append(transforms.RandomResizedCrop(size=config["augment"]["random_crop"]))
    
    if config["augment"]["color_jitter"] is not None:
        transformations.append(transforms.ColorJitter(**config["augment"]["color_jitter"]))

    augmentation_pipeline = transforms.Compose(transformations)
    train = torch.load(config["dataset"]["split"]["train"]["chemin"])
    train_augmente = augmentation_pipeline(train[config["dataset"]["columns"]["image"]])
    
    return train_augmente

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_augmentation_transforms(config)