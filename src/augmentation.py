"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""
from torchvision import transforms
from torch.utils.data import Dataset
import yaml
import os
import torch


class AugmentationDataset(Dataset):
    def __init__(self, data_path, column, transform=None):
        data = torch.load(data_path, mmap=True, map_location="cpu", weights_only=False)
        self.images = data[column]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        
        if self.transform:
            image = self.transform(image)
        return image
    

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation. À implémenter."""   
    transformations = []
    if config["augment"]["random_flip"]:
        transformations.append(transforms.RandomHorizontalFlip(p=0.5))

    if config["augment"]["random_crop"] :
        transformations.append(transforms.RandomResizedCrop(
            size=config["augment"]["random_crop"]["size"], 
            scale=config["augment"]["random_crop"]["scale"], 
            ratio=config["augment"]["random_crop"]["ratio"]
        ))
    
    if config["augment"]["color_jitter"] :
        transformations.append(transforms.ColorJitter(
            **config["augment"]["color_jitter"]
        ))
    augmentation_pipeline = transforms.Compose(transformations)
    return augmentation_pipeline



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    augmentation_pipeline = get_augmentation_transforms(config)