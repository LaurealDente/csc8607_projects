"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loading


def save_dataset(images, labels, dataset):

    dataset_to_save = {
        "image": images,
        "label": labels 
    }
    torch.save(dataset_to_save, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), "preprocessed_dataset_" 
                + dataset 
                + ".pt"))

    print("Data saved : " +  os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"), "preprocessed_dataset_" 
                + dataset 
                + ".pt"))



def preprocess_dataset(list_pil_img, mean=None, std=None):
    base_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor()
    ])
    tensors = [base_transforms(img) for img in list_pil_img]
    batch_tensor = torch.stack(tensors)

    if mean is None or std is None:
        mean = torch.mean(batch_tensor, dim=[0, 2, 3])
        std = torch.std(batch_tensor, dim=[0, 2, 3])

    normalizer = transforms.Normalize(mean=mean, std=std)
    normalized_batch = normalizer(batch_tensor)

    return normalized_batch, mean, std


def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement. À implémenter."""
    
    final_datasets, particularities = data_loading.get_data(config)
    
    # Préprocess chaque split + sauvegarde
    for dataset_name in final_datasets:
        dataset = final_datasets[dataset_name]
        
        if dataset_name == "train":
            normalized, mean, std = preprocess_dataset(dataset["image"])
        else:
            normalized, _, _ = preprocess_dataset(dataset["image"], mean, std)
        
        labels_tensor = torch.tensor(dataset['label'], dtype=torch.int64)
        save_dataset(normalized, labels_tensor, dataset_name)
    
    
    print("✓ Données prétraitées sauvées sur disque")
    return None


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(script_dir)
    sys.path.append(script_dir)
    config_path = os.path.join(script_dir, "configs/config.yaml")


    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_preprocess_transforms(config)
