"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""
import datasets
import yaml
import os
from collections import Counter
import numpy as np
from collections import defaultdict
from src import augmentation, preprocessing
from torch.utils.data import Subset, DataLoader, Dataset
import random


def detect_dataset_particularities(dataset):
    sizes = set()
    labels = set()
    multi_labels_detected = False
    channels = set()

    for sample in dataset:
        
        image = sample['image']
        label = sample['label']
        sizes.add(image.size)

        channels.add(image.mode)

        if isinstance(label, (list, tuple)):
            multi_labels_detected = True
            labels.update(label)
        else:
            labels.add(label)

    particularities = {}
    particularities['unique_image_sizes'] = sizes
    particularities['unique_labels'] = labels
    particularities['multi_labels_detected'] = multi_labels_detected
    particularities['different_image_sizes'] = (len(sizes) > 1)
    particularities['unique_image_modes'] = channels

    return particularities


def get_data(config: dict):
    """
    Crée et retourne les datasets d'entraînement/validation/test.
    """
    tiny_imagenet = datasets.load_dataset(config["dataset"]["root"])
    
    split_dataset = tiny_imagenet["train"].train_test_split(test_size=config["dataset"]["split"]["test"]["p"], 
                                        seed=config["train"]["seed"], 
                                        stratify_by_column=config["dataset"]["columns"]["label"])
    
    final_datasets = datasets.DatasetDict({
        "train": split_dataset["train"],
        "valid": tiny_imagenet["valid"],
        "test": split_dataset["test"]
    })

    for dataset in final_datasets:
        preprocessing.save_dataset(final_datasets[dataset][config["dataset"]["columns"]["image"]], final_datasets[dataset][config["dataset"]["columns"]["label"]], dataset)

    particularities = {}

    for split_name, dataset in final_datasets.items():
        labels = dataset["label"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        count_distribution = Counter(counts)
        label_dist_dict = {str(k): v for k, v in count_distribution.items()}
        
        particularities[split_name] = detect_dataset_particularities(dataset)

    return final_datasets, particularities


def get_dataloaders(split: str, augmentation_pipeline, config: dict):
    """
    Crée un DataLoader pour un split spécifique (train/val/test).
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_path = os.path.join(script_dir, config["dataset"]["split"][split]["chemin"])
    
    
    transform = augmentation_pipeline if split == "train" else None
    
    data = augmentation.AugmentationDataset(data_path=data_path, transform=transform)
    
    data_loader = DataLoader(
        data, 
        batch_size=config["train"]["batch_size"], 
        shuffle=(split == "train"),
        num_workers=config["dataset"]["num_workers"]
    )
    return data_loader



def create_stratified_subset_loader_manual(
    dataset: Dataset, 
    subset_size: int, 
    batch_size: int, 
    num_workers: int = 0
    ) -> DataLoader:
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError("Le Dataset doit avoir un attribut '.targets' ou '.labels' pour la stratification.")


    subset_size = min(subset_size, len(dataset))
    
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)


    final_indices = []
    total_size = len(dataset)
    
    for label, indices in label_to_indices.items():
        proportion = len(indices) / total_size
        num_samples_for_label = int(subset_size * proportion)
        num_samples_for_label = max(1, num_samples_for_label)
        
        sampled_indices = random.sample(indices, min(len(indices), num_samples_for_label))
        final_indices.extend(sampled_indices)
        
    random.shuffle(final_indices)
    subset_dataset = Subset(dataset, final_indices)
    
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    print(f"Création d'un sous-ensemble stratifié manuel de {len(subset_dataset)} échantillons.")
    return subset_loader


    
if __name__ == "__main__":
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_dataloaders(config)
