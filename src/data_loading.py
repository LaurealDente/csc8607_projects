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

    
    # log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../runs")
    # writer = SummaryWriter(log_dir=log_dir)

    particularities = {}

    for split_name, dataset in final_datasets.items():
        labels = dataset["label"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        count_distribution = Counter(counts)
        label_dist_dict = {str(k): v for k, v in count_distribution.items()}
        
        # writer.add_scalars(f"label_distribution/{split_name}", label_dist_dict, global_step=0)
        
        particularities[split_name] = detect_dataset_particularities(dataset)

    # writer.close()
    return final_datasets, particularities


def get_dataloaders(split: str, augmentation_pipeline, config: dict):  # ← split au lieu de dataset
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
    # 1. Vérifier que le dataset a bien une liste de labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError("Le Dataset doit avoir un attribut '.targets' ou '.labels' pour la stratification.")

    # S'assurer que la taille du sous-ensemble est réaliste
    subset_size = min(subset_size, len(dataset))

    # 2. Regrouper les indices de chaque image par label
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    # 3. Calculer combien d'échantillons prendre par classe
    final_indices = []
    total_size = len(dataset)
    
    for label, indices in label_to_indices.items():
        # Proportion de cette classe dans le dataset complet
        proportion = len(indices) / total_size
        # Nombre d'échantillons à prendre pour cette classe dans le sous-ensemble
        num_samples_for_label = int(subset_size * proportion)
        # Assurer qu'on prend au moins un échantillon si la classe est représentée
        num_samples_for_label = max(1, num_samples_for_label)
        
        # 4. Piocher aléatoirement les indices pour cette classe
        # `random.sample` pioche sans remise
        sampled_indices = random.sample(indices, min(len(indices), num_samples_for_label))
        final_indices.extend(sampled_indices)
        
    # 5. Créer le Subset et le DataLoader
    random.shuffle(final_indices) # Mélanger les indices de toutes les classes
    subset_dataset = Subset(dataset, final_indices)
    
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True, # Important de mélanger le sous-ensemble final
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
