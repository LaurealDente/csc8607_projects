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
from torch.utils.tensorboard import SummaryWriter
import numpy as np


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


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """

    try:
        tiny_imagenet = datasets.load_dataset(config["dataset"]["root"])
        
        split_dataset = tiny_imagenet["train"].train_test_split(test_size=config["dataset"]["split"]["test"]["p"], 
                                            seed=config["train"]["seed"], 
                                            stratify_by_column=config["dataset"]["columns"]["label"])
        
        final_datasets = datasets.DatasetDict({
            "train": split_dataset["train"],
            "valid": tiny_imagenet["valid"],
            "test": split_dataset["test"]
        })

        
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
    
    except :
        raise NotImplementedError("get_dataloaders doit être implémentée par l'étudiant·e.")


    
if __name__ == "__main__":
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_dataloaders(config)
