import numpy as np
import yaml
import os
import torch
import json 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.augmentation as augmentation
import src.preprocessing as preprocessing
from src.classes import i2d




def comparaison(original_img, augmented_img, label, index):
    """
    Affiche une image originale (PIL) et sa version augmentée (tenseur) côte à côte.
    """

    infos_path = os.path.join(os.path.dirname(__file__), 'dataset_infos.json')

    with open(infos_path, 'r') as f:
        dataset_infos = json.load(f)
    label = i2d[dataset_infos["Maysee--tiny-imagenet"]["features"]["label"]["names"][label]]

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"Comparaison pour l'image {index} - Label: {label}", fontsize=14)
    
    # --- Image Originale ---
    plt.subplot(1, 2, 1)
    original_img = original_img.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(original_img, 0, 1))
    plt.title("Originale")
    plt.axis('off')
    
    # --- Image Augmentée ---
    plt.subplot(1, 2, 2)
    augmented_img = augmented_img.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(np.clip(augmented_img, 0, 1))
    plt.title("Augmentée")
    plt.axis('off')


    plt.tight_layout()
    plt.show()


def sanity_check(augmentation_pipeline):
    """Observer un batch ainsi que visualiser des exemples d'images modifiées"""
    chemin = os.path.normpath(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), 'data', 'preprocessed_dataset_train.pt'))

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    train = torch.load(chemin, map_location=device, weights_only=False)
    label = train[config["dataset"]["columns"]["label"]]
    train = train[config["dataset"]["columns"]["image"]]

    train_augmente = augmentation.AugmentationDataset(data_path=config["dataset"]["split"]["train"]["chemin"],
                                         transform=augmentation_pipeline)
    
    indices_to_check = np.random.choice(len(train), 3, replace=False) 
    for i in indices_to_check:
        original_image = train[i]
        augmented_image = train_augmente[i]
        label_index = label[i]
        
        comparaison(original_image, augmented_image, label_index, index=i)

    loader_augmente = DataLoader(train_augmente, 
                                 batch_size=config["train"]["batch_size"], 
                                 shuffle=False, 
                                 num_workers=config["dataset"]["num_workers"])
    
    premier_batch  = next(iter(loader_augmente))

    print(premier_batch.shape)
   




if __name__ == "__main__" :
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    augmentation_pipeline = augmentation.get_augmentation_transforms(config)
    preprocessing.get_preprocess_transforms(config)
    sanity_check(augmentation_pipeline)