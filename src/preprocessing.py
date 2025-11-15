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
import src.data_loading as data_loading


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
    
    final_datasets, particularities  = data_loading.get_data(config)
    normalized_datasets = []

    for dataset in final_datasets :
        if dataset == "train":
            normalized, mean, std = preprocess_dataset(final_datasets[dataset]["image"])
            writer = SummaryWriter('csc8607_projects/runs/normalization_check')

            r_values = normalized[:, 0, :, :].detach().cpu().numpy().flatten()
            g_values = normalized[:, 1, :, :].detach().cpu().numpy().flatten()
            b_values = normalized[:, 2, :, :].detach().cpu().numpy().flatten()

            writer.add_histogram('Distribution/Red Channel', r_values, global_step=0)
            writer.add_histogram('Distribution/Green Channel', g_values, global_step=0)
            writer.add_histogram('Distribution/Blue Channel', b_values, global_step=0)

            writer.close()
        else :
            normalized, mean, std = preprocess_dataset(final_datasets[dataset]["image"], mean, std)

        
        list_of_labels = final_datasets[dataset]['label'] 
        labels_tensor = torch.tensor(list_of_labels, dtype=torch.int64)
        
        normalized_datasets.append(normalized)
        data_loading.save_dataset(normalized, labels_tensor, dataset)

    return normalized_datasets


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_preprocess_transforms(config)
