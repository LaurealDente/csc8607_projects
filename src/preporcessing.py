"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""
import data_loading
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import torch
import pandas as pd

# def normalisation_dataset(list_pil_img, flag) :
#     formatted_pil = []
    
#     for pil_img in list_pil_img:
#         if pil_img.mode == 'L':
#             pil_img = pil_img.convert('RGB')
#         image_array = np.array(pil_img, dtype=np.float32) / 255.0
#         formatted_pil.append(image_array)

#     all_images = np.stack(formatted_pil, axis=0)

#     if flag:
#         MEAN = np.mean(all_images, axis=(0, 1, 2))
#         STD_DEVIATION = np.std(all_images, axis=(0, 1, 2))
#     normalized_images = (all_images - MEAN) / STD_DEVIATION
      
#     writer = SummaryWriter('runs/normalization_check')

#     r_values = normalized_images[:, :, :, 0].flatten()
#     g_values = normalized_images[:, :, :, 1].flatten()
#     b_values = normalized_images[:, :, :, 2].flatten()

#     writer.add_histogram('Normalized/Red Channel', r_values, 0)
#     writer.add_histogram('Normalized/Green Channel', g_values, 0)
#     writer.add_histogram('Normalized/Blue Channel', b_values, 0)

#     writer.close()
    
#     return normalized_images

def preprocess_dataset(list_pil_img, mean=None, std=None):
    base_transforms = T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.ToTensor()
    ])
    tensors = [base_transforms(img) for img in list_pil_img]
    batch_tensor = torch.stack(tensors)

    if mean is None or std is None:
        mean = torch.mean(batch_tensor, dim=[0, 2, 3])
        std = torch.std(batch_tensor, dim=[0, 2, 3])

    normalizer = T.Normalize(mean=mean, std=std)
    normalized_batch = normalizer(batch_tensor)

    image_tensor = normalized_batch[0] 
    image_reshaped = image_tensor.permute(1, 2, 0).reshape(-1, 3)

    df_image = pd.DataFrame(
        image_reshaped.detach().cpu().numpy(),
        columns=['Red', 'Green', 'Blue']
    )

    return normalized_batch, mean, std


def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement. À implémenter."""
    
    final_datasets, particularities  = data_loading.get_dataloaders(config)
    normalized_datasets = []

    for dataset in final_datasets :
        if dataset == "train":
            normalized, mean, std = preprocess_dataset(final_datasets[dataset]["image"])
            writer = SummaryWriter('runs/normalization_check')

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

        dataset_to_save = {
            'images': normalized,
            'labels': labels_tensor 
        }
        torch.save(dataset_to_save, "../data/preprocessed_dataset_" + dataset + ".pt")
        normalized_datasets.append(normalized)

    return normalized_datasets


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    get_preprocess_transforms(config)
