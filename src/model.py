"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from torchvision import transforms
import yaml
import os


def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config. À implémenter."""
    model = []
    if config["augment"]["random_flip"]:
        model.append(model.RandomHorizontalFlip(p=0.5))

    if config["augment"]["random_crop"] is not None:
        model.append(model.RandomResizedCrop(size=config["augment"]["random_crop"]))
    
    if config["augment"]["color_jitter"] is not None:
        model.append(model.ColorJitter(**config["augment"]["color_jitter"]))


    augmentation_pipeline = transforms.Compose(model)
    return augmentation_pipeline


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    build_model(config)