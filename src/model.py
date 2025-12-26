"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

from torch import nn
import torch
import yaml
import os

class BlocResiduel(nn.Module):
    """
    Implémente un bloc résiduel standard avec deux convolutions 3x3.
    Gère la projection de la connexion résiduelle si les dimensions changent.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.1, fonction_activation=nn.ReLU, fonction = "relu", residual = True, batch_norm = True):
        super(BlocResiduel, self).__init__()

        self.residual = residual

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)]
        if batch_norm :
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(fonction_activation(inplace=True if fonction == "relu" else False))
        layers.append(nn.Dropout2d(p=dropout_p))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        if batch_norm :
            layers.append(nn.BatchNorm2d(out_channels))

        self.main_path = nn.Sequential(*layers)

        self.shortcut = nn.Sequential() 
        
        if stride != 1 or in_channels != out_channels:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            if batch_norm :
                layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*layers)
        
        self.final_relu = fonction_activation(inplace=True if fonction == "relu" else False)

    def forward(self, x):
        if not self.residual:
            return self.final_relu(self.main_path(x))
        
        residual_out = self.shortcut(x)
        
        out = self.main_path(x)
        
        out += residual_out
        
        out = self.final_relu(out)
        
        return out


class ResNet(nn.Module):
    """
    Assemble le réseau ResNet complet en utilisant les BlocResiduel.
    """
    def __init__(self, B1, B2, B3, dropout_p=0.1, num_classes=200,
                 residual=True, batch_norm=True, fonction="relu"):
        super(ResNet, self).__init__()
        
        fonction_activation = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "selu": nn.SELU
        }

        fonction_activation = fonction_activation[fonction]
        
        self.in_channels = 64
        
        layers = [nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),]
        if batch_norm:
            layers.append(nn.BatchNorm2d(self.in_channels))
        layers.append(fonction_activation(inplace=True if fonction == "relu" else False))

        self.initiale = nn.Sequential(*layers)

        self.stage1 = self._make_stage(out_channels=64, num_blocks=B1, stride=1, dropout_p=dropout_p, fonction_activation = fonction_activation, fonction = fonction, residual=residual, batch_norm = batch_norm)
        self.stage2 = self._make_stage(out_channels=128, num_blocks=B2, stride=2, dropout_p=dropout_p, fonction_activation = fonction_activation, fonction = fonction, residual=residual, batch_norm = batch_norm)
        self.stage3 = self._make_stage(out_channels=256, num_blocks=B3, stride=2, dropout_p=dropout_p, fonction_activation = fonction_activation, fonction = fonction, residual=residual, batch_norm = batch_norm)


        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def _make_stage(self, out_channels, num_blocks, stride, dropout_p, fonction_activation=nn.ReLU, fonction = "relu", residual = True, batch_norm = True):
        """
        Fonction utilitaire qui construit un stage en empilant num_blocks de BlocResiduel.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BlocResiduel(self.in_channels, out_channels, stride=s, dropout_p=dropout_p, fonction_activation = fonction_activation, fonction = fonction, residual = residual, batch_norm = batch_norm))

            self.in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initiale(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config. À implémenter."""
    modele = ResNet(*config["model"]["residual_blocks"], config["model"]["dropout"],
                    config["model"]["num_classes"], config["model"]["residual"],
                    config["model"]["batch_norm"], fonction=config["model"]["activation"])
    return modele


def get_optimizer(model, config, weight_decay=None, lr=None):
    """
    Crée un optimiseur.
    Accepte weight_decay et lr comme arguments optionnels pour surcharger la config.
    """
    wd_to_use = weight_decay if weight_decay is not None else config["train"]["optimizer"].get("weight_decay", 0)
    lr_to_use = lr if lr is not None else config["train"]["optimizer"]["lr"]

    optimizer_name = config["train"]["optimizer"].get("name", "AdamW")
    
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr_to_use, weight_decay=wd_to_use)
    elif optimizer_name == 'SGD':
        momentum = config["train"]["optimizer"].get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr_to_use, weight_decay=wd_to_use, momentum=momentum)
    else:
        return torch.optim.Adam(model.parameters(), lr=float(lr_to_use), weight_decay=float(wd_to_use))
    



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    resnet = build_model(config)
    print(resnet)
    print()