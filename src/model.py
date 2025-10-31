"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import os
import yaml

def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config. À implémenter."""
        

    raise NotImplementedError("build_model doit être implémentée par l'étudiant·e.")



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    build_model(config)