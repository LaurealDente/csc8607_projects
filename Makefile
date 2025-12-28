PYTHON := python
CONFIG := configs/config.yaml
CHECKPOINT := artifacts/best.ckpt

.PHONY: install sanity_check overfit train final_run lr_finder grid_search eval clean

install:
	$(PYTHON) -m pip install -r requirements.txt

# Vérification rapide (loss initiale + backward + chargement données)
sanity_check:
	$(PYTHON) -m src.train --config $(CONFIG) --perte_initiale --charge_datasets

# Test de sur-apprentissage sur petit batch
overfit:
	$(PYTHON) -m src.train --config $(CONFIG) --overfit_small

# Entraînement standard
train:
	$(PYTHON) -m src.train --config $(CONFIG)

# Entraînement final optimisé
final_run:
	$(PYTHON) -m src.train --config $(CONFIG) --final_run

lr_finder:
	$(PYTHON) -m src.lr_finder --config $(CONFIG)

grid_search:
	$(PYTHON) -m src.grid_search --config $(CONFIG)

eval:
	$(PYTHON) -m src.evaluate --config $(CONFIG) --checkpoint $(CHECKPOINT)

clean:
	rm -rf runs/* artifacts/* __pycache__ src/__pycache__
