# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : _Lauret, Alexandre_
- **Projet** : _Projet 23 (tiny imagenet × Blocs résiduels + Dropout2d dans les bloc)_
- **Dépôt Git** : _URL publique_
- **Environnement** : `python == 3.10.18`, `torch == 2.5.1`, `cuda == 12.0.140`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) : https://huggingface.co/datasets/zh-plus/tiny-imagenet
- **Type d’entrée** (image / texte / audio / séries) : images
- **Tâche** (multiclasses, multi-label, régression) : classification multiclasses
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : 3x64x64
- **Nombre de classes** (`meta["num_classes"]`) : 200

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

Le dataset que je vais utiliser est zh-plus/tiny-imagenet stocké sur HuggingFace Datasets.
Celui-ci est composé de 110,000 lignes et de 2 colonnes. La première colonne, nommée "image", sont les images. La deuxième colonne, nommée label, est composée des labels de chacune de ces images.


### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
|------:|----------:|--------------------------------------------------------|
| Train |           |                                                        |
| Val   |           |                                                        |
| Test  |           |                                                        |

**D2.** Donnez la taille de chaque split et le nombre de classes. 

Le dataset est composé de deux split, le premier est celui d'entraînement (train), le deuxième est le split de validation (valid). Dans chacun des deux splits, nous pouvons trouver l'ensemble des 200 classes.

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

Le dataset proposait déjà deux datasets un train et un de validation. J'ai créé un split de test à partir du dataset de train.
Pour cela, j'ai utilisé un ratio de 0.1 afin d'avoir un même nombre de valeurs entre le split de test et de validation qui ont 1000 lignes.
Le split de train a alors 9000 lignes restantes pour l'entraînement.

Pour la stratification, j'ai ciblé les valeurs de la colonne label afin d'équilibrer les trois splits. Ceci permettra un apprentissage de qualité et une meilleure évaluation du modèle.

La seed utilisée est stockée dans le fichier de configuration la variable par défaut était 42, valeur que j'ai utilisée.


**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.  

Grâce à la répartition égale de chaque classe (même nombre de label), nous allons pouvoir entraîner chacune des classes équitablement.
Cela permettra d'avoir une meilleure précision sur la matrice de confusion.


**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

Une particularité est détectée après traitement. Les tailles des images sont égales avec un format de 64x64 avec 3 channel (RGB). Sauf 2% des images qui n'ont qu'un seul canal (L).

Les longeurs de variables sont donc aussi similaires.
Les labels sont tous des entiers entre 0 et 199, il n'y a aucun multi labels.


### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = , center-crop = None, normalize = (mean=[0.4802, 0.4480, 0.3974], std=[0.2765, 0.2689, 0.2816])

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?  

Les prétraitements appliqués au dataset sont une convertion de toutes les images en format RGB car 2% sont stockées en format L.
La deuxième transformation est la conversion en tensor. Les valeurs des tensors sont comprises entre 0 et 1, le calcul de la moyenne et de l'écart type permet de normaliser ces tensors. La normalisation va permettre d'avoir des entrées sur des échelles similaires. Elle accélère la convergence du modèle et stabilise les calculs numériques.

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

Les prétraitrements sont similaires entre les trois datasets, la normalisation est calculée sur les moyennes et les ecarts types du datset train afin de ne pas faire de data leaking.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - Flip horizontal p=0.5, RandomResizedCrop scale=100%, ratio=1 

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  

Afin d'augmenter la volumétrie des données d'entraînement, j'ai utilisé le random flip, le random crop et le color jitter.
Le random flip est défini à 0.5 dans le code pour retourner aléatoirement l'image lors de l'entraînement avec une probabilité d'1/2. Il est possible de l'utiliser sur les images de notre dataset car le modèle ne se focalise pas sur des textes ou équivalents nécessitant un sens défini.
Le random crop permet de ne prendre qu'une partie de l'image de base pour entraîner le modèle sur des centrages différents. Ici les images sont en format 64x64 et le crop est réglé à 64x64, ça ne modifiera pas la photo.
Le color jitter permet d'influencer les paramètres de la photo représentant des contextes photographiques pouvant être changeants. La luminosité, le contrast, la saturation et la teinte de manière aléatoire entre x1.2 et x0.8 car tous les paramètres sont réglés à 0.2. 


**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

Les labels sont conservés pour chacune des augmentations, en effet ces modifications s'appliqueront sur l'image récupérées de base lors du DataLoader. Au cours de cet appel 

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `_____` → score = `_____`
- **Prédiction aléatoire uniforme** — Métrique : `_____` → score = `_____`  
_Commentez en 2 lignes ce que ces chiffres impliquent._

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR    | WD     | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
|---------------------|-------|--------|-------|-------|-------------------------|----------|-------|
|                     |       |        |       |       |                         |          |       |
|                     |       |        |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


