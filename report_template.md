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
  - Flip horizontal p=0.5, RandomResizedCrop size=64x64, scale=80-100%, ratio=1 

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  

Afin d'augmenter la volumétrie des données d'entraînement, j'ai utilisé le random flip, le random crop et le color jitter.
Le random flip est défini à 0.5 dans le code pour retourner aléatoirement l'image lors de l'entraînement avec une probabilité d'1/2. Il est possible de l'utiliser sur les images de notre dataset car le modèle ne se focalise pas sur des textes ou équivalents nécessitant un sens défini.
Le random crop permet de ne prendre qu'une partie de l'image de base pour entraîner le modèle sur des centrages différents. Ici les images sont cropées au maximum de 20% qui va permettre d'éviter la perte complète de l'objet examiné.
Le color jitter permet d'influencer les paramètres de la photo représentant des contextes photographiques pouvant être changeants. La luminosité, le contrast, la saturation et la teinte de manière aléatoire entre x1.2 et x0.8 car tous les paramètres sont réglés à 0.2. 


**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

Les labels sont conservés pour chacune des augmentations, en effet ces modifications s'appliqueront sur l'image récupérées de base lors du DataLoader. Au cours de l'entraînement, DataLoader appel _getitem() qui modifiera les images sur la base des probabilités citées auparavant.
Le randomhorizontalfip retourne une image, il ne modifie pas le label hormis lorsque ce sont des textes où nous perdons de l'information. Ici il est applicable sans perte la classification ne se repose pas sur des images avec des textes.
Le color jitter modifie les paramètres de l'image mais celle-ci sont toujours autant reconnaissables, elles gardent leurs caractéristiques principales. Les labels sont gardés.
Le random crop pourrait entraîner une perte d'information sur l'image et faire perdre la logique image-label, cependant ici le crop est de 80 à 100% ce qui garde la plus grande partie de l'image et évite de perdre totalement l'objet étudié.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._
![alt text](images/check_img_77669_pole.png)
![alt text](images/check_img_31474_chain.png)
![alt text](<images/check_img_70739_dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk.png>)



**D10.** Montrez 2–3 exemples et commentez brièvement. 
On voit un changement des couleurs (colorjitter) un recadrage (randomcrop) et un horizontalflip après augmentation des images.


**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.
[32, 3, 64, 64]



---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `Accuracy` → score = `0.5%`
- **Prédiction aléatoire uniforme** — Métrique : `Accuracy` → score = `0.5%`  
_Commentez en 2 lignes ce que ces chiffres impliquent._
La classe majoritaire serait un modèle qui prédit toujours une seule classe, un seul label, celui de la classe majoritaire. Le score est de 0.5% car la classe majoritaire est composée de 500 données sur 100,000 existantes (500/100,000 = 0.005).

La prédiction aléatoire uniforme a autant de chance de prédire chacune des classes. Sachant que toutes les classes ont la même probabilité de 0.005 alors la même accuracy est calculée que pour la méthode la classe majoritaire.

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :

Ici, l'exemple d'un batch de 32 est pris. Il se peut que l'utilisation soit ensuite faite sur un autre nombre de batch selon la configuration stockée dans config.yaml et les performances que cela entraîne.

  - Input : 
      Conv2d, BatchNorm2d(option), ReLU (ou dérivée)
      Entrée (32, 3, 64, 64) Sortie (32, 64, 64, 64)
      La fonction d'activation est ReLU par défaut, l'option de choisir avec GeLU, SeLU, LeakyReLU et ELU a été ajoutée.
      Aucun Pooling utilisé dans ce ResNet
  - Stage 1 (répéter N₁ fois) : 
      Conv2d, BatchNorm2d(option), ReLU(ou dérivée), Dropout2d, Conv2d, BatchNorm2d(option)
      Entrée (32, 64, 64, 64) Sortie (32, 64, 64, 64)
  - Stage 2 (répéter N₂ fois) : 
      Conv2d, BatchNorm2d(option), ReLU(ou dérivée), Dropout2d, Conv2d, BatchNorm2d(option)
      Entrée (32, 64, 64, 64) Sortie (32, 128, 32, 32)
  - Stage 3 (répéter N₃ fois) : 
      Conv2d, BatchNorm2d(option), ReLU(ou dérivée), Dropout2d, Conv2d, BatchNorm2d(option)
      Entrée (32, 128, 32, 32) Sortie (32, 256, 16, 16)
  - Tête (GAP / linéaire) → logits (dimension = nb classes)
      AdaptativeAvgPool2d, Flatten, Linear
      Entrée (32, 256, 16, 16) Sortie (32, 200) (batch_size, num_classes)
      Pas de fonction d'ac

- **Loss function** :
  - Multi-classe : CrossEntropyLoss

- **Sortie du modèle** : forme = __(32, 200)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `2 805 448` ou `4 584 776` en fonction du nombre de blocs résiduels

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).

Il existe 3 composantes principales à ce réseau, la phase d'input, l'empilement des blocs résiduels et enfin la tête.
La phase d'input permet de préparer les données d'entrée à nos blocs résiduels en changeant le nombre de channel.
La phase de bloc résiduel permet de reconnaître les patternes qui peuvent définir un objet, une classe d'image.
La dernière phase, la tête permet de transformer ces patternes en prédiction en les structurant en une sortie compréhensible (nombre de batchs, numéro label).

Deux hyperparamètres spécifiques existent pour ce modèle. Le nombre de blocs résiduels et le taux de drop out.
Le nombre de blocs (2,2,2 ou 3,3,3) permet de réguler la profondeur du réseau, lorsque la profondeur est plus grande, le risque d'overfitting l'est aussi.
Le taux de dropout régule ce problème d'overfitting en donnant une probabilité de désactivation de neurones. Cela rend le modèle plus robuste. (0.1 ou 0.3).
Ces deux hyperparamètres vont parfaitement de paire.

### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` = 5.298
- **Observée sur un batch** : `5.2958`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

La loss initiale obtenue est de 5.2958, très proche de la loss théorique, cela signifie que le modèle est cohérent. Le batch est de forme (32,3,64,64) et la sortie du modèle de forme (32, 200).

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = 32(batch)*4 = 128` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `dropout = 0.1`, `blocs résiduels = [3,3,3]`
- **Optimisation** : LR = `0.001`, weight decay = `0` (0 ou très faible recommandé)
- **Nombre d’époques** : `100`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._
![alt text](images/image.png)
Le modèle apprend bien sur le petit ensemble.

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

Le test a été mené sur un sous-ensemble de 64 exemples avec les hyperparamètres de modèle B=(3,3,3) et dropout=0.1 afin de maximiser les potentiels d'overfitting. La courbe train/loss (voir capture ci-dessus) montre que la perte d'entraînement diminue de manière drastique pour tendre vers zéro dès 40 époques.

Ce comportement prouve l'overfitting car il démontre que le modèle a une capacité suffisante pour mémoriser parfaitement ce petit jeu de données. S'il n'arrivait pas à faire chuter la perte, cela indiquerait un problème dans l'architecture ou le pipeline d'entraînement. Le succès de ce test valide donc la capacité d'apprentissage de notre modèle.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `0.00005 , 0.00050`
- **Choix pour la suite** :
  - **LR** = `0.0001`
  - **Weight decay** = `0.0001` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._
![alt text](images/image1.png)
**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.
Classement des combinaisons (de la meilleure à la moins bonne):

    Learning Rate  Weight Decay      Loss
0         0.00010       0.00001  4.906494
1         0.00010       0.00010  4.924708
2         0.00010       0.00000  4.944475
3         0.00050       0.00100  4.953686
4         0.00005       0.00010  4.959709
5         0.00005       0.00000  4.966598
6         0.00005       0.00100  4.974095
7         0.00010       0.00100  4.976490
8         0.00005       0.00001  4.999286
9         0.00050       0.00000  5.013248
10        0.00050       0.00001  5.035567
11        0.00100       0.00000  5.042563
12        0.00050       0.00010  5.115390
13        0.00100       0.00010  5.164641
14        0.00500       0.00100  5.187430
15        0.01000       0.00000  5.221153
16        0.00001       0.00010  5.231850
17        0.00001       0.00001  5.233665
18        0.00001       0.00000  5.235673
19        0.01000       0.00001  5.239549
20        0.00001       0.00100  5.246063
21        0.00500       0.00010  5.250206
22        0.00500       0.00001  5.272990
23        0.01000       0.00010  5.286049
24        0.01000       0.00100  5.287495
25        0.00500       0.00000  5.334618
26        0.00100       0.00001  5.424565
27        0.00100       0.00100  5.496681


On observe que LR = 0.0001 et WD = 0.0001 ont la combinaison avec la meilleure performance


## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{0.0001}`
  - Weight decay : `{0.0001}`
  - Hyperparamètre modèle A : `{(2,2,2), (3,3,3)}`
  - Hyperparamètre modèle B : `{0.1, 0.3}`

- **Durée des runs** : `15` époques par run (1–5 selon dataset), même seed


> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

![alt text](images/image2.png)
Avec ces graphiques du modèle 0, on voit le dropout à 0.1, le LR à 0.0001 et le Weight decay à 0.0001.
On observe ensuite l'accuracy et la loss. 


**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

|   Model_id |   Epoch |   Dropout | Block_config   |     LR |     WD |   Val_Accuracy (%) |   Val_Loss |
|-----------:|--------:|----------:|:---------------|-------:|-------:|-------------------:|-----------:|
|          1 |      14 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           16.2333  |    3.83013 |
|          0 |      14 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           15.6     |    3.91378 |
|          1 |      13 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           15.5333  |    3.85172 |
|          0 |      13 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           15.3333  |    3.91986 |
|          1 |      12 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           15.3333  |    3.9107  |
|          0 |      11 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           14.8667  |    3.98614 |
|          1 |       9 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           14.3667  |    3.99894 |
|          0 |      12 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           14.2667  |    3.99412 |
|          1 |      11 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           14.2     |    3.91523 |
|          0 |      10 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           13.2333  |    4.04634 |
|          0 |       9 |       0.1 | [2, 2, 2]      | 0.0001 | 0.0001 |           13.1333  |    4.08398 |
|          1 |      10 |       0.1 | [3, 3, 3]      | 0.0001 | 0.0001 |           13.0667  |    4.01201 |
|          2 |      14 |       0.3 | [2, 2, 2]      | 0.0001 | 0.0001 |           12.4333  |    4.15341 |
|          3 |      14 |       0.3 | [3, 3, 3]      | 0.0001 | 0.0001 |           11.8     |    4.15436 |

Avec ce classement des meilleurs résultats lors du grid search. Nous voyons clairement deux modèles qui le domine. Le 1 et le 0 qui ont un dropout de 0.1. Le 1 avec 3 block de 3 de config et le 0 avec 3 block de 2.

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `0.0001`
  - Weight decay = `0.0001`
  - Hyperparamètre modèle A = `dropout = 0.1, block_config = [3,3,3]`
  - Hyperparamètre modèle B = `dropout = 0.1, block_config = [2,2,2]`
  - Batch size = `32`
  - Époques = `100` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)


> _Insérer captures TensorBoard :_

Bleu Modele A
Rose Modele B

Train :
![alt text](images/image3.png)
![alt text](images/image4.png)
![alt text](images/image5.png)
![alt text](images/image6.png)

Val :
![alt text](images/image7.png)
![alt text](images/image8.png)
![alt text](images/image9.png)


On observe une meilleure performance du modèle A


> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

Il n'y a pas de sous apprentissage car on voit les courbes d'accuracy monter puis redescendre ou se stabiliser (pour le val) alors que le train continue à monter (début de sur apprentissage).

Un début de sur apprentissage apparait malgré les paramètres d'augmentation. Il pourrait être intéressant d'essayer avec des parametres d'augmentation plus aggressifs, grâce à l'enregistrement de la meilleure version, nous avons les poids optimaux.

Le modèle A reste stable jusqu'au bout alors que le modèle B arrête de l'être autour de la 20ème itération. Cela peut venir d'un LR trop grand pour la taille du réseau ou bien simplement le fait que la capacité du modèle ne permet pas de performer sur la tâche.

---


## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

Baseline :
  Pour cette comparaison, j'ai tout d'abord fait tourné le modèle initial afin d'avoir un élément de comparaison (run sans warmup et scheduler).
  On obtient un score F1 pique à 0.266152.
  ![alt text](images/image10.png)


LR haut :
  J'ai augmenté le LR pour voir si mon modèle apprenait plus vite, au moins au début, avec un LR plus haut.
  ![alt text](images/image11.png)
  Au lieu d'accélérer comme je pensais, le learning rate a été moins performant de A à Z.

WD haut : 
  En pensant pouvoir appliquer une meilleure généralisation du modèle, j'ai essayé avec un weight decay plus haut.
  ![alt text](images/image12.png)
  On voit bien que la performance du weight decay plus haut est inférieure à la performance de la baseline (0.18 au lieu de 0.26)
  La généralisation peut peut etre mieux apparaître sur un plus grand nombre d'epochs.

Blocks haut : 
  En sortant un peu du chemin indiqué dans l'énoncé j'ai voulu voir la performance du modèle si on ajoutait des blocks résiduels aux couches (4,4,4)
  ![alt text](images/image13.png)
  On constate que le modèle apprend légèrement mieux mais ce n'est pas si convaincant (0.26279)

Dropout haut : 
  En élevant le dropout, je pensais stabiliser le modèle pour réduire l'overfitting.
  ![alt text](images/image14.png)
  Le maximum est atteint vers la fin des 100 epochs ce qui montre la stabilité que procure ce dropout plus haut même s'il reste inférieur sur l'entierté de l'entraînement, c'est le seul qui n'overfit pas encore. La courbe continue de croitre, doucement mais elle croie.



=================================================
        RÉCAPITULATIF FINAL DES PERFORMANCES (GRID FINALE)
==================================================
| Exp_name     |   Epoch | Blocks    |   Dropout |     LR |     WD |   Train_Loss |   Train_Acc |   Train_F1 |   Val_Loss |   Val_Acc |     Val_F1 | Notes   |
|:-------------|--------:|:----------|----------:|-------:|-------:|-------------:|------------:|-----------:|-----------:|----------:|-----------:|:--------|
| baseline     |      56 | [3, 3, 3] |       0.1 | 0.0001 | 0.0001 |     1.98263  |      0.5204 | 0.51039    |    3.27661 | 0.279667  | 0.266152   |         |

| blocks_high  |      52 | [4, 4, 4] |       0.1 | 0.0001 | 0.0001 |     1.77169  |      0.5659 | 0.557088   |    3.26334 | 0.273333  | 0.26279    |         |

| dropout_high |      97 | [3, 3, 3] |       0.3 | 0.0001 | 0.0001 |     2.29538  |      0.4345 | 0.421385   |    3.27511 | 0.267333  | 0.250162   |         |

| lr_high      |      41 | [3, 3, 3] |       0.1 | 0.0005 | 0.0001 |     2.17922  |      0.4492 | 0.437079   |    3.6431  | 0.218     | 0.203994   |         |

| wd_high      |      57 | [3, 3, 3] |       0.1 | 0.0001 | 0.001  |     3.40785  |      0.2465 | 0.213238   |    3.6049  | 0.212333  | 0.18273    |         |


---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `Augmentation encore plus aggressif et plus grand dropout car on a vu que cela fonctionnait bien` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

La motivation de cette itération est de voir si l'on peut, avec un warmup, un restart du lr etc si on peut améliorer les performances de notre modèle sans ajouter de blocks. Simplement en étant plus aggressif dans notre stratégie. La performance vient principalement du fait qu'à la suite des variations des paramétrages on voyait dropout apprendre encore.




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

- **Seed** : `42`
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


