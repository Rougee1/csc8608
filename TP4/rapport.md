# TP4 – Graph Neural Networks (GNN) sur Cora

---

## Exercice 1 – Initialisation du TP et smoke test PyG (Cora)

### Q1 – Structure du dossier TP4

```
TP4/
├── configs/
│   ├── baseline_mlp.yaml
│   ├── gcn.yaml
│   └── sage_sampling.yaml
├── rapport.md
├── runs/
└── src/
    ├── benchmark.py
    ├── data.py
    ├── models.py
    ├── smoke_test.py
    ├── train.py
    └── utils.py
```

### Q2 – Installation de PyTorch Geometric

```bash
pip install torch-geometric scipy
```

### Q3 – Installation de pyg-lib

La commande suivante affiche la commande `pip install` appropriée pour la version de torch/CUDA installée :

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Puis suivre les instructions sur [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) pour installer `pyg-lib` ou `torch-sparse`.

**Sans pyg-lib / torch-sparse** (cas de cette exécution : Python 3.13, pas de wheels disponibles), le script `train.py` pour `--model sage` bascule automatiquement en **full-batch GraphSAGE**, ce qui reste valide sur Cora car le graphe est petit (2708 nœuds).

### Q4 – Complétion de `smoke_test.py`

Blancs complétés :
- `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — détection automatique GPU/CPU.
- `Planetoid(root=root, name="Cora")` — chargement du dataset Cora de la suite Planetoid.

### Q5 – Exécution du smoke test

> **[CAPTURE D'ÉCRAN]** : `TP4/report/smoke_test.png`

```
=== Environment ===
torch: 2.9.1+cpu
cuda available: False
device: cpu

=== Dataset (Cora) ===
num_nodes: 2708
num_edges: 10556
num_node_features: 1433
num_classes: 7
train/val/test: 140 500 1000

OK: smoke test passed.
```

Exécution en local sur CPU (pas de GPU disponible). Cora contient 2708 nœuds, 10556 arêtes, 1433 features par nœud (bag-of-words de publications) et 7 classes thématiques. Le split standard Planetoid est 140/500/1000 (train/val/test).

---

## Exercice 2 – Baseline tabulaire : MLP + entraînement et métriques

### Q1 – Création des fichiers

Fichiers créés : `data.py`, `models.py`, `train.py`, et `configs/baseline_mlp.yaml`.

### Q2 – Complétion de `data.py`

Blancs complétés :
- `name="Cora"` — nom du dataset Planetoid.
- `x=data.x` — features des nœuds (matrice bag-of-words, [2708, 1433]).
- `y=data.y` — labels (7 classes de publications).
- `train_mask=data.train_mask`, `val_mask=data.val_mask`, `test_mask=data.test_mask` — masques booléens fournis par Planetoid.
- `num_features=dataset.num_node_features`, `num_classes=dataset.num_classes`.

### Q3 – Complétion de `utils.py`

Blancs complétés :
- `0.0` — quand precision + recall = 0, le F1 pour cette classe vaut 0.
- `num_classes` — la Macro-F1 est la moyenne arithmétique des F1 par classe.

### Q4 – Complétion de `models.py` (MLP)

Blanc complété : `self.net(x)` — le forward passe simplement l'entrée `x` dans le réseau séquentiel.

### Q5 – Complétion de `baseline_mlp.yaml`

```yaml
seed: 42
device: "cuda"
epochs: 200
lr: 0.01
weight_decay: 5e-4

mlp:
  hidden_dim: 64
  dropout: 0.5
```

Valeurs classiques pour Cora : lr=0.01, weight_decay=5e-4 (régularisation L2), hidden_dim=64 (suffisant pour 1433→7), dropout=0.5.

### Q6 – Complétion de `train.py` (MLP)

Blanc complété : `model(x)` — le MLP prend uniquement les features des nœuds, pas le graphe.

### Q6bis – Pourquoi calculer les métriques sur train/val/test séparément ?

On sépare les métriques pour trois raisons concrètes :
- **train_mask** (140 nœuds) : permet de vérifier que le modèle apprend bien (la loss diminue, l'accuracy monte). Si les métriques train stagnent, il y a un problème d'optimisation.
- **val_mask** (500 nœuds) : sert à détecter le sur-apprentissage et à régler les hyperparamètres (lr, hidden_dim, dropout). On pourrait y ajouter un early stopping.
- **test_mask** (1000 nœuds) : donne l'estimation finale de la performance. On ne touche jamais aux hyperparamètres en se basant sur le test, sinon on biaise l'évaluation (data leakage indirect).

### Q7 – Exécution MLP

> **[CAPTURE D'ÉCRAN]** : `TP4/report/mlp_train.png`

```
device: cpu
model: mlp
epochs: 200
epoch=001 loss=1.9548 train_acc=0.3429 val_acc=0.3540 test_acc=0.3430 train_f1=0.2619 val_f1=0.1296 test_f1=0.1225 epoch_time_s=0.0251
epoch=100 loss=0.0128 train_acc=1.0000 val_acc=0.5580 test_acc=0.5670 train_f1=1.0000 val_f1=0.5499 test_f1=0.5558 epoch_time_s=0.0126
epoch=200 loss=0.0113 train_acc=1.0000 val_acc=0.5760 test_acc=0.5850 train_f1=1.0000 val_f1=0.5734 test_f1=0.5715 epoch_time_s=0.0170
total_train_time_s=2.8327
train_loop_time=4.2000
checkpoint_saved: TP4/runs\mlp.pt
```

Le MLP sur-apprend rapidement (train_acc=1.0 dès epoch 10), mais plafonne à ~58% sur le test. Sans accès à la structure du graphe, le modèle ne peut exploiter que le bag-of-words individuel de chaque nœud.

---

## Exercice 3 – Baseline GNN : GCN (full-batch)

### Q1 – Complétion de `gcn.yaml`

```yaml
seed: 42
device: "cuda"
epochs: 200
lr: 0.01
weight_decay: 5e-4

gcn:
  hidden_dim: 64
  dropout: 0.5
```

Mêmes hyperparamètres que le MLP pour isoler l'effet du graphe dans la comparaison.

### Q2 – Mise à jour de `data.py` avec `edge_index`

Blanc complété : `edge_index=data.edge_index` — la liste d'arêtes au format COO `[2, E]` fournie par Planetoid.

### Q3 – Complétion de `models.py` (GCN)

Blanc complété : `self.conv2(x, edge_index)` — on passe le `x` transformé (après ReLU + dropout) dans la deuxième couche GCN.

### Q4 – Mise à jour de `train.py` (MLP + GCN)

Blancs complétés :
- `model(x)` — forward MLP (features seules).
- `model(x, edge_index)` — forward GCN (features + structure du graphe).

### Q5 – Entraînement GCN + comparaison MLP vs GCN

> **[CAPTURE D'ÉCRAN]** : `TP4/report/gcn_train.png`

```
device: cpu
model: gcn
epochs: 200
epoch=001 loss=1.9582 train_acc=0.8357 val_acc=0.4660 test_acc=0.4670 train_f1=0.8334 val_f1=0.4846 test_f1=0.4749 epoch_time_s=0.1360
epoch=100 loss=0.0137 train_acc=1.0000 val_acc=0.7740 test_acc=0.8130 train_f1=1.0000 val_f1=0.7547 test_f1=0.8048 epoch_time_s=0.0202
epoch=200 loss=0.0089 train_acc=1.0000 val_acc=0.7740 test_acc=0.8020 train_f1=1.0000 val_f1=0.7581 test_f1=0.7942 epoch_time_s=0.0207
total_train_time_s=4.0038
train_loop_time=6.1439
checkpoint_saved: TP4/runs\gcn.pt
```

**Tableau comparatif :**

| Modèle | test_acc | test_f1 | total_train_time_s |
|--------|----------|---------|-------------------|
| MLP    | 0.5850   | 0.5715  | 2.8327 s |
| GCN    | 0.8020   | 0.7942  | 4.0038 s |

GCN apporte **+21.7 points** d'accuracy et **+22.3 points** de F1 par rapport au MLP, au prix d'un temps d'entraînement 1.4× supérieur.

### Q6 – Pourquoi GCN peut dépasser (ou non) le MLP sur Cora

Cora est un graphe de citations académiques avec une forte **homophilie** : les articles connectés appartiennent souvent à la même catégorie. GCN exploite cette propriété via le message passing : chaque nœud agrège les features de ses voisins, enrichissant sa représentation avec l'information relationnelle. Le MLP, lui, traite chaque nœud indépendamment — il ne "voit" que le bag-of-words du papier.

Les résultats le confirment : MLP~58% vs GCN~80%. Le gain est substantiel car le graphe est dense et homophile. Sur un graphe hétérophile (voisins de classes différentes), le GCN pourrait même dégrader les performances par rapport au MLP (le lissage "dilue" le signal discriminant).

---

## Exercice 4 – GraphSAGE + neighbor sampling (mini-batch)

### Q1 – Complétion de `sage_sampling.yaml`

```yaml
seed: 42
device: "cuda"
epochs: 200
lr: 0.01
weight_decay: 5e-4

sage:
  hidden_dim: 64
  dropout: 0.5

sampling:
  batch_size: 64
  num_neighbors_l1: 15
  num_neighbors_l2: 10
```

- `batch_size: 64` — taille raisonnable pour Cora (140 nœuds de train, ~2-3 mini-batches par epoch).
- `num_neighbors_l1: 15`, `num_neighbors_l2: 10` — fanout décroissant : 15 voisins au premier hop, 10 au second. Compromis entre contexte et coût.

### Q2 – Mise à jour de `data.py` avec `pyg_data`

Blanc complété : `pyg_data=data` — on expose l'objet PyG complet, nécessaire au `NeighborLoader`.

### Q3 – Complétion de `models.py` (GraphSAGE)

Blanc complété : `self.conv2(x, edge_index)` — même pattern que GCN, le `x` post-ReLU/dropout est passé à la couche 2.

### Q4 – Mise à jour de `train.py` (+ SAGE + NeighborLoader)

Blancs complétés :
- `input_nodes=train_mask` — on échantillonne autour des nœuds d'entraînement.
- `num_neighbors=[n1, n2]` — fanout par couche.

**Note d'exécution** : `pyg-lib` et `torch-sparse` ne sont pas disponibles pour Python 3.13 + torch 2.9.1 sur Windows. GraphSAGE a donc été entraîné en **full-batch** (mode de repli automatique du script).

### Q5 – Entraînement GraphSAGE + comparaison 3 modèles

> **[CAPTURE D'ÉCRAN]** : `TP4/report/sage_train.png`

```
device: cpu
model: sage
epochs: 200
sage_mode: full_batch (install pyg-lib or torch-sparse for NeighborLoader; Cora est petit, full-batch GraphSAGE reste valide pour le TP)
epoch=001 loss=1.9494 train_acc=0.9429 val_acc=0.6840 test_acc=0.6830 train_f1=0.9424 val_f1=0.6352 test_f1=0.6454 epoch_time_s=0.1132
epoch=100 loss=0.0041 train_acc=1.0000 val_acc=0.7680 test_acc=0.8020 train_f1=1.0000 val_f1=0.7618 test_f1=0.7949 epoch_time_s=0.0431
epoch=200 loss=0.0037 train_acc=1.0000 val_acc=0.7700 test_acc=0.8030 train_f1=1.0000 val_f1=0.7634 test_f1=0.7973 epoch_time_s=0.0497
total_train_time_s=9.3349
train_loop_time=15.8696
checkpoint_saved: TP4/runs\sage.pt
```

**Tableau comparatif complet :**

| Modèle    | test_acc | test_f1 | total_train_time_s |
|-----------|----------|---------|-------------------|
| MLP       | 0.5850   | 0.5715  | 2.8327 s |
| GCN       | 0.8020   | 0.7942  | 4.0038 s |
| GraphSAGE | 0.8030   | 0.7973  | 9.3349 s |

GraphSAGE obtient des résultats quasi-identiques au GCN (+0.1 point d'accuracy, +0.31 point de F1) mais avec un temps d'entraînement 2.3× supérieur. En full-batch, GraphSAGE est plus lent que GCN car SAGEConv agrège les features de manière différente (concaténation au lieu de normalisation spectrale), ce qui implique plus de paramètres.

### Q6 – Compromis du neighbor sampling

Le **neighbor sampling** accélère l'entraînement en remplaçant le full-batch (tous les voisins) par un sous-ensemble aléatoire de taille fixe (fanout) à chaque couche. Sur un graphe à N nœuds et degré moyen d, le coût d'un forward full-batch en 2 couches est O(N·d²), tandis qu'avec un sampling [k1, k2], le coût par nœud-seed est O(k1·k2). Sur Cora (d~4), le gain est modeste, mais sur un graphe social avec d~1000, c'est indispensable.

Le risque principal est la **variance du gradient** : chaque mini-batch ne voit qu'un sous-ensemble des voisins, ce qui rend l'estimation du gradient bruitée. Les nœuds hubs (degré élevé) sont particulièrement affectés : si un hub a 500 voisins mais qu'on n'en échantillonne que 15, on perd de l'information contextuelle à chaque forward. Augmenter le fanout améliore la qualité mais augmente la mémoire et le temps CPU de sampling.

De plus, le sampling introduit un coût CPU non négligeable (construction des sous-graphes), qui sur Cora peut dominer le temps GPU (le graphe est petit). Le sampling est donc surtout pertinent sur des graphes larges (>100k nœuds) où le full-batch ne tient pas en mémoire GPU.

---

## Exercice 5 – Benchmarks ingénieur : temps d'entraînement et latence d'inférence

### Q1 – Dossier `TP4/runs/`

Créé et ajouté au `.gitignore`.

### Q2 – Sauvegarde de checkpoint

Blancs complétés :
- `os.makedirs("TP4/runs", exist_ok=True)` — création du dossier.
- `os.path.join("TP4/runs", f"{args.model}.pt")` — chemin du checkpoint.

### Q3 – Complétion de `benchmark.py`

Blancs complétés :
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — détection GPU.
- `warmup = 10` — 10 itérations de warmup pour stabiliser les kernels GPU/caches.
- `runs = 50` — 50 itérations chronométrées pour moyenner.
- `model(x)` pour MLP, `model(x, edge_index)` pour GCN/SAGE.

### Q4 – Exécution du benchmark

> **[CAPTURE D'ÉCRAN]** : `TP4/report/benchmark.png`

```
model: mlp    device: cpu    avg_forward_ms: 3.9381    num_nodes: 2708    ms_per_node: 0.00145424
model: gcn    device: cpu    avg_forward_ms: 7.1263    num_nodes: 2708    ms_per_node: 0.00263156
model: sage   device: cpu    avg_forward_ms: 28.3087   num_nodes: 2708    ms_per_node: 0.01045373
```

**Tableau synthétique complet :**

| Modèle    | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|-----------|----------|---------------|-------------------|----------------|----------------|
| MLP       | 0.5850   | 0.5715        | 2.8327 s          | 4.2000 s       | 3.94 ms        |
| GCN       | 0.8020   | 0.7942        | 4.0038 s          | 6.1439 s       | 7.13 ms        |
| GraphSAGE | 0.8030   | 0.7973        | 9.3349 s          | 15.8696 s      | 28.31 ms       |

### Q5 – Pourquoi warmup + synchronisation CUDA

Le **warmup** est nécessaire car le premier forward sur GPU déclenche des allocations mémoire, la compilation JIT des kernels CUDA, et le remplissage des caches L2. Ces opérations ponctuelles faussent les mesures de latence si elles sont incluses dans le chronométrage. Après 10 itérations, les allocations sont stabilisées et les kernels compilés, donnant des mesures représentatives du régime permanent.

La **synchronisation CUDA** (`torch.cuda.synchronize()`) est indispensable car le GPU exécute les opérations de manière **asynchrone** par rapport au CPU. Sans synchronisation, `time.perf_counter()` mesure uniquement le temps de lancement des kernels (quelques microsecondes), pas leur exécution réelle. En appelant `synchronize()` avant de démarrer le chrono et après le forward, on force le CPU à attendre que tous les kernels GPU soient terminés.

Sans ces deux précautions, les mesures peuvent être 10-100x plus basses que la réalité sur GPU, rendant la comparaison entre modèles non fiable.

---

## Exercice 6 – Synthèse finale

### Q1 – Tableau comparatif final

| Modèle    | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|-----------|----------|---------------|-------------------|----------------|----------------|
| MLP       | 0.5850   | 0.5715        | 2.8327 s          | 4.2000 s       | 3.94 ms        |
| GCN       | 0.8020   | 0.7942        | 4.0038 s          | 6.1439 s       | 7.13 ms        |
| GraphSAGE | 0.8030   | 0.7973        | 9.3349 s          | 15.8696 s      | 28.31 ms       |

*Run propre, seed=42, CPU (pas de GPU disponible localement), GraphSAGE en mode full-batch.*

### Q2 – Recommandation ingénieur

**MLP** (test_acc=0.585, avg_forward=3.94 ms) : À choisir quand le graphe n'est pas disponible ou quand les features seules suffisent. Ici, le MLP plafonne à ~58% : les features bag-of-words ne sont pas assez discriminantes sans contexte voisin. Avantage : entraînement 1.4× plus rapide que GCN et inférence 1.8× plus rapide. Recommandé uniquement comme baseline ou sur des graphes à faible homophilie.

**GCN** (test_acc=0.802, avg_forward=7.13 ms) : Le meilleur compromis qualité/coût sur Cora. Gain de +21.7 points d'accuracy vs MLP pour seulement 1.4× plus de temps d'entraînement et 1.8× plus de latence d'inférence. Le full-batch est simple à implémenter et stable. Recommandé pour tout graphe statique tenant en mémoire GPU (<100k nœuds typiquement).

**GraphSAGE** (test_acc=0.803, avg_forward=28.31 ms) : Quasi-identique au GCN en qualité (+0.1 acc) mais 2.3× plus lent à l'entraînement et 4× plus lent à l'inférence sur CPU (en full-batch). La vraie valeur de GraphSAGE réside dans le **NeighborLoader** pour les grands graphes (>100k nœuds) où le GCN full-batch ne tient plus en mémoire GPU. Sur Cora, il n'y a pas d'avantage pratique.

En résumé : **GCN pour un graphe petit/moyen et statique, GraphSAGE + sampling pour un graphe large ou dynamique, MLP comme baseline ou sans structure de graphe.**

### Q3 – Risque de protocole

Un risque majeur est la **non-reproductibilité** liée à la seed unique. Avec seed=42, les résultats dépendent de l'initialisation des poids. Un modèle pourrait paraître meilleur simplement par chance. Dans un vrai projet, on lancerait chaque modèle avec 5-10 seeds différentes et on rapporterait moyenne ± écart-type.

Un autre risque est la **comparaison CPU/GPU non contrôlée**. Ici, tous les benchmarks sont sur CPU, ce qui est cohérent. Sur GPU, le caching cuDNN peut avantager le modèle lancé en second (kernels déjà optimisés). Le warmup atténue ce problème mais ne l'élimine pas totalement — idéalement, chaque benchmark devrait tourner dans un processus séparé.

Enfin, le split train/val/test de Cora est fixe (140/500/1000), standard académique mais un seul split. Un k-fold cross-validation donnerait des estimations plus robustes.

### Q4 – Vérification du dépôt

Le dépôt contient `TP4/rapport.md`, les scripts dans `TP4/src/`, et les configs dans `TP4/configs/`. Les fichiers volumineux (datasets dans `TP4/data/`, checkpoints dans `TP4/runs/`) sont exclus du dépôt via le `.gitignore`. Aucun fichier > 1 Mo n'est commité.
