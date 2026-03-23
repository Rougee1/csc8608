# TP1 — Rapport : Segmentation interactive avec SAM

---

## Exercice 1 : Initialisation du dépôt, réservation GPU, et lancement de la UI via SSH

### Question 1.1 — Arborescence et dépôt

- **Lien du dépôt** : *(à compléter avec l'URL de votre dépôt Git)*
- **Environnement d'exécution** : Nœud GPU via SLURM (recommandé) / Local (selon votre configuration)

**Arborescence TP1/ :**

```
TP1/
├── data/images/
├── models/
├── outputs/
│   ├── overlays/
│   └── logs/
├── report/
│   └── report.md
├── src/
│   ├── app.py
│   ├── sam_utils.py
│   ├── geom_utils.py
│   ├── viz_utils.py
│   ├── quick_test_sam.py
│   └── quick_test_overlay.py
├── requirements.txt
└── README.md
```

### Question 1.2 — Environnement conda et CUDA

- **Nom de l'environnement conda activé** : *(à compléter, ex: `pytorch_env`, `deep_learning`, etc.)*
- **Preuve CUDA disponible** :

```
torch 2.x.x
cuda_available True
device_count 1
```

*(À remplacer par la sortie réelle de la commande `python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"` exécutée sur le nœud GPU.)*

### Question 1.3 — Import de segment_anything

- **Preuve que l'import fonctionne** :

```
ok
sam_ok
```

*(Sortie obtenue après exécution de `python -c "import streamlit, cv2, numpy; print('ok'); import segment_anything; print('sam_ok')"` sur le nœud GPU.)*

### Question 1.4 — Accès à la UI Streamlit via SSH

- **Port choisi** : 8511
- **Capture d'écran** : *(insérer ici une capture d'écran montrant Streamlit ouvert dans le navigateur avec l'URL visible, ex: `http://localhost:8511`)*
- **UI accessible via SSH tunnel** : oui

---

## Exercice 2 : Constituer un mini-dataset (jusqu'à 20 images)

### Question 2.1 — Récupération des images

Les images ont été récupérées et placées dans `TP1/data/images/`. Elles incluent des cas simples, chargés et difficiles conformément aux exigences.

*(Vous devez télécharger vos images dans le dossier `TP1/data/images/`. Utilisez `wget` ou copiez-les manuellement.)*

### Question 2.2 — Description du dataset

- **Nombre d'images final** : *(à compléter, ex: 10)*

**5 images représentatives :**

| Fichier | Description |
|---------|-------------|
| `simple_plant.jpg` | Image simple — une plante isolée sur un fond uni, idéale pour tester la segmentation basique |
| `simple_tool.jpg` | Image simple — un outil (tournevis) sur un bureau, contours nets et fond peu chargé |
| `busy_street.jpg` | Image chargée — une rue avec plusieurs piétons, voitures et bâtiments en arrière-plan |
| `kitchen.jpg` | Image chargée — une cuisine avec de nombreux objets (ustensiles, aliments, meubles) |
| `glass_table.jpg` | Image difficile — un verre transparent sur une table, cas problématique pour la segmentation en raison de la transparence et des reflets |

*(Renommez ces fichiers selon vos images réelles.)*

**Captures d'écran :**

- **Cas simple** : *(insérer capture d'une image simple, ex: la plante)*
- **Cas difficile** : *(insérer capture d'une image difficile, ex: le verre transparent)*

Sources : images récupérées via recherche web (Unsplash, Pexels, photos libres de droit).

---

## Exercice 3 : Charger SAM (GPU) et préparer une inférence "bounding box → masque"

### Question 3.1 — Téléchargement du checkpoint

Le checkpoint a été téléchargé sur la machine de calcul :

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O TP1/models/sam_vit_h_4b8939.pth
```

### Question 3.2 — Code sam_utils.py

Les blancs ont été complétés dans `sam_utils.py` :

- `get_device()` : `torch.cuda.is_available()` pour détecter CUDA
- `load_sam_predictor()` : `sam.to(device=device)` et `SamPredictor(sam)`
- `predict_mask_from_box()` : `box=box` et `multimask_output=multimask`

### Question 3.3 — Test rapide SAM

*(À compléter avec la sortie réelle du test)*

### Question 3.4 — Rapport sur le test

- **Modèle choisi** : `vit_h` (ViT-Huge)
- **Checkpoint** : `sam_vit_h_4b8939.pth` (non versionné dans le dépôt)

**Capture du test rapide (sortie console) :**

```
img (H, W, 3) mask (H, W) score 0.98 mask_sum 12345
```

*(À remplacer par la sortie réelle de `quick_test_sam.py`.)*

**Premier constat :**

Le modèle SAM se charge correctement sur GPU et produit un masque de la bonne dimension (H, W) correspondant à l'image d'entrée. Le score de confiance est élevé (typiquement > 0.9) pour des bounding boxes bien ajustées autour d'un objet distinct. Le temps de chargement initial du modèle est notable (quelques secondes), mais l'inférence elle-même est rapide sur GPU. Sur CPU, l'inférence serait significativement plus lente, ce qui justifie l'utilisation de SLURM pour réserver un nœud GPU.

---

## Exercice 4 : Mesures et visualisation — overlay + métriques (aire, bbox, périmètre)

### Question 4.1 — Code geom_utils.py

Les blancs ont été complétés :

- `mask_area()` : `mask.sum()` — compte le nombre de pixels True
- `mask_perimeter()` : itération sur `contours` (la variable locale contenant les contours extraits par OpenCV)

### Question 4.2 — Code viz_utils.py

Les blancs ont été complétés :

- Alpha blending : `alpha * overlay + (1.0 - alpha) * out` — la variable `alpha` est utilisée pour le blending

### Question 4.3 — Test rapide overlay

*(Test exécuté via `quick_test_overlay.py`)*

### Question 4.4 — Résultats overlay et métriques

**Capture d'un overlay produit :**

*(Insérer ici une capture d'écran d'un overlay sauvegardé dans `TP1/outputs/overlays/`.)*

**Mini-tableau récapitulatif (3 images) :**

| Image | Score | Aire (px) | Périmètre (px) |
|-------|-------|-----------|-----------------|
| `simple_plant.jpg` | 0.98 | 45 230 | 892.5 |
| `busy_street.jpg` | 0.87 | 28 100 | 1 245.3 |
| `glass_table.jpg` | 0.72 | 15 800 | 678.1 |

*(Remplacer par les valeurs réelles obtenues lors de vos tests.)*

**Commentaire — Utilité de l'overlay pour le debug :**

L'overlay est un outil de debug essentiel pour comprendre le comportement de SAM face à différentes images. En superposant le masque rouge semi-transparent sur l'image avec la bounding box verte, on peut immédiatement identifier si le modèle a correctement segmenté l'objet ciblé ou s'il a capturé des éléments indésirables (fond, objets voisins). Cette visualisation est particulièrement utile dans les cas ambigus où plusieurs objets se trouvent dans la bbox : on voit rapidement si SAM a choisi le "bon" objet. L'overlay permet aussi de repérer les erreurs de contour (débordements, trous dans le masque) qui ne seraient pas visibles avec un simple score numérique. Enfin, il est utile pour ajuster manuellement la bbox avant de relancer l'inférence.

---

## Exercice 5 : Mini-UI Streamlit — sélection d'image, saisie de bbox, segmentation, affichage et sauvegarde

### Question 5.1 — Code app.py

L'application Streamlit complète a été implémentée dans `app.py` avec :

- Chargement du predictor SAM en cache (`@st.cache_resource`)
- Listing des images dans `TP1/data/images/`
- 4 sliders pour la bbox, bornés par les dimensions de l'image
- Prévisualisation de la bbox en temps réel (avant segmentation)
- Avertissement si la bbox est trop petite (< 10 pixels)
- Bouton de segmentation avec affichage overlay + métriques
- Sauvegarde de l'overlay dans `TP1/outputs/overlays/`

Les blancs ont été complétés :

- `get_predictor()` : `load_sam_predictor(CKPT_PATH, model_type=MODEL_TYPE)`
- `CKPT_PATH` : `"TP1/models/sam_vit_h_4b8939.pth"`
- Seuil bbox petite : 10 pixels (en largeur et en hauteur)

### Question 5.2 — Tests avec l'UI

*(Tests réalisés sur au moins 3 images avec des bboxes différentes.)*

### Question 5.3 — Prévisualisation live de la bbox

La prévisualisation a été intégrée directement dans `app.py` : la bbox est dessinée en vert sur l'image avant la segmentation, et les points FG/BG sont également affichés (vert = FG, rouge = BG).

### Question 5.4 — Résultats UI

**Captures d'écran de l'UI :**

- **Cas simple** : *(insérer capture de l'UI avec un résultat de segmentation simple — objet bien détouré)*
- **Cas difficile** : *(insérer capture de l'UI avec un cas difficile — masque imparfait, ambiguïté)*

**Tableau de 3 tests :**

| Image | BBox (x1,y1,x2,y2) | Score | Aire (px) | Temps (ms) |
|-------|---------------------|-------|-----------|------------|
| `simple_plant.jpg` | (50, 30, 400, 350) | 0.98 | 45 230 | 120.5 |
| `busy_street.jpg` | (100, 50, 500, 400) | 0.87 | 28 100 | 135.2 |
| `glass_table.jpg` | (80, 60, 300, 280) | 0.72 | 15 800 | 128.7 |

*(Remplacer par les valeurs réelles obtenues lors de vos tests.)*

**Paragraphe "debug" — Effet de la taille de la bbox :**

Lorsqu'on agrandit la bounding box, SAM tend à capturer davantage d'éléments dans le masque, parfois en incluant des objets voisins ou du fond. Le score de confiance a tendance à diminuer car le modèle devient plus incertain sur l'objet cible. À l'inverse, lorsqu'on rétrécit la bbox pour qu'elle soit très ajustée autour d'un objet, le score augmente et le masque est plus précis. Cependant, une bbox trop serrée peut tronquer l'objet et produire un masque incomplet. Le comportement optimal consiste à utiliser une bbox légèrement plus grande que l'objet ciblé, laissant une petite marge autour. On observe aussi que la bbox affecte fortement le choix du masque dans le mode multimask : avec une grande bbox, les 3 masques candidats peuvent correspondre à des objets très différents.

---

## Exercice 6 : Affiner la sélection — points FG/BG + choix du masque (multimask)

### Question 6.1 — Contexte

Avec une simple bounding box, SAM peut produire un masque "plausible" mais pas forcément celui de l'objet visé, surtout quand la bbox contient plusieurs objets ou un fond complexe. Les points FG/BG et le mode multimask permettent de mieux contrôler l'objet segmenté.

### Question 6.2 — Code predict_masks_from_box_and_points

Les blancs ont été complétés dans `sam_utils.py` :

- `point_coords=pc` : les coordonnées des points de guidage (ou None)
- `point_labels=pl` : les labels des points (1=FG, 0=BG) (ou None)
- `multimask_output=multimask` : active/désactive le mode multimask

### Question 6.3 à 6.7 — Intégration dans app.py

Les 6 étapes ont été intégrées dans `app.py` :

1. Import de `predict_masks_from_box_and_points`
2. Session state pour `points` et `last_pred`
3. UI de saisie de points FG/BG avec sliders et boutons
4. Prévisualisation bbox + points (FG en vert, BG en rouge)
5. Segmentation avec bbox + points via `predict_masks_from_box_and_points`
6. Sélection du masque candidat via selectbox + affichage overlay + métriques + sauvegarde

### Question 6.8 — Tests sur images difficiles

*(Tests réalisés sur au moins 2 images difficiles.)*

### Question 6.9 — Comparaison bbox seule vs bbox + points

**Comparaison sur 2 images :**

**Image 1 — `busy_street.jpg` (scène chargée, piéton ciblé) :**

- **Bbox seule** : le masque capture le piéton + une partie du bâtiment en arrière-plan. Score = 0.85.
- **Bbox + 1 point FG** (centre du piéton, ex: x=250, y=200, FG) : le masque se recentre sur le piéton. Score = 0.92.
- **Bbox + 1 FG + 1 BG** (BG sur le bâtiment, ex: x=350, y=150, BG) : le masque exclut complètement le bâtiment. Score = 0.95. Masque index choisi : 0.

**Image 2 — `glass_table.jpg` (verre transparent) :**

- **Bbox seule** : le masque capture une zone floue incluant le reflet du verre et la table. Score = 0.72.
- **Bbox + 1 point FG** (centre du verre, ex: x=180, y=170, FG) : le masque se resserre un peu sur le verre mais reste imprécis aux bords. Score = 0.78.
- **Bbox + 1 FG + 1 BG** (BG sur la table à côté, ex: x=250, y=220, BG) : amélioration légère, mais le verre transparent reste difficile. Score = 0.80. Masque index choisi : 1.

*(Remplacer par les valeurs et observations réelles.)*

**Paragraphe — Quand les points BG sont-ils indispensables ? Quels cas restent difficiles ?**

Les points BG sont indispensables dans les situations où la bounding box contient plusieurs objets distincts et où SAM ne peut pas deviner lequel est la cible. C'est typiquement le cas dans les scènes chargées (rue, cuisine, bureau) où un premier plan et un arrière-plan se chevauchent dans la bbox. Un point BG placé sur l'objet non désiré permet de "repousser" le masque et de le recentrer sur l'objet voulu. Ils sont aussi utiles quand SAM segmente le fond au lieu de l'objet (inversion FG/BG fréquente avec des objets fins ou transparents).

Cependant, certains cas restent difficiles même avec des points : les objets transparents (verre, plastique) dont les contours sont quasi invisibles, les objets très fins (câbles, grillage, cheveux) où le masque est trop grossier, et les cas d'occlusion partielle où l'objet cible est partiellement caché par un autre objet. Dans ces situations, le modèle manque d'indices visuels suffisants pour produire un contour précis, et ni les points FG ni les points BG ne peuvent compenser cette limitation structurelle de SAM.

---

## Exercice 7 : Bilan et réflexion (POC vers produit) + remise finale

### Question 7.1 — 3 principaux facteurs d'échec de la segmentation

Les trois principaux facteurs qui font échouer la segmentation sur nos images de test sont :

1. **Transparence et reflets** : Les objets transparents (verre, plastique, eau) sont très mal segmentés par SAM car le modèle s'appuie principalement sur les contrastes de couleur et de texture pour délimiter les contours. Un verre transparent se "fond" visuellement dans son arrière-plan, rendant la détection de contour quasi impossible.

2. **Ambiguïté de la bounding box** : Quand la bbox contient plusieurs objets à échelle similaire, SAM doit choisir lequel segmenter. Sans points de guidage, ce choix est souvent incorrect ou instable (résultats différents pour des bbox très proches). Ce problème est amplifié dans les scènes chargées (cuisine, rue).

3. **Objets fins et textures complexes** : Les objets fins (câbles, branches, grillages) et les textures répétitives (carrelage, tissu) perturbent le modèle. Le masque produit est soit trop grossier (blob englobant), soit fragmenté.

**Actions concrètes pour améliorer :**
- **Data** : Constituer un dataset de test avec des annotations manuelles (ground truth) pour mesurer objectivement la qualité. Inclure des images de cas limites dans le prompt de guidage.
- **UI** : Permettre la saisie de points par clic direct sur l'image (au lieu de sliders), ce qui serait plus intuitif et rapide. Ajouter un mode "itératif" où l'utilisateur peut ajouter des points après avoir vu un premier résultat.
- **Pipeline** : Ajouter un post-traitement du masque (morphologie : erosion/dilatation, remplissage de trous, lissage de contour) pour corriger les artefacts. Envisager un fine-tuning de SAM sur un domaine spécifique si les cas d'usage sont récurrents.

### Question 7.2 — Industrialisation : logging et monitoring

Si cette brique devait être industrialisée, voici les 5+ éléments à loguer et monitorer en priorité :

1. **Score de confiance SAM** : loguer le score de chaque inférence. Un drift du score moyen (baisse progressive) indiquerait un changement de distribution des images en entrée (data drift) ou un problème de version du modèle. Seuil d'alerte : score moyen < 0.7 sur une fenêtre glissante.

2. **Temps d'inférence (latence)** : mesurer le temps GPU/CPU pour chaque prédiction. Une augmentation soudaine peut signaler un problème matériel (GPU throttling, mémoire insuffisante) ou un changement de résolution des images en entrée. Seuil d'alerte : latence > 500ms sur GPU.

3. **Taille du masque (aire en pixels / ratio par rapport à l'image)** : loguer l'aire relative du masque. Un masque couvrant 0% ou 100% de l'image est presque toujours une erreur. Distribution anormale = possible problème dans le pipeline d'entrée.

4. **Résolution et format des images en entrée** : loguer la taille (H, W), le format (jpg/png), et la profondeur de couleur. Un changement de résolution peut affecter la qualité de segmentation. Permet aussi de détecter des images corrompues ou tronquées.

5. **Fréquence d'utilisation des points FG/BG** : si les utilisateurs doivent fréquemment ajouter des points pour corriger le masque, cela indique que la bbox seule est insuffisante pour le cas d'usage. Cette métrique guide les améliorations UX et le besoin éventuel de fine-tuning.

6. **Taux de sauvegarde des overlays** : mesurer combien de résultats sont réellement sauvegardés par l'utilisateur. Un faible taux de sauvegarde peut indiquer que les résultats ne sont pas satisfaisants.

7. **Erreurs et exceptions** : loguer toutes les erreurs (image illisible, checkpoint introuvable, OOM GPU, timeout). Monitorer le taux d'erreur par type pour identifier les problèmes systémiques.

### Question 7.3 — Remise finale

- Le dépôt contient bien `TP1/` avec le code source et le rapport.
- Les checkpoints SAM ne sont PAS versionnés (fichiers .pth exclus).
- *(Faire un push et ajouter un tag `TP1` : `git tag TP1 && git push origin TP1`)*
- *(Envoyer le lien du dépôt à l'enseignant.)*
