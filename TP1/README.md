# TP1 — Segmentation interactive avec SAM (Segment Anything Model)

## Description

Ce TP implémente une mini-application de segmentation interactive d'images à l'aide de **SAM (Segment Anything Model)**.
L'application permet de sélectionner une image, définir une bounding box et des points de guidage (FG/BG),
puis lancer la segmentation pour obtenir un masque binaire avec visualisation overlay et métriques.

## Structure

```
TP1/
├── data/images/          # Images à segmenter (max 20)
├── models/               # Checkpoints SAM (non versionnés)
├── outputs/
│   ├── overlays/         # Images overlay générées
│   └── logs/             # Logs éventuels
├── report/
│   └── report.md         # Rapport du TP
├── src/
│   ├── app.py            # Application Streamlit
│   ├── sam_utils.py      # Chargement SAM et inférence
│   ├── geom_utils.py     # Métriques géométriques (aire, bbox, périmètre)
│   ├── viz_utils.py      # Visualisation overlay
│   ├── quick_test_sam.py # Test rapide SAM
│   └── quick_test_overlay.py # Test rapide overlay
├── requirements.txt      # Dépendances Python
└── README.md             # Ce fichier
```

## Installation

```bash
conda activate <env_name>
pip install -r TP1/requirements.txt
```

## Téléchargement du checkpoint SAM

```bash
mkdir -p TP1/models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O TP1/models/sam_vit_h_4b8939.pth
```

## Lancement

```bash
PORT=8511
streamlit run TP1/src/app.py --server.port $PORT --server.address 0.0.0.0
```

Puis ouvrir `http://localhost:$PORT` dans le navigateur (ou via SSH tunnel).
