# TP2 — Génération d'images (Stable Diffusion, Diffusers)

## Dépendances

```bash
pip install -r TP2/requirements.txt
```

Depuis la racine du dépôt ou depuis `TP2/` :

## Smoke test

```bash
python TP2/smoke_test.py
```

Sortie : `TP2/outputs/smoke.png`

## Expériences

```bash
python TP2/experiments.py              # baseline uniquement -> outputs/baseline.png
python TP2/experiments.py t2i           # 6 runs text2img
python TP2/experiments.py i2i           # 3 runs img2img (+ placeholder si besoin)
python TP2/experiments.py all            # tout
```

## Streamlit

```bash
cd TP2
streamlit run app.py --server.port 8521 --server.address 0.0.0.0
```

Ne pas committer les gros modèles ni toutes les PNG générées : utiliser des captures dans `rapport.md`.
