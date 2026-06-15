# DÉMARRAGE RAPIDE — Pipeline Corpus Pular

## Structure du projet

```
projetPoular/
├── scripts/
│   ├── inventaire.py         # Étape 1 — Scanner et cataloguer tous tes fichiers
│   ├── transcription.py      # Étape 2 — Audio → Texte (Whisper)
│   ├── ocr_images.py         # Étape 3 — Images → Texte (Tesseract OCR)
│   ├── extraction_pdf.py     # Étape 4 — PDF → Texte (PyMuPDF)
│   ├── build_dataset.py      # Étape 5 — Assembler le dataset final
│   └── run_pipeline.py       # Orchestrateur (lance tout dans l'ordre)
├── corpus-pular/
│   ├── raw/                  # ← COPIE TES FICHIERS ICI
│   │   ├── audio/            #   tes fichiers .mp3, .wav, etc.
│   │   ├── images/           #   tes fichiers .jpg, .png, etc.
│   │   └── pdfs/             #   tes fichiers .pdf
│   ├── processed/            # Sorties automatiques du pipeline
│   ├── validated/            # Textes validés manuellement
│   ├── dataset/              # Dataset final (train.jsonl, validation.jsonl, test.jsonl)
│   └── metadata/             # Index, rapports, progression
├── logs/                     # Logs de chaque étape
└── requirements.txt
```

---

## Installation (une seule fois)

```bash
# 1. Installer les dépendances Python
pip install -r requirements.txt

# 2. Installer Tesseract OCR (pour les images)
#    Linux : sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
#    Windows : https://github.com/UB-Mannheim/tesseract/wiki

# 3. Installer ffmpeg (pour Whisper)
#    Linux : sudo apt install ffmpeg
#    Windows : winget install ffmpeg
```

---

## Utilisation

### Étape 0 — Copier tes fichiers

Place tes 150 000 fichiers dans :
- `corpus-pular/raw/audio/` → tous tes audios (.mp3, .wav, .ogg...)
- `corpus-pular/raw/images/` → toutes tes images (.jpg, .png, .tiff...)
- `corpus-pular/raw/pdfs/` → tous tes PDFs

Tu peux aussi pointer vers leur emplacement actuel avec `--corpus`.

---

### Étape 1 — Inventaire (COMMENCE ICI)

```bash
# Scanner tes fichiers là où ils sont actuellement
python scripts/inventaire.py --corpus /chemin/vers/tes/fichiers

# Ou s'ils sont déjà dans raw/
python scripts/inventaire.py
```

Ça génère :
- `corpus-pular/metadata/rapport_inventaire.json` → statistiques complètes
- `corpus-pular/metadata/index.csv` → un fichier = une ligne
- `corpus-pular/metadata/liste_audio.txt` → liste de tous les audios
- `corpus-pular/metadata/liste_image.txt` → liste de toutes les images
- `corpus-pular/metadata/liste_pdf.txt` → liste de tous les PDFs

---

### Test rapide sur 500 fichiers (recommandé avant le vrai lancement)

```bash
python scripts/run_pipeline.py --echantillon 500
```

---

### Pipeline complet (production)

```bash
# Avec GPU (recommandé)
python scripts/run_pipeline.py --whisper-model large-v3 --workers 1

# Sans GPU / CPU seulement
python scripts/run_pipeline.py --whisper-model small --workers 4
```

---

### Étapes individuelles

```bash
python scripts/run_pipeline.py --etape inventaire
python scripts/run_pipeline.py --etape transcription
python scripts/run_pipeline.py --etape ocr
python scripts/run_pipeline.py --etape pdf
python scripts/run_pipeline.py --etape dataset
```

---

### Voir l'état d'avancement

```bash
python scripts/run_pipeline.py --statut
```

---

## Reprise après interruption

Tous les scripts sauvegardent leur progression. Si tu coupes le process
(Ctrl+C, coupure réseau, etc.), relancer la même commande reprend
exactement où tu t'es arrêté.

---

## Hardware recommandé pour 150 000 fichiers

| Étape              | RAM   | GPU VRAM | Durée estimée         |
|--------------------|-------|----------|-----------------------|
| Inventaire         | 4 GB  | —        | 10–30 min             |
| Transcription audio| 16 GB | 8 GB+    | 50–200 heures (GPU)   |
| OCR images         | 8 GB  | —        | 20–50 heures (8 CPU)  |
| Extraction PDF     | 8 GB  | —        | 5–15 heures (4 CPU)   |
| Build dataset      | 16 GB | —        | 1–3 heures            |

**Sur ton serveur srv1359704 :** Lance d'abord le test sur 500 fichiers
pour estimer les durées réelles sur ta machine.

Pour la transcription de 80 000 audios avec GPU A100 (RunPod) :
→ ~20–40 heures à ~1.20$/heure = **25–50$**

---

## Résultat attendu

Après le pipeline complet, tu auras dans `corpus-pular/dataset/` :

```
train.jsonl           # ~90% des données — format pour fine-tuning LLM
validation.jsonl      # ~5% — évaluation pendant l'entraînement
test.jsonl            # ~5% — évaluation finale
rapport_dataset.json  # statistiques : nb tokens, répartition par source/domaine
```

Format de chaque ligne (compatible HuggingFace SFTTrainer) :
```json
{"text": "Subahi waktu wuri. Selu on daña Allah yo barke ma..."}
```

---

## Prochaine étape après le dataset

Fine-tuning sur RunPod/Vast.ai :
```bash
# Sur un serveur cloud avec GPU A100
pip install -r requirements.txt
python scripts/finetune.py  # (à créer — étape suivante)
```
