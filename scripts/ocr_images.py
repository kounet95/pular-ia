"""
ocr_images.py — Extraction de texte depuis les images du corpus pular
Supporte manuscrits, livres scannés, documents photographiés.

Usage:
    python scripts/ocr_images.py
    python scripts/ocr_images.py --corpus /chemin/vers/images --workers 8
    python scripts/ocr_images.py --echantillon 200
"""

import os
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ocr.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
DOSSIER_OUTPUT   = Path("./corpus-pular/processed/ocr")
DOSSIER_RAW      = Path("./corpus-pular/raw/images")
FICHIER_PROGRES  = Path("./corpus-pular/metadata/progres_ocr.json")

# Tesseract — langues installées pour les documents du corpus pular
# Le pular s'écrit avec l'alphabet latin ou arabe selon le contexte
# Installer les langues : sudo apt install tesseract-ocr-fra tesseract-ocr-ara
LANGUES_OCR = "fra+ara"   # français + arabe comme proxy pour pular (pas de modèle natif)


# ── Score qualité du texte extrait ────────────────────────────────────────────
def score_qualite(texte: str) -> float:
    """
    Estime la qualité de l'OCR entre 0 et 1.
    Un texte de bonne qualité a des mots reconnaissables et peu de caractères parasites.
    """
    if not texte or len(texte) < 10:
        return 0.0

    mots = texte.split()
    if not mots:
        return 0.0

    # Ratio de mots "lisibles" (longueur 2-20 caractères, surtout alpha)
    mots_valides = [
        m for m in mots
        if 2 <= len(m) <= 20 and sum(c.isalpha() for c in m) / len(m) > 0.6
    ]
    ratio = len(mots_valides) / len(mots)

    # Pénalité si trop de caractères spéciaux parasites
    chars_parasites = sum(1 for c in texte if c in "|\\~^@#%<>{}[]_=")
    penalite = min(chars_parasites / max(len(texte), 1), 0.5)

    return max(0.0, round(ratio - penalite, 3))


# ── Prétraitement image pour améliorer l'OCR ─────────────────────────────────
def pretraiter_image(img):
    """Convertit en niveaux de gris et améliore le contraste pour l'OCR."""
    import cv2
    import numpy as np

    # Convertir PIL → numpy si nécessaire
    img_np = np.array(img)

    # Niveaux de gris
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Débruitage
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Binarisation adaptative (meilleure pour texte sur fond variable)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    from PIL import Image
    return Image.fromarray(binary)


# ── OCR d'un fichier image ────────────────────────────────────────────────────
def ocr_fichier(args_tuple) -> dict:
    """Traitement d'une image — fonction appelée dans le processus worker."""
    chemin_str, langues = args_tuple
    chemin = Path(chemin_str)

    output_path = DOSSIER_OUTPUT / (chemin.stem + ".json")
    if output_path.exists():
        return {"statut": "deja_traite", "fichier": chemin_str}

    debut = time.time()
    try:
        from PIL import Image
        import pytesseract

        img = Image.open(chemin).convert("RGB")
        largeur, hauteur = img.size

        # Prétraitement si l'image est de basse qualité (petite résolution)
        if largeur < 800 or hauteur < 800:
            # Agrandir pour améliorer l'OCR
            facteur = max(800 / largeur, 800 / hauteur, 1)
            img = img.resize(
                (int(largeur * facteur), int(hauteur * facteur)),
                Image.LANCZOS
            )

        try:
            img_pre = pretraiter_image(img)
        except Exception:
            img_pre = img  # fallback sans prétraitement si cv2 absent

        # OCR avec données de confiance par mot
        data = pytesseract.image_to_data(
            img_pre,
            lang=langues,
            output_type=pytesseract.Output.DICT,
            config="--psm 6"   # PSM 6 = bloc de texte uniforme
        )

        # Reconstruction du texte avec filtrage par confiance
        mots_haute_conf = []
        mots_tous = []
        for i, mot in enumerate(data["text"]):
            conf = data["conf"][i]
            if mot.strip():
                mots_tous.append(mot)
                if conf > 30:   # seuil de confiance (0-100)
                    mots_haute_conf.append(mot)

        texte_complet    = " ".join(mots_tous).strip()
        texte_filtre     = " ".join(mots_haute_conf).strip()
        qualite          = score_qualite(texte_filtre)

        sortie = {
            "fichier":          chemin_str,
            "nom":              chemin.name,
            "texte":            texte_filtre,
            "texte_brut":       texte_complet,
            "score_qualite":    qualite,
            "categorie_qualite": (
                "haute"   if qualite >= 0.7 else
                "moyenne" if qualite >= 0.4 else
                "basse"
            ),
            "meta": {
                "langues_ocr":         langues,
                "resolution_originale": f"{largeur}x{hauteur}",
                "nb_mots":             len(mots_tous),
                "nb_mots_valides":     len(mots_haute_conf),
                "duree_traitement_s":  round(time.time() - debut, 2),
                "date":                datetime.now().isoformat(),
            },
            "statut": "ok",
        }

        DOSSIER_OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sortie, f, ensure_ascii=False, indent=2)

        return sortie

    except Exception as e:
        return {
            "fichier": chemin_str,
            "statut":  "erreur",
            "erreur":  str(e),
            "trace":   traceback.format_exc(),
        }


# ── Collecte des fichiers ─────────────────────────────────────────────────────
def collecter_images(dossier=None, liste_txt=None, echantillon=None) -> list:
    fichiers = []
    if liste_txt:
        with open(liste_txt, encoding="utf-8") as f:
            fichiers = [line.strip() for line in f if line.strip()]
    elif dossier:
        for ext in EXTENSIONS_IMAGE:
            fichiers.extend(str(p) for p in Path(dossier).rglob(f"*{ext}"))
    else:
        for ext in EXTENSIONS_IMAGE:
            fichiers.extend(str(p) for p in DOSSIER_RAW.rglob(f"*{ext}"))

    fichiers = [f for f in fichiers if Path(f).exists()]

    if echantillon and echantillon < len(fichiers):
        import random
        random.shuffle(fichiers)
        fichiers = fichiers[:echantillon]
        log.info(f"Mode échantillon : {echantillon} images sélectionnées")

    return fichiers


def charger_progres() -> set:
    if FICHIER_PROGRES.exists():
        with open(FICHIER_PROGRES, encoding="utf-8") as f:
            return set(json.load(f).get("traites", []))
    return set()


def sauvegarder_progres(traites: set):
    FICHIER_PROGRES.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_PROGRES, "w", encoding="utf-8") as f:
        json.dump({"traites": list(traites), "derniere_maj": datetime.now().isoformat()}, f)


# ── Pipeline principal ─────────────────────────────────────────────────────────
def lancer_ocr(fichiers: list, langues: str, workers: int):
    deja_traites = charger_progres()
    a_traiter    = [f for f in fichiers if f not in deja_traites]

    log.info(f"Total images     : {len(fichiers):,}")
    log.info(f"Déjà traitées    : {len(deja_traites):,}")
    log.info(f"À traiter        : {len(a_traiter):,}")
    log.info(f"Langues OCR      : {langues}")
    log.info(f"Workers          : {workers}")

    if not a_traiter:
        log.info("✅ Toutes les images ont déjà été traitées !")
        return

    stats        = {"ok": 0, "erreur": 0, "deja_traite": 0}
    scores       = {"haute": 0, "moyenne": 0, "basse": 0}
    traites      = set(deja_traites)
    debut_global = time.time()

    # ProcessPoolExecutor pour vrai parallélisme (OCR est CPU-bound)
    args_list = [(f, langues) for f in a_traiter]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ocr_fichier, arg): arg[0] for arg in args_list}

        for i, future in enumerate(as_completed(futures), 1):
            fichier = futures[future]
            try:
                result = future.result()
                statut = result.get("statut", "erreur")
                stats[statut if statut in stats else "erreur"] += 1

                if statut == "ok":
                    traites.add(fichier)
                    cat = result.get("categorie_qualite", "?")
                    scores[cat] = scores.get(cat, 0) + 1
                    log.info(
                        f"[{i:>6}/{len(a_traiter):,}] ✅ {Path(fichier).name} "
                        f"— qualité {cat} ({result['score_qualite']:.2f})"
                    )
                elif statut == "erreur":
                    log.warning(f"[{i:>6}/{len(a_traiter):,}] ❌ {Path(fichier).name}: {result.get('erreur', '')[:80]}")

            except Exception as e:
                stats["erreur"] += 1
                log.error(f"Exception future {fichier}: {e}")

            if i % 200 == 0:
                sauvegarder_progres(traites)
                elapsed = time.time() - debut_global
                vitesse = i / elapsed if elapsed > 0 else 1
                restant = (len(a_traiter) - i) / vitesse
                log.info(f"📊 {i:,}/{len(a_traiter):,} | {vitesse:.1f} img/s | restant ~{restant/60:.0f} min")

    sauvegarder_progres(traites)
    elapsed = time.time() - debut_global

    log.info("\n" + "═" * 50)
    log.info("  OCR TERMINÉ")
    log.info(f"  ✅ Succès       : {stats['ok']:,}")
    log.info(f"  ❌ Erreurs      : {stats['erreur']:,}")
    log.info(f"  Qualité haute   : {scores.get('haute', 0):,}")
    log.info(f"  Qualité moyenne : {scores.get('moyenne', 0):,}")
    log.info(f"  Qualité basse   : {scores.get('basse', 0):,}")
    log.info(f"  ⏱️  Durée        : {elapsed/60:.1f} min")
    log.info("═" * 50)


def main():
    parser = argparse.ArgumentParser(description="OCR images du corpus pular")
    parser.add_argument("--corpus",      default=None,         help="Dossier contenant les images")
    parser.add_argument("--liste",       default=None,         help="Fichier .txt avec chemins d'images")
    parser.add_argument("--langues",     default=LANGUES_OCR,  help="Langues Tesseract (ex: fra+ara)")
    parser.add_argument("--workers",     default=4, type=int,  help="Processus parallèles")
    parser.add_argument("--echantillon", default=None,type=int,help="Traiter seulement N images (test)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    log.info("=" * 50)
    log.info("  PIPELINE OCR IMAGES PULAR")
    log.info(f"  Démarré : {datetime.now().isoformat()}")
    log.info("=" * 50)

    fichiers = collecter_images(dossier=args.corpus, liste_txt=args.liste, echantillon=args.echantillon)

    if not fichiers:
        log.error("❌ Aucune image trouvée. Utilise --corpus pour spécifier le dossier")
        return

    log.info(f"Images trouvées : {len(fichiers):,}")
    lancer_ocr(fichiers, langues=args.langues, workers=args.workers)
    log.info("\n✅ Prochaine étape : python scripts/extraction_pdf.py")


if __name__ == "__main__":
    main()
