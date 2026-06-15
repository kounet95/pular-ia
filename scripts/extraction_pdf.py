"""
extraction_pdf.py — Extraction de texte depuis les PDFs du corpus pular
Gère PDFs natifs (texte sélectionnable) et PDFs scannés (via OCR).

Usage:
    python scripts/extraction_pdf.py
    python scripts/extraction_pdf.py --corpus /chemin/vers/pdfs --workers 8
    python scripts/extraction_pdf.py --echantillon 100
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
        logging.FileHandler("logs/extraction_pdf.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DOSSIER_OUTPUT  = Path("./corpus-pular/processed/pdf_text")
DOSSIER_RAW     = Path("./corpus-pular/raw/pdfs")
FICHIER_PROGRES = Path("./corpus-pular/metadata/progres_pdf.json")

# Seuil minimum de texte natif avant de recourir à l'OCR
MIN_CHARS_NATIF = 100


# ── Détection type de PDF ─────────────────────────────────────────────────────
def est_pdf_scanne(chemin: Path) -> bool:
    """Retourne True si le PDF semble être un scan (peu ou pas de texte natif)."""
    try:
        import pymupdf
        doc = pymupdf.open(str(chemin))
        total_chars = sum(len(page.get_text().strip()) for page in doc)
        doc.close()
        return total_chars < MIN_CHARS_NATIF
    except Exception:
        return False


# ── Extraction texte natif ────────────────────────────────────────────────────
def extraire_texte_natif(chemin: Path) -> dict:
    """Extrait le texte des PDFs qui ont du texte sélectionnable."""
    import pymupdf

    doc = pymupdf.open(str(chemin))
    pages_data = []
    texte_total = []

    for i, page in enumerate(doc):
        texte_page = page.get_text("text").strip()
        texte_total.append(texte_page)

        # Récupérer aussi les blocs avec position (utile pour les hadiths paginés)
        blocs = page.get_text("blocks")
        blocs_propres = [
            {
                "texte": b[4].strip(),
                "bbox":  [round(v, 1) for v in b[:4]],
            }
            for b in blocs if b[4].strip()
        ]

        pages_data.append({
            "numero":     i + 1,
            "texte":      texte_page,
            "nb_chars":   len(texte_page),
            "nb_blocs":   len(blocs_propres),
        })

    doc.close()
    texte_complet = "\n\n".join(texte_total).strip()

    return {
        "methode":      "natif",
        "texte":        texte_complet,
        "nb_pages":     len(pages_data),
        "nb_chars":     len(texte_complet),
        "nb_tokens_est": len(texte_complet.split()),
        "pages":        pages_data,
    }


# ── Extraction par OCR pour PDFs scannés ─────────────────────────────────────
def extraire_texte_ocr(chemin: Path, langues: str = "fra+ara") -> dict:
    """Convertit chaque page du PDF en image puis applique l'OCR."""
    import pymupdf
    from PIL import Image
    import pytesseract
    import io

    doc = pymupdf.open(str(chemin))
    pages_data = []
    texte_total = []

    for i, page in enumerate(doc):
        # Rasteriser la page à 300 DPI pour bonne qualité OCR
        mat = pymupdf.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convertir en image PIL
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        texte_page = pytesseract.image_to_string(img, lang=langues, config="--psm 6")
        texte_page = texte_page.strip()
        texte_total.append(texte_page)

        pages_data.append({
            "numero":   i + 1,
            "texte":    texte_page,
            "nb_chars": len(texte_page),
        })

    doc.close()
    texte_complet = "\n\n".join(texte_total).strip()

    return {
        "methode":      "ocr",
        "texte":        texte_complet,
        "nb_pages":     len(pages_data),
        "nb_chars":     len(texte_complet),
        "nb_tokens_est": len(texte_complet.split()),
        "pages":        pages_data,
    }


# ── Nettoyage du texte extrait ────────────────────────────────────────────────
def nettoyer_texte(texte: str) -> str:
    """Nettoyage basique du texte extrait — conserver la structure."""
    import re
    # Supprimer caractères de contrôle sauf newlines
    texte = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", texte)
    # Normaliser espaces multiples
    texte = re.sub(r"[ \t]{2,}", " ", texte)
    # Normaliser lignes vides multiples
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    # Supprimer lignes composées uniquement de symboles parasites
    lignes = [l for l in texte.splitlines() if len(l.strip()) > 1 or l.strip().isalpha()]
    return "\n".join(lignes).strip()


# ── Traitement d'un PDF (appelé dans worker) ──────────────────────────────────
def traiter_pdf(args_tuple) -> dict:
    chemin_str, langues_ocr = args_tuple
    chemin = Path(chemin_str)

    output_path = DOSSIER_OUTPUT / (chemin.stem + ".json")
    if output_path.exists():
        return {"statut": "deja_traite", "fichier": chemin_str}

    debut = time.time()
    try:
        # Décider méthode
        scanne = est_pdf_scanne(chemin)

        if scanne:
            extraction = extraire_texte_ocr(chemin, langues=langues_ocr)
        else:
            extraction = extraire_texte_natif(chemin)

        # Nettoyage
        extraction["texte"] = nettoyer_texte(extraction["texte"])

        # Score qualité simple
        mots = extraction["texte"].split()
        score = min(len(mots) / 100, 1.0)  # score proportionnel au nb de mots (max 100)

        sortie = {
            "fichier":          chemin_str,
            "nom":              chemin.name,
            "texte":            extraction["texte"],
            "methode":          extraction["methode"],
            "nb_pages":         extraction["nb_pages"],
            "nb_chars":         extraction["nb_chars"],
            "nb_tokens_est":    extraction["nb_tokens_est"],
            "score_qualite":    round(score, 3),
            "categorie_qualite": (
                "haute"   if score >= 0.7 else
                "moyenne" if score >= 0.3 else
                "basse"
            ),
            "pages_detail":     extraction.get("pages", []),
            "meta": {
                "duree_s":       round(time.time() - debut, 2),
                "langues_ocr":   langues_ocr if scanne else None,
                "date":          datetime.now().isoformat(),
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


# ── Collecte ──────────────────────────────────────────────────────────────────
def collecter_pdfs(dossier=None, liste_txt=None, echantillon=None) -> list:
    fichiers = []
    if liste_txt:
        with open(liste_txt, encoding="utf-8") as f:
            fichiers = [l.strip() for l in f if l.strip()]
    elif dossier:
        fichiers = [str(p) for p in Path(dossier).rglob("*.pdf")]
    else:
        fichiers = [str(p) for p in DOSSIER_RAW.rglob("*.pdf")]

    fichiers = [f for f in fichiers if Path(f).exists()]

    if echantillon and echantillon < len(fichiers):
        import random
        random.shuffle(fichiers)
        fichiers = fichiers[:echantillon]

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


# ── Pipeline ──────────────────────────────────────────────────────────────────
def lancer_extraction(fichiers: list, langues_ocr: str, workers: int):
    deja_traites = charger_progres()
    a_traiter    = [f for f in fichiers if f not in deja_traites]

    log.info(f"Total PDFs       : {len(fichiers):,}")
    log.info(f"Déjà traités     : {len(deja_traites):,}")
    log.info(f"À traiter        : {len(a_traiter):,}")
    log.info(f"Workers          : {workers}")

    if not a_traiter:
        log.info("✅ Tous les PDFs ont déjà été traités !")
        return

    stats        = {"ok": 0, "erreur": 0, "deja_traite": 0}
    methodes     = {"natif": 0, "ocr": 0}
    traites      = set(deja_traites)
    debut_global = time.time()
    total_tokens = 0

    args_list = [(f, langues_ocr) for f in a_traiter]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(traiter_pdf, arg): arg[0] for arg in args_list}

        for i, future in enumerate(as_completed(futures), 1):
            fichier = futures[future]
            try:
                result = future.result()
                statut = result.get("statut", "erreur")
                stats[statut if statut in stats else "erreur"] += 1

                if statut == "ok":
                    traites.add(fichier)
                    methode  = result.get("methode", "?")
                    methodes[methode] = methodes.get(methode, 0) + 1
                    tokens   = result.get("nb_tokens_est", 0)
                    total_tokens += tokens
                    log.info(
                        f"[{i:>5}/{len(a_traiter):,}] ✅ {Path(fichier).name} "
                        f"— {result['nb_pages']}p | {tokens:,} mots | {methode}"
                    )
                elif statut == "erreur":
                    log.warning(f"[{i:>5}/{len(a_traiter):,}] ❌ {Path(fichier).name}")

            except Exception as e:
                stats["erreur"] += 1
                log.error(f"Exception : {e}")

            if i % 100 == 0:
                sauvegarder_progres(traites)

    sauvegarder_progres(traites)
    elapsed = time.time() - debut_global

    log.info("\n" + "═" * 50)
    log.info("  EXTRACTION PDF TERMINÉE")
    log.info(f"  ✅ Succès   : {stats['ok']:,}")
    log.info(f"  ❌ Erreurs  : {stats['erreur']:,}")
    log.info(f"  📄 Natif    : {methodes.get('natif', 0):,}")
    log.info(f"  🔍 OCR      : {methodes.get('ocr', 0):,}")
    log.info(f"  📝 Tokens   : {total_tokens:,} estimés")
    log.info(f"  ⏱️  Durée    : {elapsed/60:.1f} min")
    log.info("═" * 50)


def main():
    parser = argparse.ArgumentParser(description="Extraction texte PDFs corpus pular")
    parser.add_argument("--corpus",      default=None,         help="Dossier PDFs")
    parser.add_argument("--liste",       default=None,         help="Fichier .txt chemins PDFs")
    parser.add_argument("--langues",     default="fra+ara",    help="Langues OCR pour PDFs scannés")
    parser.add_argument("--workers",     default=4, type=int,  help="Processus parallèles")
    parser.add_argument("--echantillon", default=None,type=int,help="N fichiers pour test")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    log.info("=" * 50)
    log.info("  PIPELINE EXTRACTION PDF PULAR")
    log.info(f"  Démarré : {datetime.now().isoformat()}")
    log.info("=" * 50)

    fichiers = collecter_pdfs(dossier=args.corpus, liste_txt=args.liste, echantillon=args.echantillon)

    if not fichiers:
        log.error("❌ Aucun PDF trouvé. Utilise --corpus pour spécifier le dossier")
        return

    log.info(f"PDFs trouvés : {len(fichiers):,}")
    lancer_extraction(fichiers, langues_ocr=args.langues, workers=args.workers)
    log.info("\n✅ Prochaine étape : python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
