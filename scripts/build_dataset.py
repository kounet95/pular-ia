"""
build_dataset.py — Assemblage et nettoyage du dataset final pular
Agrège les sorties de :
  - transcription.py    (audios → texte)
  - ocr_images.py       (images → texte)
  - extraction_pdf.py   (PDFs → texte)
  - telegram_scraper.py (canaux Telegram → texte + transcriptions)

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --min-tokens 20 --split 0.9
    python scripts/build_dataset.py --sans-telegram   # ignorer les données Telegram
"""

import json
import argparse
import logging
import hashlib
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/build_dataset.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DOSSIER_TRANSCRIPTIONS = Path("./corpus-pular/processed/transcriptions")
DOSSIER_OCR            = Path("./corpus-pular/processed/ocr")
DOSSIER_PDF            = Path("./corpus-pular/processed/pdf_text")
DOSSIER_TELEGRAM       = Path("./corpus-pular/processed/telegram/jsonl")
DOSSIER_DATASET        = Path("./corpus-pular/dataset")


# ── Détection de domaine du texte ─────────────────────────────────────────────
MOTS_ISLAMIQUES = {
    # Pular
    "juulde", "salaat", "subahi", "fajri", "aljumaa", "ramadan", "koorka",
    "qur", "hadith", "annabi", "allah", "allahu", "bismillah", "barke",
    "zakkat", "haajira", "wudu", "kiblat", "rak'a", "takkiiru",
    # Arabe translittéré
    "sallallahu", "alayhi", "wasallam", "inshallah", "mashallah", "alhamdulillah",
    # Français contexte islamique
    "prière", "mosquée", "prophète", "coran", "hadith", "oumma",
}

MOTS_CULTURELS = {
    "pullo", "peul", "fula", "fulbe", "fulani", "pular", "djeli",
    "laobe", "mbooro", "wodaabe", "boode",
}


def detecter_domaine(texte: str) -> str:
    texte_lower = texte.lower()
    mots = set(texte_lower.split())

    score_islamique = len(mots & MOTS_ISLAMIQUES)
    score_culturel  = len(mots & MOTS_CULTURELS)

    if score_islamique >= 2:
        return "islamique"
    elif score_culturel >= 2:
        return "culturel_peul"
    elif score_islamique == 1:
        return "islamique_probable"
    else:
        return "general"


# ── Nettoyage et normalisation du texte ───────────────────────────────────────
def nettoyer(texte: str) -> str:
    if not texte:
        return ""
    # Supprimer caractères de contrôle
    texte = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", texte)
    # Normaliser espaces
    texte = re.sub(r"[ \t]{2,}", " ", texte)
    # Normaliser sauts de ligne
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    return texte.strip()


def hash_texte(texte: str) -> str:
    """Hash pour déduplication."""
    return hashlib.md5(texte.encode("utf-8")).hexdigest()


def est_valide(texte: str, min_tokens: int, min_alpha_ratio: float = 0.5) -> tuple[bool, str]:
    """
    Vérifie si un texte est suffisamment propre pour le dataset.
    Retourne (valide: bool, raison: str)
    """
    if not texte or not texte.strip():
        return False, "texte_vide"

    tokens = texte.split()
    if len(tokens) < min_tokens:
        return False, f"trop_court ({len(tokens)} < {min_tokens} tokens)"

    # Ratio de caractères alphabétiques
    nb_alpha = sum(c.isalpha() for c in texte)
    ratio    = nb_alpha / max(len(texte), 1)
    if ratio < min_alpha_ratio:
        return False, f"trop_de_symboles (ratio_alpha={ratio:.2f})"

    # Longueur maximale raisonnable (éviter les dumps complets)
    if len(tokens) > 50_000:
        return False, "trop_long"

    return True, "ok"


# ── Chargement des sources ────────────────────────────────────────────────────
def charger_transcriptions(min_tokens: int) -> list[dict]:
    entrees = []
    jsons   = list(DOSSIER_TRANSCRIPTIONS.glob("*.json"))
    log.info(f"Chargement transcriptions : {len(jsons):,} fichiers")

    for f in jsons:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            if data.get("statut") != "ok":
                continue

            texte = nettoyer(data.get("texte", ""))
            valide, raison = est_valide(texte, min_tokens)
            if not valide:
                continue

            entrees.append({
                "texte":        texte,
                "source":       "audio_transcrit",
                "fichier_orig": data.get("fichier", ""),
                "langue":       "pular",
                "domaine":      detecter_domaine(texte),
                "nb_tokens":    len(texte.split()),
                "hash":         hash_texte(texte),
                "meta": {
                    "modele_whisper": data.get("meta", {}).get("modele", ""),
                    "langue_detect":  data.get("langue_detect", ""),
                    "nb_segments":    data.get("meta", {}).get("nb_segments", 0),
                },
            })

            # Ajouter aussi les segments individuels (phrases)
            for seg in data.get("segments", []):
                texte_seg = nettoyer(seg.get("texte", ""))
                valide_seg, _ = est_valide(texte_seg, min_tokens // 2)
                if valide_seg and not seg.get("sans_voix"):
                    entrees.append({
                        "texte":        texte_seg,
                        "source":       "segment_audio",
                        "fichier_orig": data.get("fichier", ""),
                        "langue":       "pular",
                        "domaine":      detecter_domaine(texte_seg),
                        "nb_tokens":    len(texte_seg.split()),
                        "hash":         hash_texte(texte_seg),
                        "meta": {
                            "debut_s": seg.get("debut"),
                            "fin_s":   seg.get("fin"),
                        },
                    })
        except Exception as e:
            log.warning(f"Erreur lecture {f.name}: {e}")

    log.info(f"  → {len(entrees):,} entrées chargées depuis les transcriptions")
    return entrees


def charger_ocr(min_tokens: int) -> list[dict]:
    entrees = []
    jsons   = list(DOSSIER_OCR.glob("*.json"))
    log.info(f"Chargement OCR : {len(jsons):,} fichiers")

    for f in jsons:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            if data.get("statut") != "ok":
                continue
            if data.get("score_qualite", 0) < 0.3:   # filtrer OCR de mauvaise qualité
                continue

            texte = nettoyer(data.get("texte", ""))
            valide, _ = est_valide(texte, min_tokens)
            if not valide:
                continue

            entrees.append({
                "texte":        texte,
                "source":       "ocr_image",
                "fichier_orig": data.get("fichier", ""),
                "langue":       "pular",
                "domaine":      detecter_domaine(texte),
                "nb_tokens":    len(texte.split()),
                "hash":         hash_texte(texte),
                "meta": {
                    "score_ocr":     data.get("score_qualite", 0),
                    "categorie_ocr": data.get("categorie_qualite", ""),
                },
            })
        except Exception as e:
            log.warning(f"Erreur lecture {f.name}: {e}")

    log.info(f"  → {len(entrees):,} entrées chargées depuis OCR images")
    return entrees


def charger_pdfs(min_tokens: int) -> list[dict]:
    entrees = []
    jsons   = list(DOSSIER_PDF.glob("*.json"))
    log.info(f"Chargement PDFs : {len(jsons):,} fichiers")

    for f in jsons:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)

            if data.get("statut") != "ok":
                continue

            # Traiter page par page pour les longs documents
            pages = data.get("pages_detail", [])
            if pages:
                for page in pages:
                    texte_page = nettoyer(page.get("texte", ""))
                    valide, _ = est_valide(texte_page, min_tokens)
                    if valide:
                        entrees.append({
                            "texte":        texte_page,
                            "source":       "pdf_page",
                            "fichier_orig": data.get("fichier", ""),
                            "langue":       "pular",
                            "domaine":      detecter_domaine(texte_page),
                            "nb_tokens":    len(texte_page.split()),
                            "hash":         hash_texte(texte_page),
                            "meta": {
                                "page":    page.get("numero"),
                                "methode": data.get("methode", ""),
                            },
                        })
            else:
                texte = nettoyer(data.get("texte", ""))
                valide, _ = est_valide(texte, min_tokens)
                if valide:
                    entrees.append({
                        "texte":        texte,
                        "source":       "pdf_complet",
                        "fichier_orig": data.get("fichier", ""),
                        "langue":       "pular",
                        "domaine":      detecter_domaine(texte),
                        "nb_tokens":    len(texte.split()),
                        "hash":         hash_texte(texte),
                        "meta":         {"methode": data.get("methode", "")},
                    })
        except Exception as e:
            log.warning(f"Erreur lecture {f.name}: {e}")

    log.info(f"  → {len(entrees):,} entrées chargées depuis PDFs")
    return entrees


# ── Chargement données Telegram ───────────────────────────────────────────────
def charger_telegram(min_tokens: int) -> list[dict]:
    """
    Lit les fichiers JSONL produits par telegram_scraper.py.
    Chaque ligne est déjà au bon format — on filtre juste sur la longueur.
    """
    entrees = []

    if not DOSSIER_TELEGRAM.exists():
        log.info("Dossier Telegram introuvable — étape Telegram skippée")
        return entrees

    jsonls = list(DOSSIER_TELEGRAM.glob("*.jsonl"))
    log.info(f"Chargement Telegram : {len(jsonls)} fichier(s) JSONL")

    for f in jsonls:
        try:
            with open(f, encoding="utf-8") as fp:
                for ligne in fp:
                    ligne = ligne.strip()
                    if not ligne:
                        continue
                    data = json.loads(ligne)

                    if data.get("statut") != "ok":
                        continue

                    texte = nettoyer(data.get("texte", ""))
                    valide, _ = est_valide(texte, min_tokens)
                    if not valide:
                        continue

                    entrees.append({
                        "texte":        texte,
                        "source":       data.get("source", "telegram"),
                        "fichier_orig": data.get("fichier", ""),
                        "langue":       data.get("langue", "pular"),
                        "domaine":      data.get("domaine", detecter_domaine(texte)),
                        "nb_tokens":    len(texte.split()),
                        "hash":         data.get("hash", hash_texte(texte)),
                        "meta":         data.get("meta", {}),
                    })
        except Exception as e:
            log.warning(f"Erreur lecture Telegram {f.name}: {e}")

    log.info(f"  → {len(entrees):,} entrées chargées depuis Telegram")
    return entrees


# ── Déduplication ─────────────────────────────────────────────────────────────
def dedupliquer(entrees: list[dict]) -> list[dict]:
    vus  = set()
    uniq = []
    for e in entrees:
        h = e["hash"]
        if h not in vus:
            vus.add(h)
            uniq.append(e)
    n_doublons = len(entrees) - len(uniq)
    log.info(f"Déduplication : {len(entrees):,} → {len(uniq):,} (supprimé {n_doublons:,} doublons)")
    return uniq


# ── Split train / validation / test ───────────────────────────────────────────
def splitter(entrees: list[dict], ratio_train: float = 0.9, ratio_val: float = 0.05) -> tuple:
    import random
    random.shuffle(entrees)
    n = len(entrees)
    n_train = int(n * ratio_train)
    n_val   = int(n * ratio_val)

    train = entrees[:n_train]
    val   = entrees[n_train:n_train + n_val]
    test  = entrees[n_train + n_val:]

    return train, val, test


# ── Sauvegarder en JSONL (format HuggingFace) ─────────────────────────────────
def sauvegarder_jsonl(entrees: list[dict], chemin: Path, format_llm: bool = True):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        for e in entrees:
            if format_llm:
                # Format pour SFTTrainer / fine-tuning LLM
                ligne = {"text": e["texte"]}
            else:
                ligne = e  # format complet avec métadonnées
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    log.info(f"  💾 {chemin.name} → {len(entrees):,} entrées")


# ── Rapport final ─────────────────────────────────────────────────────────────
def generer_rapport(train, val, test, chemin: Path):
    tout = train + val + test
    tokens_total = sum(e["nb_tokens"] for e in tout)
    sources      = Counter(e["source"] for e in tout)
    domaines     = Counter(e["domaine"] for e in tout)

    rapport = {
        "date":            datetime.now().isoformat(),
        "total_entrees":   len(tout),
        "tokens_total":    tokens_total,
        "tokens_millions": round(tokens_total / 1_000_000, 2),
        "split": {
            "train":      len(train),
            "validation": len(val),
            "test":       len(test),
        },
        "par_source":  dict(sources),
        "par_domaine": dict(domaines),
    }

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    log.info("\n" + "═" * 55)
    log.info("  📊 DATASET PULAR — RÉSUMÉ FINAL")
    log.info("═" * 55)
    log.info(f"  Total entrées    : {len(tout):,}")
    log.info(f"  Tokens estimés   : {tokens_total:,} ({tokens_total/1e6:.1f}M)")
    log.info(f"  Train            : {len(train):,}")
    log.info(f"  Validation       : {len(val):,}")
    log.info(f"  Test             : {len(test):,}")
    log.info("")
    log.info("  Par source :")
    for src, count in sources.most_common():
        log.info(f"    {src:<25} : {count:,}")
    log.info("")
    log.info("  Par domaine :")
    for dom, count in domaines.most_common():
        log.info(f"    {dom:<25} : {count:,}")
    log.info("═" * 55)


def main():
    parser = argparse.ArgumentParser(description="Construction dataset pular final")
    parser.add_argument("--min-tokens",    default=20,   type=int,   help="Nb minimum de tokens par entrée")
    parser.add_argument("--split",         default=0.9,  type=float, help="Ratio train (défaut: 0.9)")
    parser.add_argument("--format-llm",    action="store_true", default=True,
                        help="Exporter au format LLM {text: ...} (défaut: True)")
    parser.add_argument("--sans-telegram", action="store_true",
                        help="Ignorer les données Telegram (build_dataset sans Telegram)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    log.info("=" * 55)
    log.info("  BUILD DATASET PULAR")
    log.info(f"  Démarré : {datetime.now().isoformat()}")
    log.info("=" * 55)

    # Charger toutes les sources
    toutes_entrees = []
    toutes_entrees.extend(charger_transcriptions(args.min_tokens))
    toutes_entrees.extend(charger_ocr(args.min_tokens))
    toutes_entrees.extend(charger_pdfs(args.min_tokens))

    if not args.sans_telegram:
        toutes_entrees.extend(charger_telegram(args.min_tokens))

    if not toutes_entrees:
        log.error(
            "❌ Aucune entrée trouvée. Lance d'abord :\n"
            "   python scripts/transcription.py\n"
            "   python scripts/ocr_images.py\n"
            "   python scripts/extraction_pdf.py\n"
            "   python scripts/telegram_scraper.py"
        )
        return

    log.info(f"\nTotal avant déduplication : {len(toutes_entrees):,}")

    # Déduplication
    toutes_entrees = dedupliquer(toutes_entrees)

    # Split
    train, val, test = splitter(toutes_entrees, ratio_train=args.split)

    # Sauvegarder — format LLM (pour fine-tuning)
    sauvegarder_jsonl(train, DOSSIER_DATASET / "train.jsonl",      format_llm=True)
    sauvegarder_jsonl(val,   DOSSIER_DATASET / "validation.jsonl", format_llm=True)
    sauvegarder_jsonl(test,  DOSSIER_DATASET / "test.jsonl",       format_llm=True)

    # Sauvegarder aussi avec métadonnées complètes
    sauvegarder_jsonl(train, DOSSIER_DATASET / "train_meta.jsonl",      format_llm=False)
    sauvegarder_jsonl(val,   DOSSIER_DATASET / "validation_meta.jsonl", format_llm=False)
    sauvegarder_jsonl(test,  DOSSIER_DATASET / "test_meta.jsonl",       format_llm=False)

    # Rapport
    generer_rapport(train, val, test, DOSSIER_DATASET / "rapport_dataset.json")

    log.info("\n✅ Dataset prêt ! Prochaine étape : python scripts/run_pipeline.py --etape finetune")


if __name__ == "__main__":
    main()
