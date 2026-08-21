"""
import_dictionnaire.py — Ingestion du dictionnaire pular-français dans le RAG
Lit corpus-pular/dataset/wordlist.json (dictionnaire déjà présent dans le
projet) et l'indexe via rag_livres, comme un "livre" :
  - Recherche sémantique / réponse aux questions de vocabulaire dans l'app.
  - Génération d'un fichier de vocabulaire Whisper avec les mots canoniques
    du dictionnaire (les plus fiables pour biaiser la transcription).

Usage:
    python scripts/import_dictionnaire.py
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import rag_livres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/import_dictionnaire.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

FICHIER_WORDLIST = Path("./corpus-pular/dataset/wordlist.json")
LIVRE_ID = "dictionnaire_pular"
TITRE    = "Dictionnaire Pular-Français"
AUTEUR   = "Corpus Pular"


def entree_vers_texte(entree: dict) -> str:
    """Convertit une entrée du dictionnaire en un paragraphe texte pour le RAG."""
    lignes = [entree.get("headword", "").strip()]

    dialectes = entree.get("dialects") or {}
    if dialectes:
        variantes = ", ".join(f"{k}: {v}" for k, v in dialectes.items())
        lignes.append(f"(variantes dialectales — {variantes})")

    for sens in entree.get("senses") or []:
        definition = (sens.get("definition_fr") or "").strip()
        if definition:
            lignes.append(f"= {definition}")
        for ex in sens.get("examples") or []:
            pular = (ex.get("pular") or "").strip()
            fr    = (ex.get("fr") or "").strip()
            if pular:
                lignes.append(f"Ex: {pular}" + (f" — {fr}" if fr else ""))

    return "\n".join(l for l in lignes if l.strip())


def main():
    Path("logs").mkdir(exist_ok=True)

    if not FICHIER_WORDLIST.exists():
        log.error(f"Fichier introuvable : {FICHIER_WORDLIST}")
        return

    with open(FICHIER_WORDLIST, encoding="utf-8") as f:
        entrees = json.load(f)
    log.info(f"{len(entrees):,} entrées chargées depuis {FICHIER_WORDLIST}")

    paragraphes = [entree_vers_texte(e) for e in entrees]
    texte = "\n\n".join(p for p in paragraphes if p.strip())
    log.info(f"Texte du dictionnaire construit : {len(texte):,} caractères")

    livres = rag_livres.charger_index()
    if not any(l.get("livre_id") == LIVRE_ID for l in livres):
        import time
        livres.append({
            "livre_id": LIVRE_ID,
            "titre":    TITRE,
            "auteur":   AUTEUR,
            "langue":   "ff",
            "source":   "corpus-pular/dataset/wordlist.json",
            "date_ajout": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        rag_livres.sauver_index(livres)

    log.info("Indexation RAG du dictionnaire (chunking + embeddings)...")
    ajoutes = rag_livres.indexer_livre(
        titre=TITRE, auteur=AUTEUR, langue="ff",
        texte=texte, livre_id=LIVRE_ID,
    )
    log.info(f"✅ {ajoutes} nouveaux chunks indexés — vocabulaire Whisper généré "
             f"(corpus-pular/livres/metadata/{LIVRE_ID}_vocab.json)")


if __name__ == "__main__":
    main()
