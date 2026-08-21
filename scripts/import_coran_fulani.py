"""
import_coran_fulani.py — Ingestion des traductions fulani du Coran (quranenc.com)
Récupère deux textes en fulani via l'API publique de quranenc.com :
  - fulani_rwwad     : traduction littérale (Rowwad Translation Center)
  - fulani_mokhtasar : Tafsir Al-Mukhtasar (exégèse explicative, texte plus riche)

Les deux sont indexés via rag_livres (chunking + embeddings ChromaDB) ce qui :
  1. Permet la recherche sémantique / réponse aux questions (RAG) dans l'app.
  2. Génère automatiquement un fichier de vocabulaire pular par texte
     (corpus-pular/livres/metadata/<id>_vocab.json), réutilisé par
     transcription.py pour améliorer le prompt initial de Whisper.

⚠️ Attribution obligatoire : ces textes sont des œuvres traduites, diffusées
par quranenc.com / IslamHouse.com (Rowwad Translation Center et Tafsir Center
for Quranic Studies). Avant toute redistribution publique de ce corpus,
vérifie les conditions d'utilisation du site — l'attribution est conservée
dans les métadonnées (corpus-pular/livres/index.json et *_raw.json) mais ne
remplace pas une vérification de licence de ta part.

Usage:
    python scripts/import_coran_fulani.py
    python scripts/import_coran_fulani.py --seulement fulani_rwwad
    python scripts/import_coran_fulani.py --sleep 0.3
"""

import sys
import json
import time
import argparse
import logging
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import rag_livres

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/import_coran_fulani.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

API_BASE   = "https://quranenc.com/api/v1/translation/sura"
NB_SOURATES = 114

TRADUCTIONS = {
    "fulani_rwwad": {
        "titre":  "Coran — Traduction fulani (Rowwad Translation Center)",
        "auteur": "Rowwad Translation Center / IslamHouse.com",
    },
    "fulani_mokhtasar": {
        "titre":  "Coran — Tafsir Al-Mukhtasar en fulani (exégèse)",
        "auteur": "Tafsir Center for Quranic Studies / IslamHouse.com",
    },
}

DOSSIER_RAW_TXT  = Path("./corpus-pular/livres/raw")
DOSSIER_RAW_JSON = Path("./corpus-pular/livres/metadata")


HEADERS = {
    # Le serveur renvoie 403 sur le User-Agent par défaut d'urllib —
    # un User-Agent de navigateur classique passe sans problème.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def recuperer_sourate(cle: str, numero: int, essais: int = 3) -> list[dict]:
    url = f"{API_BASE}/{cle}/{numero}"
    for tentative in range(1, essais + 1):
        try:
            requete = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(requete, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            resultat = data.get("result", [])
            return resultat if isinstance(resultat, list) else [resultat]
        except Exception as e:
            log.warning(f"{cle} sourate {numero} — tentative {tentative}/{essais} échouée : {e}")
            time.sleep(1.5 * tentative)
    log.error(f"{cle} sourate {numero} — abandon après {essais} tentatives")
    return []


def telecharger_traduction(cle: str, pause: float) -> tuple[str, list[dict]]:
    """Télécharge les 114 sourates et retourne (texte_brut, tous_les_versets)."""
    blocs = []
    tous_versets = []
    for numero in range(1, NB_SOURATES + 1):
        versets = recuperer_sourate(cle, numero, )
        if not versets:
            continue
        lignes = [f"Sourate {numero}"]
        for v in versets:
            lignes.append(f"{v.get('aya')}. {v.get('translation', '').strip()}")
        blocs.append("\n".join(lignes))
        tous_versets.extend(versets)
        if numero % 20 == 0:
            log.info(f"{cle} — {numero}/{NB_SOURATES} sourates récupérées")
        time.sleep(pause)
    return "\n\n".join(blocs), tous_versets


def importer(cle: str, meta: dict, pause: float, forcer: bool):
    DOSSIER_RAW_TXT.mkdir(parents=True, exist_ok=True)
    DOSSIER_RAW_JSON.mkdir(parents=True, exist_ok=True)

    fichier_txt  = DOSSIER_RAW_TXT / f"{cle}.txt"
    fichier_json = DOSSIER_RAW_JSON / f"{cle}_raw.json"

    if fichier_txt.exists() and not forcer:
        log.info(f"{cle} — déjà téléchargé ({fichier_txt}), utilisation du cache. "
                  f"Utilise --forcer pour retélécharger.")
        texte = fichier_txt.read_text(encoding="utf-8")
    else:
        log.info(f"{cle} — téléchargement des {NB_SOURATES} sourates depuis quranenc.com...")
        texte, versets = telecharger_traduction(cle, pause)
        fichier_txt.write_text(texte, encoding="utf-8")
        with open(fichier_json, "w", encoding="utf-8") as f:
            json.dump({
                "cle": cle,
                "source": "https://quranenc.com",
                "auteur": meta["auteur"],
                "nb_versets": len(versets),
                "versets": versets,
            }, f, ensure_ascii=False)
        log.info(f"{cle} — {len(versets)} versets sauvegardés → {fichier_txt}")

    if not texte.strip():
        log.error(f"{cle} — texte vide, indexation annulée")
        return

    # Enregistrer dans l'index des livres (même format que l'espace "Livres" du site)
    livres = rag_livres.charger_index()
    if not any(l.get("livre_id") == cle for l in livres):
        livres.append({
            "livre_id": cle,
            "titre":    meta["titre"],
            "auteur":   meta["auteur"],
            "langue":   "ff",
            "source":   "quranenc.com",
            "date_ajout": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        rag_livres.sauver_index(livres)

    log.info(f"{cle} — indexation RAG (chunking + embeddings)...")
    ajoutes = rag_livres.indexer_livre(
        titre=meta["titre"], auteur=meta["auteur"], langue="ff",
        texte=texte, livre_id=cle,
    )
    log.info(f"{cle} — {ajoutes} nouveaux chunks indexés (vocabulaire Whisper généré)")


def main():
    parser = argparse.ArgumentParser(description="Import des traductions fulani du Coran (quranenc.com)")
    parser.add_argument("--seulement", choices=list(TRADUCTIONS), default=None,
                         help="N'importer qu'une seule traduction")
    parser.add_argument("--sleep", type=float, default=0.2,
                         help="Pause entre requêtes API (secondes)")
    parser.add_argument("--forcer", action="store_true",
                         help="Retélécharger même si déjà en cache")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    cibles = {args.seulement: TRADUCTIONS[args.seulement]} if args.seulement else TRADUCTIONS
    for cle, meta in cibles.items():
        importer(cle, meta, pause=args.sleep, forcer=args.forcer)

    log.info("✅ Import Coran fulani terminé. "
             "Prochaine étape : python scripts/import_dictionnaire.py")


if __name__ == "__main__":
    main()
