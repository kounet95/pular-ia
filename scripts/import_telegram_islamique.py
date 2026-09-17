"""
import_telegram_islamique.py — Indexation RAG du contenu islamique Telegram

Les canaux scrapés par telegram_scraper.py (ex: DDTV226 = "Doudhe Diina TV",
questions/réponses religieuses ; laawolsunna = enseignements) contiennent des
lectures coraniques commentées et des questions/réponses en pular, en plus de
la traduction officielle du Coran déjà indexée par import_coran_fulani.py.
Ce script les ajoute au même RAG (ChromaDB via rag_livres) afin que
coran_pular.repondre_question() (utilisé par /coran et le menu vocal du bot)
puisse aussi s'appuyer dessus.

Le contenu utile est dans la TRANSCRIPTION de l'audio — vérifié sur le
corpus déjà scrapé (base_connaissance.json) : les messages texte de ces
canaux ne sont presque jamais du contenu religieux lui-même, mais des
légendes/annonces ("Fatwa Cheick X — 14 Novembre 2024", liens, parfois des
polémiques doctrinales entre courants) qui n'ont pas leur place dans les
réponses d'un bot censé rester neutre. Par défaut, SEULES les transcriptions
sont donc indexées (--inclure-texte permet d'inclure aussi les messages
texte assez longs, à tes risques — relis-les d'abord). Tant que peu d'audios
sont transcrits (cf. le panel prof > Telegram sur le site, qui lance
scraping + transcription), peu de matière est réellement indexable ici.
Relancer ce script au fur et à mesure que de nouveaux audios sont transcrits
est sans risque : indexer_livre() ignore les chunks déjà présents
(déduplication par id).

⚠️ Contrairement à la traduction quranenc.com (source académique vérifiée),
ce contenu est une transcription automatique (Whisper) de prêches communautaires
d'un canal Telegram précis — donc une opinion/interprétation particulière, pas
une vérité générale, et potentiellement entachée d'erreurs de transcription.
coran_pular.repondre_question() le signale explicitement à l'utilisateur.

Usage:
    python scripts/import_telegram_islamique.py
    python scripts/import_telegram_islamique.py --canaux DDTV226
    python scripts/import_telegram_islamique.py --inclure-texte --min-mots 40
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/import_telegram_islamique.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

FICHIER_BASE = Path("./corpus-pular/processed/telegram/base_connaissance.json")

# Noms/descriptions lisibles — complète au besoin si d'autres canaux
# islamiques sont ajoutés à CANAUX_DEFAUT dans telegram_scraper.py.
NOMS_CANAUX = {
    "DDTV226":     "Doudhe Diina TV — Questions/réponses islamiques (Telegram)",
    "laawolsunna": "Laawol Sunna — Enseignements islamiques (Telegram)",
}


def contenu_message(m: dict, inclure_texte: bool, min_mots: int) -> str:
    """Texte utilisable pour l'indexation : la transcription audio (le vrai
    contenu religieux sur ces canaux). Le texte du message n'est utilisé que
    si --inclure-texte est passé explicitement — vérifié sur le corpus réel
    que ce sont presque toujours des légendes/annonces, pas du contenu
    religieux exploitable tel quel."""
    transcription = (m.get("transcription") or "").strip()
    if transcription:
        return transcription
    if inclure_texte:
        texte = (m.get("texte") or "").strip()
        if texte and len(texte.split()) >= min_mots:
            return texte
    return ""


def importer(canaux: list[str] | None, inclure_texte: bool, min_mots: int):
    if not FICHIER_BASE.exists():
        log.error(
            f"{FICHIER_BASE} introuvable — lance d'abord un scraping "
            "(python scripts/telegram_scraper.py, ou le panel prof > Telegram sur le site)."
        )
        return

    data = json.loads(FICHIER_BASE.read_text(encoding="utf-8"))
    canaux_presents = sorted({m["canal"] for m in data})
    cibles = canaux or canaux_presents

    from scripts import rag_livres

    for canal in cibles:
        if canal not in canaux_presents:
            log.warning(f"{canal} — aucun message scrapé pour ce canal, ignoré.")
            continue

        msgs = sorted(
            (m for m in data if m["canal"] == canal),
            key=lambda m: m.get("date") or "",
        )
        blocs = [c for m in msgs if (c := contenu_message(m, inclure_texte, min_mots))]

        if not blocs:
            log.info(f"{canal} — aucun contenu transcrit/textuel exploitable pour l'instant.")
            continue

        texte_complet = "\n\n".join(blocs)
        livre_id = f"telegram_{canal}"
        titre    = NOMS_CANAUX.get(canal, f"Telegram @{canal}")

        ajoutes = rag_livres.indexer_livre(
            titre=titre, auteur=f"Communauté Telegram @{canal}",
            langue="ff", texte=texte_complet, livre_id=livre_id,
        )
        log.info(
            f"{canal} — {ajoutes} nouveaux chunks indexés "
            f"({len(blocs)}/{len(msgs)} messages avec contenu exploitable)"
        )

        livres = rag_livres.charger_index()
        if not any(l.get("livre_id") == livre_id for l in livres):
            livres.append({
                "livre_id":   livre_id,
                "titre":      titre,
                "auteur":     f"Communauté Telegram @{canal}",
                "langue":     "ff",
                "source":     "telegram_scraper",
                "date_ajout": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            })
            rag_livres.sauver_index(livres)


def main():
    parser = argparse.ArgumentParser(description="Indexation RAG du contenu islamique Telegram")
    parser.add_argument("--canaux", nargs="+", default=None,
                         help="Canaux à indexer (défaut: tous ceux présents dans base_connaissance.json)")
    parser.add_argument("--inclure-texte", action="store_true",
                         help="Indexer aussi les messages TEXTE sans transcription (déconseillé par défaut : "
                              "ce sont presque toujours des légendes/annonces, pas du contenu religieux)")
    parser.add_argument("--min-mots", type=int, default=40,
                         help="Avec --inclure-texte : nb de mots minimum pour indexer un message texte (défaut: 40)")
    args = parser.parse_args()

    importer(args.canaux, args.inclure_texte, args.min_mots)
    log.info("✅ Indexation terminée. Relance ce script après chaque nouveau lot de transcriptions.")


if __name__ == "__main__":
    main()
