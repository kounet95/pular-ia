"""
telegram_scraper.py — Scraping canaux Telegram pular + transcription
Intégré au pipeline corpus pular (sorties compatibles avec build_dataset.py)

Usage:
    python scripts/telegram_scraper.py
    python scripts/telegram_scraper.py --canaux bts224 kacfusubuhat
    python scripts/telegram_scraper.py --limite 1000          # 1000 messages par canal
    python scripts/telegram_scraper.py --sans-audio           # textes uniquement
    python scripts/telegram_scraper.py --whisper-model large-v3

PRÉREQUIS :
    pip install telethon openai-whisper python-dotenv tqdm
    sudo apt install ffmpeg

CONFIGURATION :
    Copie .env.example → .env et remplis tes identifiants Telegram.
    Va sur https://my.telegram.org → API development tools → crée une app.
"""

import os
import json
import asyncio
import logging
import hashlib
from datetime import datetime
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    DocumentAttributeAudio,
    DocumentAttributeVideo,
)
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/telegram_scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Credentials (depuis .env) ──────────────────────────────────────────────────
API_ID   = int(os.getenv("TELEGRAM_API_ID",   "0"))
API_HASH = os.getenv("TELEGRAM_API_HASH",      "")
PHONE    = os.getenv("TELEGRAM_PHONE",         "")

# ── Canaux par défaut ──────────────────────────────────────────────────────────
CANAUX_DEFAUT = [
    #"bts224",                           # Bagnan Technologie Service — formations islamiques
    #"kacfusubuhat",                     # Kac Fusubuhat
    "DDTV226",                          # Questions/Réponses — Doudhe Diina TV
    #"wadjoudjOustazcherifalkoundary",   # Cheikh Oustad Al-Koundary
    "laawolsunna",                      # Laawol Sunna
]

# ── Dossiers (compatibles avec le pipeline existant) ──────────────────────────
DOSSIER_TELEGRAM  = Path("./corpus-pular/processed/telegram")
DOSSIER_AUDIO_TG  = DOSSIER_TELEGRAM / "audio"
DOSSIER_JSONL     = DOSSIER_TELEGRAM / "jsonl"
FICHIER_PROGRES   = DOSSIER_TELEGRAM / "progres.json"
FICHIER_BASE      = DOSSIER_TELEGRAM / "base_connaissance.json"

# ── Whisper ────────────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info(f"Chargement Whisper {model_name}...")
        _whisper_model = whisper.load_model(model_name)
        log.info("✅ Whisper prêt")
    return _whisper_model


# ── Hash pour déduplication (compatible build_dataset.py) ─────────────────────
def hash_texte(texte: str) -> str:
    return hashlib.md5(texte.encode("utf-8")).hexdigest()


# ── Détection du domaine (réutilise la même logique que build_dataset.py) ──────
MOTS_ISLAMIQUES = {
    "juulde", "salaat", "subahi", "fajri", "aljumaa", "ramadan", "koorka",
    "qur", "hadith", "annabi", "allah", "allahu", "bismillah", "barke",
    "zakkat", "haajira", "wudu", "kiblat", "rak'a",
    "sallallahu", "alayhi", "wasallam", "inshallah", "mashallah", "alhamdulillah",
    "prière", "mosquée", "prophète", "coran", "oumma",
}

def detecter_domaine(texte: str) -> str:
    mots = set(texte.lower().split())
    score = len(mots & MOTS_ISLAMIQUES)
    if score >= 2:   return "islamique"
    if score == 1:   return "islamique_probable"
    return "general"


# ── Progression ────────────────────────────────────────────────────────────────
def charger_progres() -> dict:
    if FICHIER_PROGRES.exists():
        with open(FICHIER_PROGRES, encoding="utf-8") as f:
            return json.load(f)
    return {"canaux_termines": [], "messages_ids": {}}


def sauvegarder_progres(progres: dict):
    FICHIER_PROGRES.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_PROGRES, "w", encoding="utf-8") as f:
        json.dump(progres, f, ensure_ascii=False, indent=2)


def charger_base() -> list:
    if FICHIER_BASE.exists():
        with open(FICHIER_BASE, encoding="utf-8") as f:
            return json.load(f)
    return []


def sauvegarder_base(messages: list):
    FICHIER_BASE.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_BASE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ── Détecter si le document est audio/vidéo ───────────────────────────────────
def est_media_audio(media) -> tuple[bool, str]:
    """Retourne (est_audio, mime_type)."""
    if not isinstance(media, MessageMediaDocument):
        return False, ""
    doc = media.document
    mime = getattr(doc, "mime_type", "") or ""
    est_audio = any(t in mime for t in ["audio", "ogg", "video/mp4"])
    return est_audio, mime


# ── Scraping d'un canal ────────────────────────────────────────────────────────
async def scraper_canal(
    client: TelegramClient,
    canal: str,
    limite: int,
    ids_deja_vus: set,
) -> list[dict]:
    log.info(f"⬇️  Scraping @{canal} (limite={limite or 'tous'})")
    messages = []

    try:
        entity    = await client.get_entity(canal)
        nom_canal = getattr(entity, "title", canal)

        async for msg in client.iter_messages(entity, limit=limite):
            if msg.id in ids_deja_vus:
                continue

            texte = msg.text or ""
            est_audio, mime = est_media_audio(msg.media)

            entree = {
                "canal":         canal,
                "canal_nom":     nom_canal,
                "message_id":    msg.id,
                "date":          msg.date.isoformat() if msg.date else None,
                "texte":         texte,
                "type":          "audio" if est_audio else ("photo" if isinstance(msg.media, MessageMediaPhoto) else "texte"),
                "mime_type":     mime,
                "fichier_local": None,
                "transcription": None,
                "langue_detect": None,
                "source":        "telegram",
                "hash":          hash_texte(texte) if texte else None,
                "domaine":       detecter_domaine(texte) if texte else "general",
            }
            messages.append(entree)

        log.info(f"✅ @{canal} : {len(messages)} nouveaux messages")

    except Exception as e:
        log.error(f"❌ Erreur @{canal} : {e}")

    return messages


# ── Téléchargement des médias ─────────────────────────────────────────────────
async def telecharger_medias(
    client: TelegramClient,
    canal: str,
    messages: list[dict],
) -> list[dict]:
    medias = [m for m in messages if m["type"] == "audio" and not m["fichier_local"]]
    if not medias:
        return messages

    log.info(f"📥 Téléchargement {len(medias)} fichiers audio depuis @{canal}")

    try:
        entity = await client.get_entity(canal)
    except Exception as e:
        log.error(f"Impossible d'accéder à @{canal} pour les médias : {e}")
        return messages

    DOSSIER_AUDIO_TG.mkdir(parents=True, exist_ok=True)

    for entree in tqdm(medias, desc=f"@{canal} médias"):
        try:
            msg = await client.get_messages(entity, ids=entree["message_id"])
            if not msg or not msg.media:
                continue

            # Choisir l'extension selon le mime type
            mime = entree.get("mime_type", "")
            if "ogg" in mime:      ext = "ogg"
            elif "mp4" in mime:    ext = "mp4"
            elif "mpeg" in mime:   ext = "mp3"
            else:                  ext = "bin"

            chemin = DOSSIER_AUDIO_TG / f"{canal}_{entree['message_id']}.{ext}"

            if not chemin.exists():
                await client.download_media(msg, file=str(chemin))
                log.debug(f"  ↓ {chemin.name}")

            entree["fichier_local"] = str(chemin)

        except Exception as e:
            log.warning(f"  ⚠️ Média {entree['message_id']} : {e}")

    return messages


# ── Transcription ──────────────────────────────────────────────────────────────
def transcrire_messages(messages: list[dict], model_name: str) -> list[dict]:
    a_transcrire = [
        m for m in messages
        if m["fichier_local"] and not m["transcription"] and Path(m["fichier_local"]).exists()
    ]

    if not a_transcrire:
        log.info("Aucun audio à transcrire.")
        return messages

    log.info(f"🎙️  Transcription de {len(a_transcrire)} fichiers avec Whisper {model_name}")
    model = get_whisper(model_name)

    for entree in tqdm(a_transcrire, desc="Transcription"):
        try:
            result = model.transcribe(
                entree["fichier_local"],
                language=None,       # détection automatique de langue
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
            )
            texte_transcrit = result["text"].strip()
            entree["transcription"] = texte_transcrit
            entree["langue_detect"] = result.get("language", "?")

            # Mettre à jour le hash et domaine avec la transcription
            if texte_transcrit:
                entree["hash"]    = hash_texte(texte_transcrit)
                entree["domaine"] = detecter_domaine(texte_transcrit)

            log.debug(f"  ✅ [{entree['langue_detect']}] {entree['message_id']}")

        except Exception as e:
            log.warning(f"  ⚠️ Transcription {entree['message_id']} : {e}")

    return messages


# ── Export JSONL (format compatible build_dataset.py) ─────────────────────────
def exporter_jsonl(messages: list[dict]):
    """
    Exporte chaque canal en JSONL compatible avec le pipeline build_dataset.py.
    Format identique aux sorties de transcription.py et ocr_images.py.
    """
    DOSSIER_JSONL.mkdir(parents=True, exist_ok=True)

    canaux = set(m["canal"] for m in messages)

    for canal in canaux:
        msgs_canal = [m for m in messages if m["canal"] == canal]
        chemin_jsonl = DOSSIER_JSONL / f"{canal}.jsonl"

        with open(chemin_jsonl, "w", encoding="utf-8") as f:
            for m in msgs_canal:
                # Texte principal (message Telegram)
                if m["texte"] and len(m["texte"].split()) >= 5:
                    f.write(json.dumps({
                        "fichier":      f"telegram://{canal}/{m['message_id']}",
                        "nom":          f"{canal}_{m['message_id']}_texte",
                        "texte":        m["texte"],
                        "source":       "telegram_texte",
                        "langue":       "pular",
                        "domaine":      m["domaine"],
                        "nb_tokens":    len(m["texte"].split()),
                        "hash":         hash_texte(m["texte"]),
                        "statut":       "ok",
                        "meta": {
                            "canal":    canal,
                            "date":     m["date"],
                            "id":       m["message_id"],
                        },
                    }, ensure_ascii=False) + "\n")

                # Transcription audio
                if m.get("transcription") and len(m["transcription"].split()) >= 5:
                    f.write(json.dumps({
                        "fichier":      m.get("fichier_local", ""),
                        "nom":          f"{canal}_{m['message_id']}_audio",
                        "texte":        m["transcription"],
                        "source":       "telegram_audio",
                        "langue":       m.get("langue_detect", "pular"),
                        "domaine":      detecter_domaine(m["transcription"]),
                        "nb_tokens":    len(m["transcription"].split()),
                        "hash":         hash_texte(m["transcription"]),
                        "statut":       "ok",
                        "meta": {
                            "canal":         canal,
                            "date":          m["date"],
                            "id":            m["message_id"],
                            "whisper_lang":  m.get("langue_detect"),
                        },
                    }, ensure_ascii=False) + "\n")

        log.info(f"📁 JSONL exporté : {chemin_jsonl}")


# ── Résumé final ───────────────────────────────────────────────────────────────
def afficher_resume(messages: list[dict]):
    total      = len(messages)
    textes     = sum(1 for m in messages if m["texte"])
    audios     = sum(1 for m in messages if m["type"] == "audio")
    transcrits = sum(1 for m in messages if m.get("transcription"))
    islamique  = sum(1 for m in messages if m.get("domaine") == "islamique")

    canaux_stats = {}
    for m in messages:
        c = m["canal"]
        if c not in canaux_stats:
            canaux_stats[c] = {"total": 0, "audio": 0, "transcrit": 0}
        canaux_stats[c]["total"] += 1
        if m["type"] == "audio":    canaux_stats[c]["audio"] += 1
        if m.get("transcription"):  canaux_stats[c]["transcrit"] += 1

    print("\n" + "═" * 55)
    print("  📊 RÉSUMÉ SCRAPING TELEGRAM PULAR")
    print("═" * 55)
    print(f"  Messages totaux      : {total:,}")
    print(f"  Avec texte           : {textes:,}")
    print(f"  Fichiers audio/vidéo : {audios:,}")
    print(f"  Transcrits           : {transcrits:,}")
    print(f"  Domaine islamique    : {islamique:,}")
    print()
    print("  Par canal :")
    for canal, s in canaux_stats.items():
        print(f"    @{canal:<35} {s['total']:>5} msgs  {s['audio']:>4} audio  {s['transcrit']:>4} transcrits")
    print("═" * 55)
    print(f"\n  Données JSONL → corpus-pular/processed/telegram/jsonl/")
    print(f"  Ces fichiers sont automatiquement lus par build_dataset.py\n")


# ── Main ───────────────────────────────────────────────────────────────────────
async def main(canaux: list, limite: int, sans_audio: bool, whisper_model: str):
    # Vérifier les credentials
    if not API_ID or not API_HASH or not PHONE:
        log.error("❌ Remplis TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE dans le fichier .env")
        return

    for d in [DOSSIER_TELEGRAM, DOSSIER_AUDIO_TG, DOSSIER_JSONL]:
        d.mkdir(parents=True, exist_ok=True)

    progres      = charger_progres()
    deja_termines = set(progres.get("canaux_termines", []))
    ids_par_canal = progres.get("messages_ids", {})
    tous_messages = charger_base()

    canaux_restants = [c for c in canaux if c not in deja_termines]

    if deja_termines:
        log.info(f"🔄 Reprise — déjà terminés : {list(deja_termines)}")
    if canaux_restants:
        log.info(f"📋 À scraper : {canaux_restants}")
    else:
        log.info("✅ Tous les canaux ont déjà été scrapés — passage à la transcription")

    async with TelegramClient("session_pular", API_ID, API_HASH) as client:
        await client.start(phone=PHONE)
        log.info("🔗 Connecté à Telegram")

        for canal in canaux_restants:
            ids_vus = set(ids_par_canal.get(canal, []))

            # 1. Scraping textes
            nouveaux = await scraper_canal(client, canal, limite, ids_vus)

            # 2. Téléchargement audio (sauf si --sans-audio)
            if not sans_audio:
                nouveaux = await telecharger_medias(client, canal, nouveaux)

            # Fusion avec la base existante
            tous_messages.extend(nouveaux)
            ids_par_canal[canal] = list(ids_vus | {m["message_id"] for m in nouveaux})

            # Sauvegarde immédiate
            deja_termines.add(canal)
            progres["canaux_termines"] = list(deja_termines)
            progres["messages_ids"]    = ids_par_canal
            sauvegarder_progres(progres)
            sauvegarder_base(tous_messages)
            log.info(f"💾 Progression sauvegardée après @{canal}")

    # 3. Transcription de tous les audios (y compris anciens non transcrits)
    if not sans_audio:
        tous_messages = transcrire_messages(tous_messages, whisper_model)
        sauvegarder_base(tous_messages)

    # 4. Export JSONL compatible pipeline
    exporter_jsonl(tous_messages)

    afficher_resume(tous_messages)
    log.info("✅ Prochaine étape : python scripts/build_dataset.py")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scraper Telegram corpus pular")
    parser.add_argument(
        "--canaux", nargs="+", default=CANAUX_DEFAUT,
        help="Liste des canaux Telegram (sans @)",
    )
    parser.add_argument(
        "--limite", default=None, type=int,
        help="Nombre max de messages par canal (défaut: tous)",
    )
    parser.add_argument(
        "--sans-audio", action="store_true",
        help="Ne pas télécharger ni transcrire les fichiers audio",
    )
    parser.add_argument(
        "--whisper-model", default="large-v3",
        help="Modèle Whisper : tiny/base/small/medium/large-v3 (défaut: large-v3)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Recommencer depuis zéro (ignore la progression sauvegardée)",
    )
    args = parser.parse_args()

    if args.reset and FICHIER_PROGRES.exists():
        FICHIER_PROGRES.unlink()
        log.info("🔄 Progression réinitialisée")

    log.info("=" * 55)
    log.info("  TELEGRAM SCRAPER — CORPUS PULAR")
    log.info(f"  Canaux : {args.canaux}")
    log.info(f"  Whisper: {args.whisper_model}")
    log.info("=" * 55)

    asyncio.run(main(
        canaux=args.canaux,
        limite=args.limite,
        sans_audio=args.sans_audio,
        whisper_model=args.whisper_model,
    ))
