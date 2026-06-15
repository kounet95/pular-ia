"""
community_bot.py — Bot Telegram de contribution communautaire Pular
La communauté envoie des messages vocaux → Whisper transcrit → validation → dataset

Usage:
    Ajouter TELEGRAM_BOT_TOKEN dans .env (obtenu via @BotFather)
    python scripts/community_bot.py

Commandes disponibles:
    /start  — Accueil et instructions
    /stats  — Statistiques de la communauté
    /top    — Top contributeurs
    /aide   — Aide complète
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes,
)
from telegram.request import HTTPXRequest

load_dotenv(override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/community_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
DOSSIER_CONTRIB = Path("./corpus-pular/community/contributions")
DOSSIER_AUDIO   = Path("./corpus-pular/community/audio")
FICHIER_STATS   = Path("./corpus-pular/community/stats.json")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_BOT", "base")  # base = rapide pour le bot

for d in [DOSSIER_CONTRIB, DOSSIER_AUDIO]:
    d.mkdir(parents=True, exist_ok=True)

# ── Whisper (chargé une seule fois) ──────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info(f"Chargement Whisper '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        log.info("✅ Whisper prêt")
    return _whisper_model

def transcrire(audio_path: str) -> str:
    model = get_whisper()
    result = model.transcribe(
        audio_path,
        task="transcribe",
        no_speech_threshold=0.3,
        initial_prompt="Pular fulfulde fulani langue africaine.",
        logprob_threshold=-1.5,
        condition_on_previous_text=False,
        fp16=False,
    )
    texte = result["text"].strip()
    log.info(f"Langue détectée: {result.get('language','?')} | texte: '{texte[:80]}'")
    return texte

# ── Persistance stats ─────────────────────────────────────────────────────────
def charger_stats() -> dict:
    if FICHIER_STATS.exists():
        with open(FICHIER_STATS, encoding="utf-8") as f:
            return json.load(f)
    return {"total_contributions": 0, "total_validations": 0, "contributeurs": {}}

def sauver_stats(stats: dict):
    with open(FICHIER_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def enregistrer_contribution(user_id: int, username: str, texte_auto: str,
                              audio_path: str, valide: bool, correction: str = None):
    stats = charger_stats()
    uid = str(user_id)
    stats["contributeurs"].setdefault(uid, {"nom": username, "contributions": 0, "validations": 0})

    entry = {
        "id": f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "user_id": user_id,
        "username": username,
        "transcription_auto": texte_auto,
        "texte_final": correction if correction else texte_auto,
        "valide": valide,
        "audio": audio_path,
        "timestamp": datetime.now().isoformat(),
        "source": "community_bot",
    }

    with open(DOSSIER_CONTRIB / f"{entry['id']}.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    stats["total_contributions"] += 1
    stats["contributeurs"][uid]["contributions"] += 1
    if valide:
        stats["total_validations"] += 1
        stats["contributeurs"][uid]["validations"] += 1

    sauver_stats(stats)
    log.info(f"Contribution enregistrée: {entry['id']} | validé={valide}")

# ── Commandes ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nom = update.effective_user.first_name
    await update.message.reply_text(
        f"Assalaamu alaykum {nom}! 🌙\n\n"
        "*Projet Pular IA* — aide-nous à construire le premier modèle d'intelligence "
        "artificielle pour la langue pular!\n\n"
        "📢 *Comment contribuer:*\n"
        "1️⃣ Envoie un message vocal en pular\n"
        "2️⃣ Je transcris automatiquement avec l'IA\n"
        "3️⃣ Tu valides ✅ ou corriges ✏️\n"
        "4️⃣ Ta contribution enrichit le corpus!\n\n"
        "📊 /stats — Statistiques communauté\n"
        "🏆 /top — Top contributeurs\n"
        "❓ /aide — Aide complète\n\n"
        "_Baŋ-baŋ! 🙏_",
        parse_mode="Markdown",
    )

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = charger_stats()
    total_c = stats["total_contributions"]
    total_v = stats["total_validations"]
    nb = len(stats["contributeurs"])
    taux = int(total_v / total_c * 100) if total_c > 0 else 0
    await update.message.reply_text(
        f"📊 *Statistiques Pular IA*\n\n"
        f"🎙️ Contributions totales: *{total_c}*\n"
        f"✅ Validées: *{total_v}* ({taux}%)\n"
        f"👥 Contributeurs: *{nb}*\n\n"
        f"_Chaque vocal compte! Baŋ-baŋ 🙏_",
        parse_mode="Markdown",
    )

async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = charger_stats()
    contribs = sorted(
        stats["contributeurs"].values(),
        key=lambda x: x["contributions"],
        reverse=True,
    )[:10]
    if not contribs:
        await update.message.reply_text("Pas encore de contributeurs. Sois le premier! 🚀")
        return
    lignes = ["🏆 *Top contributeurs Pular IA*\n"]
    medailles = ["🥇", "🥈", "🥉"] + ["🔹"] * 7
    for i, c in enumerate(contribs):
        lignes.append(f"{medailles[i]} {c['nom']} — {c['contributions']} vocaux")
    await update.message.reply_text("\n".join(lignes), parse_mode="Markdown")

async def cmd_aide(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Aide — Bot Pular IA*\n\n"
        "🎙️ *Envoyer un vocal:*\n"
        "Appuie sur le micro dans Telegram, parle en pular, relâche.\n\n"
        "✅ *Valider:* La transcription est correcte → ✅ Correct\n"
        "✏️ *Corriger:* Des erreurs → ✏️ Corriger, puis envoie le bon texte\n"
        "❌ *Ignorer:* Ne pas sauvegarder ce vocal\n\n"
        "📌 *Conseils pour une bonne qualité:*\n"
        "• Parle clairement, micro proche\n"
        "• Messages de 5 à 60 secondes idéaux\n"
        "• N'importe quel sujet en pular!\n"
        "• Évite les bruits de fond\n\n"
        "_Baŋ-baŋ! 🙏_",
        parse_mode="Markdown",
    )

# ── Gestion des messages vocaux ───────────────────────────────────────────────
async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg  = update.message
    user = update.effective_user
    voice = msg.voice or msg.audio

    attente = await msg.reply_text("🎙️ Reçu! Transcription en cours...")

    fichier = await ctx.bot.get_file(voice.file_id)
    audio_path = DOSSIER_AUDIO / f"{user.id}_{voice.file_id}.ogg"
    await fichier.download_to_drive(str(audio_path))

    try:
        texte = await asyncio.to_thread(transcrire, str(audio_path))
    except Exception as e:
        log.error(f"Erreur Whisper: {e}")
        await attente.edit_text("❌ Erreur lors de la transcription. Réessaie!")
        return

    # Transcription vide — demander de réessayer
    if not texte:
        await attente.edit_text(
            "⚠️ *Aucune parole détectée dans ce vocal.*\n\n"
            "Conseils pour une meilleure capture:\n"
            "• Parle plus fort et plus près du micro\n"
            "• Enregistrement de 5 à 60 secondes idéal\n"
            "• Évite les bruits de fond\n"
            "• Commence à parler dès que tu appuies sur le micro\n\n"
            "_Envoie un nouveau vocal!_ 🎙️",
            parse_mode="Markdown",
        )
        return

    ctx.user_data["pending"] = {
        "texte": texte,
        "audio_path": str(audio_path),
        "user_id": user.id,
        "username": user.username or user.first_name,
    }

    clavier = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Correct", callback_data="valider"),
            InlineKeyboardButton("✏️ Corriger", callback_data="corriger"),
        ],
        [InlineKeyboardButton("❌ Ignorer", callback_data="ignorer")],
    ])

    # Échapper les underscores dans le texte pour éviter les erreurs de formatage Markdown
    texte_md = texte.replace("_", "\\_").replace("*", "\\*")
    await attente.edit_text(
        f"📝 *Transcription automatique:*\n\n{texte_md}\n\n"
        "Est-ce correct?",
        reply_markup=clavier,
        parse_mode="Markdown",
    )

async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()
    pending = ctx.user_data.get("pending")

    if not pending:
        await query.edit_message_text("⚠️ Session expirée. Envoie un nouveau vocal.")
        return

    if query.data == "valider":
        enregistrer_contribution(
            pending["user_id"], pending["username"],
            pending["texte"], pending["audio_path"], valide=True,
        )
        ctx.user_data.pop("pending", None)
        await query.edit_message_text(
            f"✅ *Validé! Baŋ-baŋ!*\n\n_{pending['texte']}_\n\n"
            "Ta contribution aide à construire l'IA pular! 🚀",
            parse_mode="Markdown",
        )

    elif query.data == "corriger":
        ctx.user_data["en_correction"] = True
        await query.edit_message_text(
            f"✏️ *Correction*\n\nTranscription actuelle:\n_{pending['texte']}_\n\n"
            "Envoie maintenant le texte correct en pular:",
            parse_mode="Markdown",
        )

    elif query.data == "ignorer":
        ctx.user_data.pop("pending", None)
        await query.edit_message_text("❌ Ignoré. Envoie un autre vocal!")

async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.user_data.get("en_correction"):
        return
    pending = ctx.user_data.get("pending")
    if not pending:
        return

    correction = update.message.text.strip()
    enregistrer_contribution(
        pending["user_id"], pending["username"],
        pending["texte"], pending["audio_path"],
        valide=True, correction=correction,
    )
    ctx.user_data.pop("pending", None)
    ctx.user_data.pop("en_correction", None)

    await update.message.reply_text(
        f"✅ *Correction sauvée! Baŋ-baŋ!*\n\n_{correction}_\n\n"
        "Ta correction améliore le modèle Pular! 🚀",
        parse_mode="Markdown",
    )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN manquant dans .env")
        log.error("   1. Va sur Telegram → @BotFather → /newbot")
        log.error("   2. Copie le token dans .env : TELEGRAM_BOT_TOKEN=ton_token")
        return

    request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
    )

    app = Application.builder().token(BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("top",   cmd_top))
    app.add_handler(CommandHandler("aide",  cmd_aide))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("🤖 Bot Pular IA démarré! Ctrl+C pour arrêter.")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)

if __name__ == "__main__":
    main()
