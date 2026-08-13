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
    /duel   — Défier un ami en duel de vocabulaire (+ /duel CODE pour rejoindre)
    /classement — Classement des duels
    /aide   — Aide complète
"""

import os
import time
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import duels as DU
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
    # Lien d'invitation à un duel : https://t.me/<bot>?start=duel_<code>
    if ctx.args and ctx.args[0].startswith("duel_"):
        await _duel_rejoindre(update, ctx, ctx.args[0][5:].upper())
        return

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
        "⚔️ /duel — Défier un ami en duel de vocabulaire\n"
        "🏅 /classement — Classement des duels\n"
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
    # Pas de parse_mode : c['nom'] vient du prénom/pseudo Telegram de
    # l'utilisateur (texte libre) — un underscore ou astérisque dedans
    # casse le parseur Markdown et fait échouer l'envoi silencieusement.
    lignes = ["🏆 Top contributeurs Pular IA\n"]
    medailles = ["🥇", "🥈", "🥉"] + ["🔹"] * 7
    for i, c in enumerate(contribs):
        lignes.append(f"{medailles[i]} {c['nom']} — {c['contributions']} vocaux")
    await update.message.reply_text("\n".join(lignes))

async def cmd_aide(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Aide — Bot Pular IA*\n\n"
        "🎙️ *Envoyer un vocal:*\n"
        "Appuie sur le micro dans Telegram, parle en pular, relâche.\n\n"
        "✅ *Valider:* La transcription est correcte → ✅ Correct\n"
        "✏️ *Corriger:* Des erreurs → ✏️ Corriger, puis envoie le bon texte\n"
        "❌ *Ignorer:* Ne pas sauvegarder ce vocal\n\n"
        "⚔️ *Défier un ami:*\n"
        "/duel — crée un duel et partage le lien/code\n"
        "/duel CODE — rejoins le duel d'un ami\n"
        "🏅 /classement — voir qui domine\n\n"
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

# ── Duels en temps réel ───────────────────────────────────────────────────────
# État partagé entre joueurs, gardé dans ctx.bot_data (unique pour toute
# l'Application, contrairement à user_data/chat_data qui sont par utilisateur) :
#   duel_chats[code]   = {pseudo: chat_id}   — où envoyer les messages de chacun
#   duel_q_debut[code] = timestamp           — pour calculer le bonus de vitesse

def _pseudo_telegram(user) -> str:
    return (user.first_name or user.username or f"Joueur{user.id}").strip()[:40]

async def cmd_duel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        await _duel_rejoindre(update, ctx, ctx.args[0].upper())
    else:
        await _duel_creer(update, ctx)

async def _duel_creer(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pseudo = _pseudo_telegram(update.effective_user)
    mots = DU.charger_mots_pour_duel()
    if len(mots) < 4:
        await update.message.reply_text("⚠️ Pas assez de mots dans le jeu pour lancer un duel pour l'instant.")
        return

    duel = await asyncio.to_thread(DU.creer_duel, pseudo, mots, "telegram")
    code = duel["code"]
    ctx.bot_data.setdefault("duel_chats", {})[code] = {pseudo: update.effective_chat.id}

    moi = await ctx.bot.get_me()
    lien = f"https://t.me/{moi.username}?start=duel_{code}"
    # Pas de parse_mode ici : le lien contient "duel_" (underscore) et le
    # pseudo/nom Telegram de l'adversaire est arbitraire — l'un ou l'autre
    # peut casser le parseur Markdown de Telegram et faire échouer l'envoi
    # du message *silencieusement* (aucune erreur visible côté utilisateur).
    await update.message.reply_text(
        f"⚔️ Duel créé!\n\nCode : {code}\n\n"
        f"Envoie ce lien à un ami :\n{lien}\n\n"
        f"Ou il tape /duel {code} pour te rejoindre directement.\n\n"
        "En attente d'un adversaire..."
    )

async def _duel_rejoindre(update: Update, ctx: ContextTypes.DEFAULT_TYPE, code: str):
    pseudo = _pseudo_telegram(update.effective_user)
    duel = await asyncio.to_thread(DU.rejoindre_duel, code, pseudo)
    if not duel:
        await update.effective_message.reply_text(
            "❌ Code invalide, duel déjà complet, ou pseudo déjà pris par l'autre joueur."
        )
        return

    chats = ctx.bot_data.setdefault("duel_chats", {})
    chats.setdefault(code, {})[pseudo] = update.effective_chat.id
    adversaire = duel["joueurs"][0]["pseudo"]
    await update.effective_message.reply_text(
        f"⚔️ Tu affrontes {adversaire}! Que le meilleur gagne 🔥"
    )

    ctx.bot_data.setdefault("duel_q_debut", {})[code] = time.time()
    await _duel_envoyer_question(ctx, duel, code)

async def _duel_envoyer_question(ctx: ContextTypes.DEFAULT_TYPE, duel: dict, code: str):
    chats = ctx.bot_data.get("duel_chats", {}).get(code, {})
    qidx = duel["question_actuelle"]
    q = duel["questions"][qidx]
    clavier = InlineKeyboardMarkup([
        [InlineKeyboardButton(opt, callback_data=f"duel:{code}:{qidx}:{i}")]
        for i, opt in enumerate(q["options"])
    ])
    texte = f"⚔️ Question {qidx+1}/{len(duel['questions'])}\n\n{q['emoji']} Comment dit-on {q['fr']} en pular?"
    for pseudo, chat_id in list(chats.items()):
        try:
            await ctx.bot.send_message(chat_id, texte, reply_markup=clavier)
        except Exception as e:
            log.warning(f"Envoi question duel {code} à {pseudo}: {e}")

async def handle_duel_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    _, code, qidx_str, opt_idx_str = query.data.split(":")
    qidx = int(qidx_str)

    duel_avant = await asyncio.to_thread(DU.obtenir_duel, code)
    if not duel_avant or duel_avant["statut"] != "en_cours" or duel_avant["question_actuelle"] != qidx:
        await query.answer("⏱️ Cette question n'est plus active.", show_alert=True)
        return
    await query.answer()

    pseudo = _pseudo_telegram(update.effective_user)
    option = duel_avant["questions"][qidx]["options"][int(opt_idx_str)]
    debut = ctx.bot_data.get("duel_q_debut", {}).get(code, time.time())
    temps_ms = int((time.time() - debut) * 1000)

    duel = await asyncio.to_thread(DU.enregistrer_reponse, code, pseudo, qidx, option, temps_ms)
    if not duel:
        return

    moi = next((j for j in duel["joueurs"] if j["pseudo"] == pseudo), None)
    ma_reponse = next((r for r in moi["reponses"] if r["q"] == qidx), None) if moi else None
    if ma_reponse and ma_reponse["correct"]:
        resultat = f"✅ Bonne réponse! +{ma_reponse['points']} points"
    else:
        resultat = f"❌ Raté — c'était {duel_avant['questions'][qidx]['reponse']}"
    await query.edit_message_text(f"{resultat}\n\nEn attente de la question suivante...")

    if duel["statut"] == "termine":
        await _duel_terminer(ctx, duel, code)
    elif duel["question_actuelle"] != qidx:
        ctx.bot_data.setdefault("duel_q_debut", {})[code] = time.time()
        await _duel_envoyer_question(ctx, duel, code)

async def _duel_terminer(ctx: ContextTypes.DEFAULT_TYPE, duel: dict, code: str):
    chats = ctx.bot_data.get("duel_chats", {}).get(code, {})
    j1, j2 = duel["joueurs"][0], duel["joueurs"][1]
    for pseudo, chat_id in chats.items():
        moi, adversaire = (j1, j2) if j1["pseudo"] == pseudo else (j2, j1)
        if moi["score"] > adversaire["score"]:
            titre = "🏆 Tu as gagné!"
        elif moi["score"] < adversaire["score"]:
            titre = "😅 Perdu — la revanche t'attend!"
        else:
            titre = "🤝 Égalité!"
        try:
            await ctx.bot.send_message(
                chat_id,
                f"{titre}\n\nToi: {moi['score']} pts\n{adversaire['pseudo']}: {adversaire['score']} pts\n\n"
                "Tape /duel pour relancer un défi, ou /classement pour voir le classement!",
            )
        except Exception as e:
            log.warning(f"Envoi résultats duel {code} à {pseudo}: {e}")

    for cle in ("duel_chats", "duel_q_debut"):
        ctx.bot_data.get(cle, {}).pop(code, None)

async def cmd_classement(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    top = await asyncio.to_thread(DU.classement, 10)
    if not top:
        await update.message.reply_text("Aucun duel terminé pour l'instant. Lance-toi avec /duel! ⚔️")
        return
    medailles = ["🥇", "🥈", "🥉"] + ["⚔️"] * 7
    # Pas de parse_mode : les pseudos viennent aussi du site web, en texte
    # libre — un underscore ou astérisque dedans casserait le Markdown.
    lignes = ["🏆 Classement des duels Pular IA\n"]
    for i, e in enumerate(top):
        lignes.append(f"{medailles[i]} {e['pseudo']} — {e['victoires']} victoire(s), {e['points']} pts")
    await update.message.reply_text("\n".join(lignes))

# ── Erreurs ───────────────────────────────────────────────────────────────────
async def handle_error(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    """Filet de sécurité global : sans ça, une exception dans un handler
    (ex: message Markdown mal formé rejeté par Telegram) échoue en silence
    totale — rien dans le chat, juste une ligne perdue dans les logs."""
    log.error(f"Exception non gérée: {ctx.error}", exc_info=ctx.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "⚠️ Une erreur s'est produite. Réessaie dans un instant."
            )
        except Exception:
            pass

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

    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("stats",      cmd_stats))
    app.add_handler(CommandHandler("top",        cmd_top))
    app.add_handler(CommandHandler("duel",       cmd_duel))
    app.add_handler(CommandHandler("classement", cmd_classement))
    app.add_handler(CommandHandler("aide",       cmd_aide))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    # Le handler de duel (motif "duel:") doit être enregistré avant le
    # handler générique ci-dessous, sinon celui-ci intercepterait tout.
    app.add_handler(CallbackQueryHandler(handle_duel_callback, pattern=r"^duel:"))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(handle_error)

    log.info("🤖 Bot Pular IA démarré! Ctrl+C pour arrêter.")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)

if __name__ == "__main__":
    main()
