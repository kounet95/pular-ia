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
    /livres — Acheter des livres (numérique ou papier)
    /don    — Faire un don au projet
    /quiz   — Rejoindre un Quiz Live (+ /quiz CODE)
    /aide   — Aide complète
"""

import os
import time
import json
import asyncio
import logging
import httpx
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import duels as DU
import comptes as CP
import espace_editorial as EE
import notifications as NOTIF
import quizlive as QL
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

# URL publique du site — utilisée pour construire les liens de paiement
# Stripe depuis le bot (contrairement au webapp, il n'y a pas de Request
# HTTP dont dériver l'origine ici).
SITE_URL = os.getenv("SITE_URL", "").rstrip("/")
if not SITE_URL:
    _base = os.getenv("APP_BASE_URL", "").rstrip("/")
    SITE_URL = _base[:-4] if _base.endswith("/api") else _base
if not SITE_URL:
    SITE_URL = "http://localhost:8080"

# Appel interne au webapp (même conteneur, voir start.sh) plutôt que SITE_URL
# (domaine public) pour les actions Quiz Live qui doivent déclencher une
# diffusion WebSocket immédiate vers l'écran de l'animateur — inutile de
# sortir sur Internet pour un appel entre deux processus du même conteneur.
WEBAPP_LOCAL_URL = f"http://127.0.0.1:{int(os.getenv('PORT', os.getenv('WEBAPP_PORT', 8080)))}"

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

    # Lien de connexion au compte web : https://t.me/<bot>?start=connexion_<code>
    if ctx.args and ctx.args[0].startswith("connexion_"):
        await _connexion_confirmer(update, ctx, ctx.args[0][len("connexion_"):].upper())
        return

    # Lien d'abonnement depuis le site web : https://t.me/<bot>?start=abonner
    if ctx.args and ctx.args[0] == "abonner":
        await cmd_abonner(update, ctx)
        return

    # Lien de partie Quiz Live affiché sur l'écran animateur : https://t.me/<bot>?start=quiz_<code>
    if ctx.args and ctx.args[0].startswith("quiz_"):
        await _quizlive_rejoindre(update, ctx, ctx.args[0][len("quiz_"):].upper())
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
        "📚 /livres — Acheter des livres (numérique ou papier)\n"
        "💛 /don — Faire un don au projet\n"
        "🎮 /quiz CODE — Rejoindre un Quiz Live\n"
        "🔔 /abonner — Être averti des nouveaux livres et éditos\n"
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
        "📚 /livres — voir et acheter les livres (numérique ou papier)\n"
        "💛 /don — faire un don au projet\n"
        "🎮 /quiz CODE — rejoindre un Quiz Live animé sur l'écran de quelqu'un\n"
        "🔔 /abonner — être averti des nouveaux livres et éditos\n\n"
        "📌 *Conseils pour une bonne qualité:*\n"
        "• Parle clairement, micro proche\n"
        "• Messages de 5 à 60 secondes idéaux\n"
        "• N'importe quel sujet en pular!\n"
        "• Évite les bruits de fond\n\n"
        "_Baŋ-baŋ! 🙏_",
        parse_mode="Markdown",
    )

# ── Notifications (nouveaux livres, nouveaux éditos) ───────────────────────────
async def cmd_abonner(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    pseudo = _pseudo_telegram(update.effective_user)
    nouveau = await asyncio.to_thread(NOTIF.abonner, chat_id, pseudo)
    if nouveau:
        await update.effective_message.reply_text(
            "🔔 Abonnement activé ! Tu recevras un message ici à chaque nouveau livre "
            "ou nouvel édito publié.\n\nTape /desabonner à tout moment pour arrêter."
        )
    else:
        await update.effective_message.reply_text("🔔 Tu es déjà abonné aux nouveautés.")

async def cmd_desabonner(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    retire = await asyncio.to_thread(NOTIF.desabonner, chat_id)
    if retire:
        await update.effective_message.reply_text(
            "🔕 Abonnement désactivé. Tape /abonner pour te réabonner à tout moment."
        )
    else:
        await update.effective_message.reply_text("Tu n'étais pas abonné aux nouveautés.")

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

# ── Connexion au compte web via Telegram ─────────────────────────────────

async def _connexion_confirmer(update: Update, ctx: ContextTypes.DEFAULT_TYPE, code: str):
    user = update.effective_user
    ok = await asyncio.to_thread(
        CP.confirmer_code_telegram, code, user.id, user.username, user.first_name,
    )
    if not ok:
        await update.effective_message.reply_text(
            "⚠️ Ce lien de connexion a expiré ou n'est plus valide. "
            "Retourne sur le site et clique à nouveau sur \"Se connecter avec Telegram\"."
        )
        return
    await update.effective_message.reply_text(
        "✅ Connexion confirmée ! Reviens sur le site, la page se met à jour automatiquement."
    )

async def cmd_duel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args:
        await _duel_rejoindre(update, ctx, ctx.args[0].upper())
    else:
        await _duel_creer(update, ctx)

async def _duel_creer(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Affiche le choix du thème (bouton par catégorie) avant de créer."""
    lignes = [DU.THEMES[i:i + 2] for i in range(0, len(DU.THEMES), 2)]
    clavier = InlineKeyboardMarkup([
        [InlineKeyboardButton(t, callback_data=f"duelcfg:theme:{t}") for t in ligne]
        for ligne in lignes
    ])
    await update.message.reply_text("⚔️ Quel thème pour ce duel ?", reply_markup=clavier)

async def handle_duelcfg_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")  # ["duelcfg", "theme", Theme] ou ["duelcfg", "len", Theme, N]

    if parts[1] == "theme":
        theme = parts[2]
        clavier = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"⚡ Rapide (5)",    callback_data=f"duelcfg:len:{theme}:5"),
            InlineKeyboardButton(f"🎯 Standard (10)", callback_data=f"duelcfg:len:{theme}:10"),
            InlineKeyboardButton(f"🏁 Marathon (20)", callback_data=f"duelcfg:len:{theme}:20"),
        ]])
        await query.edit_message_text(f"⚔️ Thème : {theme}\n\nCombien de questions ?", reply_markup=clavier)
        return

    if parts[1] == "len":
        theme, nb_questions = parts[2], int(parts[3])
        await _duel_creer_avec_options(update, ctx, theme, nb_questions)

async def _duel_creer_avec_options(update: Update, ctx: ContextTypes.DEFAULT_TYPE, theme: str, nb_questions: int):
    pseudo = _pseudo_telegram(update.effective_user)
    mots = DU.charger_mots_pour_duel()
    if len(mots) < 4:
        await update.effective_message.reply_text("⚠️ Pas assez de mots dans le jeu pour lancer un duel pour l'instant.")
        return

    duel = await asyncio.to_thread(DU.creer_duel, pseudo, mots, "telegram", theme, nb_questions)
    code = duel["code"]
    ctx.bot_data.setdefault("duel_chats", {})[code] = {pseudo: update.effective_chat.id}

    moi = await ctx.bot.get_me()
    lien = f"https://t.me/{moi.username}?start=duel_{code}"
    # Pas de parse_mode ici : le lien contient "duel_" (underscore) et le
    # pseudo/nom Telegram de l'adversaire est arbitraire — l'un ou l'autre
    # peut casser le parseur Markdown de Telegram et faire échouer l'envoi
    # du message *silencieusement* (aucune erreur visible côté utilisateur).
    texte = (
        f"⚔️ Duel créé!\n\nThème : {duel['theme']} · {len(duel['questions'])} questions\n"
        f"Code : {code}\n\n"
        f"Envoie ce lien à un ami :\n{lien}\n\n"
        f"Ou il tape /duel {code} pour te rejoindre directement.\n\n"
        "En attente d'un adversaire..."
    )
    if update.callback_query:
        await update.callback_query.edit_message_text(texte)
    else:
        await update.effective_message.reply_text(texte)

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

# ── Achat de livres ───────────────────────────────────────────────────────────
# Le catalogue, les zones de livraison et les commandes vivent dans les mêmes
# fichiers JSON que le webapp (espace_editorial.py) — pas besoin d'appel HTTP
# entre les deux processus, ils partagent le même volume/filesystem.

def _prix_lisible(centimes: int, devise: str) -> str:
    if devise.lower() in EE.DEVISES_ZERO_DECIMALE:
        return f"{centimes:,} {devise}".replace(",", " ")
    return f"{centimes / 100:.2f} {devise}"

async def cmd_livres(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    livres = await asyncio.to_thread(EE.charger_catalogue)
    if not livres:
        await update.message.reply_text("📚 Aucun livre disponible pour l'instant.")
        return
    rep = EE.REPARTITION_REVENUS
    await update.message.reply_text(
        "📚 Catalogue Pular IA\n\n"
        f"💛 100% du prix de chaque livre est réparti : {rep['auteur']}% pour l'auteur, "
        f"{rep['plateforme']}% pour l'entretien de la plateforme, "
        f"{rep['projet']}% pour financer le projet — une IA qui comprend le pular, "
        "des projets de restauration du Fouta, et d'autres initiatives communautaires."
    )
    for livre in livres:
        await _envoyer_fiche_livre(update.effective_chat.id, ctx, livre)

async def _envoyer_fiche_livre(chat_id: int, ctx: ContextTypes.DEFAULT_TYPE, livre: dict):
    devise = livre.get("devise", "gnf").upper()
    prix_num = livre.get("prix_numerique_centimes")
    prix_pap = livre.get("prix_papier_centimes")

    morceaux = [livre["titre"]]
    if livre.get("auteur"):
        morceaux.append(f"✍️ {livre['auteur']}")
    if livre.get("description"):
        desc = livre["description"].strip()
        morceaux.append(desc[:400] + ("…" if len(desc) > 400 else ""))
    # Pas de parse_mode : titre/auteur/description viennent du catalogue
    # (texte libre saisi par l'admin) et peuvent contenir _ ou * sans
    # échappement — même leçon que pour les duels.
    caption = "\n\n".join(morceaux)[:1000]

    boutons = []
    if prix_num is not None:
        boutons.append([InlineKeyboardButton(
            f"📱 Numérique — {_prix_lisible(prix_num, devise)}",
            callback_data=f"livrefmt:{livre['id']}:numerique",
        )])
    if prix_pap is not None:
        boutons.append([InlineKeyboardButton(
            f"📦 Papier — {_prix_lisible(prix_pap, devise)}",
            callback_data=f"livrefmt:{livre['id']}:papier",
        )])
    clavier = InlineKeyboardMarkup(boutons)

    chemin_couverture = EE.DOSSIER_COUVERTURES / livre["couverture"] if livre.get("couverture") else None
    if chemin_couverture and chemin_couverture.exists():
        with open(chemin_couverture, "rb") as f:
            await ctx.bot.send_photo(chat_id, photo=f, caption=caption, reply_markup=clavier)
    else:
        await ctx.bot.send_message(chat_id, caption, reply_markup=clavier)

async def handle_livre_format_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, livre_id, format_achete = query.data.split(":")
    livres = await asyncio.to_thread(EE.charger_catalogue)
    livre = next((l for l in livres if l["id"] == livre_id), None)
    if not livre:
        await query.message.reply_text("⚠️ Ce livre n'est plus disponible.")
        return

    if format_achete == "papier":
        zones = await asyncio.to_thread(EE.charger_zones_livraison)
        boutons = [
            [InlineKeyboardButton(
                f"📍 {z['nom']} (+{_prix_lisible(z['frais_centimes'], z['devise'].upper())})",
                callback_data=f"livrezone:{livre_id}:{z['id']}",
            )]
            for z in zones
        ]
        boutons.append([InlineKeyboardButton(
            "🤝 Retrait / à organiser (sans frais)", callback_data=f"livrezone:{livre_id}:none",
        )])
        await query.message.reply_text(
            f"📦 Livraison papier pour « {livre['titre']} »\nChoisis ta zone :",
            reply_markup=InlineKeyboardMarkup(boutons),
        )
        return

    await _demarrer_paiement(update, ctx, livre, "numerique", None)

async def handle_livre_zone_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    _, livre_id, zone_id = query.data.split(":")
    livres = await asyncio.to_thread(EE.charger_catalogue)
    livre = next((l for l in livres if l["id"] == livre_id), None)
    if not livre:
        await query.message.reply_text("⚠️ Ce livre n'est plus disponible.")
        return
    zone = None
    if zone_id != "none":
        zones = await asyncio.to_thread(EE.charger_zones_livraison)
        zone = next((z for z in zones if z["id"] == zone_id), None)
    await _demarrer_paiement(update, ctx, livre, "papier", zone)

async def _demarrer_paiement(update: Update, ctx: ContextTypes.DEFAULT_TYPE, livre: dict, format_achete: str, zone: dict | None):
    chat = update.effective_chat
    user = update.effective_user
    try:
        session = await asyncio.to_thread(
            EE.creer_session_paiement, livre, format_achete, zone, SITE_URL, "",
        )
    except ValueError as e:
        await chat.send_message(f"⚠️ {e}")
        return
    except RuntimeError as e:
        await chat.send_message(f"⚠️ Paiement indisponible : {e}")
        return
    except Exception as e:
        log.error(f"Erreur création session Stripe (bot): {e}")
        await chat.send_message("⚠️ Erreur lors de la création du paiement, réessaie plus tard.")
        return

    commande = await asyncio.to_thread(
        EE.enregistrer_commande, session, livre, format_achete, zone, user.id,
    )

    nom_format = "Numérique" if format_achete == "numerique" else "Papier"
    await chat.send_message(
        f"💳 Paiement — {livre['titre']} ({nom_format})\n\n"
        f"Ouvre ce lien pour payer en toute sécurité (Stripe) :\n{session.url}\n\n"
        "Reviens ici une fois payé, je te préviens automatiquement ✅"
    )
    asyncio.create_task(_surveiller_paiement(ctx, chat.id, commande["id"]))

async def _surveiller_paiement(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, commande_id: str,
                                tentatives: int = 240, intervalle: int = 5):
    """
    Vérifie périodiquement (jusqu'à ~20 min) si la commande a été payée, et
    livre automatiquement : le fichier pour le numérique, une confirmation
    pour le papier. Si le bot redémarre entre-temps, cette tâche en mémoire
    est perdue — le paiement reste enregistré (page /livre-achete côté web
    fait la même vérification en parallèle), seule la livraison automatique
    dans le chat ne se déclenche pas ; acceptable pour une v1.
    """
    for _ in range(tentatives):
        await asyncio.sleep(intervalle)
        commande = await asyncio.to_thread(EE.obtenir_commande, commande_id)
        if not commande:
            return
        if commande["statut"] == "paye":
            await _livrer_commande(ctx, chat_id, commande)
            return

async def _livrer_commande(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, commande: dict):
    rep = EE.REPARTITION_REVENUS
    if commande["format"] == "numerique" and commande.get("fichier_numerique"):
        chemin = EE.DOSSIER_LIVRES_FICHIERS / commande["fichier_numerique"]
        if chemin.exists():
            with open(chemin, "rb") as f:
                await ctx.bot.send_document(
                    chat_id, document=f, filename=f"{commande['titre']}{chemin.suffix}",
                    caption=f"✅ Paiement confirmé — merci ! Voici « {commande['titre']} ».",
                )
        else:
            await ctx.bot.send_message(chat_id, "✅ Paiement confirmé, mais le fichier est introuvable — contacte l'équipe.")
    else:
        zone_txt = f" pour {commande['zone']['nom']}" if commande.get("zone") else ""
        await ctx.bot.send_message(
            chat_id,
            f"✅ Paiement confirmé — commande papier enregistrée{zone_txt}.\n"
            "On te contacte bientôt pour organiser la remise.",
        )
    await ctx.bot.send_message(
        chat_id,
        f"💛 Répartition : {rep['auteur']}% auteur · {rep['plateforme']}% plateforme · "
        f"{rep['projet']}% projet (IA pular, restauration du Fouta...). Merci pour ton soutien 🙏",
    )

# ── Dons ──────────────────────────────────────────────────────────────────────
# Contrairement aux livres, un don n'a pas d'auteur à rémunérer : 100% va au
# projet — voir EE.creer_session_don / la mention affichée après paiement.

_DON_DEVISE_BOT = "gnf"  # devise proposée dans le bot (montant libre non géré côté Telegram)

async def cmd_don(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    montants = EE.MONTANTS_DON_SUGGERES.get(_DON_DEVISE_BOT, [])
    boutons = [[InlineKeyboardButton(
        f"💛 {_prix_lisible(EE.calculer_montant_stripe(m, _DON_DEVISE_BOT), _DON_DEVISE_BOT.upper())}",
        callback_data=f"donmontant:{m}",
    )] for m in montants]
    await update.message.reply_text(
        "💛 Faire un don au projet Pular IA\n\n"
        "100% de ton don finance directement le projet : une IA qui comprend le pular, "
        "des projets de restauration du Fouta, et l'entretien de la plateforme.\n\n"
        "Choisis un montant :",
        reply_markup=InlineKeyboardMarkup(boutons),
    )

async def handle_don_montant_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    montant = float(query.data.split(":")[1])
    montant_centimes = EE.calculer_montant_stripe(montant, _DON_DEVISE_BOT)

    chat = update.effective_chat
    user = update.effective_user
    try:
        session = await asyncio.to_thread(
            EE.creer_session_don, montant_centimes, _DON_DEVISE_BOT, SITE_URL, "", _pseudo_telegram(user),
        )
    except RuntimeError as e:
        await chat.send_message(f"⚠️ Don indisponible : {e}")
        return
    except Exception as e:
        log.error(f"Erreur création session don Stripe (bot): {e}")
        await chat.send_message("⚠️ Erreur lors de la création du don, réessaie plus tard.")
        return

    don = await asyncio.to_thread(
        EE.enregistrer_don, session, montant_centimes, _DON_DEVISE_BOT, _pseudo_telegram(user), user.id,
    )

    await query.message.reply_text(
        f"💳 Ouvre ce lien pour faire ton don en toute sécurité (Stripe) :\n{session.url}\n\n"
        "Reviens ici une fois payé, je te préviens automatiquement ✅"
    )
    asyncio.create_task(_surveiller_don(ctx, chat.id, don["id"]))

async def _surveiller_don(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, don_id: str,
                           tentatives: int = 240, intervalle: int = 5):
    """Même logique/limite que _surveiller_paiement (voir sa docstring) :
    tâche en mémoire, perdue si le bot redémarre, paiement lui-même intact."""
    for _ in range(tentatives):
        await asyncio.sleep(intervalle)
        don = await asyncio.to_thread(EE.obtenir_don, don_id)
        if not don:
            return
        if don["statut"] == "paye":
            await ctx.bot.send_message(
                chat_id,
                "✅ Don confirmé — merci infiniment pour ton soutien ! 🙏\n\n"
                "💛 100% finance le projet : IA pular, restauration du Fouta, entretien de la plateforme.",
            )
            return

# ── Quiz Live (façon Kahoot) — les joueurs répondent depuis Telegram, ─────────
# l'animateur garde l'écran web pour le partager pendant son live. Le jeu vit
# dans les mêmes fichiers JSON que le webapp (voir quizlive.py) : ce bot
# n'a pas de canal temps réel vers l'hôte web, donc il *surveille* la partie
# (un seul sondage partagé par tous les joueurs Telegram d'une même partie,
# pas un par joueur) et pousse les mises à jour dans Telegram à chaque
# changement d'état — même principe que la surveillance de paiement.
_QUIZLIVE_FORMES = ["▲", "◆", "●", "■"]

async def cmd_quiz(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.effective_message.reply_text(
            "🎮 Pour rejoindre un quiz live, tape /quiz suivi du code affiché "
            "par l'animateur sur son écran, par exemple : /quiz ABC123"
        )
        return
    await _quizlive_rejoindre(update, ctx, ctx.args[0].upper())

async def _quizlive_rejoindre(update: Update, ctx: ContextTypes.DEFAULT_TYPE, code: str):
    pseudo = _pseudo_telegram(update.effective_user)
    chat_id = update.effective_chat.id

    partie = await asyncio.to_thread(QL.obtenir_partie, code)
    if not partie:
        await update.effective_message.reply_text("⚠️ Code de partie introuvable.")
        return
    if partie["hote"].strip().lower() == pseudo.strip().lower():
        await update.effective_message.reply_text("⚠️ Tu es l'animateur de cette partie — anime-la depuis l'écran web, pas ici.")
        return

    deja = any(j["pseudo"] == pseudo for j in partie["joueurs"])
    if not deja:
        if partie["statut"] != "attente":
            await update.effective_message.reply_text("⚠️ Cette partie a déjà démarré — trop tard pour la rejoindre.")
            return
        # Passe par l'API HTTP du webapp (pas un appel direct à QL.rejoindre_partie) :
        # c'est ce process qui possède les connexions WebSocket de l'animateur,
        # un appel direct ici laisserait son écran sans notification.
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(f"{WEBAPP_LOCAL_URL}/api/quizlive/{code}/rejoindre", data={"pseudo": pseudo})
            if r.status_code != 200:
                await update.effective_message.reply_text("⚠️ Impossible de rejoindre (pseudo déjà pris, ou partie complète/démarrée).")
                return
        except Exception as e:
            log.error(f"Quiz live — rejoindre {code} via HTTP: {e}")
            await update.effective_message.reply_text("⚠️ Erreur de connexion, réessaie dans un instant.")
            return

    msg = await update.effective_message.reply_text(
        f"✅ Tu as rejoint la partie {code} !\n⏳ En attente que l'animateur démarre…"
    )
    joueurs_tg = ctx.bot_data.setdefault("quizlive_joueurs", {}).setdefault(code, {})
    joueurs_tg[pseudo] = {"chat_id": chat_id, "message_id": msg.message_id}

    pollers = ctx.bot_data.setdefault("quizlive_pollers", set())
    if code not in pollers:
        pollers.add(code)
        asyncio.create_task(_quizlive_surveiller(ctx, code))

async def _quizlive_surveiller(ctx: ContextTypes.DEFAULT_TYPE, code: str, max_ticks: int = 2400):
    """Un seul sondage par partie, partagé par tous ses joueurs Telegram —
    évite de relire le fichier une fois par joueur par seconde. S'arrête
    dès que la partie est terminée ou après ~1h (garde-fou)."""
    try:
        dernier_etat = None
        for _ in range(max_ticks):
            await asyncio.sleep(1.5)
            partie = await asyncio.to_thread(QL.obtenir_partie, code)
            if not partie:
                return
            etat = (partie["statut"], partie["question_actuelle"])
            if etat != dernier_etat:
                dernier_etat = etat
                await _quizlive_diffuser_telegram(ctx, code, partie)
            if partie["statut"] == "termine":
                return
    finally:
        ctx.bot_data.get("quizlive_pollers", set()).discard(code)

async def _quizlive_diffuser_telegram(ctx: ContextTypes.DEFAULT_TYPE, code: str, partie: dict):
    joueurs_tg = ctx.bot_data.get("quizlive_joueurs", {}).get(code, {})
    etat = QL.etat_public(partie)

    for pseudo, info in list(joueurs_tg.items()):
        chat_id, message_id = info["chat_id"], info.get("message_id")
        try:
            if partie["statut"] == "en_cours":
                q = etat["question_actuelle"]
                clavier = InlineKeyboardMarkup([[
                    InlineKeyboardButton(f, callback_data=f"quizrep:{code}:{q['index']}:{i}")
                    for i, f in enumerate(_QUIZLIVE_FORMES)
                ]])
                texte = f"❓ Question {q['index']+1}/{q['total']} — regarde l'écran de l'animateur et tape la forme qui correspond !"
                if message_id:
                    nouveau = await ctx.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=texte, reply_markup=clavier)
                else:
                    nouveau = await ctx.bot.send_message(chat_id, texte, reply_markup=clavier)
                info["message_id"] = nouveau.message_id

            elif partie["statut"] == "revelation":
                q = etat["question_actuelle"]
                joueur = next((j for j in partie["joueurs"] if j["pseudo"] == pseudo), None)
                derniere = next((r for r in joueur["reponses"] if r["q"] == q["index"]), None) if joueur else None
                if derniere is None:
                    texte = f"⌛ Pas de réponse à temps.\nLa bonne réponse était : {q['reponse']}"
                elif derniere["correct"]:
                    texte = f"✅ Bonne réponse ! +{derniere['points']} pts\nScore total : {joueur['score']}"
                else:
                    texte = f"❌ Mauvaise réponse.\nLa bonne réponse était : {q['reponse']}\nScore total : {joueur['score']}"
                if message_id:
                    await ctx.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=texte)
                else:
                    await ctx.bot.send_message(chat_id, texte)

            elif partie["statut"] == "termine":
                rang = next((i for i, j in enumerate(etat["classement"]) if j["pseudo"] == pseudo), None)
                score = etat["classement"][rang]["score"] if rang is not None else 0
                if rang is None:
                    medaille = "?"
                elif rang < 3:
                    medaille = ["🥇", "🥈", "🥉"][rang]
                else:
                    medaille = f"#{rang + 1}"
                await ctx.bot.send_message(
                    chat_id,
                    f"🏁 Partie terminée !\n{medaille} — {score} points\n\n"
                    f"🏆 Palmarès des champions : {SITE_URL}/quizlive/classement",
                )
        except Exception as e:
            log.warning(f"Quiz live Telegram — envoi à {pseudo} ({code}) échoué: {e}")

async def handle_quizlive_reponse_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    _, code, q_index_str, opt_index_str = query.data.split(":")
    q_index, opt_index = int(q_index_str), int(opt_index_str)
    pseudo = _pseudo_telegram(update.effective_user)

    partie = await asyncio.to_thread(QL.obtenir_partie, code)
    if not partie or partie["statut"] != "en_cours" or q_index != partie["question_actuelle"]:
        await query.answer("⏱️ Trop tard pour cette question !")
        return
    options = partie["questions"][q_index]["options"]
    if opt_index >= len(options):
        await query.answer()
        return
    option = options[opt_index]

    temps_ms = 0
    if partie.get("question_ouverte_le"):
        ouverte = datetime.fromisoformat(partie["question_ouverte_le"])
        temps_ms = max(0, int((datetime.now() - ouverte).total_seconds() * 1000))

    # Même remarque que pour rejoindre : passe par l'API HTTP du webapp pour
    # que la diffusion WebSocket vers l'écran de l'animateur se déclenche
    # immédiatement (compteur "X ont répondu", etc.), pas seulement l'écriture.
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"{WEBAPP_LOCAL_URL}/api/quizlive/{code}/reponse",
                data={"pseudo": pseudo, "q": q_index, "reponse": option, "temps_ms": temps_ms},
            )
        if r.status_code == 200:
            await query.answer("✅ Réponse envoyée !")
            try:
                await query.edit_message_reply_markup(reply_markup=None)  # empêche de re-cliquer
            except Exception:
                pass
        else:
            await query.answer("⚠️ Réponse non prise en compte (trop tard, ou partie introuvable).")
    except Exception as e:
        log.error(f"Quiz live — réponse {code} via HTTP: {e}")
        await query.answer("⚠️ Erreur de connexion, réessaie.")

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
    app.add_handler(CommandHandler("livres",     cmd_livres))
    app.add_handler(CommandHandler("don",        cmd_don))
    app.add_handler(CommandHandler("quiz",       cmd_quiz))
    app.add_handler(CommandHandler("abonner",    cmd_abonner))
    app.add_handler(CommandHandler("desabonner", cmd_desabonner))
    app.add_handler(CommandHandler("aide",       cmd_aide))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    # Les handlers à motif précis doivent être enregistrés avant le handler
    # générique ci-dessous, sinon celui-ci intercepterait tout.
    app.add_handler(CallbackQueryHandler(handle_duelcfg_callback, pattern=r"^duelcfg:"))
    app.add_handler(CallbackQueryHandler(handle_duel_callback, pattern=r"^duel:"))
    app.add_handler(CallbackQueryHandler(handle_livre_format_callback, pattern=r"^livrefmt:"))
    app.add_handler(CallbackQueryHandler(handle_livre_zone_callback, pattern=r"^livrezone:"))
    app.add_handler(CallbackQueryHandler(handle_don_montant_callback, pattern=r"^donmontant:"))
    app.add_handler(CallbackQueryHandler(handle_quizlive_reponse_callback, pattern=r"^quizrep:"))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(handle_error)

    log.info("🤖 Bot Pular IA démarré! Ctrl+C pour arrêter.")
    app.run_polling(drop_pending_updates=True, bootstrap_retries=5)

if __name__ == "__main__":
    main()
