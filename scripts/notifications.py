"""
notifications.py — Abonnement Telegram aux nouveautés (nouveaux livres,
nouveaux éditos).

Liste d'abonnés stockée en JSON (comme le reste du projet). Les
notifications sont envoyées par appel HTTP direct à l'API Telegram plutôt
que via une instance python-telegram-bot vivante — nécessaire car la
création de contenu (ajout d'un livre, publication d'un édito) arrive côté
webapp, un processus séparé du bot (voir start.sh).
"""

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

PROJET_ROOT            = Path(__file__).resolve().parent.parent
DOSSIER_NOTIFICATIONS  = PROJET_ROOT / "corpus-pular" / "notifications"
FICHIER_ABONNES        = DOSSIER_NOTIFICATIONS / "abonnes.json"
DOSSIER_NOTIFICATIONS.mkdir(parents=True, exist_ok=True)

def charger_abonnes() -> list[dict]:
    if FICHIER_ABONNES.exists():
        with open(FICHIER_ABONNES, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_abonnes(abonnes: list[dict]):
    with open(FICHIER_ABONNES, "w", encoding="utf-8") as f:
        json.dump(abonnes, f, ensure_ascii=False, indent=2)

def est_abonne(chat_id: int) -> bool:
    return any(a["chat_id"] == chat_id for a in charger_abonnes())

def abonner(chat_id: int, pseudo: str) -> bool:
    """Retourne True si nouvellement abonné, False si déjà abonné."""
    abonnes = charger_abonnes()
    if any(a["chat_id"] == chat_id for a in abonnes):
        return False
    abonnes.append({
        "chat_id": chat_id,
        "pseudo":  pseudo.strip()[:40],
        "date":    datetime.now().isoformat(),
    })
    sauver_abonnes(abonnes)
    return True

def desabonner(chat_id: int) -> bool:
    abonnes = charger_abonnes()
    apres = [a for a in abonnes if a["chat_id"] != chat_id]
    if len(apres) == len(abonnes):
        return False
    sauver_abonnes(apres)
    return True

def notifier_tous(bot_token: str, texte: str):
    """
    Envoie `texte` à tous les abonnés, un par un — appel HTTP direct à
    l'API Telegram (fonctionne même sans instance python-telegram-bot
    vivante, donc utilisable depuis le processus webapp). Bloquant : à
    appeler via asyncio.to_thread côté appelant. Désabonne automatiquement
    quiconque a bloqué le bot (erreur 403) pour ne pas s'acharner sur des
    envois morts indéfiniment.
    """
    if not bot_token:
        log.warning("Notification annulée : TELEGRAM_BOT_TOKEN manquant.")
        return
    abonnes = charger_abonnes()
    if not abonnes:
        return
    a_retirer = []
    for a in abonnes:
        try:
            data = json.dumps({"chat_id": a["chat_id"], "text": texte}).encode("utf-8")
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data=data, headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                a_retirer.append(a["chat_id"])
            else:
                log.warning(f"Notification à {a['chat_id']} échouée ({e.code}): {e}")
        except Exception as e:
            log.warning(f"Notification à {a['chat_id']} échouée: {e}")
        time.sleep(0.05)  # reste large sous la limite ~30 msg/s de l'API Telegram

    if a_retirer:
        restants = [a for a in charger_abonnes() if a["chat_id"] not in a_retirer]
        sauver_abonnes(restants)
        log.info(f"{len(a_retirer)} abonné(s) retiré(s) (bot bloqué).")
