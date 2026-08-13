#!/bin/sh
# Lance le bot Telegram en arrière-plan (silencieux si TELEGRAM_BOT_TOKEN
# n'est pas configuré — community_bot.py logge une erreur et s'arrête tout
# seul, sans faire échouer ce script) puis le serveur web au premier plan,
# via exec pour que Railway puisse lui envoyer proprement SIGTERM à l'arrêt.
python scripts/community_bot.py &
exec python scripts/community_webapp.py
