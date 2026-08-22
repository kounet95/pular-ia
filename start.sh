#!/bin/sh
# Lance le bot Telegram en arrière-plan (silencieux si TELEGRAM_BOT_TOKEN
# n'est pas configuré — community_bot.py logge une erreur et s'arrête tout
# seul, sans faire échouer ce script) puis le serveur web au premier plan,
# via exec pour que Railway puisse lui envoyer proprement SIGTERM à l'arrêt.
python scripts/community_bot.py &

# Le disque persistant (corpus-pular/) est vide au premier déploiement, ou
# après ajout d'une nouvelle source de données — le Dockerfile ne copie que
# scripts/ et web/, jamais corpus-pular/. On télécharge donc le Coran fulani
# ici si absent, en arrière-plan et sans indexation RAG (--sans-rag, léger)
# pour ne pas retarder le démarrage du serveur ni le healthcheck.
if [ ! -f corpus-pular/livres/metadata/fulani_rwwad_raw.json ]; then
  ( python scripts/import_coran_fulani.py --sans-rag || echo "Import Coran fulani échoué au démarrage" ) &
fi

exec python scripts/community_webapp.py
