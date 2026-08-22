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

# Scraping Telegram en continu (enrichit corpus-pular avec les nouveaux
# messages/audios des canaux configurés — voir CANAUX_DEFAUT dans
# telegram_scraper.py). Nécessite qu'une première connexion interactive
# (téléphone + code SMS) ait déjà été faite au moins une fois : sans le
# fichier de session, Telethon resterait bloqué à attendre une saisie qui ne
# viendra jamais sur un serveur headless — on ne lance donc rien dans ce cas.
TELEGRAM_INTERVALLE_S="${TELEGRAM_SCRAPE_INTERVALLE_S:-21600}"  # 6h par défaut
if [ -n "$TELEGRAM_API_ID" ] && [ -n "$TELEGRAM_API_HASH" ] && [ -n "$TELEGRAM_PHONE" ] \
   && [ -f corpus-pular/processed/telegram/session_pular.session ]; then
  (
    while true; do
      python scripts/telegram_scraper.py || echo "Scraping Telegram échoué — nouvelle tentative dans ${TELEGRAM_INTERVALLE_S}s"
      sleep "$TELEGRAM_INTERVALLE_S"
    done
  ) &
else
  echo "Scraping Telegram désactivé : identifiants absents ou première connexion interactive pas encore faite (voir README)."
fi

exec python scripts/community_webapp.py
