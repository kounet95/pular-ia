"""
courriel.py — Envoi d'emails transactionnels via le SMTP de Gmail (compte
personnel + mot de passe d'application, aucun service tiers, aucune
nouvelle dépendance — juste smtplib de la bibliothèque standard).

Configuration (.env) :
    GMAIL_USER         — l'adresse Gmail expéditrice
    GMAIL_APP_PASSWORD — mot de passe d'application (PAS le mot de passe du
                          compte Google lui-même) — généré sur
                          myaccount.google.com/apppasswords, nécessite la
                          validation en deux étapes activée sur le compte.

Gmail limite les comptes personnels à environ 500 envois/jour — largement
suffisant pour des emails transactionnels (réinitialisation de mot de
passe, vérification) sur ce projet. Si le volume grandit un jour, migrer
vers un service dédié (Resend, Brevo...) sans changer les appelants —
seule cette fonction envoyer_email() aurait à changer.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

GMAIL_ADDRESS      = os.getenv("GMAIL_USER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")

def courriel_configure() -> bool:
    return bool(GMAIL_ADDRESS and GMAIL_APP_PASSWORD)

def envoyer_email(destinataire: str, sujet: str, corps_texte: str, corps_html: str | None = None) -> bool:
    """
    Envoie un email via le SMTP de Gmail. Retourne False (sans lever
    d'exception) si non configuré ou en cas d'échec — comme
    stripe_configure()/anthropic_configure() dans espace_editorial.py,
    pour qu'un envoi d'email raté ne fasse jamais planter le serveur.
    Bloquant (SMTP synchrone) : à appeler via asyncio.to_thread côté appelant.
    """
    if not courriel_configure():
        log.warning(f"Email non envoyé à {destinataire} — GMAIL_ADDRESS/GMAIL_APP_PASSWORD manquant(s) dans .env.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = sujet
        msg["From"] = f"Pular IA <{GMAIL_ADDRESS}>"
        msg["To"] = destinataire
        msg.attach(MIMEText(corps_texte, "plain", "utf-8"))
        if corps_html:
            msg.attach(MIMEText(corps_html, "html", "utf-8"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as serveur:
            serveur.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            serveur.sendmail(GMAIL_ADDRESS, [destinataire], msg.as_string())
        log.info(f"Email envoyé à {destinataire}: {sujet}")
        return True
    except Exception as e:
        log.error(f"Échec envoi email à {destinataire}: {e}")
        return False
