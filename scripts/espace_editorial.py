"""
espace_editorial.py — Espace Éditorial : vente de livre(s) + éditos communautaires

  - Catalogue  : livre(s) mis en vente par l'auteur (titre, description, prix,
                 couverture) avec paiement Stripe Checkout intégré.
  - Commandes  : trace des sessions de paiement Stripe et de leur statut,
                 pour permettre à l'admin de livrer manuellement (email/envoi).
  - Éditos     : articles courts écrits par la communauté (titre, auteur,
                 contenu), soumis en attente puis publiés par l'admin.

Le paiement passe par Stripe Checkout (page hébergée par Stripe) : le
serveur ne touche jamais les données de carte bancaire. `STRIPE_SECRET_KEY`
doit être défini dans l'environnement pour activer la vente ; sans elle,
`stripe_configure()` renvoie False et les routes de paiement le signalent
proprement au lieu de planter.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

PROJET_ROOT       = Path(__file__).resolve().parent.parent
DOSSIER_EDITORIAL = PROJET_ROOT / "corpus-pular" / "editorial"
DOSSIER_COUVERTURES = DOSSIER_EDITORIAL / "couvertures"
FICHIER_CATALOGUE = DOSSIER_EDITORIAL / "catalogue.json"
FICHIER_COMMANDES = DOSSIER_EDITORIAL / "commandes.json"
FICHIER_EDITOS    = DOSSIER_EDITORIAL / "editos.json"

for d in [DOSSIER_EDITORIAL, DOSSIER_COUVERTURES]:
    d.mkdir(parents=True, exist_ok=True)

DEVISES_ACCEPTEES = ["eur", "usd"]
EXTENSIONS_COUVERTURE = {".jpg", ".jpeg", ".png", ".webp"}

# ── Stripe ───────────────────────────────────────────────────────────────────

def stripe_configure():
    """Configure la clé Stripe si dispo. Retourne le module stripe ou None."""
    cle = os.getenv("STRIPE_SECRET_KEY", "")
    if not cle:
        return None
    try:
        import stripe
        stripe.api_key = cle
        return stripe
    except ImportError:
        log.warning("Le paquet 'stripe' n'est pas installé — pip install stripe")
        return None

def creer_session_paiement(livre: dict, origin: str, email: str = "") -> "stripe.checkout.Session":
    """Crée une session de paiement Stripe Checkout pour un livre du catalogue."""
    stripe = stripe_configure()
    if stripe is None:
        raise RuntimeError("Paiement non configuré (STRIPE_SECRET_KEY manquante).")

    origin = origin.rstrip("/")
    params = dict(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": livre["devise"],
                "unit_amount": livre["prix_centimes"],
                "product_data": {
                    "name": livre["titre"],
                    "description": livre.get("description", "")[:500] or None,
                },
            },
            "quantity": 1,
        }],
        success_url=f"{origin}/?edito_paiement=succes&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{origin}/?edito_paiement=annule",
        metadata={"livre_id": livre["id"]},
    )
    if email:
        params["customer_email"] = email
    return stripe.checkout.Session.create(**params)

def verifier_signature_webhook(payload: bytes, sig_header: str):
    """Vérifie la signature d'un événement webhook Stripe. Lève une exception si invalide."""
    stripe = stripe_configure()
    if stripe is None:
        raise RuntimeError("Stripe non configuré.")
    # Nom dédié pour ne pas entrer en conflit avec un STRIPE_WEBHOOK_SECRET
    # utilisé par un autre projet partageant le même compte Stripe.
    secret = os.getenv("STRIPE_WEBHOOK_SECRET_EDITORIAL", "")
    if not secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET_EDITORIAL manquant — impossible de vérifier le webhook.")
    return stripe.Webhook.construct_event(payload, sig_header, secret)

# ── Catalogue (livres en vente) ─────────────────────────────────────────────

def charger_catalogue() -> list[dict]:
    if FICHIER_CATALOGUE.exists():
        with open(FICHIER_CATALOGUE, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_catalogue(livres: list[dict]):
    with open(FICHIER_CATALOGUE, "w", encoding="utf-8") as f:
        json.dump(livres, f, ensure_ascii=False, indent=2)

def ajouter_livre(
    titre: str,
    description: str,
    prix_centimes: int,
    devise: str,
    format_livre: str,
    couverture_nom: str = "",
) -> dict:
    livres = charger_catalogue()
    fiche = {
        "id":              str(uuid.uuid4())[:8],
        "titre":           titre,
        "description":     description,
        "prix_centimes":   prix_centimes,
        "devise":          devise,
        "format":          format_livre,
        "couverture":      couverture_nom,
        "date":            datetime.now().isoformat(),
    }
    livres.append(fiche)
    sauver_catalogue(livres)
    return fiche

def supprimer_livre(livre_id: str) -> bool:
    livres = charger_catalogue()
    trouve = next((l for l in livres if l["id"] == livre_id), None)
    if not trouve:
        return False
    if trouve.get("couverture"):
        (DOSSIER_COUVERTURES / trouve["couverture"]).unlink(missing_ok=True)
    livres = [l for l in livres if l["id"] != livre_id]
    sauver_catalogue(livres)
    return True

# ── Commandes ────────────────────────────────────────────────────────────────

def charger_commandes() -> list[dict]:
    if FICHIER_COMMANDES.exists():
        with open(FICHIER_COMMANDES, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_commandes(commandes: list[dict]):
    with open(FICHIER_COMMANDES, "w", encoding="utf-8") as f:
        json.dump(commandes, f, ensure_ascii=False, indent=2)

def enregistrer_commande(session, livre: dict) -> dict:
    commandes = charger_commandes()
    commande = {
        "id":               str(uuid.uuid4())[:8],
        "stripe_session_id": session.id,
        "livre_id":         livre["id"],
        "titre":            livre["titre"],
        "prix_centimes":    livre["prix_centimes"],
        "devise":           livre["devise"],
        "email":            "",
        "statut":           "en_attente",
        "date":             datetime.now().isoformat(),
    }
    commandes.append(commande)
    sauver_commandes(commandes)
    return commande

def marquer_commande_payee(session_id: str, email: str) -> dict | None:
    commandes = charger_commandes()
    for c in commandes:
        if c["stripe_session_id"] == session_id:
            c["statut"] = "paye"
            c["email"] = email or c.get("email", "")
            c["date_paiement"] = datetime.now().isoformat()
            sauver_commandes(commandes)
            log.info(f"Éditorial — commande payée: {c['titre']} ({email})")
            return c
    log.warning(f"Éditorial — webhook pour session inconnue: {session_id}")
    return None

# ── Éditos (articles communautaires) ────────────────────────────────────────

def charger_editos() -> list[dict]:
    if FICHIER_EDITOS.exists():
        with open(FICHIER_EDITOS, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_editos(editos: list[dict]):
    with open(FICHIER_EDITOS, "w", encoding="utf-8") as f:
        json.dump(editos, f, ensure_ascii=False, indent=2)

def ajouter_edito(titre: str, auteur: str, contenu: str) -> dict:
    editos = charger_editos()
    edito = {
        "id":               str(uuid.uuid4())[:8],
        "titre":            titre,
        "auteur":           auteur or "Anonyme",
        "contenu":          contenu,
        "statut":           "en_attente",
        "date_soumission":  datetime.now().isoformat(),
        "date_publication": None,
    }
    editos.append(edito)
    sauver_editos(editos)
    return edito

def publier_edito(edito_id: str) -> bool:
    editos = charger_editos()
    for e in editos:
        if e["id"] == edito_id:
            e["statut"] = "publie"
            e["date_publication"] = datetime.now().isoformat()
            sauver_editos(editos)
            return True
    return False

def supprimer_edito(edito_id: str) -> bool:
    editos = charger_editos()
    if not any(e["id"] == edito_id for e in editos):
        return False
    editos = [e for e in editos if e["id"] != edito_id]
    sauver_editos(editos)
    return True
