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
import secrets
import uuid
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

PROJET_ROOT       = Path(__file__).resolve().parent.parent
DOSSIER_EDITORIAL = PROJET_ROOT / "corpus-pular" / "editorial"
DOSSIER_COUVERTURES = DOSSIER_EDITORIAL / "couvertures"
DOSSIER_EDITOS_IMAGES = DOSSIER_EDITORIAL / "editos_images"
DOSSIER_EDITOS_SOURCES = DOSSIER_EDITORIAL / "editos_sources"
FICHIER_CATALOGUE = DOSSIER_EDITORIAL / "catalogue.json"
FICHIER_COMMANDES = DOSSIER_EDITORIAL / "commandes.json"
FICHIER_EDITOS    = DOSSIER_EDITORIAL / "editos.json"
FICHIER_DONS      = DOSSIER_EDITORIAL / "dons.json"

for d in [DOSSIER_EDITORIAL, DOSSIER_COUVERTURES, DOSSIER_EDITOS_IMAGES, DOSSIER_EDITOS_SOURCES]:
    d.mkdir(parents=True, exist_ok=True)

DEVISES_ACCEPTEES = ["eur", "usd", "gnf", "xof"]
EXTENSIONS_COUVERTURE = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSIONS_SOURCE = {".pdf", ".txt", ".docx", ".doc", ".html", ".htm", ".md"}
EXTENSIONS_FICHIER_NUMERIQUE = {".pdf", ".epub"}

DOSSIER_LIVRES_FICHIERS = DOSSIER_EDITORIAL / "livres_fichiers"
FICHIER_ZONES_LIVRAISON = DOSSIER_EDITORIAL / "zones_livraison.json"
DOSSIER_LIVRES_FICHIERS.mkdir(parents=True, exist_ok=True)

FORMATS_LIVRE = ["numerique", "papier"]

# Répartition affichée aux acheteurs (mention légale/transparence) — ceci ne
# déclenche AUCUN versement automatique séparé : Stripe encaisse la totalité
# sur le compte du projet, la répartition réelle vers les auteurs se fait
# manuellement pour l'instant (un vrai partage automatique demanderait Stripe
# Connect + un compte par auteur, hors périmètre ici).
REPARTITION_REVENUS = {"auteur": 60, "plateforme": 20, "projet": 20}

# Montants de don suggérés par devise (unité affichée, pas centimes) —
# purement indicatif côté UI, un montant libre reste toujours possible.
MONTANTS_DON_SUGGERES = {
    "gnf": [10000, 25000, 50000, 100000],
    "xof": [1000, 2500, 5000, 10000],
    "eur": [5, 10, 25, 50],
    "usd": [5, 10, 25, 50],
}

# Devises "zéro décimale" chez Stripe : `unit_amount` est le montant entier
# tel quel (pas de centimes) — le GNF (Guinée) et le XOF (Franc CFA — BCEAO,
# zone ouest-africaine) en font partie.
DEVISES_ZERO_DECIMALE = {
    "bif", "clp", "djf", "gnf", "jpy", "kmf", "krw", "mga", "pyg",
    "rwf", "ugx", "vnd", "vuv", "xaf", "xof", "xpf",
}

def calculer_montant_stripe(prix: float, devise: str) -> int:
    """Convertit un prix affiché (ex: 15.5) vers l'unité attendue par Stripe."""
    if devise in DEVISES_ZERO_DECIMALE:
        return round(prix)
    return round(prix * 100)

# ── Stripe ───────────────────────────────────────────────────────────────────

def stripe_configure():
    """
    Configure la clé Stripe. Retourne le module stripe prêt à l'emploi, ou
    lève RuntimeError avec un message qui distingue clé manquante et paquet
    non installé — les deux se traduisaient avant en la même erreur, ce qui
    rendait le diagnostic impossible à distance (clé ajoutée mais Dockerfile
    n'installant pas réellement le paquet, par exemple).
    """
    cle = os.getenv("STRIPE_SECRET_KEY", "")
    if not cle:
        raise RuntimeError("Paiement non configuré (STRIPE_SECRET_KEY manquante).")
    try:
        import stripe
    except ImportError:
        raise RuntimeError("Paiement non disponible sur le serveur (paquet 'stripe' non installé).")
    stripe.api_key = cle
    return stripe

def prix_format(livre: dict, format_achete: str) -> int | None:
    """Prix (centimes) du livre pour ce format, ou None si le format n'est
    pas proposé pour ce livre."""
    cle = "prix_numerique_centimes" if format_achete == "numerique" else "prix_papier_centimes"
    return livre.get(cle)

def creer_session_paiement(
    livre: dict, format_achete: str, zone: dict | None, origin: str, email: str = "",
    stripe_account_id: str | None = None,
) -> "stripe.checkout.Session":
    """
    Crée une session de paiement Stripe Checkout pour un format précis d'un
    livre du catalogue (numérique ou papier). `zone` (uniquement pertinent
    pour le papier) ajoute une ligne de frais de livraison séparée — la
    livraison reste possible sans zone définie (retrait / à organiser),
    auquel cas aucun frais n'est ajouté.

    `stripe_account_id` : si l'auteur a un compte Stripe Connect actif, la
    part qui lui revient (REPARTITION_REVENUS["auteur"]) est transférée
    automatiquement vers son compte via une "destination charge" — calculée
    uniquement sur le prix du livre, jamais sur les frais de livraison (qui
    couvrent un coût réel, pas un revenu à partager). Sans compte connecté
    (ou compte pas encore actif), la totalité reste sur le compte de la
    plateforme, comme avant — la répartition se fait alors manuellement.
    """
    stripe = stripe_configure()
    if format_achete not in FORMATS_LIVRE:
        raise ValueError("Format de livre invalide.")
    prix = prix_format(livre, format_achete)
    if prix is None:
        raise ValueError("Ce format n'est pas disponible pour ce livre.")

    origin = origin.rstrip("/")
    nom_format = "Numérique" if format_achete == "numerique" else "Papier"
    line_items = [{
        "price_data": {
            "currency": livre["devise"],
            "unit_amount": prix,
            "product_data": {
                "name": f"{livre['titre']} ({nom_format})",
                "description": livre.get("description", "")[:500] or None,
            },
        },
        "quantity": 1,
    }]
    if format_achete == "papier" and zone:
        line_items.append({
            "price_data": {
                "currency": livre["devise"],
                "unit_amount": zone["frais_centimes"],
                "product_data": {"name": f"Livraison — {zone['nom']}"},
            },
            "quantity": 1,
        })

    params = dict(
        mode="payment",
        line_items=line_items,
        success_url=f"{origin}/livre-achete?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{origin}/?edito_paiement=annule",
        metadata={
            "livre_id": livre["id"],
            "format": format_achete,
            "zone_id": zone["id"] if zone else "",
        },
    )
    if email:
        params["customer_email"] = email
    if stripe_account_id:
        part_auteur = round(prix * REPARTITION_REVENUS["auteur"] / 100)
        params["payment_intent_data"] = {
            "transfer_data": {"destination": stripe_account_id, "amount": part_auteur},
        }
    return stripe.checkout.Session.create(**params)

# ── Stripe Connect (comptes vendeurs) ───────────────────────────────────────
# Chaque auteur/vendeur connecte son propre compte Stripe Express : Stripe
# héberge entièrement l'inscription (identité, coordonnées bancaires — KYC),
# ce projet n'a jamais accès à ces données. Une fois le compte actif
# (charges_enabled), les achats liés à cet auteur sont automatiquement
# partagés via creer_session_paiement ci-dessus.

def creer_compte_stripe_connecte(email: str = "") -> str:
    """Crée un compte Stripe Express vide, à faire finaliser par l'auteur via
    un lien d'onboarding (creer_lien_onboarding_stripe). Retourne l'id du
    compte Stripe (à sauvegarder sur le compte utilisateur)."""
    stripe = stripe_configure()
    params = {"type": "express"}
    if email:
        params["email"] = email
    compte = stripe.Account.create(**params)
    return compte.id

def creer_lien_onboarding_stripe(stripe_account_id: str, return_url: str, refresh_url: str) -> str:
    """Lien hébergé par Stripe où l'auteur complète (ou reprend) son
    inscription. À usage unique et de courte durée — en générer un nouveau
    à chaque fois plutôt que de le garder en cache."""
    stripe = stripe_configure()
    lien = stripe.AccountLink.create(
        account=stripe_account_id,
        return_url=return_url,
        refresh_url=refresh_url,
        type="account_onboarding",
    )
    return lien.url

def verifier_compte_stripe_actif(stripe_account_id: str) -> bool:
    """True si l'auteur a fini son inscription et peut recevoir des
    paiements (Stripe a validé son identité/ses coordonnées bancaires)."""
    stripe = stripe_configure()
    compte = stripe.Account.retrieve(stripe_account_id)
    return bool(compte.charges_enabled)

def verifier_signature_webhook(payload: bytes, sig_header: str):
    """Vérifie la signature d'un événement webhook Stripe. Lève une exception si invalide."""
    stripe = stripe_configure()
    # Nom dédié pour ne pas entrer en conflit avec un STRIPE_WEBHOOK_SECRET
    # utilisé par un autre projet partageant le même compte Stripe.
    secret = os.getenv("STRIPE_WEBHOOK_SECRET_EDITORIAL", "")
    if not secret:
        raise RuntimeError("STRIPE_WEBHOOK_SECRET_EDITORIAL manquant — impossible de vérifier le webhook.")
    return stripe.Webhook.construct_event(payload, sig_header, secret)

# ── Assistant IA (Claude) — brouillon d'édito ───────────────────────────────

def anthropic_configure():
    """Configure le client Anthropic. Retourne le client, ou lève RuntimeError
    avec un message qui distingue clé manquante et paquet non installé."""
    cle = os.getenv("ANTHROPIC_API_KEY", "")
    if not cle:
        raise RuntimeError("Assistant IA non configuré (ANTHROPIC_API_KEY manquante).")
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Assistant IA non disponible sur le serveur (paquet 'anthropic' non installé).")
    return anthropic.Anthropic(api_key=cle)

def extraire_extrait_source(chemin: Path, max_chars: int = 3000) -> str:
    """
    Extrait le texte d'un document source déposé par l'auteur (réutilise
    l'extraction du RAG livres), tronqué pour tenir dans le contexte du
    brouillon généré.
    """
    from rag_livres import extraire_texte
    texte = extraire_texte(chemin).strip()
    if len(texte) > max_chars:
        texte = texte[:max_chars] + "…"
    return texte

def rechercher_sources_existantes(sujet: str, n: int = 3) -> list[dict]:
    """
    Réutilise les corpus RAG déjà indexés (Livres + Histoire) pour trouver
    des passages pertinents au sujet — des suggestions de sources déjà
    présentes dans le corpus communautaire, jamais de contenu privé (le RAG
    Histoire exclut déjà les documents marqués 'prive').
    """
    resultats = []
    try:
        from rag_livres import rechercher as rag_rechercher
        for r in rag_rechercher(sujet, n):
            resultats.append({
                "origine": "livres",
                "titre":   r.get("titre", "?"),
                "texte":   r.get("texte", ""),
            })
    except Exception as e:
        log.warning(f"Recherche corpus livres pour édito: {e}")
    try:
        import espace_histoire as EH
        for r in EH.rechercher(sujet, n):
            resultats.append({
                "origine": "histoire",
                "titre":   r.get("titre", "?"),
                "texte":   r.get("texte", ""),
            })
    except Exception as e:
        log.warning(f"Recherche corpus histoire pour édito: {e}")
    return resultats

def generer_brouillon_edito(sujet: str, angle: str = "", sources: list[dict] | None = None) -> str:
    """
    Génère un premier brouillon d'édito avec Claude, pour aider les éditeurs
    communautaires à démarrer vite. Le texte reste à relire et vérifier par
    l'auteur avant soumission — ce n'est jamais publié tel quel.

    `sources` (optionnel) : passages issus de documents déposés par l'auteur
    et/ou trouvés dans les corpus RAG Livres/Histoire — chaque entrée porte
    `origine`, `titre`, `texte`. Quand fourni, le brouillon doit s'appuyer
    dessus plutôt que d'inventer.
    """
    client = anthropic_configure()

    consigne = f"Sujet: {sujet}"
    if angle.strip():
        consigne += f"\nAngle / précisions données par l'auteur: {angle.strip()}"

    if sources:
        consigne += "\n\nSOURCES fournies (documents déposés par l'auteur ou passages du corpus communautaire) :"
        for i, s in enumerate(sources, 1):
            consigne += f"\n\n[Source {i} — {s['origine']} — {s['titre']}]\n{s['texte']}"

    system = (
        "Tu es un assistant d'écriture pour l'espace éditorial d'un site communautaire "
        "consacré à la langue, la culture et l'histoire peules (Fouta Djallon, Macina, "
        "Fouta Toro et les autres foyers peuls). Rédige un premier brouillon d'édito en "
        "français, clair et engageant, d'environ 300 à 500 mots. Commence par une première "
        "ligne au format exact 'TITRE: <titre proposé>', puis une ligne vide, puis le corps "
        "du texte. Ce brouillon est un point de départ que l'auteur relira et adaptera avant "
        "publication — ce n'est pas un texte définitif."
    )
    if sources:
        system += (
            " Des SOURCES sont fournies dans le message : appuie-toi dessus en priorité pour "
            "les faits, dates, noms et détails concrets — reformule avec tes propres mots plutôt "
            "que de copier de longs passages tels quels, et indique entre crochets [Source N] "
            "d'où vient chaque information factuelle reprise. Pour tout ce qui n'est pas couvert "
            "par les sources, reste général et n'invente pas de détails précis non vérifiables."
        )
    else:
        system += (
            " N'invente pas de faits historiques précis (dates, noms, chiffres) dont tu n'es "
            "pas certain ; reste général sur ces points ou indique explicitement entre crochets "
            "ce qui doit être vérifié par l'auteur."
        )

    message = client.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": consigne}],
    )
    return "".join(b.text for b in message.content if b.type == "text")

# ── Assistant IA (OpenAI) — image d'illustration ────────────────────────────
# Claude ne génère pas d'images (texte/vision uniquement) : on réutilise la
# clé OpenAI déjà présente dans l'environnement pour ça, indépendamment de
# l'assistant de rédaction ci-dessus qui reste sur Claude.

def openai_configure():
    """Configure le client OpenAI. Retourne le client, ou lève RuntimeError
    avec un message qui distingue clé manquante et paquet non installé."""
    cle = os.getenv("OPENAI_API_KEY", "")
    if not cle:
        raise RuntimeError("Génération d'image non configurée (OPENAI_API_KEY manquante).")
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Génération d'image non disponible sur le serveur (paquet 'openai' non installé).")
    return OpenAI(api_key=cle)

def generer_image_edito(sujet: str) -> bytes:
    """Génère une image d'illustration pour un édito avec gpt-image-1. Retourne les octets PNG."""
    client = openai_configure()

    prompt = (
        f"Illustration éditoriale sobre et évocatrice pour un article sur : {sujet}. "
        "Style photographique ou peinture digitale, ambiance culturelle ouest-africaine / "
        "peule (Fouta Djallon, Sahel), sans texte ni typographie incrustée dans l'image."
    )
    # gpt-image-1 (pas dall-e-3 : indisponible sur certains projets/clés OpenAI
    # restreints par modèle) renvoie directement du b64_json, sans response_format.
    reponse = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
        n=1,
    )
    import base64
    return base64.b64decode(reponse.data[0].b64_json)

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
    auteur: str,
    description: str,
    devise: str,
    couverture_nom: str = "",
    prix_numerique_centimes: int | None = None,
    prix_papier_centimes: int | None = None,
    fichier_numerique_nom: str = "",
    auteur_compte_id: str | None = None,
) -> dict:
    """
    Un livre peut être proposé en numérique, en papier, ou les deux — chacun
    avec son propre prix (`None` = format non proposé pour ce livre).
    `fichier_numerique_nom` : fichier PDF/EPUB livré automatiquement après
    paiement pour le format numérique.
    `auteur_compte_id` : compte utilisateur (comptes.py) de l'auteur, si
    connu — sert à retrouver son compte Stripe Connect au moment du
    paiement pour partager automatiquement les revenus. Optionnel : sans
    ça, la vente fonctionne pareil, juste sans partage automatique.
    """
    if prix_numerique_centimes is None and prix_papier_centimes is None:
        raise ValueError("Choisis au moins un format (numérique ou papier) avec un prix.")
    if prix_numerique_centimes is not None and not fichier_numerique_nom:
        raise ValueError("Un fichier (PDF/EPUB) est requis pour proposer le format numérique.")
    livres = charger_catalogue()
    fiche = {
        "id":                      str(uuid.uuid4())[:8],
        "titre":                   titre,
        "auteur":                  auteur or "Auteur inconnu",
        "auteur_compte_id":        auteur_compte_id,
        "description":             description,
        "devise":                  devise,
        "couverture":              couverture_nom,
        "prix_numerique_centimes": prix_numerique_centimes,
        "prix_papier_centimes":    prix_papier_centimes,
        "fichier_numerique":       fichier_numerique_nom,
        "date":                    datetime.now().isoformat(),
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
    if trouve.get("fichier_numerique"):
        (DOSSIER_LIVRES_FICHIERS / trouve["fichier_numerique"]).unlink(missing_ok=True)
    livres = [l for l in livres if l["id"] != livre_id]
    sauver_catalogue(livres)
    return True

# ── Zones de livraison (papier) ─────────────────────────────────────────────
# Liste générique et modifiable par l'admin — la livraison papier reste
# disponible même sans zone définie (retrait / organisation manuelle),
# simplement sans frais de livraison ajoutés automatiquement.

def charger_zones_livraison() -> list[dict]:
    if FICHIER_ZONES_LIVRAISON.exists():
        with open(FICHIER_ZONES_LIVRAISON, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_zones_livraison(zones: list[dict]):
    with open(FICHIER_ZONES_LIVRAISON, "w", encoding="utf-8") as f:
        json.dump(zones, f, ensure_ascii=False, indent=2)

def ajouter_zone_livraison(nom: str, frais_centimes: int, devise: str) -> dict:
    zones = charger_zones_livraison()
    zone = {
        "id":             str(uuid.uuid4())[:8],
        "nom":            nom.strip()[:80],
        "frais_centimes": frais_centimes,
        "devise":         devise,
    }
    zones.append(zone)
    sauver_zones_livraison(zones)
    return zone

def supprimer_zone_livraison(zone_id: str) -> bool:
    zones = charger_zones_livraison()
    apres = [z for z in zones if z["id"] != zone_id]
    if len(apres) == len(zones):
        return False
    sauver_zones_livraison(apres)
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

def enregistrer_commande(
    session, livre: dict, format_achete: str, zone: dict | None = None, telegram_id: int | None = None,
) -> dict:
    commandes = charger_commandes()
    commande = {
        "id":                       str(uuid.uuid4())[:8],
        "jeton":                    secrets.token_urlsafe(16),
        "stripe_session_id":        session.id,
        "livre_id":                 livre["id"],
        "titre":                    livre["titre"],
        "auteur":                   livre.get("auteur", ""),
        "format":                   format_achete,
        "zone":                     zone,
        "prix_centimes":            prix_format(livre, format_achete) or 0,
        "frais_livraison_centimes": zone["frais_centimes"] if zone else 0,
        "devise":                   livre["devise"],
        "fichier_numerique":        livre.get("fichier_numerique", "") if format_achete == "numerique" else "",
        "email":                    "",
        "telegram_id":              telegram_id,
        "statut":                   "en_attente",
        "date":                     datetime.now().isoformat(),
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
            log.info(f"Éditorial — commande payée: {c['titre']} ({c['format']}) — {email or c.get('telegram_id', '')}")
            return c
    log.warning(f"Éditorial — webhook pour session inconnue: {session_id}")
    return None

def obtenir_commande(commande_id: str) -> dict | None:
    return next((c for c in charger_commandes() if c["id"] == commande_id), None)

def obtenir_commande_par_session(session_id: str) -> dict | None:
    return next((c for c in charger_commandes() if c["stripe_session_id"] == session_id), None)

# ── Dons ─────────────────────────────────────────────────────────────────
# Contrairement aux livres, un don n'a pas d'auteur à rémunérer : 100% va
# directement financer le projet (voir REPARTITION_REVENUS pour les livres,
# qui ne s'applique pas ici).

def creer_session_don(
    montant_centimes: int, devise: str, origin: str, email: str = "", nom_donateur: str = "",
) -> "stripe.checkout.Session":
    stripe = stripe_configure()
    if montant_centimes <= 0:
        raise ValueError("Le montant du don doit être positif.")
    origin = origin.rstrip("/")
    params = dict(
        mode="payment",
        line_items=[{
            "price_data": {
                "currency": devise,
                "unit_amount": montant_centimes,
                "product_data": {
                    "name": "Don au projet Pular IA",
                    "description": "Soutien à la préservation de la langue et de la culture peules.",
                },
            },
            "quantity": 1,
        }],
        success_url=f"{origin}/don-merci?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{origin}/?edito_paiement=annule",
        metadata={"type": "don", "nom_donateur": (nom_donateur or "Anonyme").strip()[:80]},
    )
    if email:
        params["customer_email"] = email
    return stripe.checkout.Session.create(**params)

def charger_dons() -> list[dict]:
    if FICHIER_DONS.exists():
        with open(FICHIER_DONS, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_dons(dons: list[dict]):
    with open(FICHIER_DONS, "w", encoding="utf-8") as f:
        json.dump(dons, f, ensure_ascii=False, indent=2)

def enregistrer_don(
    session, montant_centimes: int, devise: str, nom_donateur: str = "", telegram_id: int | None = None,
) -> dict:
    dons = charger_dons()
    don = {
        "id":                 str(uuid.uuid4())[:8],
        "jeton":              secrets.token_urlsafe(16),
        "stripe_session_id":  session.id,
        "montant_centimes":   montant_centimes,
        "devise":             devise,
        "nom_donateur":       (nom_donateur or "Anonyme").strip()[:80] or "Anonyme",
        "email":              "",
        "telegram_id":        telegram_id,
        "statut":             "en_attente",
        "date":               datetime.now().isoformat(),
    }
    dons.append(don)
    sauver_dons(dons)
    return don

def marquer_don_paye(session_id: str, email: str) -> dict | None:
    dons = charger_dons()
    for d in dons:
        if d["stripe_session_id"] == session_id:
            d["statut"] = "paye"
            d["email"] = email or d.get("email", "")
            d["date_paiement"] = datetime.now().isoformat()
            sauver_dons(dons)
            log.info(f"Don payé: {d['montant_centimes']} {d['devise']} ({d['nom_donateur']})")
            return d
    log.warning(f"Éditorial — webhook don pour session inconnue: {session_id}")
    return None

def obtenir_don(don_id: str) -> dict | None:
    return next((d for d in charger_dons() if d["id"] == don_id), None)

def obtenir_don_par_session(session_id: str) -> dict | None:
    return next((d for d in charger_dons() if d["stripe_session_id"] == session_id), None)

def total_dons_payes() -> dict:
    """Total des dons payés, groupé par devise (pas de conversion inter-devises)."""
    totaux: dict[str, int] = {}
    for d in charger_dons():
        if d["statut"] == "paye":
            totaux[d["devise"]] = totaux.get(d["devise"], 0) + d["montant_centimes"]
    return totaux

# ── Éditos (articles communautaires) ────────────────────────────────────────

def charger_editos() -> list[dict]:
    if FICHIER_EDITOS.exists():
        with open(FICHIER_EDITOS, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_editos(editos: list[dict]):
    with open(FICHIER_EDITOS, "w", encoding="utf-8") as f:
        json.dump(editos, f, ensure_ascii=False, indent=2)

def ajouter_edito(titre: str, auteur: str, contenu: str, image: str, compte_id: str | None = None) -> dict:
    """Publie l'édito immédiatement — pas de file d'attente de relecture.
    `compte_id` : rempli si l'auteur est connecté (voir comptes.py), sinon
    None pour une soumission anonyme (pseudo libre)."""
    maintenant = datetime.now().isoformat()
    editos = charger_editos()
    edito = {
        "id":               str(uuid.uuid4())[:8],
        "titre":            titre,
        "auteur":           auteur or "Anonyme",
        "compte_id":        compte_id,
        "contenu":          contenu,
        "image":            image,
        "statut":           "publie",
        "date_soumission":  maintenant,
        "date_publication": maintenant,
        "likes":            [],
        "commentaires":     [],
    }
    editos.append(edito)
    sauver_editos(editos)
    return edito

def basculer_like(edito_id: str, visiteur_id: str) -> dict | None:
    """
    Ajoute ou retire le like d'un visiteur. Pas de compte utilisateur sur ce
    site : `visiteur_id` est un identifiant anonyme généré côté client
    (localStorage) — assez pour éviter les doublons/spam de clics, pas pensé
    pour résister à un utilisateur qui vide son navigateur exprès.
    Retourne {likes, aime} ou None si l'édito n'existe pas.
    """
    editos = charger_editos()
    for e in editos:
        if e["id"] == edito_id:
            likes = e.setdefault("likes", [])
            if visiteur_id in likes:
                likes.remove(visiteur_id)
                aime = False
            else:
                likes.append(visiteur_id)
                aime = True
            sauver_editos(editos)
            return {"likes": len(likes), "aime": aime}
    return None

def ajouter_commentaire(edito_id: str, auteur: str, texte: str, compte_id: str | None = None) -> dict | None:
    """Ajoute un commentaire, publié immédiatement (même logique que les
    éditos eux-mêmes : pas de file de modération a priori)."""
    editos = charger_editos()
    for e in editos:
        if e["id"] == edito_id:
            commentaire = {
                "id":        str(uuid.uuid4())[:8],
                "auteur":    (auteur or "Anonyme").strip()[:60] or "Anonyme",
                "compte_id": compte_id,
                "texte":     texte.strip()[:1000],
                "date":      datetime.now().isoformat(),
            }
            e.setdefault("commentaires", []).append(commentaire)
            sauver_editos(editos)
            return commentaire
    return None

def supprimer_commentaire(edito_id: str, commentaire_id: str) -> bool:
    """Modération a posteriori (admin) d'un commentaire abusif."""
    editos = charger_editos()
    for e in editos:
        if e["id"] == edito_id:
            avant = e.get("commentaires", [])
            apres = [c for c in avant if c["id"] != commentaire_id]
            if len(apres) == len(avant):
                return False
            e["commentaires"] = apres
            sauver_editos(editos)
            return True
    return False

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
    trouve = next((e for e in editos if e["id"] == edito_id), None)
    if not trouve:
        return False
    if trouve.get("image"):
        (DOSSIER_EDITOS_IMAGES / trouve["image"]).unlink(missing_ok=True)
    editos = [e for e in editos if e["id"] != edito_id]
    sauver_editos(editos)
    return True
