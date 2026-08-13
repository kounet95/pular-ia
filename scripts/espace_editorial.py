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
DOSSIER_EDITOS_IMAGES = DOSSIER_EDITORIAL / "editos_images"
DOSSIER_EDITOS_SOURCES = DOSSIER_EDITORIAL / "editos_sources"
FICHIER_CATALOGUE = DOSSIER_EDITORIAL / "catalogue.json"
FICHIER_COMMANDES = DOSSIER_EDITORIAL / "commandes.json"
FICHIER_EDITOS    = DOSSIER_EDITORIAL / "editos.json"

for d in [DOSSIER_EDITORIAL, DOSSIER_COUVERTURES, DOSSIER_EDITOS_IMAGES, DOSSIER_EDITOS_SOURCES]:
    d.mkdir(parents=True, exist_ok=True)

DEVISES_ACCEPTEES = ["eur", "usd", "gnf", "xof"]
EXTENSIONS_COUVERTURE = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSIONS_SOURCE = {".pdf", ".txt", ".docx", ".doc", ".html", ".htm", ".md"}

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

def creer_session_paiement(livre: dict, origin: str, email: str = "") -> "stripe.checkout.Session":
    """Crée une session de paiement Stripe Checkout pour un livre du catalogue."""
    stripe = stripe_configure()

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

def ajouter_edito(titre: str, auteur: str, contenu: str, image: str) -> dict:
    """Publie l'édito immédiatement — pas de file d'attente de relecture."""
    maintenant = datetime.now().isoformat()
    editos = charger_editos()
    edito = {
        "id":               str(uuid.uuid4())[:8],
        "titre":            titre,
        "auteur":           auteur or "Anonyme",
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

def ajouter_commentaire(edito_id: str, auteur: str, texte: str) -> dict | None:
    """Ajoute un commentaire, publié immédiatement (même logique que les
    éditos eux-mêmes : pas de file de modération a priori)."""
    editos = charger_editos()
    for e in editos:
        if e["id"] == edito_id:
            commentaire = {
                "id":     str(uuid.uuid4())[:8],
                "auteur": (auteur or "Anonyme").strip()[:60] or "Anonyme",
                "texte":  texte.strip()[:1000],
                "date":   datetime.now().isoformat(),
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
