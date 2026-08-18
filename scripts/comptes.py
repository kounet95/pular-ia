"""
comptes.py — Comptes utilisateurs du site (inscription email/mot de passe,
connexion via Telegram, sessions). Stockage JSON comme le reste du projet
(pas de base de données) — cohérent avec espace_editorial.py / duels.py.

Le bot Telegram identifie déjà ses utilisateurs par leur compte Telegram :
la connexion via Telegram ici sert surtout à donner au web une identité
équivalente sans mot de passe à retenir, en réutilisant le bot déjà en
place (même mécanique de lien d'invitation que les duels : /start <code>).

Mots de passe : PBKDF2-HMAC-SHA256 (stdlib `hashlib`, pas de dépendance
externe comme bcrypt/passlib — cohérent avec l'historique du projet où
chaque nouvelle dépendance doit être ajoutée explicitement au Dockerfile,
et une bibliothèque en moins qui peut manquer au déploiement).
"""

import hashlib
import hmac
import json
import logging
import re
import secrets
import uuid
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

PROJET_ROOT      = Path(__file__).resolve().parent.parent
DOSSIER_COMPTES  = PROJET_ROOT / "corpus-pular" / "comptes"
FICHIER_COMPTES  = DOSSIER_COMPTES / "comptes.json"
FICHIER_SESSIONS = DOSSIER_COMPTES / "sessions.json"
FICHIER_CODES_TG = DOSSIER_COMPTES / "codes_telegram.json"
DOSSIER_COMPTES.mkdir(parents=True, exist_ok=True)

DUREE_SESSION        = timedelta(days=30)
DUREE_CODE_TELEGRAM  = timedelta(minutes=10)
DUREE_RESET_MDP      = timedelta(minutes=30)

REGEX_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ── Stockage bas niveau ──────────────────────────────────────────────────

def _charger(fichier: Path) -> list[dict]:
    if fichier.exists():
        with open(fichier, encoding="utf-8") as f:
            return json.load(f)
    return []

def _sauver(fichier: Path, data: list[dict]):
    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def charger_comptes() -> list[dict]:
    return _charger(FICHIER_COMPTES)

def sauver_comptes(comptes: list[dict]):
    _sauver(FICHIER_COMPTES, comptes)

# ── Mots de passe ────────────────────────────────────────────────────────

def _hacher_mot_de_passe(mdp: str) -> str:
    sel = secrets.token_hex(16)
    empreinte = hashlib.pbkdf2_hmac("sha256", mdp.encode("utf-8"), bytes.fromhex(sel), 200_000).hex()
    return f"{sel}${empreinte}"

def _verifier_mot_de_passe(mdp: str, hash_stocke: str) -> bool:
    try:
        sel, empreinte = hash_stocke.split("$")
    except (ValueError, AttributeError):
        return False
    calcul = hashlib.pbkdf2_hmac("sha256", mdp.encode("utf-8"), bytes.fromhex(sel), 200_000).hex()
    return hmac.compare_digest(calcul, empreinte)

# ── Comptes ──────────────────────────────────────────────────────────────

def compte_public(c: dict) -> dict:
    """Vue exposable au client : jamais le hash de mot de passe."""
    return {
        "id":            c["id"],
        "pseudo":        c["pseudo"],
        "email":         c.get("email", ""),
        "telegram_lie":  bool(c.get("telegram_id")),
        "stripe_lie":    bool(c.get("stripe_account_id")),
        "stripe_actif":  bool(c.get("stripe_actif")),
        "date_creation": c.get("date_creation", ""),
    }

def compte_admin(c: dict) -> dict:
    """Vue pour le panneau admin : quelques détails de plus que compte_public
    (dernière connexion, username Telegram) — toujours sans le hash du mot
    de passe, aucune raison légitime pour l'admin de le voir non plus."""
    return {
        "id":                      c["id"],
        "pseudo":                  c["pseudo"],
        "email":                   c.get("email", ""),
        "telegram_lie":            bool(c.get("telegram_id")),
        "telegram_username":       c.get("telegram_username") or "",
        "stripe_lie":              bool(c.get("stripe_account_id")),
        "stripe_actif":            bool(c.get("stripe_actif")),
        "date_creation":           c.get("date_creation", ""),
        "date_derniere_connexion": c.get("date_derniere_connexion", ""),
    }

def creer_compte(pseudo: str, email: str, mot_de_passe: str) -> dict:
    pseudo = pseudo.strip()[:40]
    email = email.strip().lower()[:120]
    if len(pseudo) < 2:
        raise ValueError("Pseudo trop court (2 caractères minimum).")
    if not REGEX_EMAIL.match(email):
        raise ValueError("Adresse email invalide.")
    if len(mot_de_passe) < 6:
        raise ValueError("Mot de passe trop court (6 caractères minimum).")
    comptes = charger_comptes()
    if any(c.get("email") == email for c in comptes):
        raise ValueError("Un compte existe déjà avec cet email.")
    if any(c["pseudo"].lower() == pseudo.lower() for c in comptes):
        raise ValueError("Ce pseudo est déjà pris.")
    maintenant = datetime.now().isoformat()
    compte = {
        "id":                      str(uuid.uuid4())[:8],
        "pseudo":                  pseudo,
        "email":                   email,
        "mot_de_passe_hash":       _hacher_mot_de_passe(mot_de_passe),
        "telegram_id":             None,
        "telegram_username":       None,
        "stripe_account_id":       None,
        "stripe_actif":            False,
        "date_creation":           maintenant,
        "date_derniere_connexion": maintenant,
    }
    comptes.append(compte)
    sauver_comptes(comptes)
    return compte

def verifier_connexion(email: str, mot_de_passe: str) -> dict | None:
    email = email.strip().lower()
    comptes = charger_comptes()
    compte = next((c for c in comptes if c.get("email") == email), None)
    if not compte or not compte.get("mot_de_passe_hash"):
        return None
    if not _verifier_mot_de_passe(mot_de_passe, compte["mot_de_passe_hash"]):
        return None
    compte["date_derniere_connexion"] = datetime.now().isoformat()
    sauver_comptes(comptes)
    return compte

def compte_par_id(compte_id: str) -> dict | None:
    return next((c for c in charger_comptes() if c["id"] == compte_id), None)

def compte_par_telegram(telegram_id: int) -> dict | None:
    return next((c for c in charger_comptes() if c.get("telegram_id") == telegram_id), None)

def compte_par_email(email: str) -> dict | None:
    email = email.strip().lower()
    return next((c for c in charger_comptes() if c.get("email") == email), None)

def definir_stripe_account(compte_id: str, stripe_account_id: str) -> bool:
    """Associe un compte Stripe Connect (Express) au compte utilisateur —
    stripe_actif repart à False, à confirmer ensuite via l'API Stripe une
    fois l'auteur revenu de l'onboarding hébergé par Stripe."""
    comptes = charger_comptes()
    for c in comptes:
        if c["id"] == compte_id:
            c["stripe_account_id"] = stripe_account_id
            c["stripe_actif"] = False
            sauver_comptes(comptes)
            return True
    return False

def definir_stripe_actif(compte_id: str, actif: bool) -> bool:
    comptes = charger_comptes()
    for c in comptes:
        if c["id"] == compte_id:
            c["stripe_actif"] = actif
            sauver_comptes(comptes)
            return True
    return False

def creer_compte_telegram(pseudo: str, telegram_id: int, telegram_username: str | None) -> dict:
    """Crée un compte sans mot de passe pour un utilisateur qui se connecte
    directement via Telegram (pas encore de compte email lié)."""
    comptes = charger_comptes()
    pseudo_final = pseudo
    n = 1
    while any(c["pseudo"].lower() == pseudo_final.lower() for c in comptes):
        n += 1
        pseudo_final = f"{pseudo}{n}"
    maintenant = datetime.now().isoformat()
    compte = {
        "id":                      str(uuid.uuid4())[:8],
        "pseudo":                  pseudo_final,
        "email":                   "",
        "mot_de_passe_hash":       "",
        "telegram_id":             telegram_id,
        "telegram_username":       telegram_username,
        "stripe_account_id":       None,
        "stripe_actif":            False,
        "date_creation":           maintenant,
        "date_derniere_connexion": maintenant,
    }
    comptes.append(compte)
    sauver_comptes(comptes)
    return compte

def lier_telegram(compte_id: str, telegram_id: int, telegram_username: str | None) -> bool:
    """Lie un compte Telegram à un compte web déjà connecté."""
    if compte_par_telegram(telegram_id):
        raise ValueError("Ce compte Telegram est déjà lié à un autre compte.")
    comptes = charger_comptes()
    for c in comptes:
        if c["id"] == compte_id:
            c["telegram_id"] = telegram_id
            c["telegram_username"] = telegram_username
            sauver_comptes(comptes)
            return True
    return False

# ── Sessions ─────────────────────────────────────────────────────────────
# Le cookie ne porte qu'un jeton aléatoire opaque ; le serveur garde la
# correspondance jeton (haché) → compte, ce qui permet de révoquer une
# session à tout moment (déconnexion) sans dépendre d'un secret de signature.

def creer_session(compte_id: str) -> str:
    token = secrets.token_urlsafe(32)
    maintenant = datetime.now().isoformat()
    sessions = [s for s in _charger(FICHIER_SESSIONS) if s["expire_le"] > maintenant]
    sessions.append({
        "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "compte_id":  compte_id,
        "expire_le":  (datetime.now() + DUREE_SESSION).isoformat(),
    })
    _sauver(FICHIER_SESSIONS, sessions)
    return token

def compte_depuis_session(token: str) -> dict | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    maintenant = datetime.now().isoformat()
    session = next(
        (s for s in _charger(FICHIER_SESSIONS) if s["token_hash"] == token_hash and s["expire_le"] > maintenant),
        None,
    )
    if not session:
        return None
    return compte_par_id(session["compte_id"])

def supprimer_session(token: str):
    if not token:
        return
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    sessions = [s for s in _charger(FICHIER_SESSIONS) if s["token_hash"] != token_hash]
    _sauver(FICHIER_SESSIONS, sessions)

def revoquer_sessions_compte(compte_id: str) -> int:
    """Supprime toutes les sessions actives d'un compte (déconnexion forcée
    partout, ex: en cas de compte suspecté compromis). Retourne le nombre de
    sessions révoquées."""
    sessions = _charger(FICHIER_SESSIONS)
    avant = len(sessions)
    sessions = [s for s in sessions if s["compte_id"] != compte_id]
    _sauver(FICHIER_SESSIONS, sessions)
    return avant - len(sessions)

# ── Mot de passe oublié ──────────────────────────────────────────────────
# Jeton stocké haché sur le compte lui-même (comme les sessions) — un seul
# jeton actif à la fois par compte, le précédent devient invalide dès qu'un
# nouveau est demandé.

def generer_token_reset(email: str) -> tuple[dict, str] | None:
    """Génère un jeton de réinitialisation pour le compte associé à cet
    email. None si aucun compte avec un mot de passe pour cet email
    (compte inexistant, ou compte 100% Telegram sans mot de passe) —
    l'appelant ne doit jamais révéler cette distinction au client, pour ne
    pas laisser deviner quels emails sont enregistrés."""
    compte = compte_par_email(email)
    if not compte or not compte.get("mot_de_passe_hash"):
        return None
    jeton = secrets.token_urlsafe(32)
    comptes = charger_comptes()
    for c in comptes:
        if c["id"] == compte["id"]:
            c["reset_token_hash"]   = hashlib.sha256(jeton.encode()).hexdigest()
            c["reset_token_expire"] = (datetime.now() + DUREE_RESET_MDP).isoformat()
            sauver_comptes(comptes)
            compte = c
            break
    return compte, jeton

def compte_par_reset_token(jeton: str) -> dict | None:
    token_hash = hashlib.sha256(jeton.encode()).hexdigest()
    maintenant = datetime.now().isoformat()
    return next(
        (c for c in charger_comptes()
         if c.get("reset_token_hash") == token_hash and c.get("reset_token_expire", "") > maintenant),
        None,
    )

def reinitialiser_mot_de_passe(jeton: str, nouveau_mdp: str) -> bool:
    """Change le mot de passe si le jeton est valide, le consomme (usage
    unique), et déconnecte le compte de toutes ses sessions actives —
    précaution standard après un changement de mot de passe."""
    if len(nouveau_mdp) < 6:
        raise ValueError("Mot de passe trop court (6 caractères minimum).")
    compte = compte_par_reset_token(jeton)
    if not compte:
        return False
    comptes = charger_comptes()
    for c in comptes:
        if c["id"] == compte["id"]:
            c["mot_de_passe_hash"] = _hacher_mot_de_passe(nouveau_mdp)
            c.pop("reset_token_hash", None)
            c.pop("reset_token_expire", None)
            sauver_comptes(comptes)
            break
    revoquer_sessions_compte(compte["id"])
    return True

def supprimer_compte(compte_id: str) -> bool:
    """Supprime un compte (admin) et révoque toutes ses sessions. Le
    contenu déjà publié par ce compte (éditos, commentaires, duels...)
    n'est pas supprimé — son `compte_id` devient simplement orphelin,
    comme pour tout contenu créé anonymement."""
    comptes = charger_comptes()
    apres = [c for c in comptes if c["id"] != compte_id]
    if len(apres) == len(comptes):
        return False
    sauver_comptes(apres)
    revoquer_sessions_compte(compte_id)
    return True

# ── Connexion via Telegram (réutilise le bot déjà en place) ─────────────
# Flux : le web génère un code court + lien https://t.me/<bot>?start=connexion_<code> ;
# l'utilisateur l'ouvre, le bot confirme le code ; le web (qui interroge le
# statut en boucle courte) détecte la confirmation et connecte/crée le compte.

def generer_code_telegram() -> str:
    maintenant = datetime.now().isoformat()
    codes = [c for c in _charger(FICHIER_CODES_TG) if c["expire_le"] > maintenant]
    code = secrets.token_hex(4).upper()
    codes.append({
        "code":              code,
        "statut":            "attente",
        "telegram_id":       None,
        "telegram_username": None,
        "telegram_prenom":   None,
        "expire_le":         (datetime.now() + DUREE_CODE_TELEGRAM).isoformat(),
    })
    _sauver(FICHIER_CODES_TG, codes)
    return code

def confirmer_code_telegram(code: str, telegram_id: int, telegram_username: str | None, telegram_prenom: str | None) -> bool:
    """Appelé par le bot quand l'utilisateur envoie /start connexion_<code>."""
    maintenant = datetime.now().isoformat()
    codes = _charger(FICHIER_CODES_TG)
    trouve = next((c for c in codes if c["code"] == code and c["expire_le"] > maintenant), None)
    if not trouve:
        return False
    trouve["statut"] = "confirme"
    trouve["telegram_id"] = telegram_id
    trouve["telegram_username"] = telegram_username
    trouve["telegram_prenom"] = telegram_prenom
    _sauver(FICHIER_CODES_TG, codes)
    return True

def resultat_code_telegram(code: str) -> dict | None:
    """Consulté par le web (polling). Ne décide pas seul de créer/lier un
    compte — c'est l'appelant qui sait s'il y a déjà une session active."""
    codes = _charger(FICHIER_CODES_TG)
    trouve = next((c for c in codes if c["code"] == code), None)
    if not trouve:
        return None
    if trouve["statut"] != "confirme":
        return {"statut": trouve["statut"]}
    return {
        "statut":            "confirme",
        "telegram_id":       trouve["telegram_id"],
        "telegram_username": trouve.get("telegram_username"),
        "telegram_prenom":   trouve.get("telegram_prenom"),
    }

def consommer_code_telegram(code: str):
    """Usage unique : à appeler une fois le code exploité par le web."""
    codes = [c for c in _charger(FICHIER_CODES_TG) if c["code"] != code]
    _sauver(FICHIER_CODES_TG, codes)
