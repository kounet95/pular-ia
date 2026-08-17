"""
quizlive.py — Quiz en direct multijoueur façon Kahoot, pensé pour être animé
pendant un live (TikTok, etc.) : l'animateur partage son écran avec un code
PIN, les spectateurs rejoignent depuis leur téléphone et répondent en tapant
une forme colorée — le texte des réponses n'est visible que côté écran de
l'animateur, comme sur Kahoot, pour garder tout le monde les yeux rivés sur
l'écran partagé plutôt que chacun sur son propre texte.

Contrairement aux duels (2 joueurs, avance automatique dès que les deux ont
répondu), ici c'est l'animateur qui garde la main : il démarre, révèle la
réponse, passe à la question suivante — indispensable avec un nombre de
joueurs imprévisible et variable en direct.

Persistance JSON (comme le reste du projet). Ce module ne connaît rien de
WebSocket : la diffusion temps réel reste côté appelant (community_webapp.py).
"""

import json
import random
import threading
from pathlib import Path
from datetime import datetime

import duels as DU  # réutilise la banque de mots + génération de questions QCM

PROJET_ROOT      = Path(__file__).resolve().parent.parent
DOSSIER_QUIZLIVE = PROJET_ROOT / "corpus-pular" / "quizlive"
FICHIER_PARTIES  = DOSSIER_QUIZLIVE / "parties.json"
DOSSIER_QUIZLIVE.mkdir(parents=True, exist_ok=True)

NB_QUESTIONS_DEFAUT = 10
LONGUEURS_VALIDES   = [10, 15, 20]
DUREE_QUESTION_MS   = 20000  # 20s par question — indicatif, le minuteur vit côté client
THEMES = DU.THEMES

# Forme + couleur par position d'option (0-3) — façon Kahoot : le joueur ne
# voit que ça sur son téléphone, jamais le texte de la réponse.
FORMES = [
    {"id": "triangle", "emoji": "▲", "couleur": "#e21b3c"},
    {"id": "losange",  "emoji": "◆", "couleur": "#1368ce"},
    {"id": "cercle",   "emoji": "●", "couleur": "#d89e00"},
    {"id": "carre",    "emoji": "■", "couleur": "#26890c"},
]

# ── Persistance ──────────────────────────────────────────────────────────
# Verrou process-wide : plusieurs joueurs qui rejoignent/répondent en même
# temps arrivent chacun via asyncio.to_thread() sur un thread séparé du pool
# — sans verrou, deux lectures-modifications-écritures concurrentes sur le
# même fichier JSON se marchent dessus (le second write écrase le premier,
# un joueur qui vient de rejoindre disparaît silencieusement). Un
# threading.Lock (pas asyncio.Lock, qui ne protège pas entre threads réels)
# sérialise ces accès. Reproduit et confirmé par test avant ce correctif.
_VERROU = threading.Lock()

def charger_parties() -> list[dict]:
    if FICHIER_PARTIES.exists():
        return json.loads(FICHIER_PARTIES.read_text(encoding="utf-8"))
    return []

def sauver_parties(parties: list[dict]):
    FICHIER_PARTIES.write_text(json.dumps(parties, ensure_ascii=False, indent=2), encoding="utf-8")

def _sauver_une_partie(partie: dict):
    parties = charger_parties()
    for i, p in enumerate(parties):
        if p["code"] == partie["code"]:
            parties[i] = partie
            break
    else:
        parties.append(partie)
    sauver_parties(parties)

def _generer_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sans caractères ambigus (0/O, 1/I)
    return "".join(random.choices(alphabet, k=6))

# ── Cycle de vie d'une partie ────────────────────────────────────────────

def creer_partie(
    hote_pseudo: str,
    mots: list[dict],
    theme: str = "Tout",
    nb_questions: int = NB_QUESTIONS_DEFAUT,
    duree_question_ms: int = DUREE_QUESTION_MS,
) -> dict:
    theme = theme if theme in THEMES else "Tout"
    nb_questions = nb_questions if nb_questions in LONGUEURS_VALIDES else NB_QUESTIONS_DEFAUT

    mots_theme = mots if theme == "Tout" else [m for m in mots if m.get("cat") == theme]
    if len(mots_theme) < 4:
        theme = "Tout"
        mots_theme = mots

    with _VERROU:
        parties = charger_parties()
        code = _generer_code()
        while any(p["code"] == code for p in parties):
            code = _generer_code()

        partie = {
            "code":                code,
            "statut":              "attente",  # attente → en_cours → revelation → termine
            "hote":                hote_pseudo,
            "theme":               theme,
            "questions":           DU.generer_questions(mots_theme, nb_questions),
            "question_actuelle":   -1,
            "question_ouverte_le": None,
            "duree_question_ms":   duree_question_ms,
            "joueurs":             [],
            "cree_le":             datetime.now().isoformat(),
            "demarre_le":          None,
            "termine_le":          None,
        }
        parties.append(partie)
        sauver_parties(parties)
        return partie

def obtenir_partie(code: str) -> dict | None:
    return next((p for p in charger_parties() if p["code"] == code.upper()), None)

def rejoindre_partie(code: str, pseudo: str) -> dict | None:
    """Ajoute un joueur à la salle d'attente. None si code invalide, partie
    déjà démarrée/terminée, ou pseudo déjà pris (par un joueur ou l'hôte)."""
    with _VERROU:
        partie = obtenir_partie(code)
        if not partie or partie["statut"] != "attente":
            return None
        pseudo_norm = pseudo.strip().lower()
        if pseudo_norm == partie["hote"].strip().lower():
            return None
        if any(j["pseudo"].strip().lower() == pseudo_norm for j in partie["joueurs"]):
            return None
        partie["joueurs"].append({"pseudo": pseudo, "score": 0, "reponses": []})
        _sauver_une_partie(partie)
        return partie

def demarrer_partie(code: str, hote_pseudo: str) -> dict | None:
    with _VERROU:
        partie = obtenir_partie(code)
        if not partie or partie["statut"] != "attente" or partie["hote"] != hote_pseudo:
            return None
        if not partie["joueurs"]:
            return None
        partie["statut"] = "en_cours"
        partie["question_actuelle"] = 0
        partie["question_ouverte_le"] = datetime.now().isoformat()
        partie["demarre_le"] = datetime.now().isoformat()
        _sauver_une_partie(partie)
        return partie

def _calculer_points(correct: bool, temps_ms: int, duree_question_ms: int) -> int:
    if not correct:
        return 0
    bonus_vitesse = max(0, round((duree_question_ms - min(temps_ms, duree_question_ms)) / duree_question_ms * 50))
    return 100 + bonus_vitesse

def enregistrer_reponse(code: str, pseudo: str, q_index: int, reponse: str | None, temps_ms: int) -> dict | None:
    """Enregistre la réponse d'un joueur à la question en cours. N'avance
    jamais la partie tout seul — c'est l'animateur qui décide (reveler_reponse
    / question_suivante), impossible d'attendre "que tout le monde ait
    répondu" avec un nombre de joueurs imprévisible en direct."""
    with _VERROU:
        partie = obtenir_partie(code)
        if not partie or partie["statut"] != "en_cours" or q_index != partie["question_actuelle"]:
            return None
        joueur = next((j for j in partie["joueurs"] if j["pseudo"] == pseudo), None)
        if not joueur:
            return None
        if any(r["q"] == q_index for r in joueur["reponses"]):
            return partie  # déjà répondu — pas de double comptage

        correct = reponse is not None and reponse == partie["questions"][q_index]["reponse"]
        points  = _calculer_points(correct, temps_ms, partie["duree_question_ms"])
        joueur["reponses"].append({
            "q": q_index, "reponse": reponse, "correct": correct,
            "temps_ms": temps_ms, "points": points,
        })
        joueur["score"] += points
        _sauver_une_partie(partie)
        return partie

def reveler_reponse(code: str, hote_pseudo: str) -> dict | None:
    """L'animateur clôt la question en cours : les réponses tardives ne
    comptent plus, l'écran passe en mode révélation (réponse correcte +
    classement à jour)."""
    with _VERROU:
        partie = obtenir_partie(code)
        if not partie or partie["statut"] != "en_cours" or partie["hote"] != hote_pseudo:
            return None
        partie["statut"] = "revelation"
        _sauver_une_partie(partie)
        return partie

def question_suivante(code: str, hote_pseudo: str) -> dict | None:
    with _VERROU:
        partie = obtenir_partie(code)
        if not partie or partie["statut"] != "revelation" or partie["hote"] != hote_pseudo:
            return None
        if partie["question_actuelle"] + 1 < len(partie["questions"]):
            partie["question_actuelle"] += 1
            partie["question_ouverte_le"] = datetime.now().isoformat()
            partie["statut"] = "en_cours"
        else:
            partie["statut"] = "termine"
            partie["termine_le"] = datetime.now().isoformat()
        _sauver_une_partie(partie)
        return partie

# ── Classements ──────────────────────────────────────────────────────────

def classement_partie(partie: dict) -> list[dict]:
    """Classement courant au sein d'une partie, trié par score décroissant."""
    return sorted(
        [{"pseudo": j["pseudo"], "score": j["score"]} for j in partie["joueurs"]],
        key=lambda j: -j["score"],
    )

def classement_champions(limite: int = 20) -> list[dict]:
    """Palmarès agrégé sur toutes les parties terminées : nombre de fois
    champion (1re place), parties jouées, points totaux cumulés."""
    agrege: dict[str, dict] = {}
    for p in charger_parties():
        if p["statut"] != "termine" or not p["joueurs"]:
            continue
        classement = classement_partie(p)
        meilleur_score = classement[0]["score"]
        champions = [j["pseudo"] for j in classement if j["score"] == meilleur_score]
        egalite = len(champions) > 1
        for j in p["joueurs"]:
            e = agrege.setdefault(j["pseudo"], {"pseudo": j["pseudo"], "titres": 0, "parties": 0, "points": 0})
            e["parties"] += 1
            e["points"]  += j["score"]
            if not egalite and j["pseudo"] in champions:
                e["titres"] += 1
    return sorted(agrege.values(), key=lambda e: (-e["titres"], -e["points"]))[:limite]

# ── Vue publique (diffusée en WebSocket) ────────────────────────────────
# Ne jamais renvoyer partie["questions"] brut : ça exposerait TOUTES les
# réponses de la partie (y compris les questions futures) dès la première
# connexion. On ne renvoie que la question en cours, réponse masquée tant
# que l'animateur ne l'a pas révélée.

def etat_public(partie: dict) -> dict:
    q_idx = partie["question_actuelle"]
    q_actuelle = None
    nb_repondu = 0
    if 0 <= q_idx < len(partie["questions"]):
        q = partie["questions"][q_idx]
        nb_repondu = sum(1 for j in partie["joueurs"] if any(r["q"] == q_idx for r in j["reponses"]))
        q_actuelle = {
            "index": q_idx, "total": len(partie["questions"]),
            "emoji": q["emoji"], "fr": q["fr"], "options": q["options"],
        }
        if partie["statut"] in ("revelation", "termine"):
            q_actuelle["reponse"] = q["reponse"]
            comptage = [0] * len(q["options"])
            for j in partie["joueurs"]:
                r = next((r for r in j["reponses"] if r["q"] == q_idx), None)
                if r and r["reponse"] in q["options"]:
                    comptage[q["options"].index(r["reponse"])] += 1
            q_actuelle["comptage"] = comptage

    classement = []
    if partie["statut"] in ("attente", "revelation", "termine"):
        classement = classement_partie(partie)

    return {
        "code":                partie["code"],
        "statut":              partie["statut"],
        "hote":                partie["hote"],
        "theme":               partie["theme"],
        "nb_questions":        len(partie["questions"]),
        "duree_question_ms":   partie["duree_question_ms"],
        "question_actuelle":   q_actuelle,
        "question_ouverte_le": partie["question_ouverte_le"],
        "nb_joueurs":          len(partie["joueurs"]),
        "nb_repondu":          nb_repondu,
        "noms_joueurs":        [j["pseudo"] for j in partie["joueurs"]],
        "classement":          classement,
    }
