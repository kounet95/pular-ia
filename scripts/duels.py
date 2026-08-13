"""
duels.py — Duels en temps réel : QCM de vocabulaire à deux joueurs, partagé
entre le site web (WebSocket) et le bot Telegram (messages + boutons inline).

Un duel :
  - a un code à 6 caractères, partageable par lien
  - a 2 joueurs (créateur + invité), identifiés par pseudo
  - propose la même série de questions QCM aux deux joueurs
  - le score de chaque réponse dépend de la justesse ET de la rapidité
  - le classement global agrège les victoires/points de tous les duels terminés

Persistance en JSON (comme le reste du projet) — pas de base de données.
Ce module ne connaît rien de WebSocket ni de Telegram : la diffusion en
temps réel (registre de connexions, envoi de messages) reste côté appelant,
pour que la même logique de duel serve les deux surfaces.
"""

import json
import random
from pathlib import Path
from datetime import datetime

PROJET_ROOT   = Path(__file__).resolve().parent.parent
DOSSIER_DUELS = PROJET_ROOT / "corpus-pular" / "duels"
FICHIER_DUELS = DOSSIER_DUELS / "duels.json"
DOSSIER_DUELS.mkdir(parents=True, exist_ok=True)

# Mêmes fichiers que le jeu QCM du site web (scripts/community_webapp.py) —
# lus directement ici pour que le bot Telegram n'ait pas besoin d'importer
# tout le serveur FastAPI juste pour la banque de mots. Le seed ci-dessous
# ne sert que si aucun des deux processus n'a encore initialisé le fichier.
DOSSIER_JEU         = PROJET_ROOT / "corpus-pular" / "jeu"
FICHIER_MOTS_BASE   = DOSSIER_JEU / "mots_base.json"
FICHIER_MOTS_CUSTOM = DOSSIER_JEU / "mots_custom.json"

_MOTS_SEED = [
    ("🐄","Vache","nagge","Animaux"),    ("🐑","Mouton","mbabba","Animaux"),
    ("🐐","Chèvre","mbewa","Animaux"),   ("🐎","Cheval","puccu","Animaux"),
    ("🐓","Poule","gertooɗe","Animaux"), ("🐕","Chien","rawaandu","Animaux"),
    ("🐈","Chat","muuse","Animaux"),     ("🦁","Lion","liingu","Animaux"),
    ("🐊","Caïman","baylo","Animaux"),   ("🦅","Aigle","galeejo","Animaux"),
    ("🐍","Serpent","mbeewa","Animaux"), ("🦋","Papillon","lekki","Animaux"),
    ("🏠","Maison","galle","Objets"),    ("💧","Eau","ndiyam","Objets"),
    ("🔥","Feu","jaango","Objets"),      ("☀️","Soleil","naange","Objets"),
    ("🌙","Lune","lewru","Objets"),      ("🌳","Arbre","ledde","Objets"),
    ("🥛","Lait","kosam","Objets"),      ("🍚","Riz","nyiiri","Objets"),
    ("📚","Livre","defte","Objets"),     ("🥁","Tambour","tammbari","Objets"),
    ("🔪","Couteau","lahal","Objets"),   ("🪣","Calebasse","hoore","Objets"),
    ("👁️","Œil","yitere","Corps"),      ("👂","Oreille","nowru","Corps"),
    ("👃","Nez","hinere","Corps"),       ("👄","Bouche","hunuko","Corps"),
    ("✋","Main","juuɗe","Corps"),       ("🦶","Pied","koyngal","Corps"),
    ("🦷","Dents","ñiiɓe","Corps"),      ("💪","Bras","hakke","Corps"),
    ("🌧️","Pluie","ndungu","Nature"),   ("🌊","Fleuve","maayo","Nature"),
    ("🌾","Champ","ngesa","Nature"),     ("🌿","Herbe","gaawri","Nature"),
    ("⛰️","Montagne","tugal","Nature"),  ("🌬️","Vent","hendu","Nature"),
    ("🌑","Nuit","jamma","Nature"),
    ("👨","Père","baaba","Famille"),     ("👩","Mère","yinaande","Famille"),
    ("👶","Enfant","ɓiɗɗo","Famille"),  ("👴","Grand-père","kaawu","Famille"),
    ("👵","Grand-mère","mawndoo","Famille"),("👫","Époux","gorko","Famille"),
    ("👭","Femme","debbo","Famille"),
]

def charger_mots_pour_duel() -> list[dict]:
    """Banque de mots (base + custom) pour générer les questions de duel —
    mêmes fichiers que le jeu QCM web ; seed local si rien n'existe encore."""
    def _lire(chemin):
        if chemin.exists():
            return json.loads(chemin.read_text(encoding="utf-8"))
        return None

    base = _lire(FICHIER_MOTS_BASE)
    if base is None:
        base = [{"id": f"m{i}", "emoji": e, "fr": f, "pular": p, "cat": c}
                 for i, (e, f, p, c) in enumerate(_MOTS_SEED)]
    custom = _lire(FICHIER_MOTS_CUSTOM) or []
    base_pular = {m["pular"] for m in base}
    return base + [m for m in custom if m.get("pular") not in base_pular]

NB_QUESTIONS      = 10  # défaut si non précisé
LONGUEURS_VALIDES = [5, 10, 20]
DUREE_QUESTION_MS = 15000  # 15s — indicatif, le minuteur vit côté client/bot
THEMES = ["Tout", "Animaux", "Objets", "Corps", "Nature", "Famille"]

def charger_duels() -> list[dict]:
    if FICHIER_DUELS.exists():
        return json.loads(FICHIER_DUELS.read_text(encoding="utf-8"))
    return []

def sauver_duels(duels: list[dict]):
    FICHIER_DUELS.write_text(json.dumps(duels, ensure_ascii=False, indent=2), encoding="utf-8")

def _sauver_un_duel(duel: dict):
    duels = charger_duels()
    for i, d in enumerate(duels):
        if d["code"] == duel["code"]:
            duels[i] = duel
            break
    sauver_duels(duels)

def _generer_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sans caractères ambigus (0/O, 1/I)
    return "".join(random.choices(alphabet, k=6))

def generer_questions(mots: list[dict], n: int = NB_QUESTIONS) -> list[dict]:
    """Construit n questions QCM à partir de la banque de mots (emoji/fr → pular)."""
    n = min(n, len(mots))
    choisis = random.sample(mots, n)
    questions = []
    for m in choisis:
        pool = [x["pular"] for x in mots if x["pular"] != m["pular"]]
        distracteurs = random.sample(pool, min(3, len(pool)))
        options = distracteurs + [m["pular"]]
        random.shuffle(options)
        questions.append({
            "emoji": m["emoji"], "fr": m["fr"],
            "reponse": m["pular"], "options": options,
        })
    return questions

def creer_duel(
    pseudo: str,
    mots: list[dict],
    surface: str = "web",
    theme: str = "Tout",
    nb_questions: int = NB_QUESTIONS,
) -> dict:
    """Crée un duel en attente d'un second joueur, filtré sur un thème et
    une longueur donnés (repli sur toute la banque si le thème est trop
    pauvre pour fournir assez de questions et de distracteurs)."""
    theme = theme if theme in THEMES else "Tout"
    nb_questions = nb_questions if nb_questions in LONGUEURS_VALIDES else NB_QUESTIONS

    mots_theme = mots if theme == "Tout" else [m for m in mots if m.get("cat") == theme]
    if len(mots_theme) < 4:
        theme = "Tout"
        mots_theme = mots

    duels = charger_duels()
    code = _generer_code()
    while any(d["code"] == code for d in duels):
        code = _generer_code()
    duel = {
        "code":              code,
        "statut":            "attente",  # attente → en_cours → termine
        "surface":           surface,
        "theme":             theme,
        "questions":         generer_questions(mots_theme, nb_questions),
        "question_actuelle": 0,
        "joueurs":           [{"pseudo": pseudo, "score": 0, "reponses": []}],
        "cree_le":           datetime.now().isoformat(),
        "demarre_le":        None,
        "termine_le":        None,
    }
    duels.append(duel)
    sauver_duels(duels)
    return duel

def rejoindre_duel(code: str, pseudo: str) -> dict | None:
    """Fait entrer un second joueur et démarre le duel. None si code invalide,
    duel déjà complet, ou pseudo déjà présent (pas de duel contre soi-même)."""
    duels = charger_duels()
    duel = next((d for d in duels if d["code"] == code.upper()), None)
    if not duel or duel["statut"] != "attente":
        return None
    if any(j["pseudo"].strip().lower() == pseudo.strip().lower() for j in duel["joueurs"]):
        return None
    duel["joueurs"].append({"pseudo": pseudo, "score": 0, "reponses": []})
    duel["statut"] = "en_cours"
    duel["demarre_le"] = datetime.now().isoformat()
    sauver_duels(duels)
    return duel

def obtenir_duel(code: str) -> dict | None:
    return next((d for d in charger_duels() if d["code"] == code.upper()), None)

def calculer_points(correct: bool, temps_ms: int) -> int:
    if not correct:
        return 0
    bonus_vitesse = max(0, round((DUREE_QUESTION_MS - min(temps_ms, DUREE_QUESTION_MS)) / DUREE_QUESTION_MS * 50))
    return 100 + bonus_vitesse

def enregistrer_reponse(code: str, pseudo: str, q_index: int, reponse: str | None, temps_ms: int) -> dict | None:
    """
    Enregistre la réponse d'un joueur à la question q_index (reponse=None si
    temps écoulé sans réponse). Fait avancer le duel à la question suivante
    — ou le termine — une fois que tous les joueurs ont répondu. Retourne le
    duel à jour, ou None si code/pseudo invalide ou duel pas en cours.
    """
    duel = obtenir_duel(code)
    if not duel or duel["statut"] != "en_cours":
        return None
    joueur = next((j for j in duel["joueurs"] if j["pseudo"] == pseudo), None)
    if not joueur:
        return None
    if any(r["q"] == q_index for r in joueur["reponses"]):
        return duel  # déjà répondu — pas de double comptage

    correct = reponse is not None and reponse == duel["questions"][q_index]["reponse"]
    points  = calculer_points(correct, temps_ms)
    joueur["reponses"].append({
        "q": q_index, "reponse": reponse, "correct": correct,
        "temps_ms": temps_ms, "points": points,
    })
    joueur["score"] += points

    if all(any(r["q"] == q_index for r in j["reponses"]) for j in duel["joueurs"]):
        if q_index + 1 < len(duel["questions"]):
            duel["question_actuelle"] = q_index + 1
        else:
            duel["statut"] = "termine"
            duel["termine_le"] = datetime.now().isoformat()

    _sauver_un_duel(duel)
    return duel

def classement(limite: int = 20) -> list[dict]:
    """Classement agrégé (victoires, duels joués, points totaux) par pseudo,
    calculé sur tous les duels terminés à deux joueurs."""
    agrege: dict[str, dict] = {}
    for d in charger_duels():
        if d["statut"] != "termine" or len(d["joueurs"]) < 2:
            continue
        meilleur_score = max(j["score"] for j in d["joueurs"])
        gagnants = [j["pseudo"] for j in d["joueurs"] if j["score"] == meilleur_score]
        egalite = len(gagnants) > 1
        for j in d["joueurs"]:
            e = agrege.setdefault(j["pseudo"], {"pseudo": j["pseudo"], "victoires": 0, "duels": 0, "points": 0})
            e["duels"]  += 1
            e["points"] += j["score"]
            if not egalite and j["pseudo"] in gagnants:
                e["victoires"] += 1
    return sorted(agrege.values(), key=lambda e: (-e["victoires"], -e["points"]))[:limite]
