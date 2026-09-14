"""
coran_pular.py — Recherche de versets coraniques en pular
Fusionne, verset par verset, les deux traductions téléchargées par
import_coran_fulani.py :
  - fulani_rwwad     : traduction littérale (Rowwad Translation Center)
  - fulani_mokhtasar : Tafsir Al-Mukhtasar (exégèse détaillée)

Expose une recherche tolérante (référence "sourate:verset", mot-clé, ou
phrase/extrait de verset même incomplet ou légèrement différent) réutilisable
par l'API web (community_webapp.py).

Usage:
    from scripts.coran_pular import rechercher_versets, obtenir_verset
"""

import json
import re
from pathlib import Path

PROJET_ROOT  = Path(__file__).resolve().parent.parent
DOSSIER_META = PROJET_ROOT / "corpus-pular" / "livres" / "metadata"

FICHIER_LITTERAL = DOSSIER_META / "fulani_rwwad_raw.json"
FICHIER_TAFSIR   = DOSSIER_META / "fulani_mokhtasar_raw.json"
FICHIER_INDEX    = DOSSIER_META / "coran_versets_index.json"

TRADUCTION_SOURCE  = "Rowwad Translation Center (fulani_rwwad)"
EXPLICATION_SOURCE = "Tafsir Al-Mukhtasar (fulani_mokhtasar)"

# Noms usuels (translittération standard) des 114 sourates — pure référence
# factuelle (identique à peu près partout : quran.com, tanzil.net...), pas
# un extrait de traduction protégée.
NOMS_SOURATES = [
    "Al-Fatiha", "Al-Baqarah", "Aal-E-Imran", "An-Nisa", "Al-Ma'idah",
    "Al-An'am", "Al-A'raf", "Al-Anfal", "At-Tawbah", "Yunus",
    "Hud", "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr",
    "An-Nahl", "Al-Isra", "Al-Kahf", "Maryam", "Ta-Ha",
    "Al-Anbiya", "Al-Hajj", "Al-Mu'minun", "An-Nur", "Al-Furqan",
    "Ash-Shu'ara", "An-Naml", "Al-Qasas", "Al-Ankabut", "Ar-Rum",
    "Luqman", "As-Sajdah", "Al-Ahzab", "Saba", "Fatir",
    "Ya-Sin", "As-Saffat", "Sad", "Az-Zumar", "Ghafir",
    "Fussilat", "Ash-Shura", "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah",
    "Al-Ahqaf", "Muhammad", "Al-Fath", "Al-Hujurat", "Qaf",
    "Adh-Dhariyat", "At-Tur", "An-Najm", "Al-Qamar", "Ar-Rahman",
    "Al-Waqi'ah", "Al-Hadid", "Al-Mujadilah", "Al-Hashr", "Al-Mumtahanah",
    "As-Saff", "Al-Jumu'ah", "Al-Munafiqun", "At-Taghabun", "At-Talaq",
    "At-Tahrim", "Al-Mulk", "Al-Qalam", "Al-Haqqah", "Al-Ma'arij",
    "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir", "Al-Qiyamah",
    "Al-Insan", "Al-Mursalat", "An-Naba", "An-Nazi'at", "Abasa",
    "At-Takwir", "Al-Infitar", "Al-Mutaffifin", "Al-Inshiqaq", "Al-Buruj",
    "At-Tariq", "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad",
    "Ash-Shams", "Al-Layl", "Ad-Duha", "Ash-Sharh", "At-Tin",
    "Al-Alaq", "Al-Qadr", "Al-Bayyinah", "Az-Zalzalah", "Al-Adiyat",
    "Al-Qari'ah", "At-Takathur", "Al-Asr", "Al-Humazah", "Al-Fil",
    "Quraysh", "Al-Ma'un", "Al-Kawthar", "Al-Kafirun", "An-Nasr",
    "Al-Masad", "Al-Ikhlas", "Al-Falaq", "An-Nas",
]


def nom_sourate(numero: int) -> str:
    if 1 <= numero <= len(NOMS_SOURATES):
        return NOMS_SOURATES[numero - 1]
    return ""


_index_liste: list[dict] | None = None
_index_dict:  dict[tuple, dict] | None = None
_arabe_mots:  list[frozenset] | None = None

# Diacritiques arabes (harakat, tanwin, sukun, shadda, marques de pause
# coraniques...) — les retirer permet de retrouver un verset même si le
# texte collé par l'utilisateur n'a pas exactement les mêmes signes
# diacritiques que la source quranenc.com.
_DIACRITIQUES_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")


def _normaliser_arabe(texte: str) -> str:
    if not texte:
        return ""
    texte = _DIACRITIQUES_RE.sub("", texte)
    texte = texte.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    texte = texte.replace("ى", "ي").replace("ة", "ه")
    return texte.strip()


def _charger_bruts(fichier: Path) -> dict[tuple, dict]:
    if not fichier.exists():
        return {}
    with open(fichier, encoding="utf-8") as f:
        data = json.load(f)
    return {(str(v["sura"]), str(v["aya"])): v for v in data.get("versets", [])}


def construire_index(forcer: bool = False) -> list[dict]:
    """Fusionne les deux traductions par (sourate, verset) et sauvegarde l'index."""
    if FICHIER_INDEX.exists() and not forcer:
        with open(FICHIER_INDEX, encoding="utf-8") as f:
            return json.load(f)

    littéral = _charger_bruts(FICHIER_LITTERAL)
    tafsir   = _charger_bruts(FICHIER_TAFSIR)

    cles = sorted(
        set(littéral) | set(tafsir),
        key=lambda c: (int(c[0]), int(c[1])),
    )
    index = []
    for sura, aya in cles:
        l = littéral.get((sura, aya)) or {}
        t = tafsir.get((sura, aya)) or {}
        num_sourate = int(sura)
        index.append({
            "sourate":            num_sourate,
            "sourate_nom":        nom_sourate(num_sourate),
            "verset":             int(aya),
            "arabe":              l.get("arabic_text") or t.get("arabic_text") or "",
            "traduction":         l.get("translation", ""),
            "traduction_source":  TRADUCTION_SOURCE,
            "explication":        t.get("translation", ""),
            "explication_source": EXPLICATION_SOURCE,
        })

    DOSSIER_META.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    return index


def _charger() -> tuple[list[dict], dict[tuple, dict]]:
    global _index_liste, _index_dict, _arabe_mots
    if _index_liste is None:
        _index_liste = construire_index()
        _index_dict  = {(v["sourate"], v["verset"]): v for v in _index_liste}
        _arabe_mots  = [frozenset(_normaliser_arabe(v["arabe"]).split()) for v in _index_liste]
    return _index_liste, _index_dict


def obtenir_verset(sourate: int, verset: int) -> dict | None:
    """Retourne un verset précis (arabe + traduction + explication)."""
    _, index_dict = _charger()
    return index_dict.get((sourate, verset))


_REF_RE = re.compile(r"^\s*(\d{1,3})\s*[:,.]\s*(\d{1,3})\s*$")


def _ratio_recouvrement(mots_requete: frozenset, mots_cible: frozenset) -> float:
    """Part des mots de la requête retrouvés dans le texte cible (0 à 1)."""
    if not mots_requete or not mots_cible:
        return 0.0
    return len(mots_requete & mots_cible) / len(mots_requete)


def rechercher_versets(q: str, n: int = 10, seuil: float = 0.6) -> list[dict]:
    """
    Recherche des versets :
      - "2:255" (ou "2,255" / "2.255") → renvoie directement ce verset précis.
      - texte libre → recherche tolérante par recouvrement de mots (et pas
        seulement une correspondance exacte). Colle un mot, une phrase
        entière ou même un extrait incomplet d'un verset (arabe ou pular) :
        le verset le plus proche est retrouvé même si des mots manquent ou
        diffèrent légèrement (diacritiques, coquilles...).
    """
    q = (q or "").strip()
    if not q:
        return []

    m = _REF_RE.match(q)
    if m:
        v = obtenir_verset(int(m.group(1)), int(m.group(2)))
        return [v] if v else []

    index_liste, _ = _charger()
    mots_arabe_requete = frozenset(_normaliser_arabe(q).split())
    mots_pular_requete = frozenset(q.lower().split())

    resultats = []
    for v, mots_arabe_verset in zip(index_liste, _arabe_mots):
        score = 0.0
        if mots_arabe_requete:
            score = max(score, _ratio_recouvrement(mots_arabe_requete, mots_arabe_verset))
        if mots_pular_requete:
            score = max(score, _ratio_recouvrement(mots_pular_requete, frozenset(v["traduction"].lower().split())))
            score = max(score, _ratio_recouvrement(mots_pular_requete, frozenset(v["explication"].lower().split())))
        if score >= seuil:
            resultats.append((score, v))

    resultats.sort(key=lambda t: -t[0])
    return [v for _, v in resultats[:n]]


# Canaux Telegram islamiques indexés dans le RAG par
# import_telegram_islamique.py (transcriptions d'audios + Q/R communautaires,
# cf. corpus-pular/processed/telegram/base_connaissance.json) — à distinguer
# de la traduction quranenc.com : ici, transcription automatique (Whisper)
# de prêches/réponses d'un prédicateur en particulier, pas une source vérifiée.
CANAUX_COMMUNAUTAIRES = {
    "telegram_DDTV226":     "Doudhe Diina TV (Telegram — questions/réponses)",
    "telegram_laawolsunna": "Laawol Sunna (Telegram — enseignements)",
}


def _rechercher_communaute(question: str, n: int = 4) -> list[dict]:
    """Extraits pertinents des canaux Telegram islamiques déjà indexés
    (peut être vide si import_telegram_islamique.py n'a pas encore été lancé,
    ou si rien n'y a encore été transcrit)."""
    try:
        from rag_livres import rechercher as rag_rechercher
    except Exception:
        return []
    resultats = []
    for livre_id, label in CANAUX_COMMUNAUTAIRES.items():
        try:
            for c in rag_rechercher(question, n=n, livre_id=livre_id):
                resultats.append({"source": label, "texte": c["texte"], "score": c["score"]})
        except Exception:
            continue
    resultats.sort(key=lambda c: -c["score"])
    return resultats[:n]


def repondre_question(question: str, n: int = 6, seuil: float = 0.25) -> dict:
    """
    Répond à une question libre posée en pular, français ou arabe, en
    combinant deux sources :
      - les versets trouvés par rechercher_versets (traduction/tafsir
        quranenc.com — source principale et fiable) ;
      - les extraits pertinents des canaux Telegram islamiques déjà indexés
        par import_telegram_islamique.py (enseignements/Q&R communautaires en
        pular — source secondaire, transcription automatique non vérifiée).

    Retourne {"reponse": str|None, "versets": list[dict], "communaute":
    list[dict]}. "reponse" vaut None si NI l'un NI l'autre n'a rien trouvé —
    Claude n'est alors pas appelé du tout, pour ne jamais répondre "à
    l'aveugle" sur un sujet religieux sans source à l'appui.
    """
    versets = rechercher_versets(question, n=n, seuil=seuil)
    communaute = _rechercher_communaute(question)

    if not versets and not communaute:
        return {"reponse": None, "versets": [], "communaute": []}

    from espace_editorial import anthropic_configure

    contexte = ""
    if versets:
        contexte += "SOURCE PRINCIPALE — versets du Coran (traduction officielle quranenc.com) :\n\n"
        contexte += "\n\n".join(
            f"[Sourate {v['sourate']} ({v['sourate_nom']}), verset {v['verset']}]\n"
            f"Arabe : {v['arabe']}\n"
            f"Traduction fulfulde ({TRADUCTION_SOURCE}) : {v['traduction']}\n"
            f"Tafsir fulfulde ({EXPLICATION_SOURCE}) : {v['explication']}"
            for v in versets
        )
    if communaute:
        contexte += "\n\nSOURCE SECONDAIRE — enseignements communautaires en pular (transcription "
        contexte += "automatique d'audios Telegram, à ne citer qu'en complément, jamais à la place d'un verset) :\n\n"
        contexte += "\n\n".join(f"[{e['source']}]\n{e['texte']}" for e in communaute)

    system = (
        "Tu réponds à des questions sur le Coran posées par des membres de la "
        "communauté pular (Foula/Fulfulde), en pular, en français ou en arabe. "
        "Réponds STRICTEMENT dans la même langue que la question. Base-toi "
        "UNIQUEMENT sur les extraits fournis — ne cite et n'invente aucun "
        "autre verset ni hadith. Les extraits marqués SOURCE PRINCIPALE sont "
        "la traduction officielle du Coran : appuie-toi dessus en priorité et "
        "cite toujours la référence sourate:verset de chaque verset utilisé. "
        "Les extraits marqués SOURCE SECONDAIRE sont des transcriptions "
        "automatiques (donc potentiellement imparfaites) de prêches d'un "
        "prédicateur communautaire en particulier sur Telegram : utilise-les "
        "seulement pour illustrer ou compléter, jamais comme une vérité "
        "religieuse générale, et précise qu'il s'agit de l'avis d'un "
        "enseignant communautaire (pas d'une source officielle) quand tu t'en "
        "sers. Si les extraits fournis ne répondent pas clairement à la "
        "question, dis-le honnêtement plutôt que de forcer une réponse. Pour "
        "toute question de jurisprudence (fiqh) détaillée ou de cas "
        "personnel, rappelle qu'il faut consulter un savant (alim) local. "
        "Réponse concise, 150 mots maximum, ton respectueux."
    )

    client = anthropic_configure()
    message = client.messages.create(
        model="claude-opus-5",
        max_tokens=600,
        output_config={"effort": "medium"},
        system=system,
        messages=[{
            "role": "user",
            "content": f"Extraits (fulfulde) :\n\n{contexte}\n\nQuestion : {question}",
        }],
    )
    reponse = "".join(b.text for b in message.content if b.type == "text")
    return {"reponse": reponse, "versets": versets, "communaute": communaute}


def stats_coran() -> dict:
    index_liste, _ = _charger()
    return {
        "total_versets":       len(index_liste),
        "avec_traduction":     sum(1 for v in index_liste if v["traduction"]),
        "avec_explication":    sum(1 for v in index_liste if v["explication"]),
    }
