"""
prepare_llm_dataset.py — Génère le dataset d'instruction pour fine-tuner un LLM sur le Pular

Sources utilisées (dans l'ordre de qualité décroissante) :
  1. Corrections communautaires (texte corrigé manuellement)         → gold
  2. Contributions communautaires validées par un prof               → gold
  3. Mots custom des professeurs                                      → gold
  4. Contributions communautaires non rejetées                        → silver
  5. Paires Latin/Adlam (dataset translitération)                    → selon qualite
  6. Segments propres des transcriptions Telegram                     → silver
  7. Chunks RAG (livres/poèmes)                                       → bronze
  8. Vocabulaire de base MOTS_JEU                                     → gold

Format de sortie (Alpaca) :
  {"instruction": "...", "input": "...", "output": "...", "source": "...", "qualite": "..."}

Usage :
  python scripts/prepare_llm_dataset.py            # génère train/val/test
  python scripts/prepare_llm_dataset.py --stats    # affiche les stats seulement
  python scripts/prepare_llm_dataset.py --min-qualite silver  # filtre la qualité min
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime

# ── Chemins (configurables via --root pour Google Colab) ──────────────────────
# Résolution anticipée du --root avant l'init des variables globales
_root_arg = None
for i, arg in enumerate(sys.argv):
    if arg == "--root" and i + 1 < len(sys.argv):
        _root_arg = Path(sys.argv[i + 1])
        break

PROJET_ROOT = _root_arg if _root_arg else Path(__file__).resolve().parent.parent
CORPUS      = PROJET_ROOT / "corpus-pular"

SRC_CONTRIBUTIONS  = CORPUS / "community" / "contributions"
SRC_CORRECTIONS    = CORPUS / "community" / "corrections"
SRC_TRANSCRIPTIONS = CORPUS / "processed" / "transcriptions"
SRC_TRANSLIT_TRAIN = CORPUS / "dataset" / "translit" / "train.jsonl"
SRC_TRANSLIT_VAL   = CORPUS / "dataset" / "translit" / "val.jsonl"
SRC_TRANSLIT_TEST  = CORPUS / "dataset" / "translit" / "test.jsonl"
SRC_RAG_LIVRES     = CORPUS / "dataset" / "livres" / "corpus_livres.jsonl"
SRC_MOTS_CUSTOM    = CORPUS / "jeu" / "mots_custom.json"

DST_DIR  = CORPUS / "dataset" / "llm"
DST_TRAIN = DST_DIR / "train.jsonl"
DST_VAL   = DST_DIR / "val.jsonl"
DST_TEST  = DST_DIR / "test.jsonl"
DST_STATS = DST_DIR / "stats.json"

DST_DIR.mkdir(parents=True, exist_ok=True)

POIDS_QUALITE = {"gold": 3, "silver": 1, "bronze": 0}

# ── Vocabulaire de base MOTS_JEU ──────────────────────────────────────────────
# Extrait de web/index.html — mis à jour ici si le jeu évolue
MOTS_JEU = [
    # Animaux
    {"emoji":"🐄","fr":"Vache",      "pular":"nagge",    "adlam":"𞤲𞤢𞤺𞤺𞤫",     "cat":"Animaux"},
    {"emoji":"🐐","fr":"Chèvre",     "pular":"mbabba",   "adlam":"𞤃𞤦𞤢𞤦𞤦𞤢",   "cat":"Animaux"},
    {"emoji":"🐑","fr":"Mouton",     "pular":"mbalndu",  "adlam":"𞤃𞤦𞤢𞤤𞤲𞤣𞤵",  "cat":"Animaux"},
    {"emoji":"🐓","fr":"Poulet",     "pular":"gertogal", "adlam":"𞤘𞤫𞤪𞤼𞤮𞤺𞤢𞤤", "cat":"Animaux"},
    {"emoji":"🐴","fr":"Cheval",     "pular":"puccu",    "adlam":"𞤨𞤵𞤷𞤷𞤵",    "cat":"Animaux"},
    {"emoji":"🐘","fr":"Éléphant",   "pular":"wirnde",   "adlam":"𞤱𞤭𞤪𞤲𞤣𞤫",   "cat":"Animaux"},
    {"emoji":"🦁","fr":"Lion",       "pular":"lionde",   "adlam":"𞤤𞤭𞤮𞤲𞤣𞤫",   "cat":"Animaux"},
    {"emoji":"🐊","fr":"Crocodile",  "pular":"bammbale", "adlam":"𞤄𞤢𞤃𞤦𞤢𞤤𞤫",  "cat":"Animaux"},
    {"emoji":"🐟","fr":"Poisson",    "pular":"liingu",   "adlam":"𞤤𞤭𞤭𞤲𞤺𞤵",   "cat":"Animaux"},
    {"emoji":"🐦","fr":"Oiseau",     "pular":"fowru",    "adlam":"𞤬𞤮𞤱𞤪𞤵",    "cat":"Animaux"},
    {"emoji":"🐍","fr":"Serpent",    "pular":"mbaari",   "adlam":"𞤃𞤦𞤢𞤢𞤪𞤭",   "cat":"Animaux"},
    # Objets
    {"emoji":"🥛","fr":"Lait",       "pular":"kosam",    "adlam":"𞤳𞤮𞤧𞤢𞤥",     "cat":"Objets"},
    {"emoji":"🌊","fr":"Eau",        "pular":"ndiyam",   "adlam":"𞤐𞤣𞤭𞤴𞤢𞤥",   "cat":"Objets"},
    {"emoji":"🔥","fr":"Feu",        "pular":"jaango",   "adlam":"𞤔𞤢𞤢𞤲𞤺𞤮",   "cat":"Objets"},
    {"emoji":"🏠","fr":"Maison",     "pular":"galle",    "adlam":"𞤘𞤢𞤤𞤤𞤫",     "cat":"Objets"},
    {"emoji":"🌳","fr":"Arbre",      "pular":"ledde",    "adlam":"𞤂𞤫𞤣𞤣𞤫",     "cat":"Objets"},
    {"emoji":"📖","fr":"Livre",      "pular":"deftere",  "adlam":"𞤁𞤫𞤬𞤼𞤫𞤪𞤫",  "cat":"Objets"},
    {"emoji":"🌙","fr":"Lune",       "pular":"lewru",    "adlam":"𞤂𞤫𞤱𞤪𞤵",     "cat":"Objets"},
    {"emoji":"☀️","fr":"Soleil",     "pular":"naange",   "adlam":"𞤐𞤢𞤢𞤲𞤺𞤫",   "cat":"Objets"},
    {"emoji":"⭐","fr":"Étoile",     "pular":"hoodere",  "adlam":"𞤖𞤮𞤮𞤣𞤫𞤪𞤫",  "cat":"Objets"},
    # Corps
    {"emoji":"✋","fr":"Main",       "pular":"jungo",    "adlam":"𞤔𞤵𞤲𞤺𞤮",     "cat":"Corps"},
    {"emoji":"👁️","fr":"Œil",        "pular":"yitere",   "adlam":"𞤒𞤭𞤼𞤫𞤪𞤫",   "cat":"Corps"},
    {"emoji":"👂","fr":"Oreille",    "pular":"hetorde",  "adlam":"𞤖𞤫𞤼𞤮𞤪𞤣𞤫",  "cat":"Corps"},
    {"emoji":"👄","fr":"Bouche",     "pular":"hunuko",   "adlam":"𞤖𞤵𞤲𞤵𞤳𞤮",   "cat":"Corps"},
    {"emoji":"🦵","fr":"Jambe",      "pular":"koyngal",  "adlam":"𞤑𞤮𞤴𞤲𞤺𞤢𞤤",  "cat":"Corps"},
    # Nature
    {"emoji":"🌧️","fr":"Pluie",      "pular":"ndungu",   "adlam":"𞤐𞤣𞤵𞤲𞤺𞤵",   "cat":"Nature"},
    {"emoji":"💨","fr":"Vent",       "pular":"henndu",   "adlam":"𞤖𞤫𞤲𞤲𞤣𞤵",   "cat":"Nature"},
    {"emoji":"🌍","fr":"Terre",      "pular":"leydi",    "adlam":"𞤂𞤫𞤴𞤣𞤭",     "cat":"Nature"},
    # Famille
    {"emoji":"👨","fr":"Homme",      "pular":"gorko",    "adlam":"𞤘𞤮𞤪𞤳𞤮",     "cat":"Famille"},
    {"emoji":"👩","fr":"Femme",      "pular":"debbo",    "adlam":"𞤁𞤫𞤦𞤦𞤮",     "cat":"Famille"},
    {"emoji":"👦","fr":"Enfant",     "pular":"biddo",    "adlam":"𞤄𞤭𞤣𞤣𞤮",     "cat":"Famille"},
    {"emoji":"👴","fr":"Vieux",      "pular":"mawdo",    "adlam":"𞤃𞤢𞤱𞤣𞤮",     "cat":"Famille"},
    {"emoji":"👶","fr":"Bébé",       "pular":"sukunyel", "adlam":"𞤅𞤵𞤳𞤵𞤻𞤫𞤤",  "cat":"Famille"},
]

# ── Templates d'instruction ────────────────────────────────────────────────────
# Plusieurs formulations par tâche → diversifie le dataset

TEMPLATES_VOCAB_FR_PULAR = [
    ("Comment dit-on « {fr} » en pular ?", "", "{pular}"),
    ("Traduis en pular :", "{fr}", "{pular}"),
    ("Quel est le mot pular pour {fr} ?", "", "{pular}"),
    ("Donne la traduction pular de : {fr}", "", "{pular}"),
    ("En pular, {fr} se dit :", "", "{pular}"),
]

TEMPLATES_VOCAB_PULAR_FR = [
    ("Que signifie le mot pular « {pular} » en français ?", "", "{fr}"),
    ("Traduis en français :", "{pular}", "{fr}"),
    ("Quel est le sens du mot pular {pular} ?", "", "{fr}"),
    ("En français, {pular} veut dire :", "", "{fr}"),
]

TEMPLATES_VOCAB_ADLAM = [
    ("Écris « {pular} » en alphabet Adlam.", "", "{adlam}"),
    ("Comment s'écrit {pular} en Adlam ?", "", "{adlam}"),
    ("Translitère en Adlam :", "{pular}", "{adlam}"),
    ("Donne l'écriture Adlam du mot pular {pular}.", "", "{adlam}"),
]

TEMPLATES_VOCAB_COMPLET = [
    (
        "Donne le mot pular pour « {fr} » avec son écriture Adlam.",
        "",
        "{pular} — en Adlam : {adlam}",
    ),
    (
        "Complète cette fiche de vocabulaire :\nFrançais : {fr}\nPular : ?\nAdlam : ?",
        "",
        "Pular : {pular}\nAdlam : {adlam}",
    ),
]

TEMPLATES_TRANSLIT = [
    ("Translitère ce texte pular en alphabet Adlam.", "{latin}", "{adlam}"),
    ("Écris en Adlam :", "{latin}", "{adlam}"),
    ("Convertis ce texte en écriture Adlam :", "{latin}", "{adlam}"),
    ("Ce texte est écrit en latin. Donne la version Adlam.", "{latin}", "{adlam}"),
]

TEMPLATES_CORPUS = [
    ("Continue ce texte en pular :", "{debut}", "{suite}"),
    ("Voici le début d'un texte en pular. Complète-le :", "{debut}", "{suite}"),
]

TEMPLATES_CONTRIB = [
    ("Écris une phrase en pular.", "", "{texte}"),
    ("Donne un exemple de phrase en pular.", "", "{texte}"),
    ("Formule une phrase en pular.", "", "{texte}"),
]

# ── Utilitaires ───────────────────────────────────────────────────────────────

def _nettoyer(texte: str) -> str:
    texte = re.sub(r"\s+", " ", texte).strip()
    # Supprimer les hallucinations Whisper (répétitions)
    texte = re.sub(r"(\b\w{2,}\b)( \1){4,}", "", texte)
    return texte.strip()

def _est_valide(texte: str, min_len: int = 5, max_len: int = 800) -> bool:
    if not texte:
        return False
    t = _nettoyer(texte)
    if len(t) < min_len or len(t) > max_len:
        return False
    # Rejeter si >30% de répétitions de 3 mots consécutifs
    mots = t.split()
    if len(mots) > 10:
        trigrammes = [" ".join(mots[i:i+3]) for i in range(len(mots)-2)]
        if max(trigrammes.count(t) for t in set(trigrammes)) > len(trigrammes) * 0.3:
            return False
    return True

def _choisir_template(templates: list, variables: dict) -> dict | None:
    tpl = random.choice(templates)
    try:
        instruction = tpl[0].format(**variables)
        input_      = tpl[1].format(**variables)
        output      = tpl[2].format(**variables)
        return {"instruction": instruction, "input": input_, "output": output}
    except KeyError:
        return None

def _deduplicater(exemples: list[dict]) -> list[dict]:
    vus = set()
    resultat = []
    for ex in exemples:
        cle = (ex.get("instruction","")[:60], ex.get("output","")[:80])
        if cle not in vus:
            vus.add(cle)
            resultat.append(ex)
    return resultat

# ── Extracteurs par source ────────────────────────────────────────────────────

def extraire_vocabulaire(mots: list[dict], source: str = "mots_jeu") -> list[dict]:
    """Génère des paires vocab FR↔Pular↔Adlam."""
    exemples = []
    for mot in mots:
        fr    = mot.get("fr", "").strip()
        pular = mot.get("pular", "").strip()
        adlam = mot.get("adlam", "").strip()
        if not fr or not pular:
            continue
        vars_base = {"fr": fr, "pular": pular, "adlam": adlam}

        # FR → Pular
        ex = _choisir_template(TEMPLATES_VOCAB_FR_PULAR, vars_base)
        if ex:
            exemples.append({**ex, "source": source, "qualite": "gold"})

        # Pular → FR
        ex = _choisir_template(TEMPLATES_VOCAB_PULAR_FR, vars_base)
        if ex:
            exemples.append({**ex, "source": source, "qualite": "gold"})

        # Pular → Adlam (si disponible)
        if adlam:
            ex = _choisir_template(TEMPLATES_VOCAB_ADLAM, vars_base)
            if ex:
                exemples.append({**ex, "source": source, "qualite": "gold"})

            # Fiche complète
            ex = _choisir_template(TEMPLATES_VOCAB_COMPLET, vars_base)
            if ex:
                exemples.append({**ex, "source": source, "qualite": "gold"})

    return exemples


def extraire_translit(fichier: Path, qualite_min: str = "bronze") -> list[dict]:
    """Génère des exemples de translitération Latin→Adlam."""
    if not fichier.exists():
        return []
    exemples = []
    for ligne in fichier.read_text(encoding="utf-8").strip().split("\n"):
        if not ligne.strip():
            continue
        try:
            d = json.loads(ligne)
        except json.JSONDecodeError:
            continue
        latin = d.get("latin", "").strip()
        adlam = d.get("target", "").strip()
        qualite = d.get("qualite", "silver")

        if not _est_valide(latin, 5, 400) or not adlam:
            continue
        if POIDS_QUALITE.get(qualite, 0) < POIDS_QUALITE.get(qualite_min, 0):
            continue

        ex = _choisir_template(TEMPLATES_TRANSLIT, {"latin": latin, "adlam": adlam})
        if ex:
            exemples.append({**ex, "source": "translit_dataset", "qualite": qualite})

    return exemples


def extraire_contributions(dossier: Path) -> list[dict]:
    """Génère des exemples depuis les contributions communautaires."""
    if not dossier.exists():
        return []
    exemples = []
    for fichier in dossier.glob("*.json"):
        try:
            d = json.loads(fichier.read_text(encoding="utf-8"))
        except Exception:
            continue

        status  = d.get("status", "pending")
        if status == "rejeter":
            continue

        texte = (d.get("texte_final") or d.get("texte_corrige_prof") or "").strip()
        texte = _nettoyer(texte)
        if not _est_valide(texte, 5, 300):
            continue

        qualite = "gold" if status in ("valider", "corriger") else "silver"

        ex = _choisir_template(TEMPLATES_CONTRIB, {"texte": texte})
        if ex:
            exemples.append({**ex, "source": "contribution_communautaire", "qualite": qualite})

    return exemples


def extraire_corrections(dossier: Path) -> list[dict]:
    """Génère des exemples de texte corrigé manuellement."""
    if not dossier.exists():
        return []
    exemples = []
    for fichier in dossier.glob("*.json"):
        try:
            d = json.loads(fichier.read_text(encoding="utf-8"))
        except Exception:
            continue

        auto    = _nettoyer(d.get("texte_auto", "") or d.get("transcription_auto", ""))
        corrige = _nettoyer(d.get("texte_corrige", "") or d.get("texte_corrige_prof", ""))

        if not _est_valide(corrige, 10, 600):
            continue

        # Paire brut → corrigé (utile pour le fine-tuning correction)
        if _est_valide(auto, 5) and auto != corrige:
            exemples.append({
                "instruction": "Corrige cette transcription automatique en pular.",
                "input":       auto,
                "output":      corrige,
                "source":      "correction_communautaire",
                "qualite":     "gold",
            })

        # Juste le texte corrigé comme exemple de phrase pular
        ex = _choisir_template(TEMPLATES_CONTRIB, {"texte": corrige})
        if ex:
            exemples.append({**ex, "source": "correction_communautaire", "qualite": "gold"})

    return exemples


def extraire_transcriptions(dossier: Path, conf_min: float = -1.0) -> list[dict]:
    """Génère des exemples depuis les segments de transcription Telegram."""
    if not dossier.exists():
        return []
    exemples = []
    for fichier in dossier.glob("*.json"):
        try:
            d = json.loads(fichier.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Texte complet de la transcription
        texte_complet = _nettoyer(d.get("texte", ""))
        if _est_valide(texte_complet, 30, 600):
            ex = _choisir_template(TEMPLATES_CONTRIB, {"texte": texte_complet})
            if ex:
                exemples.append({**ex, "source": "transcription_telegram", "qualite": "silver"})

        # Segments individuels (plus courts, plus propres)
        for seg in d.get("segments", []):
            conf = seg.get("confiance", -2.0)
            texte_seg = _nettoyer(seg.get("texte", ""))
            if conf < conf_min:
                continue
            if not _est_valide(texte_seg, 10, 200):
                continue
            ex = _choisir_template(TEMPLATES_CONTRIB, {"texte": texte_seg})
            if ex:
                exemples.append({**ex, "source": "segment_telegram", "qualite": "silver"})

    return exemples


def extraire_rag(fichier: Path, max_exemples: int = 2000) -> list[dict]:
    """Génère des exemples de continuation depuis les chunks RAG."""
    if not fichier.exists():
        return []
    exemples = []
    lignes = fichier.read_text(encoding="utf-8").strip().split("\n")
    random.shuffle(lignes)
    for ligne in lignes[:max_exemples * 3]:
        if not ligne.strip():
            continue
        try:
            d = json.loads(ligne)
        except Exception:
            continue

        texte = _nettoyer(d.get("text", ""))
        if not _est_valide(texte, 60, 600):
            continue

        # Couper en debut/suite (60/40)
        pivot = int(len(texte) * 0.55)
        # Chercher une coupure propre (fin de mot)
        espace = texte.rfind(" ", pivot - 30, pivot + 30)
        if espace == -1:
            espace = pivot
        debut = texte[:espace].strip()
        suite = texte[espace:].strip()

        if not _est_valide(debut, 20) or not _est_valide(suite, 20):
            continue

        ex = _choisir_template(TEMPLATES_CORPUS, {"debut": debut, "suite": suite})
        if ex:
            exemples.append({**ex, "source": "rag_livres", "qualite": "bronze"})

        if len(exemples) >= max_exemples:
            break

    return exemples


def extraire_mots_custom(fichier: Path) -> list[dict]:
    """Génère des exemples depuis les mots ajoutés par les professeurs."""
    if not fichier.exists():
        return []
    try:
        mots = json.loads(fichier.read_text(encoding="utf-8"))
    except Exception:
        return []
    return extraire_vocabulaire(mots, source="mots_custom_prof")


# ── Assemblage + split ────────────────────────────────────────────────────────

def assembler(qualite_min: str = "bronze") -> list[dict]:
    print("\nChargement des sources...")
    tous = []

    # 1. Vocabulaire de base (MOTS_JEU + custom)
    vocab_jeu = extraire_vocabulaire(MOTS_JEU, source="mots_jeu")
    print(f"  Vocabulaire MOTS_JEU  : {len(vocab_jeu):>5} exemples")
    tous.extend(vocab_jeu)

    vocab_custom = extraire_mots_custom(SRC_MOTS_CUSTOM)
    print(f"  Mots custom profs     : {len(vocab_custom):>5} exemples")
    tous.extend(vocab_custom)

    # 2. Corrections manuelles (meilleure qualité)
    corrections = extraire_corrections(SRC_CORRECTIONS)
    print(f"  Corrections           : {len(corrections):>5} exemples")
    tous.extend(corrections)

    # 3. Contributions validées
    contribs = extraire_contributions(SRC_CONTRIBUTIONS)
    print(f"  Contributions         : {len(contribs):>5} exemples")
    tous.extend(contribs)

    # 4. Translitération Latin→Adlam
    for fic in [SRC_TRANSLIT_TRAIN, SRC_TRANSLIT_VAL, SRC_TRANSLIT_TEST]:
        translit = extraire_translit(fic, qualite_min)
        print(f"  Translit ({fic.stem:<6})   : {len(translit):>5} exemples")
        tous.extend(translit)

    # 5. Transcriptions Telegram (segments propres)
    transcrip = extraire_transcriptions(SRC_TRANSCRIPTIONS, conf_min=-1.0)
    print(f"  Transcriptions Tlgram : {len(transcrip):>5} exemples")
    tous.extend(transcrip)

    # 6. RAG / Livres (bronze — optionnel)
    if POIDS_QUALITE.get(qualite_min, 0) <= 0:
        rag = extraire_rag(SRC_RAG_LIVRES)
        print(f"  RAG livres            : {len(rag):>5} exemples")
        tous.extend(rag)
    else:
        print(f"  RAG livres            :     0 exemples (filtre qualité {qualite_min})")

    # Déduplication
    avant = len(tous)
    tous = _deduplicater(tous)
    print(f"\nAvant dedup : {avant} | Apres dedup : {len(tous)}")

    return tous


def sauver_splits(tous: list[dict]):
    random.shuffle(tous)

    # Stratification par qualité
    gold   = [e for e in tous if e.get("qualite") == "gold"]
    silver = [e for e in tous if e.get("qualite") == "silver"]
    bronze = [e for e in tous if e.get("qualite") == "bronze"]

    def splitter(lst, val_ratio=0.1, test_ratio=0.1):
        n_val  = max(1, int(len(lst) * val_ratio))
        n_test = max(1, int(len(lst) * test_ratio))
        return lst[n_val+n_test:], lst[:n_val], lst[n_val:n_val+n_test]

    trains, vals, tests = [], [], []
    for groupe in [gold, silver, bronze]:
        if groupe:
            tr, va, te = splitter(groupe)
            trains.extend(tr)
            vals.extend(va)
            tests.extend(te)

    random.shuffle(trains)
    random.shuffle(vals)
    random.shuffle(tests)

    def ecrire(chemin: Path, data: list[dict]):
        with open(chemin, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    ecrire(DST_TRAIN, trains)
    ecrire(DST_VAL,   vals)
    ecrire(DST_TEST,  tests)

    # Stats
    sources_count: dict = {}
    qualite_count: dict = {}
    for ex in tous:
        s = ex.get("source", "?")
        q = ex.get("qualite", "?")
        sources_count[s] = sources_count.get(s, 0) + 1
        qualite_count[q] = qualite_count.get(q, 0) + 1

    stats = {
        "date":    datetime.now().isoformat(),
        "total":   len(tous),
        "train":   len(trains),
        "val":     len(vals),
        "test":    len(tests),
        "qualite": qualite_count,
        "sources": sources_count,
    }
    with open(DST_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def afficher_stats(stats: dict):
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  DATASET LLM PULAR -- {stats['date'][:10]}")
    print(sep)
    print(f"  Total    : {stats['total']:>6} exemples")
    print(f"  Train    : {stats['train']:>6}")
    print(f"  Val      : {stats['val']:>6}")
    print(f"  Test     : {stats['test']:>6}")
    print()
    print("  Qualite :")
    for q, n in sorted(stats["qualite"].items(), key=lambda x: -x[1]):
        bar = "#" * min(30, n // max(1, stats["total"] // 30))
        print(f"    {q:<8} {n:>5}  {bar}")
    print()
    print("  Sources :")
    for s, n in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        print(f"    {s:<32} {n:>5}")
    print(sep)
    print(f"  Fichiers -> {DST_DIR}")
    print(sep)


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare le dataset LLM pour le Pular")
    parser.add_argument("--stats",        action="store_true",
                        help="Affiche les stats du dataset existant sans le regenerer")
    parser.add_argument("--min-qualite",  default="bronze",
                        choices=["gold", "silver", "bronze"],
                        help="Qualite minimum des exemples a inclure (defaut: bronze)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Graine aleatoire pour la reproductibilite (defaut: 42)")
    parser.add_argument("--root",         type=str, default=None,
                        help="Chemin racine du projet (defaut: auto-detect)")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.stats:
        if DST_STATS.exists():
            with open(DST_STATS, encoding="utf-8") as f:
                afficher_stats(json.load(f))
        else:
            print("Aucun dataset généré encore. Lance sans --stats d'abord.")
    else:
        tous  = assembler(qualite_min=args.min_qualite)
        stats = sauver_splits(tous)
        afficher_stats(stats)
        print("\nDataset pret pour Google Colab !")
        print(f"  Upload {DST_DIR} sur Google Drive -> pular-ia/datasets/llm/")
