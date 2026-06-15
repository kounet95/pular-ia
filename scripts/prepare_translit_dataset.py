"""
prepare_translit_dataset.py — Prépare le dataset Latin↔Adlam pour fine-tuner mT5

Sources supportées (place tes fichiers dans corpus-pular/raw/translit/):
  *.csv   → colonnes: latin,adlam  OU  texte_latin,texte_adlam  OU  source,cible
  *.json  → liste de {"latin": "...", "adlam": "..."} OU dict {"mot": "traduction"}
  *.jsonl → une paire par ligne {"latin": "...", "adlam": "..."}
  *.html  → extrait les textes des balises avec data-latin/data-adlam
  *.txt   → deux colonnes séparées par TAB: latin<TAB>adlam

Les transcriptions existantes sont converties automatiquement (argent/silver labels).
Les corrections communautaires sont incluses avec la priorité la plus haute.

Usage:
    python scripts/prepare_translit_dataset.py
    python scripts/prepare_translit_dataset.py --min-chars 3 --max-chars 200
    python scripts/prepare_translit_dataset.py --no-silver

Sortie:
    corpus-pular/dataset/translit/train.jsonl
    corpus-pular/dataset/translit/val.jsonl
    corpus-pular/dataset/translit/test.jsonl
    corpus-pular/dataset/translit/stats.json
"""

import json
import re
import csv
import sys
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Ajouter le dossier scripts dans le path pour importer adlam.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adlam import latin_vers_adlam, adlam_vers_latin, est_adlam

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/prepare_dataset.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
PROJET_ROOT         = Path(__file__).resolve().parent.parent
DOSSIER_TRANSLIT    = PROJET_ROOT / "corpus-pular" / "raw" / "translit"
DOSSIER_TRANSCRIPTS = PROJET_ROOT / "corpus-pular" / "processed" / "transcriptions"
DOSSIER_CORRECTIONS = PROJET_ROOT / "corpus-pular" / "community" / "corrections"
DOSSIER_CONTRIB     = PROJET_ROOT / "corpus-pular" / "community" / "contributions"
DOSSIER_SORTIE      = PROJET_ROOT / "corpus-pular" / "dataset" / "translit"

DOSSIER_TRANSLIT.mkdir(parents=True, exist_ok=True)
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# ── Filtre qualité ─────────────────────────────────────────────────────────────
# Mots à filtrer (hallucinations Whisper courantes)
HALLUCINATIONS = {
    "kwa", "merci", "sous-titres", "sous-titrage", "transcription",
    "subtitles", "subtitle", "amara", "doublage", "merci d'avoir",
    "thank you", "thanks for watching",
}

def est_hallucination(texte: str) -> bool:
    """Détecte les hallucinations Whisper typiques."""
    mots = texte.lower().split()
    if len(mots) < 2:
        return True
    # Plus de 50% du texte répétitif = hallucination
    compteur = defaultdict(int)
    for m in mots:
        compteur[m] += 1
    max_freq = max(compteur.values())
    if max_freq / len(mots) > 0.4:
        return True
    # Mots connus d'hallucination
    if any(h in mots for h in HALLUCINATIONS):
        return True
    return False

def nettoyer_texte(texte: str) -> str:
    """Nettoie le texte latin (normalise espaces, retire caractères parasites)."""
    if not texte:
        return ""
    texte = texte.strip()
    # Normaliser espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    # Retirer URL
    texte = re.sub(r"https?://\S+", "", texte)
    # Retirer emojis et symboles
    texte = re.sub(r"[🛑⭐️🗣📊✅❌⏩⏱️🙏]", "", texte)
    return texte.strip()

def valider_paire(latin: str, adlam: str, min_chars: int, max_chars: int) -> bool:
    """Vérifie qu'une paire Latin/Adlam est utilisable pour l'entraînement."""
    if not latin or not adlam:
        return False
    if len(latin) < min_chars or len(latin) > max_chars:
        return False
    if not est_adlam(adlam):
        return False
    if est_hallucination(latin):
        return False
    return True

def formater_entree(latin: str) -> str:
    """Format d'entrée pour mT5 : préfixe de tâche."""
    return f"translitere en adlam: {latin}"

# ══════════════════════════════════════════════════════════════════════════════
# Extracteurs par format
# ══════════════════════════════════════════════════════════════════════════════

def extraire_csv(chemin: Path) -> list[dict]:
    """Extrait les paires depuis un fichier CSV."""
    paires = []
    with open(chemin, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        # Détecter les colonnes
        col_latin = next((h for h in headers if h in ("latin", "texte_latin", "source", "fr", "roman", "romanise")), None)
        col_adlam = next((h for h in headers if h in ("adlam", "texte_adlam", "cible", "ad", "adlm")), None)

        if not col_latin or not col_adlam:
            log.warning(f"CSV {chemin.name}: colonnes non reconnues {headers}. Essai avec la 1ère et 2ème colonne.")
            col_latin = headers[0] if headers else None
            col_adlam = headers[1] if len(headers) > 1 else None

        if not col_latin or not col_adlam:
            log.error(f"CSV {chemin.name}: impossible de trouver les colonnes latin/adlam")
            return paires

        for row in reader:
            # Normaliser les noms de colonnes
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            latin = row_lower.get(col_latin, "").strip()
            adlam = row_lower.get(col_adlam, "").strip()
            if latin and adlam:
                paires.append({"latin": latin, "adlam": adlam, "source": chemin.name})

    log.info(f"CSV {chemin.name}: {len(paires)} paires")
    return paires


def extraire_json(chemin: Path) -> list[dict]:
    """Extrait les paires depuis un fichier JSON."""
    paires = []
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)

    # Format: liste de {"latin": ..., "adlam": ...}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                latin = item.get("latin") or item.get("texte_latin") or item.get("roman") or item.get("source") or ""
                adlam = item.get("adlam") or item.get("texte_adlam") or item.get("adlm") or item.get("cible") or ""
                if latin and adlam:
                    paires.append({"latin": str(latin).strip(), "adlam": str(adlam).strip(), "source": chemin.name})

    # Format: dict mot → traduction
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, str):
                # Deviner lequel est Latin/Adlam
                if est_adlam(k) and not est_adlam(v):
                    paires.append({"latin": v.strip(), "adlam": k.strip(), "source": chemin.name})
                elif not est_adlam(k) and est_adlam(v):
                    paires.append({"latin": k.strip(), "adlam": v.strip(), "source": chemin.name})

    log.info(f"JSON {chemin.name}: {len(paires)} paires")
    return paires


def extraire_jsonl(chemin: Path) -> list[dict]:
    """Extrait les paires depuis un fichier JSONL (une paire JSON par ligne)."""
    paires = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                item = json.loads(ligne)
                latin = item.get("latin") or item.get("texte_latin") or item.get("input") or ""
                adlam = item.get("adlam") or item.get("texte_adlam") or item.get("target") or ""
                # Si input contient le préfixe mT5, l'extraire
                if latin.startswith("translitere en adlam:"):
                    latin = latin[len("translitere en adlam:"):].strip()
                if latin and adlam:
                    paires.append({"latin": latin.strip(), "adlam": adlam.strip(), "source": chemin.name})
            except json.JSONDecodeError:
                pass

    log.info(f"JSONL {chemin.name}: {len(paires)} paires")
    return paires


def extraire_html(chemin: Path) -> list[dict]:
    """
    Extrait les paires depuis un HTML de site de traduction.
    Cherche les attributs data-latin/data-adlam, ou les patterns courants.
    """
    paires = []
    try:
        from html.parser import HTMLParser

        class ExtracteurPaires(HTMLParser):
            def __init__(self):
                super().__init__()
                self.paires = []
                self._latin_buf = []
                self._adlam_buf = []
                self._in_latin = False
                self._in_adlam = False

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                # data-latin / data-adlam
                if "data-latin" in attrs_dict and "data-adlam" in attrs_dict:
                    self.paires.append({
                        "latin": attrs_dict["data-latin"].strip(),
                        "adlam": attrs_dict["data-adlam"].strip(),
                    })
                # Classe "latin" ou "adlam"
                classes = attrs_dict.get("class", "")
                if "latin" in classes:
                    self._in_latin = True
                    self._latin_buf = []
                elif "adlam" in classes or "adlm" in classes:
                    self._in_adlam = True
                    self._adlam_buf = []

            def handle_endtag(self, tag):
                if self._in_latin:
                    self._in_latin = False
                if self._in_adlam:
                    self._in_adlam = False
                    if self._latin_buf and self._adlam_buf:
                        self.paires.append({
                            "latin": "".join(self._latin_buf).strip(),
                            "adlam": "".join(self._adlam_buf).strip(),
                        })

            def handle_data(self, data):
                if self._in_latin:
                    self._latin_buf.append(data)
                elif self._in_adlam:
                    self._adlam_buf.append(data)

        texte_html = chemin.read_text(encoding="utf-8", errors="replace")
        extracteur = ExtracteurPaires()
        extracteur.feed(texte_html)

        for p in extracteur.paires:
            paires.append({**p, "source": chemin.name})

        # Fallback: extraire toutes les paires de textes Adlam
        if not paires:
            mots_adlam = re.findall(r"[𞤀-𞥟]+", texte_html)
            if mots_adlam:
                log.info(f"HTML {chemin.name}: trouvé {len(mots_adlam)} mots Adlam bruts (sans paires)")

    except Exception as e:
        log.error(f"HTML {chemin.name}: erreur {e}")

    log.info(f"HTML {chemin.name}: {len(paires)} paires")
    return paires


def extraire_txt(chemin: Path) -> list[dict]:
    """Extrait les paires depuis un fichier texte (latin<TAB>adlam)."""
    paires = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#"):
                continue
            parties = ligne.split("\t")
            if len(parties) >= 2:
                latin, adlam = parties[0].strip(), parties[1].strip()
                if est_adlam(adlam) and not est_adlam(latin):
                    paires.append({"latin": latin, "adlam": adlam, "source": chemin.name})
                elif est_adlam(latin) and not est_adlam(adlam):
                    paires.append({"latin": adlam, "adlam": latin, "source": chemin.name})

    log.info(f"TXT {chemin.name}: {len(paires)} paires")
    return paires


# ── Source 2: Silver labels depuis les transcriptions ────────────────────────
def extraire_transcriptions_silver(min_chars: int, max_chars: int) -> list[dict]:
    """
    Génère des paires (silver) depuis les transcriptions Whisper existantes.
    La conversion Latin→Adlam est faite par la table de règles (non-parfaite, mais utile).
    Utilise le texte complet ET les segments pour maximiser la quantité de données.
    """
    paires = []
    if not DOSSIER_TRANSCRIPTS.exists():
        return paires

    fichiers = list(DOSSIER_TRANSCRIPTS.glob("*.json"))
    log.info(f"Transcriptions: {len(fichiers)} fichiers")

    for chemin in fichiers:
        with open(chemin, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("statut") != "ok":
            continue

        # 1. Texte complet (niveau fichier)
        texte_complet = data.get("texte", "").strip()
        if texte_complet:
            texte_complet = nettoyer_texte(texte_complet)
            if min_chars <= len(texte_complet) <= max_chars and not est_hallucination(texte_complet):
                paires.append({
                    "latin":   texte_complet,
                    "adlam":   latin_vers_adlam(texte_complet),
                    "source":  "silver_transcription",
                    "qualite": "silver",
                })

        # 2. Segments individuels (phrases plus courtes = meilleur pour l'entraînement)
        for seg in data.get("segments", []):
            texte = seg.get("texte", "").strip()
            if not texte:
                continue
            # Pas de filtrage sur sans_voix — certains sont mal marqués
            # Seulement filtrer les confiances très basses (< -1.5)
            confiance = seg.get("confiance", -1.0)
            if confiance < -1.5:
                continue

            texte = nettoyer_texte(texte)
            if len(texte) < min_chars or len(texte) > max_chars:
                continue
            if est_hallucination(texte):
                continue

            paires.append({
                "latin":   texte,
                "adlam":   latin_vers_adlam(texte),
                "source":  "silver_transcription",
                "qualite": "silver",
            })

    log.info(f"Silver labels depuis transcriptions: {len(paires)} exemples")
    return paires


# ── Source 3: Corrections communautaires (or/gold) ───────────────────────────
def extraire_corrections_gold() -> list[dict]:
    """
    Extrait les corrections communautaires validées.
    Ces paires sont de qualité 'or' — elles ont été vérifiées par un locuteur natif.
    """
    paires = []

    # Corrections de transcriptions
    if DOSSIER_CORRECTIONS.exists():
        for chemin in DOSSIER_CORRECTIONS.glob("*.json"):
            with open(chemin, encoding="utf-8") as f:
                data = json.load(f)
            texte = data.get("texte_corrige", "").strip()
            if texte and not est_hallucination(texte):
                texte = nettoyer_texte(texte)
                paires.append({
                    "latin":  texte,
                    "adlam":  latin_vers_adlam(texte),
                    "source": "correction_communautaire",
                    "qualite": "gold",
                })

    # Contributions directes de la web app
    if DOSSIER_CONTRIB.exists():
        for chemin in DOSSIER_CONTRIB.glob("*.json"):
            with open(chemin, encoding="utf-8") as f:
                data = json.load(f)
            texte = data.get("texte_final", "").strip()
            if texte and not est_hallucination(texte):
                texte = nettoyer_texte(texte)
                paires.append({
                    "latin":  texte,
                    "adlam":  latin_vers_adlam(texte),
                    "source": "contribution_webapp",
                    "qualite": "gold",
                })

    log.info(f"Gold labels depuis communauté: {len(paires)} paires")
    return paires


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def deduplicater(paires: list[dict]) -> list[dict]:
    """Déduplique par texte latin (garde la version de meilleure qualité)."""
    priorite = {"gold": 3, "silver": 1, None: 2}  # None = données scraping
    vus = {}

    for p in paires:
        cle = p["latin"].lower().strip()
        if cle not in vus:
            vus[cle] = p
        else:
            # Garder la version gold si disponible
            prio_nouveau  = priorite.get(p.get("qualite"), 2)
            prio_existant = priorite.get(vus[cle].get("qualite"), 2)
            if prio_nouveau > prio_existant:
                vus[cle] = p

    return list(vus.values())


def diviser_dataset(paires: list[dict], ratio_val=0.05, ratio_test=0.05):
    """Divise en train/val/test de façon reproductible."""
    random.seed(42)
    random.shuffle(paires)
    n = len(paires)
    n_val  = max(1, int(n * ratio_val))
    n_test = max(1, int(n * ratio_test))
    test  = paires[:n_test]
    val   = paires[n_test:n_test + n_val]
    train = paires[n_test + n_val:]
    return train, val, test


def sauver_split(paires: list[dict], chemin: Path, is_train: bool):
    """Sauvegarde un split au format JSONL (entrée mT5)."""
    with open(chemin, "w", encoding="utf-8") as f:
        for p in paires:
            # Format d'entraînement mT5
            exemple = {
                "input":  formater_entree(p["latin"]),
                "target": p["adlam"],
                "latin":  p["latin"],
                "source": p.get("source", "?"),
                "qualite": p.get("qualite", "scraping"),
            }
            f.write(json.dumps(exemple, ensure_ascii=False) + "\n")
    log.info(f"{'Train' if is_train else 'Val/Test'}: {len(paires)} exemples → {chemin.name}")


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset Latin↔Adlam")
    parser.add_argument("--min-chars",  type=int, default=3,   help="Longueur min du texte latin")
    parser.add_argument("--max-chars",  type=int, default=300, help="Longueur max du texte latin")
    parser.add_argument("--no-silver",  action="store_true",   help="Pas de silver labels depuis transcriptions")
    parser.add_argument("--val-ratio",  type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  PRÉPARATION DATASET LATIN ↔ ADLAM")
    log.info("=" * 55)
    log.info(f"Dossier données scraping: {DOSSIER_TRANSLIT}")
    log.info(f"Longueur texte: {args.min_chars} – {args.max_chars} caractères")

    toutes_paires = []
    stats_sources = {}

    # ── 1. Données de scraping web ────────────────────────────────────────────
    extracteurs = {
        "*.csv":   extraire_csv,
        "*.json":  extraire_json,
        "*.jsonl": extraire_jsonl,
        "*.html":  extraire_html,
        "*.htm":   extraire_html,
        "*.txt":   extraire_txt,
    }
    total_scraping = 0
    for pattern, extracteur in extracteurs.items():
        for fichier in DOSSIER_TRANSLIT.glob(pattern):
            paires = extracteur(fichier)
            total_scraping += len(paires)
            toutes_paires.extend(paires)

    if total_scraping == 0:
        log.info(f"ℹ️  Aucune donnée scraping dans {DOSSIER_TRANSLIT}")
        log.info(f"   → Place tes fichiers CSV/JSON/HTML là avec les colonnes latin,adlam")
    stats_sources["scraping"] = total_scraping

    # ── 2. Silver labels depuis transcriptions ────────────────────────────────
    if not args.no_silver:
        silver = extraire_transcriptions_silver(args.min_chars, args.max_chars)
        toutes_paires.extend(silver)
        stats_sources["silver"] = len(silver)
    else:
        stats_sources["silver"] = 0

    # ── 3. Gold labels depuis la communauté ───────────────────────────────────
    gold = extraire_corrections_gold()
    toutes_paires.extend(gold)
    stats_sources["gold"] = len(gold)

    log.info(f"\nTotal brut: {len(toutes_paires)} paires")
    log.info(f"  Scraping : {stats_sources['scraping']}")
    log.info(f"  Silver   : {stats_sources['silver']}")
    log.info(f"  Gold     : {stats_sources['gold']}")

    if not toutes_paires:
        log.error("❌ Aucune paire trouvée. Vérifie tes fichiers dans corpus-pular/raw/translit/")
        return

    # ── 4. Validation et filtrage ─────────────────────────────────────────────
    avant = len(toutes_paires)
    toutes_paires = [
        p for p in toutes_paires
        if valider_paire(p["latin"], p["adlam"], args.min_chars, args.max_chars)
    ]
    log.info(f"Après filtrage qualité: {len(toutes_paires)} ({avant - len(toutes_paires)} rejetés)")

    # ── 5. Déduplication ──────────────────────────────────────────────────────
    avant = len(toutes_paires)
    toutes_paires = deduplicater(toutes_paires)
    log.info(f"Après déduplication: {len(toutes_paires)} ({avant - len(toutes_paires)} doublons supprimés)")

    if not toutes_paires:
        log.error("❌ Aucune paire valide après filtrage.")
        return

    # ── 6. Division train/val/test ────────────────────────────────────────────
    train, val, test = diviser_dataset(toutes_paires, args.val_ratio, args.test_ratio)

    # ── 7. Sauvegarde ─────────────────────────────────────────────────────────
    sauver_split(train, DOSSIER_SORTIE / "train.jsonl", is_train=True)
    sauver_split(val,   DOSSIER_SORTIE / "val.jsonl",   is_train=False)
    sauver_split(test,  DOSSIER_SORTIE / "test.jsonl",  is_train=False)

    stats = {
        "total":   len(toutes_paires),
        "train":   len(train),
        "val":     len(val),
        "test":    len(test),
        "sources": stats_sources,
    }
    with open(DOSSIER_SORTIE / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info("\n" + "=" * 55)
    log.info("  DATASET PRÊT")
    log.info(f"  Train: {len(train):,}")
    log.info(f"  Val  : {len(val):,}")
    log.info(f"  Test : {len(test):,}")
    log.info(f"  → {DOSSIER_SORTIE}")
    log.info("=" * 55)
    log.info("\nProchaine étape:")
    log.info("  python scripts/finetune_mt5.py")

    # Aperçu de 5 exemples
    log.info("\nAperçu (5 exemples du train):")
    for ex in train[:5]:
        log.info(f"  LATIN : {ex['latin'][:60]}")
        log.info(f"  ADLAM : {ex['adlam'][:60]}")
        log.info(f"  Source: {ex.get('source')} | Qualité: {ex.get('qualite', '?')}")
        log.info("")


if __name__ == "__main__":
    main()
