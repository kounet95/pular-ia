"""
coran_pular.py — Recherche de versets coraniques en pular
Fusionne, verset par verset, les deux traductions téléchargées par
import_coran_fulani.py :
  - fulani_rwwad     : traduction littérale (courte)
  - fulani_mokhtasar : Tafsir Al-Mukhtasar (explication détaillée)

Expose une recherche simple (référence "sourate:verset" ou mots-clés dans
le texte pular) réutilisable par l'API web (community_webapp.py).

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

_index_liste: list[dict] | None = None
_index_dict:  dict[tuple, dict] | None = None


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
        index.append({
            "sourate":     int(sura),
            "verset":      int(aya),
            "arabe":       l.get("arabic_text") or t.get("arabic_text") or "",
            "traduction":  l.get("translation", ""),
            "explication": t.get("translation", ""),
        })

    DOSSIER_META.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    return index


def _charger() -> tuple[list[dict], dict[tuple, dict]]:
    global _index_liste, _index_dict
    if _index_liste is None:
        _index_liste = construire_index()
        _index_dict  = {(v["sourate"], v["verset"]): v for v in _index_liste}
    return _index_liste, _index_dict


def obtenir_verset(sourate: int, verset: int) -> dict | None:
    """Retourne un verset précis (arabe + traduction + explication)."""
    _, index_dict = _charger()
    return index_dict.get((sourate, verset))


_REF_RE = re.compile(r"^\s*(\d{1,3})\s*[:,.]\s*(\d{1,3})\s*$")


def rechercher_versets(q: str, n: int = 10) -> list[dict]:
    """
    Recherche des versets :
      - "2:255" (ou "2,255" / "2.255") → renvoie directement ce verset précis.
      - texte libre → recherche par mots-clés (insensible à la casse) dans
        la traduction littérale ET l'explication (Tafsir), en pular.
    """
    q = (q or "").strip()
    if not q:
        return []

    m = _REF_RE.match(q)
    if m:
        v = obtenir_verset(int(m.group(1)), int(m.group(2)))
        return [v] if v else []

    index_liste, _ = _charger()
    q_lower = q.lower()
    resultats = []
    for v in index_liste:
        if q_lower in v["traduction"].lower() or q_lower in v["explication"].lower():
            resultats.append(v)
            if len(resultats) >= n:
                break
    return resultats


def stats_coran() -> dict:
    index_liste, _ = _charger()
    return {
        "total_versets":       len(index_liste),
        "avec_traduction":     sum(1 for v in index_liste if v["traduction"]),
        "avec_explication":    sum(1 for v in index_liste if v["explication"]),
    }
