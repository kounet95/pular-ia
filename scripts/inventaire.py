"""
inventaire.py — Inventaire complet du corpus pular
Scanne tous les fichiers, génère un rapport détaillé et un index CSV.

Usage:
    python scripts/inventaire.py --corpus ./corpus-pular/raw
    python scripts/inventaire.py --corpus /chemin/vers/tes/fichiers
"""

import os
import csv
import json
import argparse
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict


# Extensions reconnues par type
EXTENSIONS = {
    "audio": {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma", ".opus", ".mp4", ".webm"},
    "image": {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"},
    "pdf":   {".pdf"},
    "texte": {".txt", ".doc", ".docx", ".rtf", ".odt"},
    "autre": set(),
}

# Index inversé extension → type
EXT_TO_TYPE = {}
for type_name, exts in EXTENSIONS.items():
    for ext in exts:
        EXT_TO_TYPE[ext] = type_name


def detecter_type(fichier: Path) -> str:
    ext = fichier.suffix.lower()
    return EXT_TO_TYPE.get(ext, "autre")


def taille_lisible(octets: int) -> str:
    for unite in ["o", "Ko", "Mo", "Go", "To"]:
        if octets < 1024:
            return f"{octets:.1f} {unite}"
        octets /= 1024
    return f"{octets:.1f} Po"


def hash_fichier(chemin: Path, taille_max_hash: int = 10 * 1024 * 1024) -> str:
    """Hash MD5 rapide — lit seulement les premiers 10 Mo pour les gros fichiers."""
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        chunk = f.read(taille_max_hash)
        h.update(chunk)
    return h.hexdigest()


def scanner_corpus(dossier: str, avec_hash: bool = False) -> list[dict]:
    """
    Parcourt récursivement le dossier et retourne la liste de tous les fichiers
    avec leurs métadonnées.
    """
    dossier = Path(dossier)
    if not dossier.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dossier}")

    fichiers = []
    erreurs = []
    total = 0

    print(f"\n🔍 Scan en cours : {dossier.resolve()}")
    print("   (cela peut prendre quelques minutes pour 150k fichiers...)\n")

    for chemin in dossier.rglob("*"):
        if not chemin.is_file():
            continue

        total += 1
        if total % 5000 == 0:
            print(f"   → {total:,} fichiers scannés...")

        try:
            stat = chemin.stat()
            type_fichier = detecter_type(chemin)

            entree = {
                "chemin":      str(chemin.resolve()),
                "nom":         chemin.name,
                "extension":   chemin.suffix.lower(),
                "type":        type_fichier,
                "taille_oct":  stat.st_size,
                "taille":      taille_lisible(stat.st_size),
                "modifie":     datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "dossier":     str(chemin.parent.resolve()),
                "hash_md5":    hash_fichier(chemin) if avec_hash else "",
                "traite":      False,
            }
            fichiers.append(entree)

        except (PermissionError, OSError) as e:
            erreurs.append({"chemin": str(chemin), "erreur": str(e)})

    print(f"\n✅ Scan terminé : {total:,} fichiers trouvés, {len(erreurs)} erreurs")
    return fichiers, erreurs


def generer_rapport(fichiers: list[dict]) -> dict:
    """Génère les statistiques du corpus."""
    par_type     = Counter()
    par_ext      = Counter()
    taille_type  = defaultdict(int)
    taille_total = 0

    for f in fichiers:
        par_type[f["type"]] += 1
        par_ext[f["extension"]] += 1
        taille_type[f["type"]] += f["taille_oct"]
        taille_total += f["taille_oct"]

    # Estimation durée audio (hypothèse ~3 min/fichier en moyenne)
    nb_audio = par_type.get("audio", 0)
    duree_estimee_h = (nb_audio * 3) / 60

    rapport = {
        "date_inventaire":     datetime.now().isoformat(),
        "total_fichiers":      len(fichiers),
        "taille_totale":       taille_lisible(taille_total),
        "taille_totale_oct":   taille_total,
        "par_type":            dict(par_type),
        "par_extension":       dict(par_ext.most_common(30)),
        "taille_par_type":     {k: taille_lisible(v) for k, v in taille_type.items()},
        "estimation_audio": {
            "nb_fichiers":     nb_audio,
            "duree_min_h":     round(nb_audio * 1 / 60, 1),   # hypothèse 1 min/fichier
            "duree_moy_h":     round(duree_estimee_h, 1),      # hypothèse 3 min/fichier
            "duree_max_h":     round(nb_audio * 10 / 60, 1),   # hypothèse 10 min/fichier
        },
        "estimation_tokens": {
            "note": "Estimation brute — à affiner après traitement",
            "audio_tokens_min": nb_audio * 500,
            "audio_tokens_max": nb_audio * 5000,
            "pdf_tokens":       par_type.get("pdf", 0) * 2000,
            "image_tokens":     par_type.get("image", 0) * 200,
        },
    }
    return rapport


def afficher_rapport(rapport: dict):
    print("\n" + "═" * 60)
    print("  📊  RAPPORT D'INVENTAIRE — CORPUS PULAR")
    print("═" * 60)
    print(f"  Date            : {rapport['date_inventaire'][:19]}")
    print(f"  Total fichiers  : {rapport['total_fichiers']:,}")
    print(f"  Taille totale   : {rapport['taille_totale']}")
    print()
    print("  Par type :")
    for type_name, count in sorted(rapport["par_type"].items(), key=lambda x: -x[1]):
        taille = rapport["taille_par_type"].get(type_name, "?")
        print(f"    {type_name:<10} : {count:>8,} fichiers   ({taille})")

    print()
    print("  Extensions les plus fréquentes :")
    for ext, count in list(rapport["par_extension"].items())[:15]:
        print(f"    {ext or '(sans ext)':<12} : {count:>8,}")

    print()
    est = rapport["estimation_audio"]
    print("  Estimation durée audio :")
    print(f"    Min ({est['nb_fichiers']:,} fichiers × 1min)  : {est['duree_min_h']:,} heures")
    print(f"    Moy ({est['nb_fichiers']:,} fichiers × 3min)  : {est['duree_moy_h']:,} heures")
    print(f"    Max ({est['nb_fichiers']:,} fichiers × 10min) : {est['duree_max_h']:,} heures")

    print()
    tok = rapport["estimation_tokens"]
    print("  Estimation tokens (pour le LLM) :")
    print(f"    Audio (après transcription) : {tok['audio_tokens_min']:,} – {tok['audio_tokens_max']:,}")
    print(f"    PDF (extraction texte)      : {tok['pdf_tokens']:,}")
    print(f"    Images (OCR)                : {tok['image_tokens']:,}")
    total_tok_min = tok["audio_tokens_min"] + tok["pdf_tokens"] + tok["image_tokens"]
    total_tok_max = tok["audio_tokens_max"] + tok["pdf_tokens"] + tok["image_tokens"]
    print(f"    ─────────────────────────────────────────")
    print(f"    TOTAL estimé                : {total_tok_min:,} – {total_tok_max:,} tokens")
    print("═" * 60)


def sauvegarder(fichiers: list, rapport: dict, erreurs: list, dossier_output: str):
    dossier_output = Path(dossier_output)
    dossier_output.mkdir(parents=True, exist_ok=True)

    # 1. Index CSV complet (un fichier par ligne)
    csv_path = dossier_output / "index.csv"
    champs = ["chemin", "nom", "extension", "type", "taille", "taille_oct", "modifie", "dossier", "traite"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=champs, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fichiers)
    print(f"\n💾 Index CSV sauvegardé   → {csv_path}")

    # 2. Rapport JSON
    rapport_path = dossier_output / "rapport_inventaire.json"
    with open(rapport_path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f"💾 Rapport JSON sauvegardé → {rapport_path}")

    # 3. Erreurs (si elles existent)
    if erreurs:
        err_path = dossier_output / "erreurs_scan.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(erreurs, f, ensure_ascii=False, indent=2)
        print(f"⚠️  {len(erreurs)} erreurs sauvegardées → {err_path}")

    # 4. Liste par type — pour alimenter les autres scripts
    for type_name in ["audio", "image", "pdf", "texte"]:
        liste = [f["chemin"] for f in fichiers if f["type"] == type_name]
        if liste:
            liste_path = dossier_output / f"liste_{type_name}.txt"
            with open(liste_path, "w", encoding="utf-8") as f:
                f.write("\n".join(liste))
            print(f"💾 Liste {type_name:<8} ({len(liste):,} fichiers) → {liste_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inventaire du corpus pular — scanne et catalogue tous les fichiers"
    )
    parser.add_argument(
        "--corpus",
        default="./corpus-pular/raw",
        help="Dossier racine du corpus à scanner (default: ./corpus-pular/raw)"
    )
    parser.add_argument(
        "--output",
        default="./corpus-pular/metadata",
        help="Dossier de sortie pour l'index et les rapports (default: ./corpus-pular/metadata)"
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Calculer le hash MD5 de chaque fichier (pour déduplication — plus lent)"
    )
    args = parser.parse_args()

    fichiers, erreurs = scanner_corpus(args.corpus, avec_hash=args.hash)
    rapport = generer_rapport(fichiers)
    afficher_rapport(rapport)
    sauvegarder(fichiers, rapport, erreurs, args.output)

    print("\n✅ Inventaire terminé. Prochaine étape : python scripts/transcription.py")


if __name__ == "__main__":
    main()
