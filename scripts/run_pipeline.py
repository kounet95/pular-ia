"""
run_pipeline.py — Orchestrateur principal du pipeline pular
Lance toutes les étapes dans l'ordre ou une étape spécifique.

Usage:
    python scripts/run_pipeline.py                        # pipeline complet
    python scripts/run_pipeline.py --etape inventaire
    python scripts/run_pipeline.py --etape transcription
    python scripts/run_pipeline.py --etape ocr
    python scripts/run_pipeline.py --etape pdf
    python scripts/run_pipeline.py --etape dataset
    python scripts/run_pipeline.py --echantillon 500      # test sur 500 fichiers
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


# ── Couleurs terminal ──────────────────────────────────────────────────────────
class C:
    VERT   = "\033[92m"
    JAUNE  = "\033[93m"
    ROUGE  = "\033[91m"
    BLEU   = "\033[94m"
    GRAS   = "\033[1m"
    RESET  = "\033[0m"


def log(msg: str, niveau: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    couleur = {"info": C.BLEU, "ok": C.VERT, "warn": C.JAUNE, "err": C.ROUGE}.get(niveau, "")
    print(f"{couleur}[{ts}] {msg}{C.RESET}")


def print_banniere():
    print(f"""
{C.GRAS}╔══════════════════════════════════════════════════════╗
║         PIPELINE CORPUS PULAR — LLM FOUNDATION      ║
║               by Koune — projetPoular                ║
╚══════════════════════════════════════════════════════╝{C.RESET}
""")


# ── Vérification des dépendances ──────────────────────────────────────────────
def verifier_dependances() -> bool:
    dependances = {
        "whisper":       "pip install openai-whisper",
        "PIL":           "pip install Pillow",
        "pytesseract":   "pip install pytesseract  (+ installer Tesseract-OCR)",
        "pymupdf":       "pip install pymupdf",
        "torch":         "pip install torch",
        "transformers":  "pip install transformers",
        "datasets":      "pip install datasets",
        "peft":          "pip install peft",
        "trl":           "pip install trl",
    }

    manquantes = []
    for module, install in dependances.items():
        try:
            __import__(module)
        except ImportError:
            manquantes.append((module, install))

    if manquantes:
        log(f"⚠️  {len(manquantes)} dépendances manquantes :", "warn")
        for module, install in manquantes:
            print(f"   ❌ {module:15} → {install}")
        print("\n   Lance : pip install -r requirements.txt\n")
        return False

    log("✅ Toutes les dépendances sont présentes", "ok")
    return True


# ── Lancement d'une étape ─────────────────────────────────────────────────────
def lancer_etape(nom: str, commande: list, timeout_s: int = None) -> bool:
    log(f"▶  Étape : {C.GRAS}{nom}{C.RESET}", "info")
    debut = time.time()

    try:
        result = subprocess.run(
            commande,
            check=False,
            timeout=timeout_s,
        )
        elapsed = time.time() - debut

        if result.returncode == 0:
            log(f"✅ {nom} terminé en {elapsed/60:.1f} min", "ok")
            return True
        else:
            log(f"❌ {nom} échoué (code {result.returncode})", "err")
            return False

    except subprocess.TimeoutExpired:
        log(f"⏱️  {nom} — timeout dépassé", "warn")
        return False
    except KeyboardInterrupt:
        log(f"⏹️  {nom} — interrompu par l'utilisateur", "warn")
        sys.exit(0)


# ── Affichage du statut actuel ────────────────────────────────────────────────
def afficher_statut():
    log("📊 Statut du pipeline :", "info")
    print()

    checks = [
        ("Inventaire",     Path("./corpus-pular/metadata/rapport_inventaire.json")),
        ("Transcriptions", Path("./corpus-pular/metadata/progres_transcription.json")),
        ("OCR images",     Path("./corpus-pular/metadata/progres_ocr.json")),
        ("Extraction PDF", Path("./corpus-pular/metadata/progres_pdf.json")),
        ("Telegram",       Path("./corpus-pular/processed/telegram/progres.json")),
        ("Dataset",        Path("./corpus-pular/dataset/train.jsonl")),
    ]

    for nom, chemin in checks:
        if chemin.exists():
            stat = chemin.stat()
            modif = datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m %H:%M")

            # Lire infos si disponible
            if chemin.suffix == ".json":
                try:
                    with open(chemin) as f:
                        data = json.load(f)
                    if "traites" in data:
                        info = f"{len(data['traites']):,} traités"
                    elif "total_fichiers" in data:
                        info = f"{data['total_fichiers']:,} fichiers"
                    elif "total_entrees" in data:
                        info = f"{data['total_entrees']:,} entrées | {data.get('tokens_millions', '?')}M tokens"
                    else:
                        info = "ok"
                except Exception:
                    info = "ok"
            elif chemin.suffix == ".jsonl":
                nb_lignes = sum(1 for _ in open(chemin, encoding="utf-8"))
                info = f"{nb_lignes:,} entrées"
            else:
                info = "ok"

            print(f"  {C.VERT}✅{C.RESET} {nom:<20} — {info} (màj: {modif})")
        else:
            print(f"  {C.JAUNE}⏳{C.RESET} {nom:<20} — non commencé")

    print()


# ── Pipeline complet ───────────────────────────────────────────────────────────
def pipeline_complet(args):
    etapes = [
        {
            "nom":      "Inventaire corpus",
            "cmd":      [sys.executable, "scripts/inventaire.py",
                         "--corpus", args.corpus or "./corpus-pular/raw",
                         "--output", "./corpus-pular/metadata"],
            "timeout":  3600,
        },
        {
            "nom":      "Transcription audio",
            "cmd":      [sys.executable, "scripts/transcription.py",
                         "--model",   args.whisper_model,
                         "--workers", str(args.workers)] +
                        (["--echantillon", str(args.echantillon)] if args.echantillon else []),
            "timeout":  None,   # peut prendre des jours pour 150k fichiers
        },
        {
            "nom":      "OCR images",
            "cmd":      [sys.executable, "scripts/ocr_images.py",
                         "--workers", str(args.workers)] +
                        (["--echantillon", str(args.echantillon)] if args.echantillon else []),
            "timeout":  None,
        },
        {
            "nom":      "Extraction PDF",
            "cmd":      [sys.executable, "scripts/extraction_pdf.py",
                         "--workers", str(args.workers)] +
                        (["--echantillon", str(args.echantillon)] if args.echantillon else []),
            "timeout":  None,
        },
        *([{
            "nom":      "Telegram scraping",
            "cmd":      [sys.executable, "scripts/telegram_scraper.py",
                         "--whisper-model", args.whisper_model] +
                        (["--limite", str(args.telegram_limite)] if args.telegram_limite else []) +
                        (["--sans-audio"] if args.telegram_sans_audio else []),
            "timeout":  None,
            "optionnel": True,
        }] if not getattr(args, "sans_telegram", False) else []),
        {
            "nom":      "Construction dataset",
            "cmd":      [sys.executable, "scripts/build_dataset.py",
                         "--min-tokens", str(args.min_tokens)],
            "timeout":  3600,
        },
    ]

    rapport_pipeline = {
        "debut":  datetime.now().isoformat(),
        "etapes": {},
    }

    for etape in etapes:
        ok = lancer_etape(etape["nom"], etape["cmd"], etape["timeout"])
        rapport_pipeline["etapes"][etape["nom"]] = "ok" if ok else "echec"

        if not ok:
            if etape.get("optionnel"):
                log(f"⚠️  Étape optionnelle échouée ({etape['nom']}) — pipeline continue", "warn")
            elif not args.continuer_si_erreur:
                log(f"Pipeline arrêté à l'étape : {etape['nom']}", "err")
                log("Utilise --continuer-si-erreur pour ignorer les erreurs", "warn")
                break

    rapport_pipeline["fin"] = datetime.now().isoformat()

    rapport_path = Path("./corpus-pular/metadata/rapport_pipeline.json")
    with open(rapport_path, "w") as f:
        json.dump(rapport_pipeline, f, indent=2)

    log(f"\n📋 Rapport pipeline sauvegardé : {rapport_path}", "ok")


# ── Étape unique ───────────────────────────────────────────────────────────────
def etape_unique(nom: str, args):
    commandes = {
        "inventaire": [sys.executable, "scripts/inventaire.py",
                       "--corpus", args.corpus or "./corpus-pular/raw"],
        "transcription": [sys.executable, "scripts/transcription.py",
                          "--model", args.whisper_model,
                          "--workers", str(args.workers)] +
                         (["--echantillon", str(args.echantillon)] if args.echantillon else []),
        "ocr": [sys.executable, "scripts/ocr_images.py",
                "--workers", str(args.workers)] +
               (["--echantillon", str(args.echantillon)] if args.echantillon else []),
        "pdf": [sys.executable, "scripts/extraction_pdf.py",
                "--workers", str(args.workers)] +
               (["--echantillon", str(args.echantillon)] if args.echantillon else []),
        "telegram": [sys.executable, "scripts/telegram_scraper.py",
                     "--whisper-model", args.whisper_model] +
                    (["--limite", str(args.telegram_limite)] if args.telegram_limite else []) +
                    (["--sans-audio"] if args.telegram_sans_audio else []),
        "dataset": [sys.executable, "scripts/build_dataset.py",
                    "--min-tokens", str(args.min_tokens)],
    }

    if nom not in commandes:
        log(f"Étape inconnue : {nom}. Choix : {', '.join(commandes)}", "err")
        return

    lancer_etape(nom, commandes[nom])


def main():
    Path("logs").mkdir(exist_ok=True)
    print_banniere()

    parser = argparse.ArgumentParser(description="Orchestrateur pipeline corpus pular")
    parser.add_argument("--etape",         default=None,      help="Étape : inventaire/transcription/ocr/pdf/telegram/dataset")
    parser.add_argument("--corpus",        default=None,      help="Dossier racine du corpus source")
    parser.add_argument("--whisper-model", default="large-v3",help="Modèle Whisper (tiny/base/small/medium/large-v3)")
    parser.add_argument("--workers",       default=4,type=int,help="Workers parallèles (OCR/PDF)")
    parser.add_argument("--min-tokens",    default=20,type=int,help="Tokens minimum par entrée dataset")
    parser.add_argument("--echantillon",   default=None,type=int, help="N fichiers pour test rapide")
    # Options Telegram
    parser.add_argument("--telegram-limite",      default=None, type=int,
                        help="Nb max messages par canal Telegram (défaut: tous)")
    parser.add_argument("--telegram-sans-audio",  action="store_true",
                        help="Scraper Telegram textes uniquement, sans télécharger les audios")
    parser.add_argument("--sans-telegram",        action="store_true",
                        help="Exclure l'étape Telegram du pipeline complet")
    # Contrôle général
    parser.add_argument("--statut",        action="store_true", help="Afficher le statut du pipeline")
    parser.add_argument("--continuer-si-erreur", action="store_true",
                        help="Continuer le pipeline même si une étape échoue")
    parser.add_argument("--verifier-deps", action="store_true", help="Vérifier les dépendances Python")

    args = parser.parse_args()

    if args.verifier_deps:
        verifier_dependances()
        return

    if args.statut:
        afficher_statut()
        return

    if args.etape:
        etape_unique(args.etape, args)
    else:
        log("🚀 Lancement du pipeline complet", "ok")
        if args.echantillon:
            log(f"⚠️  Mode test — {args.echantillon} fichiers par type", "warn")
        pipeline_complet(args)

    afficher_statut()


if __name__ == "__main__":
    main()
