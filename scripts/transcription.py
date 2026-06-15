"""
transcription.py — Transcription audio → texte avec Whisper
Traite tous les fichiers audio du corpus pular en parallèle.

Usage:
    python scripts/transcription.py
    python scripts/transcription.py --workers 4 --model large-v3
    python scripts/transcription.py --liste ./corpus-pular/metadata/liste_audio.txt
    python scripts/transcription.py --echantillon 500    # tester sur 500 fichiers d'abord
"""

import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/transcription.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────────
EXTENSIONS_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus", ".webm"}
DOSSIER_OUTPUT   = Path("./corpus-pular/processed/transcriptions")
DOSSIER_RAW      = Path("./corpus-pular/processed/telegram/audio")
FICHIER_PROGRES  = Path("./corpus-pular/metadata/progres_transcription.json")

# Pular/Fula ("ff") n'est pas supporté par Whisper → détection automatique
LANGUE_PULAR = None


# ── Chargement modèle (une seule fois, partagé entre threads) ──────────────────
_whisper_model = None

def get_modele(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        import whisper, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Chargement Whisper {model_name} sur {device.upper()}...")
        _whisper_model = whisper.load_model(model_name, device=device)
        log.info(f"✅ Modèle Whisper prêt ({device.upper()})")
    return _whisper_model


# ── Gestion progression (reprendre où on s'est arrêté) ────────────────────────
def charger_progres() -> set:
    """Retourne les chemins déjà traités (pour reprendre un traitement interrompu)."""
    if FICHIER_PROGRES.exists():
        with open(FICHIER_PROGRES, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("traites", []))
    return set()


def sauvegarder_progres(traites: set):
    FICHIER_PROGRES.parent.mkdir(parents=True, exist_ok=True)
    with open(FICHIER_PROGRES, "w", encoding="utf-8") as f:
        json.dump({"traites": list(traites), "derniere_maj": datetime.now().isoformat()}, f)


# ── Transcription d'un fichier unique ─────────────────────────────────────────
def transcrire_fichier(chemin: Path, model_name: str) -> dict:
    """
    Transcrit un fichier audio et retourne un dict avec le texte et les segments.
    Retourne None si le fichier est déjà traité ou s'il y a une erreur.
    """
    output_path = DOSSIER_OUTPUT / (chemin.stem + ".json")

    # Déjà traité → skip
    if output_path.exists():
        return {"statut": "deja_traite", "fichier": str(chemin)}

    debut = time.time()
    try:
        model = get_modele(model_name)

        import torch
        sur_gpu = torch.cuda.is_available()
        result = model.transcribe(
            str(chemin),
            language=LANGUE_PULAR,
            task="transcribe",
            verbose=False,
            # beam_size=1 = décodage glouton : évite le crash sur segments courts
            # et est 3-5x plus rapide que beam_size=5 sur CPU
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.3,
            initial_prompt="Pular fulfulde fulani langue africaine.",
            fp16=sur_gpu,
        )

        duree = time.time() - debut
        taille = chemin.stat().st_size

        sortie = {
            "fichier":       str(chemin),
            "nom":           chemin.name,
            "texte":         result["text"].strip(),
            "langue_detect": result.get("language", "inconnu"),
            "segments":      [
                {
                    "debut":      round(s["start"], 2),
                    "fin":        round(s["end"], 2),
                    "texte":      s["text"].strip(),
                    "confiance":  round(s.get("avg_logprob", 0), 4),
                    "sans_voix":  s.get("no_speech_prob", 0) > 0.5,
                }
                for s in result["segments"]
            ],
            "meta": {
                "modele":            model_name,
                "duree_traitement_s": round(duree, 2),
                "taille_fichier_oct": taille,
                "date":              datetime.now().isoformat(),
                "nb_segments":       len(result["segments"]),
                "nb_tokens":         len(result["text"].split()),
            },
            "statut": "ok",
        }

        # Sauvegarder immédiatement
        DOSSIER_OUTPUT.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sortie, f, ensure_ascii=False, indent=2)

        return sortie

    except Exception as e:
        log.error(f"ERREUR {chemin.name}: {e}")
        return {
            "fichier": str(chemin),
            "statut":  "erreur",
            "erreur":  str(e),
            "trace":   traceback.format_exc(),
        }


# ── Collecte des fichiers à traiter ───────────────────────────────────────────
def collecter_fichiers(dossier: str = None, liste_txt: str = None, echantillon: int = None) -> list[Path]:
    fichiers = []

    if liste_txt:
        with open(liste_txt, encoding="utf-8") as f:
            fichiers = [Path(line.strip()) for line in f if line.strip()]
        log.info(f"Fichiers chargés depuis liste : {len(fichiers):,}")
    elif dossier:
        dossier = Path(dossier)
        for ext in EXTENSIONS_AUDIO:
            fichiers.extend(dossier.rglob(f"*{ext}"))
        log.info(f"Fichiers trouvés dans {dossier} : {len(fichiers):,}")
    else:
        # Chercher dans le dossier par défaut ET dans raw/
        for ext in EXTENSIONS_AUDIO:
            fichiers.extend(DOSSIER_RAW.rglob(f"*{ext}"))
        log.info(f"Fichiers trouvés dans corpus : {len(fichiers):,}")

    # Filtrer les fichiers inexistants
    fichiers = [f for f in fichiers if f.exists()]

    # Échantillon aléatoire si demandé
    if echantillon and echantillon < len(fichiers):
        import random
        random.shuffle(fichiers)
        fichiers = fichiers[:echantillon]
        log.info(f"Mode échantillon : {echantillon} fichiers sélectionnés")

    return fichiers


# ── Pipeline principal ─────────────────────────────────────────────────────────
def lancer_transcription(fichiers: list[Path], model_name: str, workers: int):
    deja_traites = charger_progres()
    a_traiter    = [f for f in fichiers if str(f) not in deja_traites]

    log.info(f"Total fichiers     : {len(fichiers):,}")
    log.info(f"Déjà traités       : {len(deja_traites):,}")
    log.info(f"À transcrire       : {len(a_traiter):,}")
    log.info(f"Modèle Whisper     : {model_name}")
    log.info(f"Workers parallèles : {workers}")

    if not a_traiter:
        log.info("✅ Tous les fichiers ont déjà été traités !")
        return

    stats = {"ok": 0, "erreur": 0, "deja_traite": 0}
    debut_global = time.time()
    traites = set(deja_traites)

    # Note: Whisper n'est pas thread-safe avec GPU.
    # Avec workers=1 on est séquentiel mais sûr.
    # Avec workers>1, utiliser CPU uniquement.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(transcrire_fichier, f, model_name): f
            for f in a_traiter
        }

        for i, future in enumerate(as_completed(futures), 1):
            fichier = futures[future]
            try:
                result = future.result()
                statut = result.get("statut", "erreur")
                stats[statut if statut in stats else "erreur"] += 1

                if statut == "ok":
                    traites.add(str(fichier))
                    tokens = result["meta"]["nb_tokens"]
                    duree  = result["meta"]["duree_traitement_s"]
                    log.info(f"[{i:>6}/{len(a_traiter):,}] ✅ {fichier.name} — {tokens} mots en {duree:.1f}s")

                elif statut == "erreur":
                    log.warning(f"[{i:>6}/{len(a_traiter):,}] ❌ {fichier.name}")

            except Exception as e:
                stats["erreur"] += 1
                log.error(f"Future exception pour {fichier}: {e}")

            # Sauvegarder la progression toutes les 100 transcriptions
            if i % 100 == 0:
                sauvegarder_progres(traites)
                elapsed = time.time() - debut_global
                vitesse = i / elapsed if elapsed > 0 else 0
                restant = (len(a_traiter) - i) / vitesse if vitesse > 0 else 0
                log.info(
                    f"📊 Progression: {i:,}/{len(a_traiter):,} | "
                    f"Vitesse: {vitesse:.1f} fichiers/s | "
                    f"Restant: {restant/60:.0f} min"
                )

    # Sauvegarde finale
    sauvegarder_progres(traites)

    elapsed = time.time() - debut_global
    log.info("\n" + "═" * 50)
    log.info(f"  TRANSCRIPTION TERMINÉE")
    log.info(f"  ✅ Succès   : {stats['ok']:,}")
    log.info(f"  ❌ Erreurs  : {stats['erreur']:,}")
    log.info(f"  ⏩ Skippés  : {stats['deja_traite']:,}")
    log.info(f"  ⏱️  Durée    : {elapsed/60:.1f} minutes")
    log.info("═" * 50)


def main():
    parser = argparse.ArgumentParser(description="Transcription audio pular avec Whisper")
    parser.add_argument("--corpus",     default=None,          help="Dossier contenant les audios")
    parser.add_argument("--liste",      default=None,          help="Fichier .txt avec un chemin audio par ligne")
    parser.add_argument("--model",      default="large-v3",    help="Modèle Whisper (tiny/base/small/medium/large-v3)")
    parser.add_argument("--workers",    default=1, type=int,   help="Workers parallèles (1 avec GPU, 4+ avec CPU)")
    parser.add_argument("--echantillon",default=None,type=int, help="Traiter seulement N fichiers (test)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)

    log.info("=" * 50)
    log.info("  PIPELINE TRANSCRIPTION PULAR")
    log.info(f"  Démarré : {datetime.now().isoformat()}")
    log.info("=" * 50)

    fichiers = collecter_fichiers(
        dossier=args.corpus,
        liste_txt=args.liste,
        echantillon=args.echantillon,
    )

    if not fichiers:
        log.error("❌ Aucun fichier audio trouvé. Vérifie le chemin avec --corpus")
        return

    lancer_transcription(fichiers, model_name=args.model, workers=args.workers)
    log.info("\n✅ Prochaine étape : python scripts/ocr_images.py")


if __name__ == "__main__":
    main()
