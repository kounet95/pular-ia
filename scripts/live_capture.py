"""
live_capture.py — Capture et transcription en temps réel des lives TikTok / Facebook / YouTube

Usage:
    python scripts/live_capture.py --url "https://www.facebook.com/xxx/videos/xxx"
    python scripts/live_capture.py --url "https://www.tiktok.com/@xxx/live"
    python scripts/live_capture.py --url "https://www.youtube.com/watch?v=xxx"
    python scripts/live_capture.py --url URL --duree 3600    # limiter à 1h
    python scripts/live_capture.py --url URL --segment 30   # segments de 30s

Plateformes supportées: Facebook Live, TikTok Live, YouTube Live, et tout ce que yt-dlp supporte.

PRÉREQUIS:
    pip install yt-dlp
    ffmpeg installé sur le système
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/live_capture.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DOSSIER_LIVES   = Path("./corpus-pular/community/lives")
DOSSIER_SEGM    = Path("./corpus-pular/community/lives/segments")
DOSSIER_TRANS   = Path("./corpus-pular/community/lives/transcriptions")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_BOT", "base")

for d in [DOSSIER_SEGM, DOSSIER_TRANS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Vérifications dépendances ─────────────────────────────────────────────────
def verifier_deps():
    erreurs = []
    for cmd in [["yt-dlp", "--version"], ["ffmpeg", "-version"]]:
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            erreurs.append(cmd[0])
    if erreurs:
        log.error(f"❌ Dépendances manquantes: {', '.join(erreurs)}")
        log.error("   pip install yt-dlp")
        log.error("   winget install ffmpeg  (Windows)")
        sys.exit(1)
    log.info("✅ yt-dlp et ffmpeg trouvés")

# ── Extraction URL du flux live ───────────────────────────────────────────────
def obtenir_url_flux(url_live: str) -> str | None:
    """Utilise yt-dlp pour extraire l'URL du flux audio du live."""
    log.info(f"Résolution du flux: {url_live}")
    try:
        result = subprocess.run(
            ["yt-dlp", "-g", "--no-playlist",
             "-f", "bestaudio/best",
             "--no-warnings", url_live],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            log.error(f"yt-dlp error: {result.stderr.strip()}")
            return None
        flux_url = result.stdout.strip().split("\n")[0]
        log.info(f"✅ Flux trouvé: {flux_url[:80]}...")
        return flux_url
    except subprocess.TimeoutExpired:
        log.error("Timeout lors de la résolution du flux")
        return None

# ── Téléchargement d'un segment ───────────────────────────────────────────────
def telecharger_segment(flux_url: str, duree: int, chemin_sortie: Path) -> bool:
    """Capture `duree` secondes du flux audio et le sauvegarde en WAV 16kHz."""
    cmd = [
        "ffmpeg", "-y",
        "-i", flux_url,
        "-t", str(duree),
        "-ar", "16000", "-ac", "1",
        "-vn",
        str(chemin_sortie),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=duree + 30)
        return chemin_sortie.exists() and chemin_sortie.stat().st_size > 1000
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.error(f"Erreur capture segment: {e}")
        return False

# ── Transcription d'un segment ────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        log.info(f"Chargement Whisper '{WHISPER_MODEL}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model

def transcrire_segment(audio_path: Path) -> dict | None:
    try:
        model  = get_whisper()
        result = model.transcribe(str(audio_path), language="ff", task="transcribe")
        texte  = result["text"].strip()
        if not texte:
            return None
        return {
            "texte": texte,
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        log.error(f"Erreur Whisper sur {audio_path.name}: {e}")
        return None

# ── Sauvegarde d'un segment transcrit ────────────────────────────────────────
def sauver_transcription(session_id: str, idx: int, url_live: str,
                          audio_path: Path, transcription: dict):
    entry = {
        "session_id": session_id,
        "segment":    idx,
        "url_live":   url_live,
        "audio":      str(audio_path),
        "texte":      transcription["texte"],
        "segments":   transcription["segments"],
        "timestamp":  datetime.now().isoformat(),
        "source":     "live_capture",
    }
    chemin = DOSSIER_TRANS / f"{session_id}_{idx:04d}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    # Afficher en temps réel
    print(f"\n[{idx:04d}] 📝 {transcription['texte']}\n")
    log.info(f"Segment {idx} transcrit ({len(transcription['texte'])} chars)")

# ── Boucle principale de capture ─────────────────────────────────────────────
_continuer = True

def arreter(sig, frame):
    global _continuer
    log.info("Signal d'arrêt reçu — arrêt après le segment en cours...")
    _continuer = False

signal.signal(signal.SIGINT,  arreter)
signal.signal(signal.SIGTERM, arreter)

def capturer_live(url_live: str, duree_segment: int = 60, duree_max: int | None = None):
    """Capture le live en segments, transcrit chaque segment en continu."""
    verifier_deps()

    session_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    debut_global = time.time()
    idx          = 0

    log.info(f"=== Session live {session_id} ===")
    log.info(f"URL     : {url_live}")
    log.info(f"Segment : {duree_segment}s")
    if duree_max:
        log.info(f"Durée max: {duree_max}s")

    # Résoudre l'URL du flux une seule fois (re-résoudre si elle expire)
    flux_url = obtenir_url_flux(url_live)
    if not flux_url:
        log.error("❌ Impossible d'accéder au live. Vérifie que le live est actif.")
        return

    while _continuer:
        # Limite de durée globale
        if duree_max and (time.time() - debut_global) >= duree_max:
            log.info("Durée maximale atteinte.")
            break

        idx += 1
        log.info(f"▶️  Capture segment {idx}...")
        audio_path = DOSSIER_SEGM / f"{session_id}_{idx:04d}.wav"

        ok = telecharger_segment(flux_url, duree_segment, audio_path)
        if not ok:
            log.warning("Segment vide ou erreur — tentative de re-résolution du flux...")
            flux_url = obtenir_url_flux(url_live)
            if not flux_url:
                log.error("❌ Le live semble terminé.")
                break
            continue

        # Transcription dans un thread séparé pour ne pas bloquer la capture
        def traiter(ap=audio_path, i=idx):
            tr = transcrire_segment(ap)
            if tr:
                sauver_transcription(session_id, i, url_live, ap, tr)
            else:
                log.info(f"Segment {i}: silence ou langue non détectée, ignoré.")

        threading.Thread(target=traiter, daemon=True).start()

    log.info(f"=== Session {session_id} terminée — {idx} segments capturés ===")

    # Rapport de session
    rapport = {
        "session_id":  session_id,
        "url_live":    url_live,
        "nb_segments": idx,
        "duree_s":     int(time.time() - debut_global),
        "timestamp":   datetime.now().isoformat(),
    }
    with open(DOSSIER_LIVES / f"rapport_{session_id}.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    log.info(f"Rapport sauvé: rapport_{session_id}.json")

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Capture et transcription en temps réel des lives TikTok/Facebook/YouTube"
    )
    parser.add_argument("--url",     required=True, help="URL du live (Facebook, TikTok, YouTube...)")
    parser.add_argument("--segment", type=int, default=60,  help="Durée d'un segment en secondes (défaut: 60)")
    parser.add_argument("--duree",   type=int, default=None, help="Durée totale maximale en secondes")
    args = parser.parse_args()

    capturer_live(
        url_live      = args.url,
        duree_segment = args.segment,
        duree_max     = args.duree,
    )

if __name__ == "__main__":
    main()
