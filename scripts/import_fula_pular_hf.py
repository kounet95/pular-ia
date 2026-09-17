"""
import_fula_pular_hf.py — Ingestion du dataset HuggingFace Fula-pular
Source: https://huggingface.co/datasets/Pullo-Africa-Protagonist/Fula-pular
9 761 paires (audio 16kHz, texte) en pular du Fouta Djallon, ~1,8 Go.

⚠️ Aucune licence n'est déclarée sur ce dataset (pas de tag `license:`, rien
dans le README) — avant un usage en production ou pour l'entraînement d'un
modèle distribué, vérifie les conditions de réutilisation auprès de l'auteur
(Pullo-Africa-Protagonist sur HuggingFace), comme le rappelaient déjà les
scripts d'import précédents (Coran fulani) pour leurs propres sources.

Sortie compatible avec le pipeline existant (même format que
telegram_scraper.py) :
  - Audio  → corpus-pular/processed/hf_fula_pular/audio/*.wav
  - JSONL  → corpus-pular/processed/hf_fula_pular/jsonl/fula_pular_hf.jsonl
    (lu automatiquement par build_dataset.py, comme les sorties Telegram)

Usage:
    python scripts/import_fula_pular_hf.py
    python scripts/import_fula_pular_hf.py --limite 200   # tester sur 200 exemples
"""

import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/import_fula_pular_hf.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DATASET_ID  = "Pullo-Africa-Protagonist/Fula-pular"
DOSSIER_OUT = Path("./corpus-pular/processed/hf_fula_pular")
DOSSIER_AUDIO = DOSSIER_OUT / "audio"
DOSSIER_JSONL = DOSSIER_OUT / "jsonl"
FICHIER_JSONL = DOSSIER_JSONL / "fula_pular_hf.jsonl"
FICHIER_PROGRES = DOSSIER_OUT / "progres.json"


def hash_texte(texte: str) -> str:
    return hashlib.md5(texte.encode("utf-8")).hexdigest()


def charger_progres() -> set:
    if FICHIER_PROGRES.exists():
        return set(json.loads(FICHIER_PROGRES.read_text(encoding="utf-8")).get("traites", []))
    return set()


def sauvegarder_progres(traites: set):
    FICHIER_PROGRES.write_text(json.dumps({"traites": list(traites)}, ensure_ascii=False), encoding="utf-8")


def importer(limite: int | None, forcer: bool):
    for d in [DOSSIER_AUDIO, DOSSIER_JSONL]:
        d.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset, Audio
    import soundfile as sf
    import io

    log.info(f"Chargement du dataset {DATASET_ID} (streaming)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    # decode=False : évite le décodage automatique de `datasets`, qui exige
    # torchcodec (dépendance native FFmpeg peu fiable sous Windows) — on
    # récupère les octets bruts et on décode nous-mêmes avec soundfile
    # (libsndfile, portable, aucune dépendance FFmpeg).
    ds = ds.cast_column("path", Audio(decode=False))

    deja_traites = set() if forcer else charger_progres()
    mode_ecriture = "w" if forcer or not FICHIER_JSONL.exists() else "a"

    compte = 0
    with open(FICHIER_JSONL, mode_ecriture, encoding="utf-8") as f_out:
        for i, exemple in enumerate(ds):
            if limite and compte >= limite:
                break

            cle = f"fula_pular_hf_{i:05d}"
            if cle in deja_traites:
                continue

            texte = (exemple.get("text") or "").strip()
            audio_brut = (exemple.get("path") or {}).get("bytes")
            tableau, sr = (None, None)
            if audio_brut:
                try:
                    tableau, sr = sf.read(io.BytesIO(audio_brut))
                except Exception as e:
                    log.warning(f"{cle} — décodage audio échoué: {e}")

            if not texte or tableau is None:
                deja_traites.add(cle)
                continue

            fichier_wav = DOSSIER_AUDIO / f"{cle}.wav"
            if not fichier_wav.exists():
                sf.write(str(fichier_wav), tableau, sr)

            f_out.write(json.dumps({
                "fichier":   str(fichier_wav),
                "nom":       f"{cle}_audio",
                "texte":     texte,
                "source":    "hf_fula_pular",
                "langue":    "pular",
                "domaine":   "general",
                "nb_tokens": len(texte.split()),
                "hash":      hash_texte(texte),
                "statut":    "ok",
                "meta": {
                    "dataset": DATASET_ID,
                    "index":   i,
                },
            }, ensure_ascii=False) + "\n")

            deja_traites.add(cle)
            compte += 1
            if compte % 200 == 0:
                sauvegarder_progres(deja_traites)
                log.info(f"{compte} exemples importés...")

    sauvegarder_progres(deja_traites)
    log.info(f"✅ {compte} nouveaux exemples importés → {FICHIER_JSONL}")
    log.info("Prochaine étape : python scripts/build_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="Import du dataset HF Fula-pular")
    parser.add_argument("--limite", type=int, default=None, help="Nombre max d'exemples (test)")
    parser.add_argument("--forcer", action="store_true", help="Réimporter même si déjà fait")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    importer(limite=args.limite, forcer=args.forcer)


if __name__ == "__main__":
    main()
