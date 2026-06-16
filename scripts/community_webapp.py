"""
community_webapp.py — Application web de contribution communautaire Pular
À partager pendant les lives TikTok / Facebook pour collecter des données

Usage:
    python scripts/community_webapp.py
    → http://localhost:8000

Pour accès public pendant un live:
    pip install ngrok
    ngrok http 8000
    → Partage l'URL https://xxx.ngrok.io dans les commentaires du live
"""

import os
import json
import uuid
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/community_webapp.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = int(os.getenv("WEBAPP_PORT", 8000))
DOSSIER_CONTRIB = Path("./corpus-pular/community/contributions")
DOSSIER_AUDIO   = Path("./corpus-pular/community/audio")
FICHIER_STATS   = Path("./corpus-pular/community/stats.json")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_BOT", "base")

for d in [DOSSIER_CONTRIB, DOSSIER_AUDIO]:
    d.mkdir(parents=True, exist_ok=True)

# ── Whisper (chargé uniquement au premier appel, jamais au démarrage) ─────────
_whisper_model  = None
_whisper_chargement = False

def get_whisper():
    global _whisper_model, _whisper_chargement
    if _whisper_model is None and not _whisper_chargement:
        _whisper_chargement = True
        import whisper, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Chargement Whisper '{WHISPER_MODEL}' sur {device.upper()}...")
        _whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        log.info(f"✅ Whisper prêt ({device.upper()})")
        _whisper_chargement = False
    return _whisper_model

def transcrire(audio_path: str) -> dict:
    model = get_whisper()
    result = model.transcribe(
        audio_path,
        task="transcribe",
        # Seuil bas = Whisper transcrit même si la parole est peu claire
        no_speech_threshold=0.3,
        # Indice de contexte pour orienter la détection vers le pular/français
        initial_prompt="Pular fulfulde fulani langue africaine.",
        # Désactiver la compression de log (évite les hallucinations silencieuses)
        logprob_threshold=-1.5,
        condition_on_previous_text=False,
        fp16=False,
    )
    texte = result["text"].strip()
    langue = result.get("language", "?")
    log.info(f"Langue détectée: {langue} | texte: '{texte[:80]}'")
    return {
        "text": texte,
        "language": langue,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ],
    }

# ── Stats ─────────────────────────────────────────────────────────────────────
def charger_stats() -> dict:
    if FICHIER_STATS.exists():
        with open(FICHIER_STATS, encoding="utf-8") as f:
            return json.load(f)
    return {"total_contributions": 0, "total_validations": 0, "contributeurs": {}}

def sauver_stats(stats: dict):
    with open(FICHIER_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Pular IA — Contribution communautaire")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemin absolu du projet (indépendant du répertoire de lancement)
PROJET_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH   = PROJET_ROOT / "web" / "index.html"

# ── Page principale ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    log.info(f"Chargement HTML: {HTML_PATH}")
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    log.error(f"❌ index.html introuvable: {HTML_PATH}")
    return HTMLResponse(
        "<h1 style='font-family:sans-serif;color:#1a6b3c'>Pular IA</h1>"
        f"<p>Fichier introuvable: {HTML_PATH}</p>",
        status_code=200,
    )

# ── API: Transcription ─────────────────────────────────────────────────────────
@app.post("/api/transcrire")
async def api_transcrire(audio: UploadFile = File(...)):
    ext      = Path(audio.filename).suffix or ".webm"
    tmp      = Path(tempfile.mktemp(suffix=ext))
    wav_path = tmp.with_suffix(".wav")

    try:
        contenu = await audio.read()
        log.info(f"Audio reçu: {len(contenu)} octets | format: {ext}")
        if len(contenu) < 100:
            raise HTTPException(400, "Fichier audio vide ou trop court.")
        tmp.write_bytes(contenu)

        # Conversion WAV dans un thread (ne bloque pas l'event loop)
        log.info("Conversion ffmpeg en cours...")
        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(wav_path)],
            capture_output=True, check=True,
        )
        log.info(f"Conversion OK — {wav_path.stat().st_size} octets")

        # Transcription Whisper dans un thread (lent sur CPU, ne pas bloquer)
        log.info("Transcription Whisper en cours (peut prendre 30-60s sur CPU)...")
        resultat = await asyncio.to_thread(transcrire, str(wav_path))

        texte = resultat["text"]
        log.info(f"Transcription OK: '{texte[:80]}'")

        if not texte:
            return JSONResponse({"ok": True, "text": "", "text_adlam": "",
                                 "segments": [],
                                 "warning": "Aucun texte détecté — parle plus près du micro."})
        texte_adlam = latin_vers_adlam(texte)
        return JSONResponse({"ok": True, **resultat, "text_adlam": texte_adlam})

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else ""
        log.error(f"Erreur ffmpeg: {err[:300]}")
        raise HTTPException(500, f"Conversion audio échouée: {err[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Erreur transcription: {type(e).__name__}: {e}")
        raise HTTPException(500, str(e))
    finally:
        tmp.unlink(missing_ok=True)
        if wav_path.exists():
            wav_path.unlink()

# ── API: Valider/sauvegarder ───────────────────────────────────────────────────
@app.post("/api/valider")
async def api_valider(
    audio:        UploadFile = File(...),
    transcription: str       = Form(...),
    texte_final:  str        = Form(...),
    pseudo:       str        = Form("anonyme"),
):
    """Sauvegarde la contribution validée dans le corpus."""
    contrib_id = str(uuid.uuid4())[:8]
    ext = Path(audio.filename).suffix or ".webm"
    audio_path = DOSSIER_AUDIO / f"{contrib_id}{ext}"
    audio_path.write_bytes(await audio.read())

    entry = {
        "id": contrib_id,
        "pseudo": pseudo[:50],
        "transcription_auto": transcription,
        "texte_final": texte_final.strip(),
        "audio": str(audio_path),
        "timestamp": datetime.now().isoformat(),
        "source": "community_webapp",
    }

    with open(DOSSIER_CONTRIB / f"{contrib_id}.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    stats = charger_stats()
    stats["total_contributions"] += 1
    stats["total_validations"]   += 1
    uid = pseudo[:50]
    stats["contributeurs"].setdefault(uid, {"nom": uid, "contributions": 0})
    stats["contributeurs"][uid]["contributions"] += 1
    sauver_stats(stats)

    log.info(f"Contribution web sauvée: {contrib_id} | pseudo={pseudo}")
    return JSONResponse({"ok": True, "id": contrib_id})

# ══════════════════════════════════════════════════════════════════════════════
# ADLAM — Conversion Latin ↔ Adlam + transcription MMS
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(PROJET_ROOT / "scripts"))
from adlam import latin_vers_adlam, adlam_vers_latin, est_adlam, CLAVIER_ADLAM

@app.get("/api/clavier-adlam")
async def api_clavier_adlam():
    """Retourne la disposition du clavier Adlam pour le frontend."""
    return JSONResponse(CLAVIER_ADLAM)

@app.post("/api/convertir")
async def api_convertir(texte: str = Form(...), vers: str = Form("adlam")):
    """Convertit Latin → Adlam ou Adlam → Latin."""
    if vers == "adlam":
        return JSONResponse({"ok": True, "resultat": latin_vers_adlam(texte), "script": "adlam"})
    else:
        return JSONResponse({"ok": True, "resultat": adlam_vers_latin(texte), "script": "latin"})

# ── API: Stats publiques ───────────────────────────────────────────────────────
@app.get("/api/stats")
async def api_stats():
    stats = charger_stats()
    corrections = list(DOSSIER_CORRECTIONS.glob("*.json"))
    return JSONResponse({
        "total_contributions": stats["total_contributions"],
        "total_validations":   stats["total_validations"],
        "nb_contributeurs":    len(stats["contributeurs"]),
        "total_corrections":   len(corrections),
    })

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION DES TRANSCRIPTIONS EXISTANTES
# ══════════════════════════════════════════════════════════════════════════════

DOSSIER_TRANSCRIPTIONS = PROJET_ROOT / "corpus-pular" / "processed" / "transcriptions"
DOSSIER_CORRECTIONS    = PROJET_ROOT / "corpus-pular" / "community" / "corrections"
FICHIER_SAUTS          = PROJET_ROOT / "corpus-pular" / "community" / "sauts.json"

DOSSIER_CORRECTIONS.mkdir(parents=True, exist_ok=True)

def charger_sauts() -> set:
    if FICHIER_SAUTS.exists():
        with open(FICHIER_SAUTS, encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def sauver_saut(nom: str):
    sauts = charger_sauts()
    sauts.add(nom)
    with open(FICHIER_SAUTS, "w", encoding="utf-8") as f:
        json.dump(list(sauts), f)

def transcriptions_a_corriger() -> list[Path]:
    """Retourne les JSON de transcription pas encore corrigés ni sautés."""
    if not DOSSIER_TRANSCRIPTIONS.exists():
        return []
    corrigees = {p.stem for p in DOSSIER_CORRECTIONS.glob("*.json")}
    sauts     = charger_sauts()
    faites    = corrigees | sauts
    return [
        p for p in sorted(DOSSIER_TRANSCRIPTIONS.glob("*.json"))
        if p.stem not in faites
    ]

# ── Servir un fichier audio du corpus (pour l'audio player) ──────────────────
from fastapi.responses import FileResponse

@app.get("/audio/{nom_fichier:path}")
async def servir_audio(nom_fichier: str):
    # Chercher dans tous les dossiers audio du corpus
    dossiers = [
        PROJET_ROOT / "corpus-pular" / "processed" / "telegram" / "audio",
        PROJET_ROOT / "corpus-pular" / "raw" / "audio",
        PROJET_ROOT / "corpus-pular" / "community" / "audio",
    ]
    for dossier in dossiers:
        chemin = dossier / nom_fichier
        if chemin.exists():
            return FileResponse(str(chemin))
    raise HTTPException(404, f"Audio introuvable: {nom_fichier}")

# ── Prochaine transcription à corriger ───────────────────────────────────────
@app.get("/api/a-corriger")
async def api_a_corriger():
    liste = transcriptions_a_corriger()
    if not liste:
        total = len(list(DOSSIER_TRANSCRIPTIONS.glob("*.json"))) if DOSSIER_TRANSCRIPTIONS.exists() else 0
        return JSONResponse({
            "ok": True,
            "fini": True,
            "message": "Toutes les transcriptions ont été corrigées! Baŋ-baŋ 🙏",
            "total": total,
        })
    fichier = liste[0]
    with open(fichier, encoding="utf-8") as f:
        data = json.load(f)
    total   = len(list(DOSSIER_TRANSCRIPTIONS.glob("*.json"))) if DOSSIER_TRANSCRIPTIONS.exists() else 0
    restant = len(liste)
    # Nom du fichier audio (juste le nom, pas le chemin complet)
    audio_nom = Path(data.get("fichier", "")).name
    return JSONResponse({
        "ok":         True,
        "fini":       False,
        "id":         fichier.stem,
        "nom":        data.get("nom", fichier.stem),
        "texte_auto": data.get("texte", ""),
        "audio_nom":  audio_nom,
        "duree_s":    data.get("duree_s", 0),
        "restant":    restant,
        "total":      total,
        "fait":       total - restant,
    })

# ── Soumettre une correction ──────────────────────────────────────────────────
@app.post("/api/corriger")
async def api_corriger(
    id:           str = Form(...),
    texte_auto:   str = Form(...),
    texte_corrige: str = Form(...),
    pseudo:       str = Form("anonyme"),
    action:       str = Form("corriger"),   # "corriger" | "sauter" | "impossible"
):
    if action == "sauter":
        sauver_saut(id)
        return JSONResponse({"ok": True, "action": "sauter"})

    if action == "impossible":
        sauver_saut(id)
        log.info(f"Marqué impossible: {id}")
        return JSONResponse({"ok": True, "action": "impossible"})

    texte_final = texte_corrige.strip()
    if not texte_final:
        raise HTTPException(400, "Le texte corrigé est vide.")

    entry = {
        "id":            id,
        "pseudo":        pseudo[:50],
        "texte_auto":    texte_auto,
        "texte_corrige": texte_final,
        "timestamp":     datetime.now().isoformat(),
        "source":        "community_correction",
    }
    with open(DOSSIER_CORRECTIONS / f"{id}.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    # Mise à jour stats
    stats = charger_stats()
    stats.setdefault("total_corrections", 0)
    stats["total_corrections"] += 1
    uid = pseudo[:50]
    stats["contributeurs"].setdefault(uid, {"nom": uid, "contributions": 0})
    stats["contributeurs"][uid]["contributions"] += 1
    sauver_stats(stats)

    log.info(f"Correction sauvée: {id} | pseudo={pseudo}")
    return JSONResponse({"ok": True, "id": id})

# ══════════════════════════════════════════════════════════════════════════════
# RAG — Livres, poèmes, articles en pular
# ══════════════════════════════════════════════════════════════════════════════

from rag_livres import (
    extraire_texte, indexer_livre, rechercher as rag_rechercher,
    charger_index, sauver_index, stats_rag, exporter_dataset,
    DOSSIER_RAW as LIVRES_RAW,
)

EXTENSIONS_ACCEPTEES = {".pdf", ".txt", ".docx", ".doc", ".html", ".htm", ".md"}

@app.post("/api/upload-livre")
async def api_upload_livre(
    fichier: UploadFile = File(...),
    titre:   str        = Form(...),
    auteur:  str        = Form("Anonyme"),
    langue:  str        = Form("pular"),
):
    """Reçoit un livre/poème, extrait le texte, l'indexe dans le RAG."""
    ext = Path(fichier.filename).suffix.lower()
    if ext not in EXTENSIONS_ACCEPTEES:
        raise HTTPException(400, f"Format non supporté: {ext}. Acceptés: PDF, TXT, DOCX, HTML, MD")

    contenu = await fichier.read()
    if len(contenu) < 10:
        raise HTTPException(400, "Fichier vide.")

    # Sauvegarder le fichier original
    livre_id  = str(uuid.uuid4())[:8]
    nom_sauve = f"{livre_id}_{fichier.filename}"
    chemin    = LIVRES_RAW / nom_sauve
    chemin.write_bytes(contenu)
    log.info(f"Livre reçu: {fichier.filename} ({len(contenu)} octets)")

    # Extraction + indexation dans un thread (peut être lent)
    try:
        texte = await asyncio.to_thread(extraire_texte, chemin)
        if not texte.strip():
            chemin.unlink(missing_ok=True)
            raise HTTPException(422, "Impossible d'extraire du texte de ce fichier.")

        nb_chunks = await asyncio.to_thread(
            indexer_livre, titre, auteur, langue, texte, livre_id
        )

        # Sauver dans l'index JSON
        livres = charger_index()
        livres.append({
            "id":        livre_id,
            "titre":     titre,
            "auteur":    auteur,
            "langue":    langue,
            "fichier":   nom_sauve,
            "nb_chunks": nb_chunks,
            "nb_chars":  len(texte),
            "date":      datetime.now().isoformat(),
        })
        sauver_index(livres)

        log.info(f"Livre indexé: '{titre}' — {nb_chunks} chunks")
        return JSONResponse({
            "ok":        True,
            "id":        livre_id,
            "nb_chunks": nb_chunks,
            "nb_chars":  len(texte),
            "message":   f"'{titre}' indexé avec succès ({nb_chunks} passages)",
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Erreur indexation livre: {e}")
        raise HTTPException(500, f"Erreur: {str(e)}")

@app.get("/api/livres")
async def api_livres():
    """Liste tous les livres indexés."""
    return JSONResponse(charger_index())

@app.get("/api/rechercher")
async def api_rechercher(q: str, n: int = 5, langue: str = None):
    """Recherche sémantique dans le corpus via RAG."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(400, "Requête trop courte.")
    try:
        resultats = await asyncio.to_thread(rag_rechercher, q, n, langue)
        return JSONResponse({"ok": True, "resultats": resultats, "query": q})
    except Exception as e:
        log.error(f"Erreur RAG recherche: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/rag-stats")
async def api_rag_stats():
    """Statistiques du corpus RAG."""
    return JSONResponse(await asyncio.to_thread(stats_rag))

def _phrases_jeu_sync(n: int) -> list:
    """Récupère n phrases courtes depuis ChromaDB (appelé dans un thread)."""
    import re, random
    from rag_livres import get_collection
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    offset = random.randint(0, max(0, total - n * 6))
    batch  = collection.get(
        limit=n * 6, offset=offset,
        include=["documents", "metadatas"],
    )
    phrases = []
    for doc, meta in zip(batch["documents"], batch["metadatas"]):
        for s in re.split(r"[.!?\n؟।]+", doc):
            s = s.strip()
            if 15 < len(s) < 180:
                phrases.append({
                    "texte":  s,
                    "titre":  meta.get("titre", "?"),
                    "langue": meta.get("langue", "?"),
                })
    random.shuffle(phrases)
    return phrases[:n]

@app.get("/api/phrases-jeu")
async def api_phrases_jeu(n: int = 5):
    """Retourne des phrases courtes issues du RAG pour le mode 'Lire' du jeu."""
    try:
        phrases = await asyncio.to_thread(_phrases_jeu_sync, n)
        return JSONResponse({"ok": True, "phrases": phrases})
    except Exception as e:
        log.warning(f"phrases-jeu: {e}")
        return JSONResponse({"ok": True, "phrases": []})

@app.post("/api/exporter-dataset")
async def api_exporter_dataset():
    """Exporte le corpus RAG en JSONL pour le fine-tuning LLM."""
    try:
        chemin = await asyncio.to_thread(exporter_dataset)
        return JSONResponse({"ok": True, "fichier": str(chemin)})
    except Exception as e:
        raise HTTPException(500, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# ESPACE PROFESSEUR — Mots personnalisés + validation contributions
# ══════════════════════════════════════════════════════════════════════════════

DOSSIER_JEU         = PROJET_ROOT / "corpus-pular" / "jeu"
FICHIER_MOTS_CUSTOM = DOSSIER_JEU / "mots_custom.json"
DOSSIER_JEU.mkdir(parents=True, exist_ok=True)

def charger_mots_custom() -> list[dict]:
    if FICHIER_MOTS_CUSTOM.exists():
        with open(FICHIER_MOTS_CUSTOM, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_mots_custom(mots: list[dict]):
    with open(FICHIER_MOTS_CUSTOM, "w", encoding="utf-8") as f:
        json.dump(mots, f, ensure_ascii=False, indent=2)

@app.get("/api/prof/mots")
async def api_prof_mots():
    """Retourne tous les mots custom ajoutés par les professeurs."""
    return JSONResponse(charger_mots_custom())

@app.post("/api/prof/mot")
async def api_prof_ajouter_mot(
    emoji:  str = Form("❓"),
    fr:     str = Form(...),
    pular:  str = Form(...),
    adlam:  str = Form(""),
    cat:    str = Form("Autre"),
    note:   str = Form(""),
    pseudo: str = Form("prof"),
):
    if not fr.strip() or not pular.strip():
        raise HTTPException(400, "Les champs 'fr' et 'pular' sont obligatoires.")
    mots = charger_mots_custom()
    nouveau = {
        "id":     str(uuid.uuid4())[:8],
        "emoji":  emoji.strip(),
        "fr":     fr.strip(),
        "pular":  pular.strip(),
        "adlam":  adlam.strip(),
        "cat":    cat.strip(),
        "note":   note.strip(),
        "pseudo": pseudo.strip(),
        "date":   datetime.now().isoformat(),
    }
    mots.append(nouveau)
    sauver_mots_custom(mots)
    log.info(f"Mot custom ajouté: {nouveau['pular']} ({pseudo})")
    return JSONResponse({"ok": True, "mot": nouveau})

@app.put("/api/prof/mot/{mot_id}")
async def api_prof_modifier_mot(
    mot_id: str,
    emoji:  str = Form("❓"),
    fr:     str = Form(...),
    pular:  str = Form(...),
    adlam:  str = Form(""),
    cat:    str = Form("Autre"),
    note:   str = Form(""),
):
    mots = charger_mots_custom()
    for m in mots:
        if m["id"] == mot_id:
            m.update({
                "emoji":   emoji.strip(),
                "fr":      fr.strip(),
                "pular":   pular.strip(),
                "adlam":   adlam.strip(),
                "cat":     cat.strip(),
                "note":    note.strip(),
                "modifie": datetime.now().isoformat(),
            })
            sauver_mots_custom(mots)
            return JSONResponse({"ok": True, "mot": m})
    raise HTTPException(404, f"Mot {mot_id} introuvable.")

@app.delete("/api/prof/mot/{mot_id}")
async def api_prof_supprimer_mot(mot_id: str):
    mots = charger_mots_custom()
    avant = len(mots)
    mots = [m for m in mots if m["id"] != mot_id]
    if len(mots) == avant:
        raise HTTPException(404, f"Mot {mot_id} introuvable.")
    sauver_mots_custom(mots)
    log.info(f"Mot supprimé: {mot_id}")
    return JSONResponse({"ok": True})

@app.get("/api/prof/contributions")
async def api_prof_contributions(limit: int = 20):
    """Liste les contributions communautaires pour validation prof."""
    contribs = []
    fichiers = sorted(DOSSIER_CONTRIB.glob("*.json"), reverse=True)[:limit * 3]
    for f in fichiers:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            contribs.append({
                "id":                 d.get("id", f.stem),
                "pseudo":             d.get("pseudo", "?"),
                "date":               d.get("timestamp", ""),
                "texte_final":        d.get("texte_final", ""),
                "transcription_auto": d.get("transcription_auto", ""),
                "audio_nom":          Path(d.get("audio", "")).name,
                "status":             d.get("status", "pending"),
            })
        except Exception:
            pass
    contribs.sort(key=lambda x: (x["status"] != "pending", x["date"]))
    total_pending = sum(1 for c in contribs if c["status"] == "pending")
    return JSONResponse({"contributions": contribs[:limit], "total_pending": total_pending})

@app.post("/api/prof/valider")
async def api_prof_valider(
    id:         str = Form(...),
    action:     str = Form(...),   # "valider" | "corriger" | "rejeter"
    correction: str = Form(""),
):
    """Valide, corrige ou rejette une contribution communautaire."""
    fichier = DOSSIER_CONTRIB / f"{id}.json"
    if not fichier.exists():
        raise HTTPException(404, f"Contribution {id} introuvable.")

    with open(fichier, encoding="utf-8") as f:
        data = json.load(f)

    data["status"]          = action
    data["date_validation"] = datetime.now().isoformat()
    if action == "corriger" and correction.strip():
        data["texte_final"]         = correction.strip()
        data["texte_corrige_prof"]  = correction.strip()

    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    stats = charger_stats()
    stats.setdefault("total_validations", 0)
    if action in ("valider", "corriger"):
        stats["total_validations"] += 1
    sauver_stats(stats)

    log.info(f"Contribution {id}: {action}")
    return JSONResponse({"ok": True, "id": id, "action": action})

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN — Export corpus + génération dataset (pour Google Colab)
# ══════════════════════════════════════════════════════════════════════════════

import io
import zipfile
from fastapi.responses import StreamingResponse

ADMIN_KEY = os.getenv("ADMIN_KEY", "")  # clé secrète définie dans .env / Railway

def _check_admin(key: str):
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(403, "Clé admin incorrecte.")

@app.get("/api/admin/export-corpus")
async def api_export_corpus(key: str = ""):
    """
    Retourne un ZIP de tout le corpus textuel (sans audio).
    Utilisé par Google Colab pour récupérer les données sans upload manuel.
    """
    _check_admin(key)

    def creer_zip() -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            dossiers = {
                "community/contributions": DOSSIER_CONTRIB,
                "community/corrections":   DOSSIER_CORRECTIONS,
                "processed/transcriptions": DOSSIER_TRANSCRIPTIONS,
            }
            for arc_prefix, dossier in dossiers.items():
                if dossier.exists():
                    for f in dossier.glob("*.json"):
                        z.write(f, f"corpus-pular/{arc_prefix}/{f.name}")

            # Dataset translit
            for split in ["train", "val", "test"]:
                f = PROJET_ROOT / "corpus-pular" / "dataset" / "translit" / f"{split}.jsonl"
                if f.exists():
                    z.write(f, f"corpus-pular/dataset/translit/{f.name}")

            # Mots custom
            if FICHIER_MOTS_CUSTOM.exists():
                z.write(FICHIER_MOTS_CUSTOM, "corpus-pular/jeu/mots_custom.json")

            # Dataset livres RAG
            livres_jsonl = PROJET_ROOT / "corpus-pular" / "dataset" / "livres" / "corpus_livres.jsonl"
            if livres_jsonl.exists():
                z.write(livres_jsonl, "corpus-pular/dataset/livres/corpus_livres.jsonl")

        buf.seek(0)
        return buf.read()

    contenu = await asyncio.to_thread(creer_zip)

    # Stats rapides pour le log
    nb_contrib = len(list(DOSSIER_CONTRIB.glob("*.json"))) if DOSSIER_CONTRIB.exists() else 0
    nb_corr    = len(list(DOSSIER_CORRECTIONS.glob("*.json"))) if DOSSIER_CORRECTIONS.exists() else 0
    log.info(f"Export corpus: {len(contenu)//1024} KB | {nb_contrib} contributions | {nb_corr} corrections")

    return StreamingResponse(
        io.BytesIO(contenu),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=corpus-pular.zip"},
    )

@app.get("/api/admin/stats-corpus")
async def api_stats_corpus(key: str = ""):
    """Statistiques du corpus pour Colab (sans téléchargement)."""
    _check_admin(key)

    nb_contrib  = len(list(DOSSIER_CONTRIB.glob("*.json")))    if DOSSIER_CONTRIB.exists()    else 0
    nb_corr     = len(list(DOSSIER_CORRECTIONS.glob("*.json"))) if DOSSIER_CORRECTIONS.exists() else 0
    nb_transcrip= len(list(DOSSIER_TRANSCRIPTIONS.glob("*.json"))) if DOSSIER_TRANSCRIPTIONS.exists() else 0
    nb_mots_cust= len(charger_mots_custom())

    # Compter par status dans contributions
    status_count: dict = {}
    if DOSSIER_CONTRIB.exists():
        for f in DOSSIER_CONTRIB.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                s = d.get("status", "pending")
                status_count[s] = status_count.get(s, 0) + 1
            except Exception:
                pass

    rag = stats_rag()

    return JSONResponse({
        "contributions":       nb_contrib,
        "contributions_status": status_count,
        "corrections":         nb_corr,
        "transcriptions":      nb_transcrip,
        "mots_custom":         nb_mots_cust,
        "rag_chunks":          rag.get("total_chunks", 0),
        "rag_livres":          rag.get("total_livres", 0),
        "timestamp":           datetime.now().isoformat(),
    })

@app.post("/api/admin/generer-dataset")
async def api_generer_dataset(key: str = ""):
    """
    Lance prepare_llm_dataset.py sur le serveur.
    Retourne les stats du dataset généré.
    Appeler depuis Colab après avoir enrichi le corpus.
    """
    _check_admin(key)

    def run_generation() -> dict:
        import subprocess, sys
        script = PROJET_ROOT / "scripts" / "prepare_llm_dataset.py"
        result = subprocess.run(
            [sys.executable, str(script), "--root", str(PROJET_ROOT), "--seed", "42"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
        # Lire les stats
        stats_path = PROJET_ROOT / "corpus-pular" / "dataset" / "llm" / "stats.json"
        if stats_path.exists():
            with open(stats_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    try:
        stats = await asyncio.to_thread(run_generation)
        log.info(f"Dataset généré: {stats.get('total', 0)} exemples")
        return JSONResponse({"ok": True, "stats": stats})
    except Exception as e:
        log.error(f"Erreur génération dataset: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/admin/telecharger-dataset")
async def api_telecharger_dataset(key: str = ""):
    """Retourne le dataset LLM (train/val/test) en ZIP pour Colab."""
    _check_admin(key)

    def creer_zip_dataset() -> bytes:
        buf = io.BytesIO()
        dossier_llm = PROJET_ROOT / "corpus-pular" / "dataset" / "llm"
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fichier in ["train.jsonl", "val.jsonl", "test.jsonl", "stats.json"]:
                chemin = dossier_llm / fichier
                if chemin.exists():
                    z.write(chemin, fichier)
        buf.seek(0)
        return buf.read()

    contenu = await asyncio.to_thread(creer_zip_dataset)
    return StreamingResponse(
        io.BytesIO(contenu),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=dataset_llm_pular.zip"},
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Web app Pular IA → http://localhost:{PORT}")
    log.info("   Pour accès public: ngrok http 8080")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
