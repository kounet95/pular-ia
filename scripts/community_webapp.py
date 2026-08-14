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
import io
import json
import html
import uuid
import wave
import base64
import hashlib
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
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
PORT            = int(os.getenv("PORT", os.getenv("WEBAPP_PORT", 8080)))
DOSSIER_CONTRIB = Path("./corpus-pular/community/contributions")
DOSSIER_AUDIO   = Path("./corpus-pular/community/audio")
FICHIER_STATS   = Path("./corpus-pular/community/stats.json")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL_BOT", "base")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

for d in [DOSSIER_CONTRIB, DOSSIER_AUDIO]:
    d.mkdir(parents=True, exist_ok=True)

# ── Prompt vocabulaire dynamique (alimenté par les mots + documents ajoutés) ──
import time as _time
_prompt_cache:        str   = ""
_prompt_last_refresh: float = 0.0

def construire_prompt_vocabulaire() -> str:
    """
    Construit le initial_prompt Whisper à partir des mots du jeu et des phrases.
    Mis en cache 5 min. Invalidé à chaque ajout de document/mot via
    invalider_cache_prompt().
    """
    global _prompt_cache, _prompt_last_refresh
    now = _time.time()
    if _prompt_cache and (now - _prompt_last_refresh) < 300:
        return _prompt_cache

    mots_uniques: list[str] = []
    vus: set[str] = set()

    def ajouter(w: str):
        w = w.strip(" .,!?;:()'\"").lower()
        if len(w) > 1 and w not in vus:
            vus.add(w)
            mots_uniques.append(w)

    # 1. Mots du jeu (base + custom) — chargés plus tard dans le fichier,
    #    on lit directement les JSON pour éviter la dépendance circulaire.
    DOSSIER_JEU_LOCAL = PROJET_ROOT / "corpus-pular" / "jeu"
    for nom_fichier in ("mots_base.json", "mots_custom.json"):
        p = DOSSIER_JEU_LOCAL / nom_fichier
        if p.exists():
            try:
                for m in json.loads(p.read_text(encoding="utf-8")):
                    pular = m.get("pular", "")
                    if pular:
                        ajouter(pular)
            except Exception:
                pass

    # 2. Phrases (extraire les tokens pular)
    for nom_fichier in ("phrases_base.json", "phrases_custom.json"):
        p = DOSSIER_JEU_LOCAL / nom_fichier
        if p.exists():
            try:
                for ph in json.loads(p.read_text(encoding="utf-8")):
                    for token in ph.get("pular", "").split():
                        ajouter(token)
            except Exception:
                pass

    # 3. Vocabulaire extrait des documents RAG (fichiers _vocab.json)
    DOSSIER_META = PROJET_ROOT / "corpus-pular" / "livres" / "metadata"
    if DOSSIER_META.exists():
        try:
            for vocab_file in sorted(DOSSIER_META.glob("*_vocab.json")):
                mots_doc = json.loads(vocab_file.read_text(encoding="utf-8"))
                for mot in mots_doc:
                    ajouter(mot)
                if len(mots_uniques) >= 200:
                    break
        except Exception:
            pass

    # Groq compte en octets UTF-8 (limite = 896) — les lettres pular comme ɓ ɗ ŋ
    # valent 2 octets chacune, d'où l'écart entre len() Python et le décompte Groq.
    # Le prompt commence par des PHRASES complètes (meilleur signal pour Whisper
    # qu'une liste de mots isolés) puis complète avec le vocabulaire disponible.
    MAX_BYTES = 870
    PHRASES_SEED = [
        "Jam waali? Jam tan, baŋ-baŋ.",
        "Mi yiɗi pular fulfulde.",
        "Hol tò innde maa?",
        "Bismillahi Rahmaani Rahiimi.",
        "Baaba am, yinaande am, ɓiɗɗo am.",
        "Nagge, mbabba, mbewa, puccu.",
        "Ndiyam, jaango, naange, lewru.",
    ]
    base   = " ".join(PHRASES_SEED)
    prompt = base
    nb_mots = 0
    for mot in mots_uniques:
        candidat = f"{prompt} {mot}"
        if len(candidat.encode("utf-8")) > MAX_BYTES:
            break
        prompt = candidat
        nb_mots += 1

    _prompt_cache        = prompt
    _prompt_last_refresh = now
    log.info(f"Prompt vocabulaire: {nb_mots} mots, {len(prompt.encode('utf-8'))} octets UTF-8")
    return prompt

def invalider_cache_prompt():
    """Appeler après tout ajout de mot ou de document pour forcer la reconstruction."""
    global _prompt_last_refresh
    _prompt_last_refresh = 0.0


# ── Transcription via Groq API (production) ───────────────────────────────────
def _transcrire_groq(audio_path: str) -> dict:
    """Groq Whisper large-v3-turbo — ~1s, gratuit, scalable."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    try:
        prompt = construire_prompt_vocabulaire()
    except Exception:
        prompt = "Pular fulfulde Fouta Djallon fulani langue africaine."
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f),
            model="whisper-large-v3-turbo",
            prompt=prompt,
            response_format="verbose_json",
            temperature=0.2,
        )
    texte   = (result.text or "").strip()
    langue  = getattr(result, "language", "?") or "?"
    segs    = getattr(result, "segments", None) or []
    log.info(f"[Groq] langue={langue} | '{texte[:80]}'")

    def _seg(s):
        # groq>=0.9 retourne des objets Pydantic (pas des dicts)
        if isinstance(s, dict):
            return {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
        return {"start": getattr(s, "start", 0), "end": getattr(s, "end", 0), "text": getattr(s, "text", "")}

    return {
        "text":     texte,
        "language": langue,
        "segments": [_seg(s) for s in segs],
    }

# ── Whisper local (fallback dev / pas de clé Groq) ────────────────────────────
_whisper_model      = None
_whisper_chargement = False

def get_whisper():
    global _whisper_model, _whisper_chargement
    if _whisper_model is None and not _whisper_chargement:
        _whisper_chargement = True
        try:
            import whisper, torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Chargement Whisper '{WHISPER_MODEL}' sur {device.upper()}...")
            _whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
            log.info(f"✅ Whisper prêt ({device.upper()})")
        except Exception as e:
            log.error(f"❌ Échec chargement Whisper: {type(e).__name__}: {e}")
            raise
        finally:
            _whisper_chargement = False
    if _whisper_model is None:
        raise RuntimeError("Whisper non disponible — configure GROQ_API_KEY pour la prod")
    return _whisper_model

def _transcrire_local(audio_path: str) -> dict:
    model  = get_whisper()
    prompt = construire_prompt_vocabulaire()
    result = model.transcribe(
        audio_path,
        task="transcribe",
        no_speech_threshold=0.3,
        initial_prompt=prompt,
        temperature=0.2,
        logprob_threshold=-1.5,
        condition_on_previous_text=False,
        fp16=False,
    )
    texte  = result["text"].strip()
    langue = result.get("language", "?")
    log.info(f"[Local] langue={langue} | '{texte[:80]}'")
    return {
        "text":     texte,
        "language": langue,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result.get("segments", [])
        ],
    }

def transcrire(audio_path: str) -> dict:
    if GROQ_API_KEY:
        return _transcrire_groq(audio_path)
    return _transcrire_local(audio_path)

# ── Stats ─────────────────────────────────────────────────────────────────────
def charger_stats() -> dict:
    if FICHIER_STATS.exists():
        with open(FICHIER_STATS, encoding="utf-8") as f:
            return json.load(f)
    return {"total_contributions": 0, "total_validations": 0, "contributeurs": {}}

def sauver_stats(stats: dict):
    with open(FICHIER_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

# ── Métriques de fiabilité ────────────────────────────────────────────────────
def calcul_wer(reference: str, hypothese: str) -> float:
    """Word Error Rate : distance d'édition sur les mots, normalisée par len(reference)."""
    ref = reference.lower().split()
    hyp = hypothese.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            temp = d[j]
            d[j] = prev if ref[i-1] == hyp[j-1] else 1 + min(prev, d[j], d[j-1])
            prev = temp
    return d[m] / n

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

# ── Health check (Railway l'appelle avant de router le trafic) ────────────────
@app.get("/health")
async def health():
    return JSONResponse({"ok": True})

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

        if GROQ_API_KEY:
            # ── Groq : envoie le fichier original, pas besoin de ffmpeg ──────
            log.info("[Groq] Transcription en cours...")
            resultat = await asyncio.to_thread(_transcrire_groq, str(tmp))
        else:
            # ── Whisper local : conversion WAV nécessaire ────────────────────
            log.info("Conversion ffmpeg en cours...")
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(wav_path)],
                capture_output=True, check=True,
            )
            log.info(f"Conversion OK — {wav_path.stat().st_size} octets")
            log.info("Transcription Whisper locale en cours (30-60s sur CPU)...")
            resultat = await asyncio.to_thread(_transcrire_local, str(wav_path))

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
        log.error(f"Erreur transcription: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
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
from arabe import CLAVIER_ARABE

@app.get("/api/clavier-adlam")
async def api_clavier_adlam():
    """Retourne la disposition du clavier Adlam pour le frontend."""
    return JSONResponse(CLAVIER_ADLAM)

@app.get("/api/clavier-arabe")
async def api_clavier_arabe():
    """Retourne la disposition du clavier arabe (ajami) pour le frontend."""
    return JSONResponse(CLAVIER_ARABE)

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
DOSSIER_CORRECTIONS          = PROJET_ROOT / "corpus-pular" / "community" / "corrections"
DOSSIER_CORRECTIONS_PHRASES  = PROJET_ROOT / "corpus-pular" / "community" / "corrections_phrases"
DOSSIER_TTS_CACHE            = PROJET_ROOT / "corpus-pular" / "community" / "tts_cache"
FICHIER_SAUTS                = PROJET_ROOT / "corpus-pular" / "community" / "sauts.json"

DOSSIER_CORRECTIONS.mkdir(parents=True, exist_ok=True)
DOSSIER_CORRECTIONS_PHRASES.mkdir(parents=True, exist_ok=True)
DOSSIER_TTS_CACHE.mkdir(parents=True, exist_ok=True)

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
    get_collection,
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
        invalider_cache_prompt()

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


@app.get("/api/livres/{livre_id}/passages")
async def api_livre_passages(livre_id: str, n: int = 8):
    """Retourne les N premiers passages indexés d'un livre."""
    def _get():
        try:
            col   = get_collection()
            total = col.count()
            if total == 0:
                return []
            res = col.get(
                where={"livre_id": livre_id},
                limit=n,
                include=["documents", "metadatas"],
            )
            docs  = res.get("documents") or []
            metas = res.get("metadatas") or []
            return [
                {"texte": d, "chunk_id": m.get("chunk_id", i)}
                for i, (d, m) in enumerate(zip(docs, metas))
            ]
        except Exception as e:
            log.warning(f"Passages {livre_id}: {e}")
            return []
    passages = await asyncio.to_thread(_get)
    return JSONResponse({"passages": passages})


@app.get("/api/livres/{livre_id}/fichier")
async def api_livre_fichier(livre_id: str):
    """Télécharger le fichier original d'un livre."""
    livres = charger_index()
    livre  = next((l for l in livres if l["id"] == livre_id), None)
    if not livre:
        raise HTTPException(404, "Livre non trouvé.")
    fichier = livre.get("fichier", "")
    chemin  = LIVRES_RAW / fichier if fichier else None
    if not chemin or not chemin.exists():
        raise HTTPException(404, "Fichier original introuvable.")
    from fastapi.responses import FileResponse
    return FileResponse(chemin, filename=chemin.name)


@app.delete("/api/livres/{livre_id}")
async def api_supprimer_livre(livre_id: str):
    """Supprime un livre : index JSON + fichier + chunks ChromaDB."""
    livres = charger_index()
    livre  = next((l for l in livres if l["id"] == livre_id), None)
    if not livre:
        raise HTTPException(404, "Livre non trouvé.")

    # 1. Supprimer le fichier original
    fichier = livre.get("fichier", "")
    if fichier:
        chemin = LIVRES_RAW / fichier
        if chemin.exists():
            chemin.unlink()

    # 2. Supprimer les chunks ChromaDB
    def _suppr_chroma():
        try:
            col = get_collection()
            col.delete(where={"livre_id": livre_id})
            log.info(f"Chunks supprimés pour livre {livre_id}")
        except Exception as e:
            log.warning(f"Suppression chunks: {e}")
    await asyncio.to_thread(_suppr_chroma)

    # 3. Mettre à jour l'index JSON
    livres = [l for l in livres if l["id"] != livre_id]
    sauver_index(livres)

    log.info(f"Livre supprimé: {livre_id} — {livre.get('titre','?')}")
    return JSONResponse({"ok": True})

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

# ══════════════════════════════════════════════════════════════════════════════
# ESPACE HISTOIRE & PATRIMOINE — manuscrits familiaux, thèses, poèmes, familles
# ══════════════════════════════════════════════════════════════════════════════

import espace_histoire as EH

@app.post("/api/histoire/upload")
async def api_histoire_upload(
    fichier:          UploadFile = File(...),
    titre:             str        = Form(...),
    auteur_detenteur:  str        = Form("Anonyme"),
    royaume:           str        = Form("Autre / Général"),
    type_source:       str        = Form("autre"),
    confidentialite:   str        = Form("public"),
    note:              str        = Form(""),
):
    """Reçoit un document historique (manuscrit, thèse, poème...) et l'indexe."""
    ext = Path(fichier.filename).suffix.lower()
    if ext not in EXTENSIONS_ACCEPTEES:
        raise HTTPException(400, f"Format non supporté: {ext}. Acceptés: PDF, TXT, DOCX, HTML, MD")
    if type_source not in EH.TYPES_SOURCE:
        raise HTTPException(400, "Type de source invalide.")
    if confidentialite not in EH.NIVEAUX_CONFIDENTIALITE:
        raise HTTPException(400, "Niveau de confidentialité invalide.")

    contenu = await fichier.read()
    if len(contenu) < 10:
        raise HTTPException(400, "Fichier vide.")

    doc_id    = str(uuid.uuid4())[:8]
    nom_sauve = f"{doc_id}_{fichier.filename}"
    chemin    = EH.DOSSIER_RAW / nom_sauve
    chemin.write_bytes(contenu)
    log.info(f"Histoire — document reçu: {fichier.filename} ({len(contenu)} octets)")

    try:
        texte = await asyncio.to_thread(extraire_texte, chemin)
        if not texte.strip():
            chemin.unlink(missing_ok=True)
            raise HTTPException(422, "Impossible d'extraire du texte de ce fichier.")

        nb_chunks = await asyncio.to_thread(
            EH.indexer_document, titre, auteur_detenteur, royaume, type_source,
            confidentialite, texte, doc_id,
        )

        docs = EH.charger_documents()
        docs.append({
            "id":                doc_id,
            "titre":             titre,
            "auteur_detenteur":  auteur_detenteur,
            "royaume":           royaume,
            "type_source":       type_source,
            "confidentialite":   confidentialite,
            "note":              note,
            "fichier":           nom_sauve,
            "nb_chunks":         nb_chunks,
            "nb_chars":          len(texte),
            "date":              datetime.now().isoformat(),
        })
        EH.sauver_documents(docs)

        log.info(f"Histoire — indexé: '{titre}' — {nb_chunks} chunks")
        return JSONResponse({
            "ok": True, "id": doc_id, "nb_chunks": nb_chunks, "nb_chars": len(texte),
            "message": f"'{titre}' ajouté à l'espace Histoire ({nb_chunks} passages)",
        })
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Erreur indexation document histoire: {e}")
        raise HTTPException(500, f"Erreur: {str(e)}")


@app.get("/api/histoire/documents")
async def api_histoire_documents(key: str = ""):
    """
    Liste les documents. Les documents 'prive' n'exposent que leur fiche
    (titre/royaume/type/note) — jamais le fichier ni les passages — sauf
    à fournir la clé admin.
    """
    docs = EH.charger_documents()
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    if is_admin:
        return JSONResponse(docs)

    allegee = []
    for d in docs:
        if d.get("confidentialite") == "prive":
            allegee.append({
                "id": d["id"], "titre": d["titre"], "royaume": d["royaume"],
                "type_source": d["type_source"], "confidentialite": "prive",
                "note": d.get("note", ""), "date": d["date"],
            })
        else:
            allegee.append(d)
    return JSONResponse(allegee)


@app.get("/api/histoire/documents/{doc_id}/passages")
async def api_histoire_passages(doc_id: str, n: int = 8, key: str = ""):
    """Retourne des passages d'un document (bloqué si privé, sauf admin)."""
    docs = EH.charger_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Document non trouvé.")
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    if doc.get("confidentialite") == "prive" and not is_admin:
        raise HTTPException(403, "Document privé — accès réservé.")

    def _get():
        try:
            col = EH.get_collection()
            if col.count() == 0:
                return []
            res = col.get(where={"doc_id": doc_id}, limit=n, include=["documents", "metadatas"])
            docs_ = res.get("documents") or []
            metas = res.get("metadatas") or []
            tronquer = doc.get("confidentialite") == "sur_demande" and not is_admin
            out = []
            for i, (t, m) in enumerate(zip(docs_, metas)):
                texte = (t[:300] + "…") if tronquer and len(t) > 300 else t
                out.append({"texte": texte, "chunk_no": m.get("chunk_no", i)})
            return out
        except Exception as e:
            log.warning(f"Passages histoire {doc_id}: {e}")
            return []
    passages = await asyncio.to_thread(_get)
    return JSONResponse({"passages": passages})


@app.get("/api/histoire/documents/{doc_id}/fichier")
async def api_histoire_fichier(doc_id: str, key: str = ""):
    """Télécharge le fichier original — uniquement si public, ou admin."""
    docs = EH.charger_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Document non trouvé.")
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    if doc.get("confidentialite") != "public" and not is_admin:
        raise HTTPException(403, "Ce document n'est pas téléchargeable publiquement.")
    chemin = EH.DOSSIER_RAW / doc.get("fichier", "")
    if not chemin.exists():
        raise HTTPException(404, "Fichier original introuvable.")
    return FileResponse(chemin, filename=chemin.name)


@app.delete("/api/histoire/documents/{doc_id}")
async def api_histoire_supprimer(doc_id: str, key: str = ""):
    """Supprime un document (admin uniquement)."""
    _check_admin(key)
    docs = EH.charger_documents()
    doc = next((d for d in docs if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Document non trouvé.")

    fichier = doc.get("fichier", "")
    if fichier:
        chemin = EH.DOSSIER_RAW / fichier
        if chemin.exists():
            chemin.unlink()

    def _suppr_chroma():
        try:
            col = EH.get_collection()
            col.delete(where={"doc_id": doc_id})
        except Exception as e:
            log.warning(f"Suppression chunks histoire: {e}")
    await asyncio.to_thread(_suppr_chroma)

    docs = [d for d in docs if d["id"] != doc_id]
    EH.sauver_documents(docs)
    log.info(f"Histoire — document supprimé: {doc_id} — {doc.get('titre','?')}")
    return JSONResponse({"ok": True})


@app.get("/api/histoire/rechercher")
async def api_histoire_rechercher(q: str, n: int = 5, royaume: str = None, type_source: str = None):
    """Recherche sémantique dans le corpus Histoire (jamais de contenu privé)."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(400, "Requête trop courte.")
    try:
        resultats = await asyncio.to_thread(EH.rechercher, q, n, royaume, type_source)
        return JSONResponse({"ok": True, "resultats": resultats, "query": q})
    except Exception as e:
        log.error(f"Erreur recherche histoire: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/histoire/stats")
async def api_histoire_stats():
    return JSONResponse(await asyncio.to_thread(EH.stats_histoire))


@app.get("/api/histoire/meta")
async def api_histoire_meta():
    """Retourne les listes de royaumes / types de source / niveaux — pour peupler les <select>."""
    return JSONResponse({
        "royaumes": EH.ROYAUMES,
        "types_source": EH.TYPES_SOURCE,
        "niveaux_confidentialite": EH.NIVEAUX_CONFIDENTIALITE,
    })

# ── Annuaire des familles détentrices ───────────────────────────────────────

@app.post("/api/histoire/familles")
async def api_histoire_ajouter_famille(
    nom_famille:     str = Form(...),
    royaume:         str = Form("Autre / Général"),
    lignage:         str = Form(""),
    localisation:    str = Form(""),
    description:     str = Form(...),
    contact:         str = Form(""),
    contact_visible: bool = Form(False),
):
    """Enregistre une famille/lignage détentrice de sources historiques."""
    if not nom_famille.strip() or not description.strip():
        raise HTTPException(400, "Nom de famille et description requis.")
    fiche = await asyncio.to_thread(
        EH.ajouter_famille, nom_famille.strip(), royaume, lignage.strip(),
        localisation.strip(), description.strip(), contact.strip(), contact_visible,
    )
    log.info(f"Histoire — famille ajoutée: {nom_famille} ({royaume})")
    return JSONResponse({"ok": True, "famille": {**fiche, "contact": ("***" if not contact_visible else fiche["contact"])}})


@app.get("/api/histoire/familles")
async def api_histoire_familles(key: str = ""):
    """Liste les familles. Le contact n'est visible que si contact_visible=true ou admin."""
    familles = EH.charger_familles()
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    out = []
    for f in familles:
        f2 = dict(f)
        if not f2.get("contact_visible") and not is_admin:
            f2["contact"] = ""
        out.append(f2)
    return JSONResponse(out)


@app.delete("/api/histoire/familles/{famille_id}")
async def api_histoire_supprimer_famille(famille_id: str, key: str = ""):
    _check_admin(key)
    familles = EH.charger_familles()
    if not any(f["id"] == famille_id for f in familles):
        raise HTTPException(404, "Famille non trouvée.")
    familles = [f for f in familles if f["id"] != famille_id]
    EH.sauver_familles(familles)
    return JSONResponse({"ok": True})

# ══════════════════════════════════════════════════════════════════════════════
# COMPTES UTILISATEURS — inscription email/mot de passe + connexion Telegram
# ══════════════════════════════════════════════════════════════════════════════

import comptes as CP

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
COOKIE_SESSION      = "pular_session"
_bot_username_cache = {"valeur": ""}

async def _bot_username() -> str:
    """Récupère (et met en cache) le @username du bot Telegram via getMe —
    le processus web n'a pas d'instance python-telegram-bot vivante (le bot
    tourne dans un processus séparé, voir start.sh), donc un simple appel
    HTTP à l'API Telegram suffit et évite d'ajouter une dépendance ici."""
    if _bot_username_cache["valeur"]:
        return _bot_username_cache["valeur"]
    if not TELEGRAM_BOT_TOKEN:
        return ""
    def _appel():
        import urllib.request
        with urllib.request.urlopen(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=8
        ) as r:
            return json.loads(r.read())
    try:
        data = await asyncio.to_thread(_appel)
        if data.get("ok"):
            _bot_username_cache["valeur"] = data["result"]["username"]
    except Exception as e:
        log.warning(f"Impossible de récupérer le username du bot Telegram: {e}")
    return _bot_username_cache["valeur"]

def _poser_cookie_session(request: Request, resp: JSONResponse, token: str):
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(
        COOKIE_SESSION, token, max_age=int(CP.DUREE_SESSION.total_seconds()),
        httponly=True, samesite="lax", secure=https, path="/",
    )

async def _compte_courant(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_SESSION, "")
    if not token:
        return None
    return await asyncio.to_thread(CP.compte_depuis_session, token)

@app.post("/api/comptes/inscription")
async def api_comptes_inscription(
    request: Request,
    pseudo:       str = Form(...),
    email:        str = Form(...),
    mot_de_passe: str = Form(...),
):
    try:
        compte = await asyncio.to_thread(CP.creer_compte, pseudo, email, mot_de_passe)
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = await asyncio.to_thread(CP.creer_session, compte["id"])
    resp = JSONResponse({"ok": True, "compte": CP.compte_public(compte)})
    _poser_cookie_session(request, resp, token)
    return resp

@app.post("/api/comptes/connexion")
async def api_comptes_connexion(
    request: Request,
    email:        str = Form(...),
    mot_de_passe: str = Form(...),
):
    compte = await asyncio.to_thread(CP.verifier_connexion, email, mot_de_passe)
    if not compte:
        raise HTTPException(401, "Email ou mot de passe incorrect.")
    token = await asyncio.to_thread(CP.creer_session, compte["id"])
    resp = JSONResponse({"ok": True, "compte": CP.compte_public(compte)})
    _poser_cookie_session(request, resp, token)
    return resp

@app.post("/api/comptes/deconnexion")
async def api_comptes_deconnexion(request: Request):
    token = request.cookies.get(COOKIE_SESSION, "")
    if token:
        await asyncio.to_thread(CP.supprimer_session, token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_SESSION, path="/")
    return resp

@app.get("/api/comptes/moi")
async def api_comptes_moi(request: Request):
    compte = await _compte_courant(request)
    if not compte:
        raise HTTPException(401, "Non connecté.")
    return JSONResponse(CP.compte_public(compte))

@app.post("/api/comptes/telegram/code")
async def api_comptes_telegram_code():
    """Génère un code court + lien d'invitation vers le bot pour se
    connecter (ou créer un compte) sans mot de passe."""
    username = await _bot_username()
    if not username:
        raise HTTPException(503, "Connexion Telegram indisponible (bot non configuré).")
    code = await asyncio.to_thread(CP.generer_code_telegram)
    return JSONResponse({
        "code": code,
        "lien": f"https://t.me/{username}?start=connexion_{code}",
        "expire_dans_secondes": int(CP.DUREE_CODE_TELEGRAM.total_seconds()),
    })

@app.get("/api/comptes/telegram/statut/{code}")
async def api_comptes_telegram_statut(code: str, request: Request):
    """Interrogé en boucle courte par le web pendant que l'utilisateur va
    confirmer le code sur Telegram. Si une session est déjà active, lie ce
    Telegram au compte connecté plutôt que d'en créer un nouveau."""
    resultat = await asyncio.to_thread(CP.resultat_code_telegram, code)
    if resultat is None:
        raise HTTPException(404, "Code invalide ou expiré.")
    if resultat["statut"] != "confirme":
        return JSONResponse({"statut": resultat["statut"]})

    tg_id = resultat["telegram_id"]
    compte_actuel = await _compte_courant(request)
    resp_cookie_token = None

    if compte_actuel:
        existant = await asyncio.to_thread(CP.compte_par_telegram, tg_id)
        if existant and existant["id"] != compte_actuel["id"]:
            raise HTTPException(409, "Ce compte Telegram est déjà lié à un autre compte.")
        if not existant:
            await asyncio.to_thread(
                CP.lier_telegram, compte_actuel["id"], tg_id, resultat.get("telegram_username"),
            )
        compte = await asyncio.to_thread(CP.compte_par_id, compte_actuel["id"])
    else:
        compte = await asyncio.to_thread(CP.compte_par_telegram, tg_id)
        if not compte:
            pseudo_base = (
                resultat.get("telegram_prenom") or resultat.get("telegram_username") or f"Ami{tg_id}"
            ).strip()[:40]
            compte = await asyncio.to_thread(
                CP.creer_compte_telegram, pseudo_base, tg_id, resultat.get("telegram_username"),
            )
        resp_cookie_token = await asyncio.to_thread(CP.creer_session, compte["id"])

    await asyncio.to_thread(CP.consommer_code_telegram, code)
    resp = JSONResponse({"statut": "confirme", "compte": CP.compte_public(compte)})
    if resp_cookie_token:
        _poser_cookie_session(request, resp, resp_cookie_token)
    return resp

# ══════════════════════════════════════════════════════════════════════════════
# ESPACE ÉDITORIAL — vente du livre (Stripe Checkout) + éditos communautaires
# ══════════════════════════════════════════════════════════════════════════════

import espace_editorial as EE

# ── Catalogue (livres en vente) ─────────────────────────────────────────────

@app.get("/api/editorial/livres")
async def api_editorial_livres():
    return JSONResponse(EE.charger_catalogue())

@app.post("/api/editorial/livres")
async def api_editorial_ajouter_livre(
    titre:              str = Form(...),
    auteur:             str = Form(""),
    description:        str = Form(""),
    devise:             str = Form("gnf"),
    prix_numerique:     str = Form(""),   # chaîne vide = format non proposé
    prix_papier:        str = Form(""),
    key:                str = Form(""),
    couverture:         UploadFile | None = File(None),
    fichier_numerique:  UploadFile | None = File(None),
):
    """Ajoute un livre au catalogue de vente (admin uniquement)."""
    _check_admin(key)
    if not titre.strip():
        raise HTTPException(400, "Titre requis.")
    devise = devise.lower()
    if devise not in EE.DEVISES_ACCEPTEES:
        raise HTTPException(400, f"Devise non supportée. Acceptées: {EE.DEVISES_ACCEPTEES}")

    def _parse_prix(brut: str) -> int | None:
        brut = brut.strip()
        if not brut:
            return None
        try:
            val = float(brut)
        except ValueError:
            raise HTTPException(400, "Prix invalide.")
        if val <= 0:
            raise HTTPException(400, "Le prix doit être positif.")
        return EE.calculer_montant_stripe(val, devise)

    prix_num_centimes = _parse_prix(prix_numerique)
    prix_pap_centimes = _parse_prix(prix_papier)

    couverture_nom = ""
    if couverture is not None and couverture.filename:
        ext = Path(couverture.filename).suffix.lower()
        if ext not in EE.EXTENSIONS_COUVERTURE:
            raise HTTPException(400, "Image de couverture: jpg, png ou webp uniquement.")
        contenu = await couverture.read()
        couverture_nom = f"{uuid.uuid4().hex[:8]}{ext}"
        (EE.DOSSIER_COUVERTURES / couverture_nom).write_bytes(contenu)

    fichier_nom = ""
    if prix_num_centimes is not None:
        if fichier_numerique is None or not fichier_numerique.filename:
            raise HTTPException(400, "Un fichier (PDF/EPUB) est requis pour proposer le format numérique.")
        ext = Path(fichier_numerique.filename).suffix.lower()
        if ext not in EE.EXTENSIONS_FICHIER_NUMERIQUE:
            raise HTTPException(400, "Fichier numérique: PDF ou EPUB uniquement.")
        contenu_fichier = await fichier_numerique.read()
        if len(contenu_fichier) < 10:
            raise HTTPException(400, "Fichier numérique invalide.")
        fichier_nom = f"{uuid.uuid4().hex[:8]}{ext}"
        (EE.DOSSIER_LIVRES_FICHIERS / fichier_nom).write_bytes(contenu_fichier)

    try:
        fiche = await asyncio.to_thread(
            EE.ajouter_livre, titre.strip(), auteur.strip(), description.strip(), devise,
            couverture_nom, prix_num_centimes, prix_pap_centimes, fichier_nom,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info(f"Éditorial — livre ajouté au catalogue: {titre}")
    return JSONResponse({"ok": True, "livre": fiche})

@app.get("/api/editorial/livres/{livre_id}/couverture")
async def api_editorial_couverture(livre_id: str):
    livres = EE.charger_catalogue()
    livre = next((l for l in livres if l["id"] == livre_id), None)
    if not livre or not livre.get("couverture"):
        raise HTTPException(404, "Couverture introuvable.")
    chemin = EE.DOSSIER_COUVERTURES / livre["couverture"]
    if not chemin.exists():
        raise HTTPException(404, "Couverture introuvable.")
    return FileResponse(chemin)

@app.delete("/api/editorial/livres/{livre_id}")
async def api_editorial_supprimer_livre(livre_id: str, key: str = ""):
    _check_admin(key)
    if not await asyncio.to_thread(EE.supprimer_livre, livre_id):
        raise HTTPException(404, "Livre non trouvé.")
    return JSONResponse({"ok": True})

# ── Zones de livraison (papier) ─────────────────────────────────────────────

@app.get("/api/editorial/zones-livraison")
async def api_editorial_zones_livraison():
    return JSONResponse(EE.charger_zones_livraison())

@app.post("/api/editorial/zones-livraison")
async def api_editorial_ajouter_zone(
    nom:    str = Form(...),
    frais:  float = Form(...),
    devise: str = Form("gnf"),
    key:    str = Form(""),
):
    _check_admin(key)
    if not nom.strip():
        raise HTTPException(400, "Nom de zone requis.")
    if frais < 0:
        raise HTTPException(400, "Les frais ne peuvent pas être négatifs.")
    devise = devise.lower()
    if devise not in EE.DEVISES_ACCEPTEES:
        raise HTTPException(400, f"Devise non supportée. Acceptées: {EE.DEVISES_ACCEPTEES}")
    frais_centimes = EE.calculer_montant_stripe(frais, devise)
    zone = await asyncio.to_thread(EE.ajouter_zone_livraison, nom.strip(), frais_centimes, devise)
    return JSONResponse({"ok": True, "zone": zone})

@app.delete("/api/editorial/zones-livraison/{zone_id}")
async def api_editorial_supprimer_zone(zone_id: str, key: str = ""):
    _check_admin(key)
    if not await asyncio.to_thread(EE.supprimer_zone_livraison, zone_id):
        raise HTTPException(404, "Zone non trouvée.")
    return JSONResponse({"ok": True})

@app.get("/livres", response_class=HTMLResponse)
async def page_livres(request: Request):
    """Librairie — catalogue complet, achat en numérique ou papier
    (avec choix de zone de livraison), rendu côté serveur."""
    base = _base_url(request)
    livres = EE.charger_catalogue()
    rep = EE.REPARTITION_REVENUS

    def _prix_html(centimes: int, devise: str) -> str:
        d = devise.lower()
        if d in ("gnf", "xof"):
            return f"{centimes:,}".replace(",", " ") + f" {devise.upper()}"
        return f"{centimes/100:.2f} {devise.upper()}"

    def _carte(l: dict) -> str:
        titre = html.escape(l["titre"])
        auteur = html.escape(l.get("auteur", ""))
        desc = html.escape(l.get("description", ""))
        image = f'{base}/api/editorial/livres/{l["id"]}/couverture' if l.get("couverture") else ""
        devise = l["devise"]

        boutons = []
        if l.get("prix_numerique_centimes") is not None:
            boutons.append(
                f'<button class="livre-achat-btn" onclick="livreAcheter(\'{l["id"]}\',\'numerique\',null)">'
                f'📱 Numérique — {_prix_html(l["prix_numerique_centimes"], devise)}</button>'
            )
        if l.get("prix_papier_centimes") is not None:
            boutons.append(
                f'<button class="livre-achat-btn" onclick="livreChoisirPapier(\'{l["id"]}\')">'
                f'📦 Papier — {_prix_html(l["prix_papier_centimes"], devise)}</button>'
            )

        return f"""
        <div class="livre-carte" id="livre-{l['id']}">
          {f'<img class="livre-couv" src="{image}" alt="">' if image else '<div class="livre-couv livre-couv-vide">📖</div>'}
          <div class="livre-corps">
            <h2>{titre}</h2>
            {f'<div class="livre-auteur">✍️ {auteur}</div>' if auteur else ''}
            <p class="livre-desc">{desc}</p>
            <div class="livre-boutons">{''.join(boutons)}</div>
            <div class="livre-zone-choix" id="zone-choix-{l['id']}" style="display:none;"></div>
          </div>
        </div>"""

    liste_html = "".join(_carte(l) for l in livres) if livres else \
        '<p style="color:#8fac97;text-align:center;padding:40px 0;">Aucun livre en vente pour l\'instant.</p>'

    return HTMLResponse(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Librairie — Pular IA</title>
<meta name="description" content="Achète les livres de la communauté Pular IA, en numérique ou en papier.">
<meta property="og:type" content="website">
<meta property="og:title" content="Librairie — Pular IA">
<meta property="og:url" content="{base}/livres">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1f15; color: #e8f5e9; font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  }}
  header {{
    width: 100%; background: linear-gradient(135deg, #0a3d20 0%, #1a6b3c 100%);
    border-bottom: 2px solid #c8a84b; padding: 14px 20px; text-align: center;
  }}
  header a {{ color: #c8a84b; text-decoration: none; font-size: .85rem; font-weight: 600; }}
  header a:hover {{ text-decoration: underline; }}
  header h1 {{ font-size: 1.2rem; color: #c8a84b; margin-top: 8px; }}
  .repartition {{
    width: 100%; max-width: 720px; margin: 16px auto 0; padding: 14px 18px; border-radius: 10px;
    background: #142b1c; border: 1px solid #1e3d28; font-size: .8rem; color: #8fac97;
  }}
  .repartition strong {{ color: #c8a84b; }}
  main {{ width: 100%; max-width: 720px; padding: 20px 16px 60px; display: flex; flex-direction: column; gap: 16px; }}
  .livre-carte {{
    display: flex; gap: 14px; background: #142b1c; border: 1px solid #1e3d28;
    border-radius: 12px; padding: 14px; scroll-margin-top: 20px;
  }}
  .livre-couv {{ width: 110px; height: 150px; object-fit: cover; border-radius: 8px; flex-shrink: 0; }}
  .livre-couv-vide {{
    display: flex; align-items: center; justify-content: center; font-size: 2rem;
    background: #0d1f15; border: 1px solid #2d5c3a;
  }}
  .livre-corps h2 {{ font-size: 1.05rem; color: #c8a84b; margin-bottom: 4px; line-height: 1.3; }}
  .livre-auteur {{ font-size: .78rem; color: #8fac97; margin-bottom: 6px; }}
  .livre-desc {{ font-size: .85rem; line-height: 1.5; color: #e8f5e9; margin-bottom: 10px; }}
  .livre-boutons {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .livre-achat-btn, .zone-btn {{
    padding: 9px 14px; border-radius: 8px; border: 1px solid #8b1e5c; background: transparent;
    color: #e8f5e9; font-size: .82rem; font-weight: 600; cursor: pointer; font-family: inherit;
  }}
  .livre-achat-btn:hover, .zone-btn:hover {{ background: #8b1e5c; }}
  .livre-zone-choix {{ margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }}
</style>
</head>
<body>
  <header>
    <a href="{base}/#carte-editorial">← Espace Éditorial Pular IA</a>
    <h1>🛒 Librairie</h1>
  </header>
  <div class="repartition">
    💛 <strong>100% du prix est réparti</strong> : {rep['auteur']}% pour l'auteur ·
    {rep['plateforme']}% pour l'entretien de la plateforme ·
    {rep['projet']}% pour financer le projet (IA qui comprend le pular, restauration du Fouta, et d'autres initiatives communautaires).
  </div>
  <main>{liste_html}</main>
  <script>
    const API = {json.dumps(base)};
    let ZONES = null;

    async function chargerZones() {{
      if (ZONES) return ZONES;
      try {{
        const r = await fetch(`${{API}}/api/editorial/zones-livraison`);
        ZONES = await r.json();
      }} catch (e) {{ ZONES = []; }}
      return ZONES;
    }}

    async function livreChoisirPapier(livreId) {{
      const zones = await chargerZones();
      const div = document.getElementById(`zone-choix-${{livreId}}`);
      const boutons = zones.map(z =>
        `<button class="zone-btn" onclick="livreAcheter('${{livreId}}','papier','${{z.id}}')">📍 ${{z.nom}}</button>`
      ).join('');
      div.innerHTML = `<p style="font-size:.78rem;color:#8fac97;">Choisis ta zone de livraison :</p>` + boutons +
        `<button class="zone-btn" onclick="livreAcheter('${{livreId}}','papier',null)">🤝 Retrait / à organiser (sans frais)</button>`;
      div.style.display = 'flex';
    }}

    async function livreAcheter(livreId, format, zoneId) {{
      try {{
        const fd = new FormData();
        fd.append('livre_id', livreId);
        fd.append('format', format);
        fd.append('origin', window.location.origin);
        if (zoneId) fd.append('zone_id', zoneId);
        const r = await fetch(`${{API}}/api/editorial/checkout`, {{ method: 'POST', body: fd }});
        const d = await r.json();
        if (!d.ok) throw new Error(d.detail || 'Erreur serveur');
        window.location.href = d.url;
      }} catch (e) {{
        alert('Erreur : ' + e.message);
      }}
    }}

    if (window.location.hash) {{
      const cible = document.querySelector(window.location.hash);
      if (cible) cible.scrollIntoView({{ behavior: 'smooth' }});
    }}
  </script>
</body>
</html>""")

# ── Paiement (Stripe Checkout) ──────────────────────────────────────────────

@app.post("/api/editorial/checkout")
async def api_editorial_checkout(
    livre_id:      str = Form(...),
    format_achete: str = Form(..., alias="format"),
    origin:        str = Form(...),
    zone_id:       str = Form(""),
    email:         str = Form(""),
):
    livres = EE.charger_catalogue()
    livre = next((l for l in livres if l["id"] == livre_id), None)
    if not livre:
        raise HTTPException(404, "Livre non trouvé.")
    zone = None
    if format_achete == "papier" and zone_id.strip():
        zones = EE.charger_zones_livraison()
        zone = next((z for z in zones if z["id"] == zone_id.strip()), None)
        if not zone:
            raise HTTPException(404, "Zone de livraison non trouvée.")
    try:
        session = await asyncio.to_thread(
            EE.creer_session_paiement, livre, format_achete, zone, origin, email.strip(),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"Erreur création session Stripe: {e}")
        raise HTTPException(500, "Erreur lors de la création du paiement.")
    commande = await asyncio.to_thread(EE.enregistrer_commande, session, livre, format_achete, zone)
    return JSONResponse({"ok": True, "url": session.url, "commande_id": commande["id"]})

@app.post("/api/editorial/webhook")
async def api_editorial_webhook(request: Request):
    """Webhook Stripe : confirme le paiement une fois la session complétée."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = await asyncio.to_thread(EE.verifier_signature_webhook, payload, sig_header)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.warning(f"Webhook Stripe rejeté: {e}")
        raise HTTPException(400, "Signature invalide.")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = (session.get("customer_details") or {}).get("email", "") or session.get("customer_email", "")
        if (session.get("metadata") or {}).get("type") == "don":
            await asyncio.to_thread(EE.marquer_don_paye, session["id"], email)
        else:
            await asyncio.to_thread(EE.marquer_commande_payee, session["id"], email)

    return JSONResponse({"received": True})

@app.get("/api/editorial/commandes")
async def api_editorial_commandes(key: str = ""):
    """Liste des commandes — admin uniquement (contient les emails acheteurs)."""
    _check_admin(key)
    return JSONResponse(EE.charger_commandes())

@app.get("/api/editorial/commandes/{commande_id}/statut")
async def api_editorial_commande_statut(commande_id: str, jeton: str = ""):
    commande = EE.obtenir_commande(commande_id)
    if not commande or not jeton or commande.get("jeton") != jeton:
        raise HTTPException(404, "Commande non trouvée.")
    return JSONResponse({
        "statut": commande["statut"], "format": commande["format"],
        "titre": commande["titre"], "zone": commande.get("zone"),
    })

@app.get("/api/editorial/commandes/{commande_id}/fichier")
async def api_editorial_commande_fichier(commande_id: str, jeton: str = ""):
    commande = EE.obtenir_commande(commande_id)
    if not commande or not jeton or commande.get("jeton") != jeton:
        raise HTTPException(404, "Commande non trouvée.")
    if commande["statut"] != "paye":
        raise HTTPException(402, "Paiement non confirmé.")
    if commande["format"] != "numerique" or not commande.get("fichier_numerique"):
        raise HTTPException(404, "Aucun fichier numérique pour cette commande.")
    chemin = EE.DOSSIER_LIVRES_FICHIERS / commande["fichier_numerique"]
    if not chemin.exists():
        raise HTTPException(404, "Fichier introuvable.")
    nom_telecharge = f"{commande['titre']}{chemin.suffix}"
    return FileResponse(chemin, filename=nom_telecharge)

@app.get("/api/dons/meta")
async def api_dons_meta():
    return JSONResponse({
        "montants_suggeres": EE.MONTANTS_DON_SUGGERES,
        "devises": EE.DEVISES_ACCEPTEES,
        "total_paye": EE.total_dons_payes(),
    })

@app.post("/api/dons/checkout")
async def api_dons_checkout(
    montant:      float = Form(...),
    devise:       str = Form("gnf"),
    origin:       str = Form(...),
    email:        str = Form(""),
    nom_donateur: str = Form(""),
):
    devise = devise.lower()
    if devise not in EE.DEVISES_ACCEPTEES:
        raise HTTPException(400, f"Devise non supportée. Acceptées: {EE.DEVISES_ACCEPTEES}")
    if montant <= 0:
        raise HTTPException(400, "Montant invalide.")
    montant_centimes = EE.calculer_montant_stripe(montant, devise)
    try:
        session = await asyncio.to_thread(
            EE.creer_session_don, montant_centimes, devise, origin, email.strip(), nom_donateur.strip(),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"Erreur création session don Stripe: {e}")
        raise HTTPException(500, "Erreur lors de la création du don.")
    don = await asyncio.to_thread(
        EE.enregistrer_don, session, montant_centimes, devise, nom_donateur.strip(),
    )
    return JSONResponse({"ok": True, "url": session.url, "don_id": don["id"]})

@app.get("/api/dons/{don_id}/statut")
async def api_dons_statut(don_id: str, jeton: str = ""):
    don = EE.obtenir_don(don_id)
    if not don or not jeton or don.get("jeton") != jeton:
        raise HTTPException(404, "Don non trouvé.")
    return JSONResponse({
        "statut": don["statut"], "montant_centimes": don["montant_centimes"], "devise": don["devise"],
    })

@app.get("/api/dons")
async def api_dons_liste(key: str = ""):
    """Liste des dons — admin uniquement (contient les emails donateurs)."""
    _check_admin(key)
    return JSONResponse(EE.charger_dons())

@app.get("/don-merci", response_class=HTMLResponse)
async def page_don_merci(request: Request, session_id: str = ""):
    """Page de confirmation après un don Stripe réussi."""
    base = _base_url(request)
    don = EE.obtenir_don_par_session(session_id) if session_id else None
    if not don:
        return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Don introuvable — Pular IA</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="background:#0d1f15;color:#e8f5e9;font-family:system-ui,sans-serif;
text-align:center;padding:60px 20px;">
<h1 style="color:#c8a84b;">Don introuvable</h1>
<a href="{base}/#carte-dons" style="color:#c8a84b;">← Retour</a>
</body></html>""", status_code=404)

    nom = html.escape(don.get("nom_donateur") or "Anonyme")

    return HTMLResponse(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Merci pour ton don — Pular IA</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1f15; color: #e8f5e9; font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  }}
  main {{ width: 100%; max-width: 480px; padding: 40px 20px; text-align: center; }}
  h1 {{ font-size: 1.5rem; color: #c8a84b; margin-bottom: 10px; }}
  .sous {{ color: #8fac97; font-size: .9rem; margin-bottom: 26px; }}
  #statut {{ padding: 18px; border-radius: 10px; border: 1px solid #2d5c3a; background: #142b1c; }}
  .info {{
    margin-top: 30px; padding: 16px; border-radius: 10px; background: #142b1c;
    border: 1px solid #1e3d28; font-size: .8rem; color: #8fac97;
  }}
  footer {{ margin-top: 30px; }}
  footer a {{ color: #c8a84b; text-decoration: none; font-size: .85rem; font-weight: 600; }}
</style>
</head>
<body>
  <main>
    <h1>🙏 Merci, {nom} !</h1>
    <p class="sous">Ton don soutient directement le projet Pular IA.</p>
    <div id="statut">⏳ Confirmation du paiement en cours...</div>
    <div class="info">
      💛 100% de ton don finance le projet : IA qui comprend le pular, restauration du Fouta,
      et l'entretien de la plateforme.
    </div>
    <footer><a href="{base}/#carte-dons">← Retour</a></footer>
  </main>
  <script>
    const DON_ID = {json.dumps(don["id"])};
    const JETON  = {json.dumps(don["jeton"])};
    async function verifier() {{
      try {{
        const r = await fetch(`/api/dons/${{DON_ID}}/statut?jeton=${{JETON}}`);
        const d = await r.json();
        const el = document.getElementById('statut');
        if (d.statut !== 'paye') {{ setTimeout(verifier, 2500); return; }}
        el.innerHTML = '✅ Paiement confirmé — merci pour ton soutien !';
      }} catch (err) {{ setTimeout(verifier, 3000); }}
    }}
    verifier();
  </script>
</body>
</html>""")

@app.get("/livre-achete", response_class=HTMLResponse)
async def page_livre_achete(request: Request, session_id: str = ""):
    """Page de confirmation après un paiement Stripe réussi — affiche le
    téléchargement (numérique) ou les infos de livraison (papier), avec un
    rappel de la répartition des revenus."""
    base = _base_url(request)
    commande = EE.obtenir_commande_par_session(session_id) if session_id else None
    if not commande:
        return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Commande introuvable — Pular IA</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="background:#0d1f15;color:#e8f5e9;font-family:system-ui,sans-serif;
text-align:center;padding:60px 20px;">
<h1 style="color:#c8a84b;">Commande introuvable</h1>
<a href="{base}/#carte-editorial" style="color:#c8a84b;">← Retour à l'espace Éditorial</a>
</body></html>""", status_code=404)

    titre = html.escape(commande["titre"])
    rep = EE.REPARTITION_REVENUS
    zone_nom = html.escape(commande["zone"]["nom"]) if commande.get("zone") else ""

    return HTMLResponse(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Merci pour ton achat — Pular IA</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1f15; color: #e8f5e9; font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  }}
  main {{ width: 100%; max-width: 520px; padding: 40px 20px; text-align: center; }}
  h1 {{ font-size: 1.5rem; color: #c8a84b; margin-bottom: 10px; }}
  .sous {{ color: #8fac97; font-size: .9rem; margin-bottom: 26px; }}
  #statut {{ padding: 18px; border-radius: 10px; border: 1px solid #2d5c3a; background: #142b1c; }}
  .telecharger {{
    display: inline-block; margin-top: 14px; padding: 12px 22px; border-radius: 8px;
    background: #8b1e5c; color: #fff; text-decoration: none; font-weight: 700;
  }}
  .repartition {{
    margin-top: 30px; padding: 16px; border-radius: 10px; background: #142b1c;
    border: 1px solid #1e3d28; font-size: .8rem; color: #8fac97; text-align: left;
  }}
  .repartition strong {{ color: #c8a84b; }}
  footer {{ margin-top: 30px; }}
  footer a {{ color: #c8a84b; text-decoration: none; font-size: .85rem; font-weight: 600; }}
</style>
</head>
<body>
  <main>
    <h1>🙏 Merci pour ton achat !</h1>
    <p class="sous">{titre}</p>
    <div id="statut">⏳ Confirmation du paiement en cours...</div>

    <div class="repartition">
      💛 <strong>100% du prix est réparti</strong> : {rep['auteur']}% pour l'auteur ·
      {rep['plateforme']}% pour l'entretien de la plateforme ·
      {rep['projet']}% pour financer le projet (IA qui comprend le pular, restauration du Fouta, et d'autres initiatives communautaires).
    </div>

    <footer><a href="{base}/#carte-editorial">← Retour à l'espace Éditorial</a></footer>
  </main>
  <script>
    const COMMANDE_ID = {json.dumps(commande["id"])};
    const JETON       = {json.dumps(commande["jeton"])};
    const ZONE_NOM     = {json.dumps(zone_nom)};

    async function verifier() {{
      try {{
        const r = await fetch(`/api/editorial/commandes/${{COMMANDE_ID}}/statut?jeton=${{JETON}}`);
        const d = await r.json();
        const el = document.getElementById('statut');
        if (d.statut !== 'paye') {{
          setTimeout(verifier, 2500);
          return;
        }}
        if (d.format === 'numerique') {{
          el.innerHTML = '✅ Paiement confirmé — ton livre est prêt !<br>' +
            `<a class="telecharger" href="/api/editorial/commandes/${{COMMANDE_ID}}/fichier?jeton=${{JETON}}">⬇️ Télécharger le livre</a>`;
        }} else {{
          el.innerHTML = '✅ Paiement confirmé — commande papier enregistrée' +
            (ZONE_NOM ? ` pour <strong>${{ZONE_NOM}}</strong>.` : '.') +
            '<br>On te contacte bientôt pour organiser la remise.';
        }}
      }} catch (err) {{ setTimeout(verifier, 3000); }}
    }}
    verifier();
  </script>
</body>
</html>""")

# ── Éditos (articles communautaires) ────────────────────────────────────────

@app.post("/api/editorial/editos/generer")
async def api_editorial_generer_edito(
    sujet:           str = Form(...),
    angle:           str = Form(""),
    chercher_corpus: bool = Form(True),
    documents:       list[UploadFile] = File(default=[]),
):
    """
    Génère un premier brouillon d'édito avec Claude — à partir du sujet,
    de documents déposés par l'auteur (réutilise l'extraction du RAG livres)
    et/ou de passages trouvés dans les corpus RAG Livres/Histoire déjà
    indexés, utilisés comme sources plutôt que de laisser le modèle inventer.
    """
    if not sujet.strip():
        raise HTTPException(400, "Indique un sujet.")

    sources: list[dict] = []
    for doc in documents:
        if not doc or not doc.filename:
            continue
        ext = Path(doc.filename).suffix.lower()
        if ext not in EE.EXTENSIONS_SOURCE:
            raise HTTPException(400, f"Format non supporté pour '{doc.filename}'. Acceptés: PDF, TXT, DOCX, HTML, MD")
        contenu = await doc.read()
        if len(contenu) < 10:
            continue
        # Nom de fichier temporaire construit à partir d'un UUID uniquement
        # (jamais du nom fourni par le client) pour éviter tout traversal.
        chemin_tmp = EE.DOSSIER_EDITOS_SOURCES / f"{uuid.uuid4().hex[:8]}{ext}"
        chemin_tmp.write_bytes(contenu)
        try:
            extrait = await asyncio.to_thread(EE.extraire_extrait_source, chemin_tmp)
            if extrait.strip():
                sources.append({"origine": "document déposé", "titre": doc.filename, "texte": extrait})
        except Exception as e:
            log.warning(f"Extraction document source édito '{doc.filename}': {e}")
        finally:
            chemin_tmp.unlink(missing_ok=True)

    if chercher_corpus:
        try:
            sources.extend(await asyncio.to_thread(EE.rechercher_sources_existantes, sujet.strip()))
        except Exception as e:
            log.warning(f"Recherche corpus pour édito: {e}")

    try:
        texte = await asyncio.to_thread(
            EE.generer_brouillon_edito, sujet.strip(), angle.strip(), sources or None,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"Erreur génération IA édito: {e}")
        raise HTTPException(500, "Erreur lors de la génération du brouillon.")

    sources_info = [{"origine": s["origine"], "titre": s["titre"]} for s in sources]
    return JSONResponse({"ok": True, "texte": texte, "sources": sources_info})

@app.post("/api/editorial/editos/generer-image")
async def api_editorial_generer_image_edito(sujet: str = Form(...)):
    """Génère une image d'illustration avec DALL·E, pour démarrer plus vite."""
    if not sujet.strip():
        raise HTTPException(400, "Décris l'image à générer.")
    try:
        image_bytes = await asyncio.to_thread(EE.generer_image_edito, sujet.strip())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(f"Erreur génération image édito: {e}")
        raise HTTPException(500, "Erreur lors de la génération de l'image.")
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return JSONResponse({"ok": True, "image_base64": f"data:image/png;base64,{image_b64}"})

@app.post("/api/editorial/editos")
async def api_editorial_ajouter_edito(
    request: Request,
    titre:   str = Form(...),
    auteur:  str = Form("Anonyme"),
    contenu: str = Form(...),
    image:   UploadFile = File(...),
):
    if not titre.strip() or not contenu.strip():
        raise HTTPException(400, "Titre et contenu requis.")
    if not image.filename:
        raise HTTPException(400, "Une image est obligatoire pour illustrer l'édito.")
    ext = Path(image.filename).suffix.lower()
    if ext not in EE.EXTENSIONS_COUVERTURE:
        raise HTTPException(400, "Image: jpg, png ou webp uniquement.")
    contenu_image = await image.read()
    if len(contenu_image) < 10:
        raise HTTPException(400, "Image invalide.")

    image_nom = f"{uuid.uuid4().hex[:8]}{ext}"
    (EE.DOSSIER_EDITOS_IMAGES / image_nom).write_bytes(contenu_image)

    # Connecté : le pseudo du compte fait foi (l'auteur libre saisi dans le
    # formulaire est ignoré) — anonyme : pseudo libre comme avant.
    compte = await _compte_courant(request)
    auteur_final = compte["pseudo"] if compte else (auteur.strip() or "Anonyme")

    edito = await asyncio.to_thread(
        EE.ajouter_edito, titre.strip(), auteur_final, contenu.strip(), image_nom,
        compte["id"] if compte else None,
    )
    log.info(f"Éditorial — édito soumis: '{titre}' par {auteur_final}")
    return JSONResponse({"ok": True, "edito": edito})

@app.get("/api/editorial/editos/{edito_id}/image")
async def api_editorial_edito_image(edito_id: str):
    editos = EE.charger_editos()
    edito = next((e for e in editos if e["id"] == edito_id), None)
    if not edito or not edito.get("image"):
        raise HTTPException(404, "Image introuvable.")
    chemin = EE.DOSSIER_EDITOS_IMAGES / edito["image"]
    if not chemin.exists():
        raise HTTPException(404, "Image introuvable.")
    return FileResponse(chemin)

def _edito_pour_client(e: dict, visiteur_id: str = "") -> dict:
    """
    Convertit la liste interne de likes (identifiants anonymes de visiteurs)
    en un simple compteur + indicateur pour CE visiteur, pour ne jamais
    exposer les identifiants des autres visiteurs dans l'API publique.
    """
    d = dict(e)
    likes = d.pop("likes", [])
    d["likes_count"] = len(likes)
    d["mon_like"] = bool(visiteur_id) and visiteur_id in likes
    d["commentaires"] = d.get("commentaires", [])
    return d

async def _identite_visiteur(request: Request, visiteur_id: str) -> str:
    """Un compte connecté fait toujours foi ; sinon on retombe sur
    l'identifiant anonyme fourni par le client (localStorage). Le préfixe
    'compte:' est réservé aux identités authentifiées — un visiteur anonyme
    qui le fournirait explicitement dans `visiteur_id` (pour usurper le
    like d'un compte) est ignoré, jamais transmis tel quel."""
    compte = await _compte_courant(request)
    if compte:
        return f"compte:{compte['id']}"
    visiteur_id = visiteur_id.strip()[:80]
    return "" if visiteur_id.startswith("compte:") else visiteur_id

@app.get("/api/editorial/editos")
async def api_editorial_editos(request: Request, key: str = "", visiteur_id: str = ""):
    """
    Par défaut, ne renvoie que les éditos publiés. Avec la clé admin,
    renvoie aussi ceux en attente de validation.
    """
    editos = EE.charger_editos()
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    if not is_admin:
        editos = [e for e in editos if e["statut"] == "publie"]
    identite = await _identite_visiteur(request, visiteur_id)
    return JSONResponse([_edito_pour_client(e, identite) for e in editos])

@app.get("/api/editorial/editos/{edito_id}")
async def api_editorial_edito_unique(edito_id: str, request: Request, key: str = "", visiteur_id: str = ""):
    """Un édito précis — public uniquement s'il est publié (sinon 404, même s'il existe)."""
    editos = EE.charger_editos()
    edito = next((e for e in editos if e["id"] == edito_id), None)
    is_admin = bool(ADMIN_KEY) and key == ADMIN_KEY
    if not edito or (edito["statut"] != "publie" and not is_admin):
        raise HTTPException(404, "Édito non trouvé.")
    identite = await _identite_visiteur(request, visiteur_id)
    return JSONResponse(_edito_pour_client(edito, identite))

@app.post("/api/editorial/editos/{edito_id}/like")
async def api_editorial_like_edito(edito_id: str, request: Request, visiteur_id: str = Form("")):
    identite = await _identite_visiteur(request, visiteur_id)
    if not identite:
        raise HTTPException(400, "Identifiant visiteur manquant.")
    resultat = await asyncio.to_thread(EE.basculer_like, edito_id, identite)
    if resultat is None:
        raise HTTPException(404, "Édito non trouvé.")
    return JSONResponse({"ok": True, **resultat})

@app.post("/api/editorial/editos/{edito_id}/commentaires")
async def api_editorial_ajouter_commentaire(
    edito_id: str,
    request: Request,
    auteur: str = Form(""),
    texte:  str = Form(...),
):
    texte = texte.strip()
    if not texte:
        raise HTTPException(400, "Commentaire vide.")
    if len(texte) > 1000:
        raise HTTPException(400, "Commentaire trop long (max 1000 caractères).")
    compte = await _compte_courant(request)
    auteur_final = compte["pseudo"] if compte else (auteur.strip() or "Anonyme")
    commentaire = await asyncio.to_thread(
        EE.ajouter_commentaire, edito_id, auteur_final, texte, compte["id"] if compte else None,
    )
    if commentaire is None:
        raise HTTPException(404, "Édito non trouvé.")
    return JSONResponse({"ok": True, "commentaire": commentaire})

@app.delete("/api/editorial/editos/{edito_id}/commentaires/{commentaire_id}")
async def api_editorial_supprimer_commentaire(edito_id: str, commentaire_id: str, key: str = ""):
    _check_admin(key)
    if not await asyncio.to_thread(EE.supprimer_commentaire, edito_id, commentaire_id):
        raise HTTPException(404, "Commentaire non trouvé.")
    return JSONResponse({"ok": True})

MOIS_FR = ["janvier", "février", "mars", "avril", "mai", "juin",
           "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

def _date_fr(iso: str) -> str:
    try:
        d = datetime.fromisoformat(iso)
        return f"{d.day} {MOIS_FR[d.month - 1]} {d.year}"
    except Exception:
        return ""

def _contenu_vers_html(texte: str) -> str:
    paragraphes = [p.strip() for p in texte.split("\n\n") if p.strip()]
    return "\n".join(
        f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paragraphes
    )

def _base_url(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    base = f"{scheme}://{request.url.hostname}"
    if request.url.port and request.url.port not in (80, 443):
        base += f":{request.url.port}"
    return base

@app.get("/editos", response_class=HTMLResponse)
async def page_editos_publies(request: Request):
    """Page listant tous les éditos publiés — rendue côté serveur, style magazine,
    chaque entrée renvoyant vers sa page de lecture /lire/{id}."""
    base = _base_url(request)
    editos = sorted(
        (e for e in EE.charger_editos() if e["statut"] == "publie"),
        key=lambda e: e.get("date_publication") or "",
        reverse=True,
    )

    def _carte(e: dict) -> str:
        titre   = html.escape(e["titre"])
        auteur  = html.escape(e.get("auteur", "Anonyme"))
        date_pub = _date_fr(e.get("date_publication") or e.get("date_soumission", ""))
        extrait = html.escape(e["contenu"].strip().replace("\n", " ")[:180])
        if len(e["contenu"]) > 180:
            extrait += "…"
        image = f'{base}/api/editorial/editos/{e["id"]}/image' if e.get("image") else ""
        nb_likes = len(e.get("likes", []))
        nb_comm  = len(e.get("commentaires", []))
        return f"""
        <a class="carte-edito" href="{base}/lire/{e['id']}">
          {f'<img src="{image}" alt="">' if image else ''}
          <div class="carte-edito-corps">
            <h2>{titre}</h2>
            <div class="carte-edito-meta">✍️ {auteur} · {date_pub} · ❤️ {nb_likes} · 💬 {nb_comm}</div>
            <p>{extrait}</p>
          </div>
        </a>"""

    liste_html = "".join(_carte(e) for e in editos) if editos else \
        '<p style="color:#8fac97;text-align:center;padding:40px 0;">Aucun édito publié pour l\'instant.</p>'

    return HTMLResponse(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Éditos publiés — Espace Éditorial Pular IA</title>
<meta name="description" content="Les articles publiés par la communauté sur la langue, la culture, l'histoire et l'actualité peules.">
<meta property="og:type" content="website">
<meta property="og:title" content="Éditos publiés — Pular IA">
<meta property="og:description" content="Les articles publiés par la communauté sur la langue, la culture, l'histoire et l'actualité peules.">
<meta property="og:url" content="{base}/editos">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1f15; color: #e8f5e9; font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  }}
  header {{
    width: 100%; background: linear-gradient(135deg, #0a3d20 0%, #1a6b3c 100%);
    border-bottom: 2px solid #c8a84b; padding: 14px 20px; text-align: center;
  }}
  header a {{ color: #c8a84b; text-decoration: none; font-size: .85rem; font-weight: 600; }}
  header a:hover {{ text-decoration: underline; }}
  header h1 {{ font-size: 1.2rem; color: #c8a84b; margin-top: 8px; }}
  main {{ width: 100%; max-width: 720px; padding: 24px 16px 60px; display: flex; flex-direction: column; gap: 16px; }}
  .carte-edito {{
    display: flex; gap: 14px; background: #142b1c; border: 1px solid #1e3d28;
    border-radius: 12px; padding: 14px; text-decoration: none; color: inherit;
    transition: border-color .15s;
  }}
  .carte-edito:hover {{ border-color: #8b1e5c; }}
  .carte-edito img {{
    width: 110px; height: 110px; object-fit: cover; border-radius: 8px; flex-shrink: 0;
  }}
  .carte-edito-corps h2 {{ font-size: 1.02rem; color: #c8a84b; margin-bottom: 4px; line-height: 1.3; }}
  .carte-edito-meta {{ font-size: .75rem; color: #8fac97; margin-bottom: 6px; }}
  .carte-edito-corps p {{ font-size: .85rem; line-height: 1.5; color: #e8f5e9; }}
</style>
</head>
<body>
  <header>
    <a href="{base}/#carte-editorial">← Espace Éditorial Pular IA</a>
    <h1>✍️ Éditos publiés</h1>
  </header>
  <main>{liste_html}</main>
</body>
</html>""")

@app.get("/lire/{edito_id}", response_class=HTMLResponse)
async def page_lire_edito(edito_id: str, request: Request):
    """Page de lecture d'un édito publié — rendue côté serveur (balises Open
    Graph correctes) pour un partage propre sur les réseaux sociaux."""
    editos = EE.charger_editos()
    edito = next((e for e in editos if e["id"] == edito_id and e["statut"] == "publie"), None)
    base = _base_url(request)

    if not edito:
        return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Édito introuvable — Pular IA</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="background:#0d1f15;color:#e8f5e9;font-family:system-ui,sans-serif;
text-align:center;padding:60px 20px;">
<h1 style="color:#c8a84b;">Édito introuvable</h1>
<p>Ce texte n'existe pas ou n'est pas encore publié.</p>
<a href="{base}/#carte-editorial" style="color:#c8a84b;">← Retour à l'espace Éditorial</a>
</body></html>""", status_code=404)

    titre       = html.escape(edito["titre"])
    auteur      = html.escape(edito.get("auteur", "Anonyme"))
    date_pub    = _date_fr(edito.get("date_publication") or edito.get("date_soumission", ""))
    contenu_html = _contenu_vers_html(edito["contenu"])
    extrait     = html.escape(edito["contenu"].strip().replace("\n", " ")[:180])
    image_url   = f"{base}/api/editorial/editos/{edito_id}/image" if edito.get("image") else ""
    page_url    = f"{base}/lire/{edito_id}"
    nb_likes    = len(edito.get("likes", []))
    commentaires = sorted(edito.get("commentaires", []), key=lambda c: c.get("date", ""))

    url_encodee   = quote(page_url, safe="")
    titre_encode  = quote(edito["titre"])
    whatsapp_texte = quote(f"{edito['titre']} — {page_url}")

    def _commentaire_html(c: dict) -> str:
        c_auteur = html.escape(c.get("auteur", "Anonyme"))
        c_texte  = html.escape(c.get("texte", "")).replace("\n", "<br>")
        c_date   = _date_fr(c.get("date", ""))
        return f"""
        <div class="commentaire" data-id="{c['id']}">
          <div class="commentaire-meta"><strong>{c_auteur}</strong> · {c_date}</div>
          <div class="commentaire-texte">{c_texte}</div>
        </div>"""

    commentaires_html = "".join(_commentaire_html(c) for c in commentaires) or \
        '<p id="pas-de-commentaires" style="color:#8fac97;font-size:.85rem;">Aucun commentaire pour l\'instant — sois le premier à réagir !</p>'

    return HTMLResponse(f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titre} — Espace Éditorial Pular IA</title>
<meta name="description" content="{extrait}">
<meta property="og:type" content="article">
<meta property="og:title" content="{titre}">
<meta property="og:description" content="{extrait}">
<meta property="og:url" content="{page_url}">
{f'<meta property="og:image" content="{image_url}">' if image_url else ''}
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{titre}">
<meta name="twitter:description" content="{extrait}">
{f'<meta name="twitter:image" content="{image_url}">' if image_url else ''}
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1f15; color: #e8f5e9; font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  }}
  header {{
    width: 100%; background: linear-gradient(135deg, #0a3d20 0%, #1a6b3c 100%);
    border-bottom: 2px solid #c8a84b; padding: 14px 20px;
  }}
  header a {{ color: #c8a84b; text-decoration: none; font-size: .85rem; font-weight: 600; }}
  header a:hover {{ text-decoration: underline; }}
  main {{ width: 100%; max-width: 720px; padding: 28px 18px 60px; }}
  .edito-image {{
    width: 100%; max-height: 420px; object-fit: cover; border-radius: 12px;
    border: 1px solid #2d5c3a; margin-bottom: 22px; display: block;
  }}
  h1 {{ font-size: 1.7rem; color: #c8a84b; line-height: 1.3; margin-bottom: 10px; }}
  .meta {{ font-size: .85rem; color: #8fac97; margin-bottom: 26px; }}
  .contenu p {{ font-size: 1.05rem; line-height: 1.8; margin-bottom: 18px; color: #e8f5e9; }}
  .partage {{ margin-top: 34px; padding-top: 22px; border-top: 1px solid #1e3d28; }}
  .partage-titre {{ font-size: .8rem; color: #8fac97; font-weight: 600; margin-bottom: 10px; }}
  .partage-liens {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .partage-liens a, .partage-liens button {{
    display: inline-flex; align-items: center; gap: 6px; padding: 9px 14px; border-radius: 8px;
    border: 1px solid #2d5c3a; background: #142b1c; color: #e8f5e9; text-decoration: none;
    font-size: .82rem; font-weight: 600; cursor: pointer; font-family: inherit;
  }}
  .partage-liens a:hover, .partage-liens button:hover {{ border-color: #c8a84b; }}
  .reactions {{ display: flex; align-items: center; gap: 14px; margin-top: 26px; }}
  #btn-like {{
    display: inline-flex; align-items: center; gap: 8px; padding: 9px 16px; border-radius: 20px;
    border: 1px solid #2d5c3a; background: #142b1c; color: #e8f5e9; cursor: pointer;
    font-size: .9rem; font-weight: 600; font-family: inherit; transition: border-color .15s, background .15s;
  }}
  #btn-like:hover {{ border-color: #8b1e5c; }}
  #btn-like.aime {{ background: #3a1224; border-color: #8b1e5c; color: #ff7fa8; }}
  .commentaires {{ margin-top: 36px; padding-top: 22px; border-top: 1px solid #1e3d28; }}
  .commentaires h2 {{ font-size: 1.05rem; color: #c8a84b; margin-bottom: 16px; }}
  .commentaire-form {{ display: flex; flex-direction: column; gap: 8px; margin-bottom: 24px; }}
  .commentaire-form input, .commentaire-form textarea {{
    background: #142b1c; border: 1px solid #2d5c3a; border-radius: 8px; color: #e8f5e9;
    padding: 10px 12px; font-family: inherit; font-size: .88rem;
  }}
  .commentaire-form textarea {{ min-height: 80px; resize: vertical; }}
  .commentaire-form button {{
    align-self: flex-start; padding: 9px 18px; border-radius: 8px; border: none;
    background: #8b1e5c; color: #fff; font-weight: 600; cursor: pointer; font-size: .85rem;
  }}
  .commentaire-form button:hover {{ background: #a3266f; }}
  .commentaire-form button:disabled {{ opacity: .6; cursor: default; }}
  .commentaire {{ padding: 12px 0; border-bottom: 1px solid #1e3d28; }}
  .commentaire-meta {{ font-size: .78rem; color: #8fac97; margin-bottom: 4px; }}
  .commentaire-meta strong {{ color: #c8a84b; }}
  .commentaire-texte {{ font-size: .9rem; line-height: 1.5; color: #e8f5e9; }}
  footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #1e3d28; width: 100%; }}
  footer a {{
    display: inline-block; padding: 10px 18px; border-radius: 8px; border: 1px solid #8b1e5c;
    color: #e8f5e9; text-decoration: none; font-size: .88rem; font-weight: 600;
  }}
  footer a:hover {{ background: #8b1e5c; }}
</style>
</head>
<body>
  <header><a href="{base}/#carte-editorial">← Espace Éditorial Pular IA</a></header>
  <main>
    {f'<img class="edito-image" src="{image_url}" alt="">' if image_url else ''}
    <h1>{titre}</h1>
    <div class="meta">✍️ {auteur} · {date_pub}</div>
    <div class="contenu">{contenu_html}</div>

    <div class="reactions">
      <button id="btn-like" onclick="basculerLike()">🤍 <span id="nb-likes">{nb_likes}</span></button>
    </div>

    <div class="partage">
      <p class="partage-titre">📤 PARTAGER CET ARTICLE</p>
      <div class="partage-liens">
        <button id="btn-partage-natif" onclick="partageNatif()" style="display:none;">📱 Partager (TikTok, Instagram...)</button>
        <a href="https://wa.me/?text={whatsapp_texte}" target="_blank" rel="noopener">🟢 WhatsApp</a>
        <a href="https://www.facebook.com/sharer/sharer.php?u={url_encodee}" target="_blank" rel="noopener">🔵 Facebook</a>
        <a href="https://twitter.com/intent/tweet?text={titre_encode}&url={url_encodee}" target="_blank" rel="noopener">⚫ X</a>
        <button id="btn-copier-lien" onclick="copierLien()">🔗 Copier le lien</button>
      </div>
      <p style="font-size:.72rem;color:#8fac97;margin-top:8px;">
        TikTok n'a pas de lien de partage direct — utilise "Partager" ci-dessus (sur mobile) pour l'envoyer vers l'appli TikTok, ou colle le lien copié dans ta bio/description.
      </p>
    </div>

    <div class="commentaires">
      <h2>💬 Commentaires (<span id="nb-commentaires">{len(commentaires)}</span>)</h2>
      <div class="commentaire-form">
        <input id="com-auteur" type="text" placeholder="Ton nom / pseudo (facultatif)" maxlength="60">
        <textarea id="com-texte" placeholder="Ton commentaire..." maxlength="1000"></textarea>
        <button id="btn-commenter" onclick="envoyerCommentaire()">Publier</button>
      </div>
      <div id="liste-commentaires">{commentaires_html}</div>
    </div>

    <footer>
      <a href="{base}/#carte-editorial">📰 Voir tous les éditos</a>
    </footer>
  </main>
  <script>
    const API       = {json.dumps(base)};
    const EDITO_ID  = {json.dumps(edito_id)};
    const PAGE_URL  = {json.dumps(page_url)};
    const PAGE_TITRE = {json.dumps(edito["titre"])};

    function copierLien() {{
      const btn = document.getElementById('btn-copier-lien');
      navigator.clipboard.writeText(PAGE_URL).then(() => {{
        const original = btn.textContent;
        btn.textContent = '✅ Lien copié!';
        setTimeout(() => btn.textContent = original, 2000);
      }});
    }}

    if (navigator.share) {{
      document.getElementById('btn-partage-natif').style.display = 'inline-flex';
    }}
    function partageNatif() {{
      navigator.share({{ title: PAGE_TITRE, url: PAGE_URL }}).catch(() => {{}});
    }}

    function visiteurId() {{
      let id = localStorage.getItem('pularVisiteurId');
      if (!id) {{
        id = 'v_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
        localStorage.setItem('pularVisiteurId', id);
      }}
      return id;
    }}

    function escHtml(s) {{
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }}

    async function chargerEtatLike() {{
      try {{
        const r = await fetch(`${{API}}/api/editorial/editos/${{EDITO_ID}}?visiteur_id=${{encodeURIComponent(visiteurId())}}`);
        const e = await r.json();
        document.getElementById('nb-likes').textContent = e.likes_count;
        document.getElementById('btn-like').classList.toggle('aime', !!e.mon_like);
        document.getElementById('btn-like').firstChild.textContent = e.mon_like ? '❤️ ' : '🤍 ';
      }} catch (err) {{}}
    }}
    chargerEtatLike();

    async function chargerCompte() {{
      try {{
        const r = await fetch(`${{API}}/api/comptes/moi`);
        if (!r.ok) return;
        const compte = await r.json();
        const auteurEl = document.getElementById('com-auteur');
        auteurEl.value = compte.pseudo;
        auteurEl.disabled = true;
        auteurEl.placeholder = 'Connecté en tant que ' + compte.pseudo;
      }} catch (err) {{}}
    }}
    chargerCompte();

    async function basculerLike() {{
      const btn = document.getElementById('btn-like');
      btn.disabled = true;
      try {{
        const fd = new FormData();
        fd.append('visiteur_id', visiteurId());
        const r = await fetch(`${{API}}/api/editorial/editos/${{EDITO_ID}}/like`, {{ method: 'POST', body: fd }});
        const d = await r.json();
        if (d.ok) {{
          document.getElementById('nb-likes').textContent = d.likes;
          btn.classList.toggle('aime', d.aime);
          btn.firstChild.textContent = d.aime ? '❤️ ' : '🤍 ';
        }}
      }} catch (err) {{}}
      btn.disabled = false;
    }}

    async function envoyerCommentaire() {{
      const auteurEl = document.getElementById('com-auteur');
      const texteEl  = document.getElementById('com-texte');
      const btn      = document.getElementById('btn-commenter');
      const texte    = texteEl.value.trim();
      if (!texte) return;
      btn.disabled = true;
      try {{
        const fd = new FormData();
        fd.append('auteur', auteurEl.value.trim());
        fd.append('texte', texte);
        const r = await fetch(`${{API}}/api/editorial/editos/${{EDITO_ID}}/commentaires`, {{ method: 'POST', body: fd }});
        const d = await r.json();
        if (!r.ok) throw new Error(d.detail || 'Erreur');
        const vide = document.getElementById('pas-de-commentaires');
        if (vide) vide.remove();
        const c = d.commentaire;
        const div = document.createElement('div');
        div.className = 'commentaire';
        div.dataset.id = c.id;
        div.innerHTML = `<div class="commentaire-meta"><strong>${{escHtml(c.auteur)}}</strong> · à l'instant</div>
          <div class="commentaire-texte">${{escHtml(c.texte).replace(/\\n/g, '<br>')}}</div>`;
        document.getElementById('liste-commentaires').appendChild(div);
        document.getElementById('nb-commentaires').textContent =
          document.querySelectorAll('#liste-commentaires .commentaire').length;
        texteEl.value = '';
      }} catch (err) {{
        alert('Impossible de publier le commentaire.');
      }}
      btn.disabled = false;
    }}
  </script>
</body>
</html>""")

@app.post("/api/editorial/editos/{edito_id}/valider")
async def api_editorial_valider_edito(edito_id: str, key: str = ""):
    _check_admin(key)
    if not await asyncio.to_thread(EE.publier_edito, edito_id):
        raise HTTPException(404, "Édito non trouvé.")
    return JSONResponse({"ok": True})

@app.delete("/api/editorial/editos/{edito_id}")
async def api_editorial_supprimer_edito(edito_id: str, key: str = ""):
    _check_admin(key)
    if not await asyncio.to_thread(EE.supprimer_edito, edito_id):
        raise HTTPException(404, "Édito non trouvé.")
    return JSONResponse({"ok": True})

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

# ══════════════════════════════════════════════════════════════════════════════
# DUELS EN TEMPS RÉEL — QCM de vocabulaire à deux, invitations, classement
# ══════════════════════════════════════════════════════════════════════════════

import duels as DU

# Registre des connexions WebSocket actives par duel, gardées par pseudo —
# en mémoire, process unique (pas de Redis/pubsub : ce projet tourne sur une
# seule instance). Indexer par pseudo (et pas juste une liste) permet de
# distinguer un vrai reconnect (l'ancien socket du pseudo n'est plus dans le
# registre) d'un pseudo actif usurpé par quelqu'un d'autre (encore présent).
DUEL_CONNEXIONS: dict[str, dict[str, WebSocket]] = {}

def _mots_jeu_pour_duel() -> list[dict]:
    base   = charger_mots_base()
    custom = charger_mots_custom()
    base_pular = {m["pular"] for m in base}
    return base + [m for m in custom if m.get("pular") not in base_pular]

async def _duel_diffuser(code: str, payload: dict):
    """Envoie l'état à jour à tous les sockets connectés sur ce duel."""
    morts = []
    for pseudo_connecte, ws in DUEL_CONNEXIONS.get(code, {}).items():
        try:
            await ws.send_json(payload)
        except Exception:
            morts.append(pseudo_connecte)
    for p in morts:
        DUEL_CONNEXIONS.get(code, {}).pop(p, None)

@app.post("/api/duel/creer")
async def api_duel_creer(
    request: Request,
    pseudo:       str = Form(...),
    theme:        str = Form("Tout"),
    nb_questions: int = Form(DU.NB_QUESTIONS),
):
    compte = await _compte_courant(request)
    pseudo = compte["pseudo"] if compte else pseudo.strip()[:40]
    if not pseudo:
        raise HTTPException(400, "Choisis un pseudo.")
    mots = _mots_jeu_pour_duel()
    if len(mots) < 4:
        raise HTTPException(503, "Pas assez de mots dans le jeu pour lancer un duel.")
    duel = await asyncio.to_thread(DU.creer_duel, pseudo, mots, "web", theme, nb_questions)
    log.info(f"Duel créé: {duel['code']} par {pseudo} — thème={duel['theme']}, {len(duel['questions'])} questions")
    return JSONResponse({"ok": True, "duel": duel})

@app.get("/api/duel/meta")
async def api_duel_meta():
    return JSONResponse({"themes": DU.THEMES, "longueurs": DU.LONGUEURS_VALIDES})

@app.get("/api/duel/{code}")
async def api_duel_etat(code: str):
    duel = await asyncio.to_thread(DU.obtenir_duel, code)
    if not duel:
        raise HTTPException(404, "Duel introuvable.")
    return JSONResponse(duel)

@app.get("/api/duel/classement/top")
async def api_duel_classement(limite: int = 20):
    return JSONResponse(await asyncio.to_thread(DU.classement, limite))

@app.websocket("/ws/duel/{code}")
async def ws_duel(websocket: WebSocket, code: str, pseudo: str = ""):
    code = code.upper()
    pseudo = pseudo.strip()[:40]
    # On accepte d'abord la connexion : fermer avant accept() renvoie un rejet
    # HTTP brut (403) côté client, sans code exploitable — un message JSON
    # explicite avant fermeture est bien plus fiable pour le navigateur.
    await websocket.accept()

    duel = await asyncio.to_thread(DU.obtenir_duel, code)
    if not duel or not pseudo:
        await websocket.send_json({"type": "erreur", "message": "Code de duel invalide."})
        await websocket.close()
        return

    connexions_duel = DUEL_CONNEXIONS.setdefault(code, {})
    deja_joueur = any(j["pseudo"] == pseudo for j in duel["joueurs"])

    if deja_joueur and pseudo in connexions_duel:
        # Un pseudo déjà activement connecté — probablement quelqu'un
        # d'autre qui a tapé le même nom, pas un reconnect légitime.
        await websocket.send_json({"type": "erreur", "message": "Ce pseudo est déjà utilisé dans ce duel — choisis-en un autre."})
        await websocket.close()
        return

    if not deja_joueur:
        duel = await asyncio.to_thread(DU.rejoindre_duel, code, pseudo)
        if not duel:
            await websocket.send_json({"type": "erreur", "message": "Duel complet, ou pseudo déjà pris par l'autre joueur."})
            await websocket.close()
            return

    connexions_duel[pseudo] = websocket

    # Un joueur qui vient de rejoindre (ou qui se reconnecte) fait avancer
    # l'état pour tout le monde (l'adversaire déjà connecté doit voir
    # "en_cours" apparaître, ou récupérer l'état courant après une coupure).
    await _duel_diffuser(code, {"type": "etat", "duel": duel})

    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "reponse":
                duel = await asyncio.to_thread(
                    DU.enregistrer_reponse, code, pseudo,
                    int(msg.get("q", -1)), msg.get("reponse"), int(msg.get("temps_ms", 0)),
                )
                if duel:
                    await _duel_diffuser(code, {"type": "etat", "duel": duel})
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Starlette peut lever RuntimeError("WebSocket is not connected...")
        # plutôt qu'un WebSocketDisconnect propre quand l'autre joueur se
        # déconnecte pendant qu'une diffusion est en cours sur ce socket —
        # dans les deux cas, la connexion est bel et bien terminée.
        pass
    except Exception as e:
        log.warning(f"Erreur websocket duel {code} ({pseudo}): {e}")
    finally:
        if connexions_duel.get(pseudo) is websocket:
            connexions_duel.pop(pseudo, None)

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
    invalider_cache_prompt()
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
            invalider_cache_prompt()
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

# ── Phrases custom ─────────────────────────────────────────────────────────────
FICHIER_PHRASES_CUSTOM = DOSSIER_JEU / "phrases_custom.json"

def charger_phrases_custom() -> list[dict]:
    if FICHIER_PHRASES_CUSTOM.exists():
        with open(FICHIER_PHRASES_CUSTOM, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_phrases_custom(phrases: list[dict]):
    with open(FICHIER_PHRASES_CUSTOM, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)

@app.get("/api/prof/phrases")
async def api_prof_phrases():
    return JSONResponse(charger_phrases_custom())

@app.post("/api/prof/phrase")
async def api_prof_ajouter_phrase(
    pular: str = Form(...),
    adlam: str = Form(""),
    fr:    str = Form(""),
    en:    str = Form(""),
    ar:    str = Form(""),
    cat:   str = Form("Autre"),
    pseudo: str = Form("prof"),
):
    if not pular.strip():
        raise HTTPException(400, "Le champ 'pular' est obligatoire.")
    phrases = charger_phrases_custom()
    nouveau = {
        "id":     str(uuid.uuid4())[:8],
        "pular":  pular.strip(),
        "adlam":  adlam.strip() or latin_vers_adlam(pular.strip()),
        "fr":     fr.strip(),
        "en":     en.strip(),
        "ar":     ar.strip(),
        "cat":    cat.strip() or "Autre",
        "pseudo": pseudo.strip(),
        "date":   datetime.now().isoformat(),
    }
    phrases.append(nouveau)
    sauver_phrases_custom(phrases)
    log.info(f"Phrase ajoutée: '{pular[:50]}' ({pseudo})")
    return JSONResponse({"ok": True, "phrase": nouveau})

@app.put("/api/prof/phrase/{phrase_id}")
async def api_prof_modifier_phrase(
    phrase_id: str,
    pular: str = Form(...),
    adlam: str = Form(""),
    fr:    str = Form(""),
    en:    str = Form(""),
    ar:    str = Form(""),
    cat:   str = Form("Autre"),
):
    phrases = charger_phrases_custom()
    for p in phrases:
        if p["id"] == phrase_id:
            p.update({
                "pular":   pular.strip(),
                "adlam":   adlam.strip() or latin_vers_adlam(pular.strip()),
                "fr":      fr.strip(),
                "en":      en.strip(),
                "ar":      ar.strip(),
                "cat":     cat.strip() or "Autre",
                "modifie": datetime.now().isoformat(),
            })
            sauver_phrases_custom(phrases)
            return JSONResponse({"ok": True, "phrase": p})
    raise HTTPException(404, f"Phrase {phrase_id} introuvable.")

@app.delete("/api/prof/phrase/{phrase_id}")
async def api_prof_supprimer_phrase(phrase_id: str):
    phrases = charger_phrases_custom()
    avant = len(phrases)
    phrases = [p for p in phrases if p["id"] != phrase_id]
    if len(phrases) == avant:
        raise HTTPException(404, f"Phrase {phrase_id} introuvable.")
    sauver_phrases_custom(phrases)
    return JSONResponse({"ok": True})

# ── Données de base : phrases + mots (éditables via le panel prof) ─────────────
FICHIER_PHRASES_BASE = DOSSIER_JEU / "phrases_base.json"
FICHIER_MOTS_BASE    = DOSSIER_JEU / "mots_base.json"

_PHRASES_SEED = [
    # (pular, fr, en, ar, cat)
    ("Jam waali? Jam tan, baŋ-baŋ.",           "Comment vas-tu? Je vais bien, merci.",                      "How are you? I'm fine, thank you.",                          "كيف حالك؟ أنا بخير، شكرًا.",                       "Salutations"),
    ("Hol tò innde maa?",                       "Comment t'appelles-tu?",                                    "What is your name?",                                         "ما اسمك؟",                                            "Salutations"),
    ("Innde am ko Amadou. Mi jooɗii e Kanade.", "Je m'appelle Amadou. J'habite au Canada.",                  "My name is Amadou. I live in Canada.",                       "اسمي أمادو. أسكن في كندا.",                           "Salutations"),
    ("A jaaraama walaa! Alla hokku jam.",        "Merci beaucoup! Qu'Allah te donne la paix.",               "Thank you very much! May Allah grant you peace.",            "شكرًا جزيلاً! أعطاك الله السلام.",                     "Salutations"),
    ("Nde ndarii? Nde warii?",                  "D'où viens-tu? Où vas-tu?",                                "Where do you come from? Where are you going?",               "من أين أتيت؟ إلى أين تذهب؟",                          "Salutations"),
    ("Bismillahi Rahmaani Rahiimi.",             "Au nom d'Allah, le Clément, le Miséricordieux.",           "In the name of Allah, the Most Gracious, the Most Merciful.", "بسم الله الرحمن الرحيم.",                             "Islam"),
    ("Alhamdulillaahi Rabbil aalamiin.",         "Louange à Allah, Seigneur des mondes.",                    "Praise be to Allah, Lord of the worlds.",                    "الحمد لله رب العالمين.",                              "Islam"),
    ("Allahu Akbar, Allah mo Moƴƴo, Allah mo Jom baawɗe fof.", "Allah est Grand, Allah est Bon, Allah est Tout-Puissant.", "Allah is Great, Allah is Good, Allah is All-Powerful.", "الله أكبر، الله الطيب، الله القادر على كل شيء.",       "Islam"),
    ("Mi andaa ko Allah yiɗi. Mi yiɗi janngude Al-Qur'aana.", "Je sais ce qu'Allah aime. J'aime lire le Coran.", "I know what Allah loves. I love reading the Quran.",     "أعرف ما يحبه الله. أحب قراءة القرآن.",                "Islam"),
    ("Ramadan woni lewru barke e naafoore.",     "Le Ramadan est un mois de bénédiction et de bienfaits.",  "Ramadan is a month of blessing and goodness.",               "رمضان شهر البركة والخير.",                            "Islam"),
    ("Minen kuɓɓi. Mi jogii debbo e ɓiɓɓe tati.", "Je suis marié. J'ai une femme et trois enfants.",       "I am married. I have a wife and three children.",            "أنا متزوج. لدي زوجة وثلاثة أطفال.",                    "Famille"),
    ("Baaba am woni ngesa. Yinaande am woni galle.", "Mon père est au champ. Ma mère est à la maison.",    "My father is in the field. My mother is at home.",           "أبي في الحقل. أمي في المنزل.",                        "Famille"),
    ("Mi yiɗi ɓiɓɓe am haa ɓuri fof.",          "J'aime mes enfants plus que tout.",                       "I love my children more than anything.",                     "أحب أطفالي أكثر من أي شيء.",                          "Famille"),
    ("Worɓe e rewɓe fof poti yiɗde famili maɓɓe.", "Les hommes et les femmes doivent aimer leur famille.", "Men and women must love their family.",                      "يجب على الرجال والنساء أن يحبوا عائلتهم.",             "Famille"),
    ("Hannde subaka, mi ñaami nyiiri e kosam.",  "Ce matin, j'ai mangé du riz avec du lait.",               "This morning, I ate rice with milk.",                        "هذا الصباح، أكلت الأرز مع الحليب.",                    "Quotidien"),
    ("Ndiyam moƴƴi. Ñaamdu moƴƴi faa jeyɗo.",   "L'eau est bonne. La nourriture est bonne pour celui qui en a.", "Water is good. Food is good for those who have it.",   "الماء طيب. الطعام طيب لمن يملكه.",                     "Quotidien"),
    ("Mi yahay suudu janngo sakkitin.",           "J'irai à l'école demain matin.",                         "I will go to school tomorrow morning.",                      "سأذهب إلى المدرسة غدًا صباحًا.",                       "Quotidien"),
    ("Leydi pular woni leydi moƴƴere.",          "Le pays peul est un beau pays.",                          "The Fula land is a beautiful country.",                      "بلاد الفولاني بلد جميل.",                             "Quotidien"),
    ("Ko waɗi-ɗaa hannde? Mi golliima tawa.",    "Qu'as-tu fait aujourd'hui? J'ai travaillé fort.",         "What did you do today? I worked hard.",                      "ماذا فعلت اليوم؟ عملت بجد.",                          "Quotidien"),
    ("Nagge am jogii ɓiɓɓe ɗiɗi yontere hee.",  "Ma vache a eu deux veaux cette semaine.",                 "My cow had two calves this week.",                           "ولدت بقرتي عجلين هذا الأسبوع.",                       "Nature"),
    ("Ladde mawndi. Ladde moƴƴi faa aynaaɓe.",  "La forêt est grande. La forêt est bonne pour les éleveurs.", "The forest is big. The forest is good for herders.",      "الغابة كبيرة. الغابة جيدة للرعاة.",                    "Nature"),
    ("Ndungu wari. Ndiyam ɓurtii e maayo.",      "La saison des pluies est arrivée. L'eau a débordé du fleuve.", "The rainy season has arrived. The river has overflowed.", "جاء موسم الأمطار. فاض النهر.",                       "Nature"),
    ("Winde mawndi woni dow ladde.",              "Le grand village est au-dessus de la forêt.",             "The big village is above the forest.",                       "القرية الكبيرة فوق الغابة.",                          "Nature"),
    ("Pulaagu woni ndimaagu e moƴƴere e muuɗum.", "Le Pulaagu c'est la noblesse, la bonté et la pudeur.",  "Pulaagu is nobility, kindness and modesty.",                 "البولاآڠو هو النبل والطيبة والحياء.",                  "Culture"),
    ("Semteende woni tiitoonde Pullo kañum.",    "La pudeur est le fondement de l'identité peule.",         "Modesty is the foundation of Fula identity.",                "الحياء هو أساس الهوية الفولانية.",                    "Culture"),
    ("Ko feewde haa ɓuri yiɗde woni gollirde.",  "Ce qui est bien et ce qu'on aime, c'est ce qu'il faut faire.", "What is good and what one loves is what should be done.", "ما هو جيد وما نحبه هو ما يجب فعله.",                  "Culture"),
    ("Gerɗol peelo woni moƴƴere e teddungal.",   "La musique peule est beauté et dignité.",                 "Fula music is beauty and dignity.",                          "الموسيقى الفولانية جمال وكرامة.",                     "Culture"),
]

# Nouveaux sujets (histoire, sciences, philosophie, politique, proverbes) :
# brouillons rédigés par Claude, PAS relus par un locuteur natif du pular.
# Marqués `a_verifier` pour ne jamais se mélanger silencieusement aux
# phrases déjà établies tant qu'un locuteur natif ne les a pas validées.
_PHRASES_NOUVEAUX_SUJETS = [
    ("Taariik amen teddi.",                    "Notre histoire est précieuse.",              "Our history is precious.",              "تاريخنا ثمين.",                      "Histoire"),
    ("Mi yiɗi anndude taariik amen.",           "J'aime connaître notre histoire.",           "I love knowing our history.",           "أحب معرفة تاريخنا.",                 "Histoire"),
    ("Mi yiɗi jangude siyaans.",                "J'aime étudier les sciences.",               "I love studying science.",              "أحب دراسة العلوم.",                  "Scientifique"),
    ("Siyaans hokku en ganndal.",               "La science nous donne du savoir.",           "Science gives us knowledge.",           "العلم يمنحنا المعرفة.",              "Scientifique"),
    ("Filosofi woni miijo.",                    "La philosophie est une réflexion.",          "Philosophy is a reflection.",           "الفلسفة تأمل.",                     "Philosophie"),
    ("Mi yiɗi miijaade ko woni goonga.",        "J'aime réfléchir à ce qui est vrai.",        "I love reflecting on what is true.",    "أحب التفكير فيما هو صحيح.",          "Philosophie"),
    ("Politik woni fii leydi.",                 "La politique concerne le pays.",             "Politics concerns the country.",        "السياسة تتعلق بالبلد.",              "Politique"),
    ("En fof poti wallitde leydi amen.",        "Nous devons tous aider notre pays.",         "We must all help our country.",         "يجب علينا جميعا مساعدة بلدنا.",       "Politique"),
    ("Ko moƴƴi woni ko poti gollirde.",         "Ce qui est bon est ce qu'il faut faire.",    "What is good is what must be done.",    "ما هو جيد هو ما يجب فعله.",          "Proverbes"),
    ("Semteende ɓuri jawdi.",                   "La pudeur vaut plus que la richesse.",       "Modesty is worth more than wealth.",    "الحياء أثمن من الثروة.",             "Proverbes"),
]

_MOTS_SEED = [
    ("🐄","Vache","nagge","Animaux"),    ("🐑","Mouton","mbabba","Animaux"),
    ("🐐","Chèvre","mbewa","Animaux"),   ("🐎","Cheval","puccu","Animaux"),
    ("🐓","Poule","gertooɗe","Animaux"), ("🐕","Chien","rawaandu","Animaux"),
    ("🐈","Chat","muuse","Animaux"),     ("🦁","Lion","liingu","Animaux"),
    ("🐊","Caïman","baylo","Animaux"),   ("🦅","Aigle","galeejo","Animaux"),
    ("🐍","Serpent","mbeewa","Animaux"), ("🦋","Papillon","lekki","Animaux"),
    ("🏠","Maison","galle","Objets"),    ("💧","Eau","ndiyam","Objets"),
    ("🔥","Feu","jaango","Objets"),      ("☀️","Soleil","naange","Objets"),
    ("🌙","Lune","lewru","Objets"),      ("🌳","Arbre","ledde","Objets"),
    ("🥛","Lait","kosam","Objets"),      ("🍚","Riz","nyiiri","Objets"),
    ("📚","Livre","defte","Objets"),     ("🥁","Tambour","tammbari","Objets"),
    ("🔪","Couteau","lahal","Objets"),   ("🪣","Calebasse","hoore","Objets"),
    ("👁️","Œil","yitere","Corps"),      ("👂","Oreille","nowru","Corps"),
    ("👃","Nez","hinere","Corps"),       ("👄","Bouche","hunuko","Corps"),
    ("✋","Main","juuɗe","Corps"),       ("🦶","Pied","koyngal","Corps"),
    ("🦷","Dents","ñiiɓe","Corps"),      ("💪","Bras","hakke","Corps"),
    ("🌧️","Pluie","ndungu","Nature"),   ("🌊","Fleuve","maayo","Nature"),
    ("🌾","Champ","ngesa","Nature"),     ("🌿","Herbe","gaawri","Nature"),
    ("⛰️","Montagne","tugal","Nature"),  ("🌬️","Vent","hendu","Nature"),
    ("🌑","Nuit","jamma","Nature"),
    ("👨","Père","baaba","Famille"),     ("👩","Mère","yinaande","Famille"),
    ("👶","Enfant","ɓiɗɗo","Famille"),  ("👴","Grand-père","kaawu","Famille"),
    ("👵","Grand-mère","mawndoo","Famille"),("👫","Époux","gorko","Famille"),
    ("👭","Femme","debbo","Famille"),
]

def charger_phrases_base() -> list[dict]:
    """
    Charge les phrases de base. Migration automatique et idempotente :
    - complète en/ar sur les phrases d'origine si le fichier a été écrit sur
      disque avant l'ajout des traductions (repérées par leur texte pular) ;
    - ajoute les phrases des nouveaux sujets (histoire, sciences...) si elles
      ne sont pas encore présentes, sans dupliquer ni écraser des phrases
      déjà éditées par un admin.
    """
    traductions_originales = {p: (e, a) for p, f, e, a, c in _PHRASES_SEED}
    nouveaux = {
        f"phn{i}": {"id": f"phn{i}", "pular": p, "fr": f, "en": e, "ar": a, "cat": c, "a_verifier": True}
        for i, (p, f, e, a, c) in enumerate(_PHRASES_NOUVEAUX_SUJETS)
    }

    if FICHIER_PHRASES_BASE.exists():
        data = json.loads(FICHIER_PHRASES_BASE.read_text(encoding="utf-8"))
        modifie = False
        for ph in data:
            if ph.get("pular") in traductions_originales:
                en, ar = traductions_originales[ph["pular"]]
                if not ph.get("en"):
                    ph["en"] = en; modifie = True
                if not ph.get("ar"):
                    ph["ar"] = ar; modifie = True
        ids_presents = {ph.get("id") for ph in data}
        for nid, nph in nouveaux.items():
            if nid not in ids_presents:
                data.append(nph)
                modifie = True
        if modifie:
            FICHIER_PHRASES_BASE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    data = [
        {"id": f"ph{i}", "pular": p, "fr": f, "en": e, "ar": a, "cat": c}
        for i, (p, f, e, a, c) in enumerate(_PHRASES_SEED)
    ] + list(nouveaux.values())
    FICHIER_PHRASES_BASE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def sauver_phrases_base(phrases: list[dict]):
    FICHIER_PHRASES_BASE.write_text(json.dumps(phrases, ensure_ascii=False, indent=2), encoding="utf-8")

def charger_mots_base() -> list[dict]:
    if FICHIER_MOTS_BASE.exists():
        return json.loads(FICHIER_MOTS_BASE.read_text(encoding="utf-8"))
    data = [{"id": f"m{i}", "emoji": e, "fr": f, "pular": p, "cat": c} for i, (e, f, p, c) in enumerate(_MOTS_SEED)]
    FICHIER_MOTS_BASE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data

def sauver_mots_base(mots: list[dict]):
    FICHIER_MOTS_BASE.write_text(json.dumps(mots, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/phrases-toutes")
async def api_phrases_toutes():
    """Toutes les phrases : base + custom fusionnées."""
    base   = charger_phrases_base()
    custom = charger_phrases_custom()
    base_pular = {p["pular"] for p in base}
    return JSONResponse(
        base + [p for p in custom if p.get("pular") not in base_pular],
        headers={"Cache-Control": "no-store"},
    )

@app.put("/api/prof/phrase-base/{phrase_id}")
async def api_modifier_phrase_base(
    phrase_id: str,
    pular: str = Form(...),
    fr:    str = Form(""),
    en:    str = Form(""),
    ar:    str = Form(""),
    cat:   str = Form("Autre"),
):
    phrases = charger_phrases_base()
    idx = next((i for i, p in enumerate(phrases) if p.get("id") == phrase_id), None)
    if idx is None:
        raise HTTPException(404, "Phrase de base non trouvée.")
    pular_clean = pular.strip()
    phrases[idx].update({
        "pular": pular_clean,
        "adlam": latin_vers_adlam(pular_clean),
        "fr":    fr.strip(),
        "en":    en.strip(),
        "ar":    ar.strip(),
        "cat":   cat,
        "modifie": datetime.now().isoformat(),
    })
    # Une modification par un admin vaut relecture : on retire le badge
    # "à vérifier" éventuellement posé sur un brouillon généré par l'IA.
    phrases[idx].pop("a_verifier", None)
    sauver_phrases_base(phrases)
    return JSONResponse({"ok": True, "phrase": phrases[idx]})

@app.delete("/api/prof/phrase-base/{phrase_id}")
async def api_supprimer_phrase_base(phrase_id: str):
    phrases = charger_phrases_base()
    sauver_phrases_base([p for p in phrases if p.get("id") != phrase_id])
    return JSONResponse({"ok": True})

# ── Suggestions de correction de phrases (communauté) ─────────────────────────
@app.post("/api/suggestion-phrase")
async def api_suggestion_phrase(
    phrase_id:      str = Form(...),
    pular_original: str = Form(...),
    pular_corrige:  str = Form(...),
    fr_corrige:     str = Form(""),
    en_corrige:     str = Form(""),
    ar_corrige:     str = Form(""),
    pseudo:         str = Form("anonyme"),
):
    pular_c = pular_corrige.strip()
    if not pular_c:
        raise HTTPException(400, "Le texte corrigé est vide.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    entry = {
        "fichier":        f"{phrase_id}_{ts}.json",
        "phrase_id":      phrase_id,
        "pseudo":         pseudo[:50],
        "pular_original": pular_original.strip(),
        "pular_corrige":  pular_c,
        "fr_corrige":     fr_corrige.strip(),
        "en_corrige":     en_corrige.strip(),
        "ar_corrige":     ar_corrige.strip(),
        "timestamp":      datetime.now().isoformat(),
    }
    path = DOSSIER_CORRECTIONS_PHRASES / entry["fichier"]
    path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Suggestion phrase: {phrase_id} | pseudo={pseudo[:20]}")
    return JSONResponse({"ok": True})

@app.get("/api/prof/suggestions-phrases")
async def api_suggestions_phrases():
    suggestions = []
    for f in sorted(DOSSIER_CORRECTIONS_PHRASES.glob("*.json")):
        try:
            suggestions.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return JSONResponse(suggestions)

@app.post("/api/prof/appliquer-suggestion-phrase")
async def api_appliquer_suggestion_phrase(
    phrase_id:     str = Form(...),
    pular_corrige: str = Form(...),
    fr_corrige:    str = Form(""),
    en_corrige:    str = Form(""),
    ar_corrige:    str = Form(""),
    fichier:       str = Form(...),
):
    pular_c = pular_corrige.strip()
    for charger, sauver in [
        (charger_phrases_base, sauver_phrases_base),
        (charger_phrases_custom, sauver_phrases_custom),
    ]:
        phrases = charger()
        for p in phrases:
            if p.get("id") == phrase_id:
                p["pular"] = pular_c
                p["adlam"] = latin_vers_adlam(pular_c)
                if fr_corrige.strip():
                    p["fr"] = fr_corrige.strip()
                if en_corrige.strip():
                    p["en"] = en_corrige.strip()
                if ar_corrige.strip():
                    p["ar"] = ar_corrige.strip()
                p["modifie"] = datetime.now().isoformat()
                sauver(phrases)
                (DOSSIER_CORRECTIONS_PHRASES / fichier).unlink(missing_ok=True)
                invalider_cache_prompt()
                return JSONResponse({"ok": True})
    raise HTTPException(404, f"Phrase {phrase_id} introuvable.")

@app.delete("/api/prof/suggestion-phrase/{nom}")
async def api_ignorer_suggestion_phrase(nom: str):
    (DOSSIER_CORRECTIONS_PHRASES / nom).unlink(missing_ok=True)
    return JSONResponse({"ok": True})

# ══════════════════════════════════════════════════════════════════════════════
# TTS — Synthèse vocale OmniVoice (k2-fsa, 600+ langues dont fub = Fulfulde)
# ══════════════════════════════════════════════════════════════════════════════

_ov_client = None

def _get_ov_client():
    global _ov_client
    if _ov_client is None:
        from gradio_client import Client
        _ov_client = Client("k2-fsa/OmniVoice")
        log.info("OmniVoice client connecté (k2-fsa/OmniVoice)")
    return _ov_client

def _numpy_vers_wav(sr: int, data) -> bytes:
    """Convertit un tableau numpy audio en bytes WAV (16-bit mono)."""
    import numpy as np
    pcm = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def _tts_generer(texte: str, langue: str) -> bytes:
    """
    Appelle OmniVoice via le Space HF (gradio_client).
    Mode clone sans ref_audio → OmniVoice génère une voix par défaut pour la langue.
    Retourne les bytes d'un fichier WAV.
    """
    client = _get_ov_client()

    # Paramètres mode clone sans audio de référence
    result = client.predict(
        texte,   # text
        langue,  # language (ex: "fub" = Fulfulde Adamawa)
        None,    # ref_audio  — pas de clonage de voix
        None,    # ref_text
        "female, adult, clear, slow",  # instruct — attributs de voix
        20,      # num_step
        3.0,     # guidance_scale
        True,    # denoise
        1.0,     # speed
        0.0,     # duration (0 = auto)
        True,    # preprocess_prompt
        True,    # postprocess_output
        api_name="/_clone_fn",
    )

    # Le résultat est (audio, status_msg) — audio = (sr, np.ndarray) ou fichier path
    audio_out = result[0] if isinstance(result, (list, tuple)) else result

    if isinstance(audio_out, str):
        # Chemin vers un fichier WAV temporaire
        with open(audio_out, "rb") as f:
            return f.read()

    if isinstance(audio_out, tuple):
        sr, data = audio_out
        return _numpy_vers_wav(sr, data)

    raise ValueError(f"Format audio OmniVoice inattendu: {type(audio_out)}")

@app.get("/api/tts")
async def api_tts(texte: str, langue: str = "fub"):
    """
    Synthèse vocale OmniVoice pour une phrase Pular/Fulfulde.
    Les résultats sont mis en cache sur disque.
    langue : code ISO 639-3 — fub = Fulfulde Adamawa (Guinée, Cameroun)
    """
    texte = texte.strip()
    if not texte:
        raise HTTPException(400, "Texte vide.")

    cle = hashlib.md5(f"{langue}:{texte}".encode()).hexdigest()
    cache_wav = DOSSIER_TTS_CACHE / f"{cle}.wav"

    if cache_wav.exists():
        return FileResponse(
            str(cache_wav), media_type="audio/wav",
            headers={"Cache-Control": "max-age=604800"},
        )

    try:
        audio_bytes = await asyncio.to_thread(_tts_generer, texte, langue)
        cache_wav.write_bytes(audio_bytes)
        return Response(
            content=audio_bytes, media_type="audio/wav",
            headers={"Cache-Control": "max-age=604800"},
        )
    except Exception as e:
        log.error(f"TTS OmniVoice: {e}")
        raise HTTPException(503, "Synthèse vocale temporairement indisponible.")

@app.get("/api/mots-tous")
async def api_mots_tous():
    """Tous les mots : base + custom fusionnés."""
    base   = charger_mots_base()
    custom = charger_mots_custom()
    base_pular = {m["pular"] for m in base}
    return JSONResponse(base + [m for m in custom if m.get("pular") not in base_pular])

@app.put("/api/prof/mot-base/{mot_id}")
async def api_modifier_mot_base(
    mot_id: str,
    emoji: str = Form("❓"),
    fr:    str = Form(...),
    pular: str = Form(...),
    cat:   str = Form("Autre"),
):
    mots = charger_mots_base()
    idx = next((i for i, m in enumerate(mots) if m.get("id") == mot_id), None)
    if idx is None:
        raise HTTPException(404, "Mot de base non trouvé.")
    mots[idx].update({"emoji": emoji.strip(), "fr": fr.strip(), "pular": pular.strip(), "cat": cat})
    sauver_mots_base(mots)
    return JSONResponse({"ok": True})

@app.delete("/api/prof/mot-base/{mot_id}")
async def api_supprimer_mot_base(mot_id: str):
    mots = charger_mots_base()
    sauver_mots_base([m for m in mots if m.get("id") != mot_id])
    return JSONResponse({"ok": True})


# ── Telegram scraping ──────────────────────────────────────────────────────────
_telegram_en_cours = False
_TELEGRAM_DOSSIER  = PROJET_ROOT / "corpus-pular" / "processed" / "telegram"
_TELEGRAM_STATUS   = _TELEGRAM_DOSSIER / "status.json"

@app.get("/api/prof/telegram/status")
async def api_telegram_status():
    nb_audio = len(list((_TELEGRAM_DOSSIER / "audio").glob("*"))) if (_TELEGRAM_DOSSIER / "audio").exists() else 0
    nb_msg   = 0
    jsonl_dir = _TELEGRAM_DOSSIER / "jsonl"
    if jsonl_dir.exists():
        for f in jsonl_dir.glob("*.jsonl"):
            try: nb_msg += sum(1 for _ in f.open(encoding="utf-8"))
            except Exception: pass
    base = {
        "configured": bool(os.getenv("TELEGRAM_API_ID") and os.getenv("TELEGRAM_API_HASH")),
        "nb_audio": nb_audio,
        "nb_messages": nb_msg,
        "en_cours": _telegram_en_cours,
    }
    if _TELEGRAM_STATUS.exists():
        try: base.update(json.loads(_TELEGRAM_STATUS.read_text(encoding="utf-8")))
        except Exception: pass
    return JSONResponse(base)

@app.post("/api/prof/telegram/lancer")
async def api_telegram_lancer(
    canaux:     str  = Form(""),
    limite:     int  = Form(200),
    sans_audio: bool = Form(False),
):
    """Lance le scraper Telegram en arrière-plan."""
    global _telegram_en_cours
    if _telegram_en_cours:
        raise HTTPException(409, "Un scraping est déjà en cours.")
    if not os.getenv("TELEGRAM_API_ID") or not os.getenv("TELEGRAM_API_HASH"):
        raise HTTPException(400, "Configure TELEGRAM_API_ID, TELEGRAM_API_HASH et TELEGRAM_PHONE dans Railway Variables.")

    _telegram_en_cours = True

    async def _run():
        global _telegram_en_cours
        import sys
        try:
            args = [sys.executable, str(PROJET_ROOT / "scripts" / "telegram_scraper.py"),
                    "--limite", str(limite)]
            if canaux.strip():
                args += ["--canaux"] + canaux.strip().split()
            if sans_audio:
                args += ["--sans-audio"]
            debut = datetime.now().isoformat()
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=str(PROJET_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=3600)
            status = {
                "dernier_run": debut,
                "fin_run": datetime.now().isoformat(),
                "ok": proc.returncode == 0,
                "canaux": canaux.strip() or "défaut",
                "limite": limite,
                "en_cours": False,
            }
            _TELEGRAM_STATUS.parent.mkdir(parents=True, exist_ok=True)
            _TELEGRAM_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"Scraping Telegram terminé: code={proc.returncode}")
        except Exception as e:
            log.error(f"Erreur scraping Telegram: {e}")
        finally:
            _telegram_en_cours = False

    asyncio.create_task(_run())
    return JSONResponse({"ok": True, "message": "Scraping lancé en arrière-plan (jusqu'à 1h)."})


# ── Contributeurs ───────────────────────────────────────────────────────────────
@app.get("/api/prof/contributeurs")
async def api_prof_contributeurs():
    contribs_par_pseudo: dict = {}
    if DOSSIER_CONTRIB.exists():
        for f in sorted(DOSSIER_CONTRIB.glob("*.json"), reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                pseudo = d.get("pseudo", "anonyme")
                if pseudo not in contribs_par_pseudo:
                    contribs_par_pseudo[pseudo] = {"pseudo": pseudo, "nb": 0, "derniere": ""}
                contribs_par_pseudo[pseudo]["nb"] += 1
                ts = d.get("timestamp", "")
                if ts > contribs_par_pseudo[pseudo]["derniere"]:
                    contribs_par_pseudo[pseudo]["derniere"] = ts
            except Exception:
                pass
    liste = sorted(contribs_par_pseudo.values(), key=lambda x: x["nb"], reverse=True)
    return JSONResponse({"contributeurs": liste, "total": len(liste)})

# ── Corrections déjà faites ────────────────────────────────────────────────────
@app.get("/api/prof/corrections")
async def api_prof_corrections(limit: int = 30):
    corrections = []
    if DOSSIER_CORRECTIONS.exists():
        for f in sorted(DOSSIER_CORRECTIONS.glob("*.json"), reverse=True)[:limit]:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                corrections.append({
                    "id":            d.get("id", f.stem),
                    "pseudo":        d.get("pseudo", "?"),
                    "date":          d.get("timestamp", ""),
                    "texte_auto":    d.get("texte_auto", ""),
                    "texte_corrige": d.get("texte_corrige", ""),
                })
            except Exception:
                pass
    return JSONResponse({"corrections": corrections, "total": len(corrections)})

@app.get("/api/prof/fiabilite")
async def api_prof_fiabilite():
    """Métriques de fiabilité des transcriptions automatiques (WER, taux de correction)."""
    def _calculer():
        total = avec_paire = corrects = 0
        wers: list[float] = []
        tranches = {"0": 0, "1-25": 0, "26-50": 0, ">50": 0}
        audio_dispo = 0

        if DOSSIER_CONTRIB.exists():
            for f in DOSSIER_CONTRIB.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    total += 1
                    auto  = d.get("transcription_auto", "").strip()
                    final = d.get("texte_final", "").strip()
                    if not auto or not final:
                        continue
                    avec_paire += 1
                    audio_rel = d.get("audio", "")
                    if audio_rel and (PROJET_ROOT / audio_rel).exists():
                        audio_dispo += 1
                    w = calcul_wer(final, auto)
                    wers.append(w)
                    if w == 0.0:
                        corrects += 1
                        tranches["0"] += 1
                    elif w <= 0.25:
                        tranches["1-25"] += 1
                    elif w <= 0.50:
                        tranches["26-50"] += 1
                    else:
                        tranches[">50"] += 1
                except Exception:
                    pass

        wer_moyen    = round(sum(wers) / len(wers) * 100, 1) if wers else 0.0
        taux_correct = round(corrects / avec_paire * 100, 1) if avec_paire else 0.0
        return {
            "total_contributions": total,
            "avec_paire":          avec_paire,
            "corrects":            corrects,
            "corriges":            avec_paire - corrects,
            "taux_correct":        taux_correct,
            "wer_moyen":           wer_moyen,
            "tranches_wer":        tranches,
            "audio_utilisables":   audio_dispo,
        }

    return JSONResponse(await asyncio.to_thread(_calculer))


@app.get("/api/prof/exporter-whisper")
async def api_prof_exporter_whisper():
    """Exporte les paires audio+texte validées en ZIP pour fine-tuner Whisper (HuggingFace format)."""
    def _creer_zip() -> tuple:
        buf = io.BytesIO()
        nb = 0
        lignes_meta = ['audio_path,text\n']
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            if DOSSIER_CONTRIB.exists():
                for f in DOSSIER_CONTRIB.glob("*.json"):
                    try:
                        d = json.loads(f.read_text(encoding="utf-8"))
                        texte     = d.get("texte_final", "").strip()
                        audio_rel = d.get("audio", "")
                        if not texte or not audio_rel:
                            continue
                        audio_path = PROJET_ROOT / audio_rel
                        if not audio_path.exists():
                            continue
                        arc_name = f"audio/{audio_path.name}"
                        z.write(audio_path, arc_name)
                        texte_esc = texte.replace('"', '""')
                        lignes_meta.append(f'"{arc_name}","{texte_esc}"\n')
                        nb += 1
                    except Exception:
                        pass
            z.writestr("metadata.csv", "".join(lignes_meta))
            readme = (
                "# Dataset Pular — Whisper Fine-tuning\n\n"
                f"Nombre de paires audio/texte : {nb}\n\n"
                "## Format\n"
                "- `audio/` : fichiers audio (.webm)\n"
                "- `metadata.csv` : colonnes `audio_path,text`\n\n"
                "## Utilisation (Google Colab)\n"
                "```python\n"
                "from datasets import load_dataset\n"
                "ds = load_dataset('csv', data_files='metadata.csv')\n"
                "```\n"
            )
            z.writestr("README.md", readme)
        buf.seek(0)
        return buf.read(), nb

    contenu, nb = await asyncio.to_thread(_creer_zip)
    log.info(f"Export Whisper dataset: {nb} paires audio/texte, {len(contenu)//1024} KB")
    return StreamingResponse(
        io.BytesIO(contenu),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=pular_whisper_dataset.zip"},
    )


# ── Dataset : consulter / ajouter / supprimer / évolution ──────────────────────

@app.get("/api/prof/dataset")
async def api_prof_dataset(page: int = 1, limit: int = 15, status: str = "all", q: str = ""):
    """Liste paginée de toutes les contributions avec WER calculé."""
    def _lire():
        items = []
        if not DOSSIER_CONTRIB.exists():
            return {"items": [], "total": 0, "page": page, "pages": 1}
        for f in sorted(DOSSIER_CONTRIB.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                st = d.get("status", "pending")
                if status != "all" and st != status:
                    continue
                texte = (d.get("texte_final") or d.get("transcription_auto") or "").lower()
                if q and q.lower() not in texte:
                    continue
                auto  = d.get("transcription_auto", "").strip()
                final = d.get("texte_final", "").strip()
                wer   = round(calcul_wer(final, auto) * 100, 1) if auto and final else None
                audio_nom = Path(d.get("audio", "")).name if d.get("audio") else ""
                items.append({
                    "id":               d.get("id", f.stem),
                    "pseudo":           d.get("pseudo", "?"),
                    "date":             d.get("timestamp", "")[:10],
                    "texte_final":      final,
                    "transcription_auto": auto,
                    "status":           st,
                    "audio_nom":        audio_nom,
                    "wer":              wer,
                    "source":           d.get("source", ""),
                })
            except Exception:
                pass
        total  = len(items)
        start  = (page - 1) * limit
        return {
            "items": items[start:start + limit],
            "total": total,
            "page":  page,
            "pages": max(1, (total + limit - 1) // limit),
        }
    return JSONResponse(await asyncio.to_thread(_lire))


@app.get("/api/prof/dataset/evolution")
async def api_prof_dataset_evolution():
    """Évolution journalière du corpus : nb contributions + WER moyen par jour."""
    def _calculer():
        par_jour: dict = {}
        if DOSSIER_CONTRIB.exists():
            for f in DOSSIER_CONTRIB.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    jour = (d.get("timestamp") or "")[:10] or "?"
                    if jour not in par_jour:
                        par_jour[jour] = {"count": 0, "wer_sum": 0.0, "wer_n": 0, "corrects": 0}
                    par_jour[jour]["count"] += 1
                    auto  = d.get("transcription_auto", "").strip()
                    final = d.get("texte_final", "").strip()
                    if auto and final:
                        w = calcul_wer(final, auto)
                        par_jour[jour]["wer_sum"] += w
                        par_jour[jour]["wer_n"]   += 1
                        if w == 0.0:
                            par_jour[jour]["corrects"] += 1
                except Exception:
                    pass

        cumul = 0
        timeline = []
        for jour, info in sorted(par_jour.items()):
            if jour == "?":
                continue
            cumul += info["count"]
            wer_moy = round(info["wer_sum"] / info["wer_n"] * 100, 1) if info["wer_n"] else None
            taux_ok = round(info["corrects"] / info["wer_n"] * 100, 1) if info["wer_n"] else None
            timeline.append({
                "date":      jour,
                "count":     info["count"],
                "cumul":     cumul,
                "wer_moy":   wer_moy,
                "taux_ok":   taux_ok,
            })

        total_n   = sum(v["wer_n"]   for v in par_jour.values())
        total_ws  = sum(v["wer_sum"] for v in par_jour.values())
        total_ok  = sum(v["corrects"]for v in par_jour.values())
        total_cnt = sum(v["count"]   for v in par_jour.values())
        return {
            "timeline":         timeline[-30:],
            "total":            total_cnt,
            "wer_global":       round(total_ws / total_n * 100, 1) if total_n else None,
            "taux_ok_global":   round(total_ok / total_n * 100, 1) if total_n else None,
        }
    return JSONResponse(await asyncio.to_thread(_calculer))


@app.post("/api/prof/dataset/ajouter")
async def api_prof_dataset_ajouter(
    pular:  str = Form(...),
    fr:     str = Form(""),
    pseudo: str = Form("prof"),
):
    """Ajoute une entrée texte directement dans le dataset (sans audio, statut validé)."""
    pular = pular.strip()
    if not pular:
        raise HTTPException(400, "Le texte pular est obligatoire.")
    id_ = uuid.uuid4().hex[:8]
    entry = {
        "id":                id_,
        "pseudo":            pseudo or "prof",
        "transcription_auto": pular,
        "texte_final":       pular,
        "fr":                fr.strip(),
        "audio":             "",
        "timestamp":         datetime.now().isoformat(),
        "source":            "ajout_manuel",
        "status":            "valider",
    }
    (DOSSIER_CONTRIB / f"{id_}.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = charger_stats()
    stats["total_contributions"] = stats.get("total_contributions", 0) + 1
    stats["total_validations"]   = stats.get("total_validations",   0) + 1
    sauver_stats(stats)
    log.info(f"Dataset — entrée manuelle: {id_} = '{pular[:60]}'")
    return JSONResponse({"ok": True, "id": id_})


@app.delete("/api/prof/dataset/{id}")
async def api_prof_dataset_supprimer(id: str):
    """Supprime une contribution du dataset (et son audio si présent)."""
    f = DOSSIER_CONTRIB / f"{id}.json"
    if not f.exists():
        raise HTTPException(404, "Entrée non trouvée.")
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        audio_rel = d.get("audio", "")
        if audio_rel:
            ap = PROJET_ROOT / audio_rel
            if ap.exists():
                ap.unlink()
    except Exception:
        pass
    f.unlink()
    log.info(f"Dataset — suppression: {id}")
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
