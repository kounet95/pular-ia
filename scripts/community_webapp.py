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
    MAX_BYTES = 870
    base   = "Pular fulfulde Fouta Djallon fulani langue africaine."
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
            temperature=0.0,
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
    cat:   str = Form("Autre"),
):
    phrases = charger_phrases_custom()
    for p in phrases:
        if p["id"] == phrase_id:
            p.update({
                "pular":   pular.strip(),
                "adlam":   adlam.strip() or latin_vers_adlam(pular.strip()),
                "fr":      fr.strip(),
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
    ("Jam waali? Jam tan, baŋ-baŋ.",           "Comment vas-tu? Je vais bien, merci.",                      "Salutations"),
    ("Hol tò innde maa?",                       "Comment t'appelles-tu?",                                    "Salutations"),
    ("Innde am ko Amadou. Mi jooɗii e Kanade.", "Je m'appelle Amadou. J'habite au Canada.",                  "Salutations"),
    ("A jaaraama walaa! Alla hokku jam.",        "Merci beaucoup! Qu'Allah te donne la paix.",               "Salutations"),
    ("Nde ndarii? Nde warii?",                  "D'où viens-tu? Où vas-tu?",                                "Salutations"),
    ("Bismillahi Rahmaani Rahiimi.",             "Au nom d'Allah, le Clément, le Miséricordieux.",           "Islam"),
    ("Alhamdulillaahi Rabbil aalamiin.",         "Louange à Allah, Seigneur des mondes.",                    "Islam"),
    ("Allahu Akbar, Allah mo Moƴƴo, Allah mo Jom baawɗe fof.", "Allah est Grand, Allah est Bon, Allah est Tout-Puissant.", "Islam"),
    ("Mi andaa ko Allah yiɗi. Mi yiɗi janngude Al-Qur'aana.", "Je sais ce qu'Allah aime. J'aime lire le Coran.", "Islam"),
    ("Ramadan woni lewru barke e naafoore.",     "Le Ramadan est un mois de bénédiction et de bienfaits.",  "Islam"),
    ("Minen kuɓɓi. Mi jogii debbo e ɓiɓɓe tati.", "Je suis marié. J'ai une femme et trois enfants.",       "Famille"),
    ("Baaba am woni ngesa. Yinaande am woni galle.", "Mon père est au champ. Ma mère est à la maison.",    "Famille"),
    ("Mi yiɗi ɓiɓɓe am haa ɓuri fof.",          "J'aime mes enfants plus que tout.",                       "Famille"),
    ("Worɓe e rewɓe fof poti yiɗde famili maɓɓe.", "Les hommes et les femmes doivent aimer leur famille.", "Famille"),
    ("Hannde subaka, mi ñaami nyiiri e kosam.",  "Ce matin, j'ai mangé du riz avec du lait.",               "Quotidien"),
    ("Ndiyam moƴƴi. Ñaamdu moƴƴi faa jeyɗo.",   "L'eau est bonne. La nourriture est bonne pour celui qui en a.", "Quotidien"),
    ("Mi yahay suudu janngo sakkitin.",           "J'irai à l'école demain matin.",                         "Quotidien"),
    ("Leydi pular woni leydi moƴƴere.",          "Le pays peul est un beau pays.",                          "Quotidien"),
    ("Ko waɗi-ɗaa hannde? Mi golliima tawa.",    "Qu'as-tu fait aujourd'hui? J'ai travaillé fort.",         "Quotidien"),
    ("Nagge am jogii ɓiɓɓe ɗiɗi yontere hee.",  "Ma vache a eu deux veaux cette semaine.",                 "Nature"),
    ("Ladde mawndi. Ladde moƴƴi faa aynaaɓe.",  "La forêt est grande. La forêt est bonne pour les éleveurs.", "Nature"),
    ("Ndungu wari. Ndiyam ɓurtii e maayo.",      "La saison des pluies est arrivée. L'eau a débordé du fleuve.", "Nature"),
    ("Winde mawndi woni dow ladde.",              "Le grand village est au-dessus de la forêt.",             "Nature"),
    ("Pulaagu woni ndimaagu e moƴƴere e muuɗum.", "Le Pulaagu c'est la noblesse, la bonté et la pudeur.",  "Culture"),
    ("Semteende woni tiitoonde Pullo kañum.",    "La pudeur est le fondement de l'identité peule.",         "Culture"),
    ("Ko feewde haa ɓuri yiɗde woni gollirde.",  "Ce qui est bien et ce qu'on aime, c'est ce qu'il faut faire.", "Culture"),
    ("Gerɗol peelo woni moƴƴere e teddungal.",   "La musique peule est beauté et dignité.",                 "Culture"),
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
    if FICHIER_PHRASES_BASE.exists():
        return json.loads(FICHIER_PHRASES_BASE.read_text(encoding="utf-8"))
    data = [{"id": f"ph{i}", "pular": p, "fr": f, "cat": c} for i, (p, f, c) in enumerate(_PHRASES_SEED)]
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
    return JSONResponse(base + [p for p in custom if p.get("pular") not in base_pular])

@app.put("/api/prof/phrase-base/{phrase_id}")
async def api_modifier_phrase_base(
    phrase_id: str,
    pular: str = Form(...),
    fr:    str = Form(""),
    cat:   str = Form("Autre"),
):
    phrases = charger_phrases_base()
    idx = next((i for i, p in enumerate(phrases) if p.get("id") == phrase_id), None)
    if idx is None:
        raise HTTPException(404, "Phrase de base non trouvée.")
    phrases[idx].update({"pular": pular.strip(), "fr": fr.strip(), "cat": cat})
    sauver_phrases_base(phrases)
    return JSONResponse({"ok": True})

@app.delete("/api/prof/phrase-base/{phrase_id}")
async def api_supprimer_phrase_base(phrase_id: str):
    phrases = charger_phrases_base()
    sauver_phrases_base([p for p in phrases if p.get("id") != phrase_id])
    return JSONResponse({"ok": True})

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
