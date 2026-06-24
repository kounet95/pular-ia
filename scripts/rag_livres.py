"""
rag_livres.py — RAG (Retrieval-Augmented Generation) pour le corpus Pular

Gère l'ingestion de livres, poèmes, articles en pular/fulfulde:
  - Extraction de texte (PDF, DOCX, TXT, HTML)
  - Chunking avec chevauchement
  - Indexation ChromaDB (embeddings multilingues)
  - Recherche sémantique
  - Export vers le dataset d'entraînement

Usage:
    from scripts.rag_livres import indexer_livre, rechercher, exporter_dataset
"""

import re
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

PROJET_ROOT    = Path(__file__).resolve().parent.parent
DOSSIER_RAW    = PROJET_ROOT / "corpus-pular" / "livres" / "raw"
DOSSIER_META   = PROJET_ROOT / "corpus-pular" / "livres" / "metadata"
DOSSIER_CHROMA = PROJET_ROOT / "corpus-pular" / "rag" / "chroma"
FICHIER_LIVRES = PROJET_ROOT / "corpus-pular" / "livres" / "index.json"

for d in [DOSSIER_RAW, DOSSIER_META, DOSSIER_CHROMA]:
    d.mkdir(parents=True, exist_ok=True)

# ── Extraction de texte ───────────────────────────────────────────────────────

def extraire_pdf(chemin: Path) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(chemin) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages)
    except ImportError:
        log.error("Installe pdfplumber: pip install pdfplumber")
        return ""
    except Exception as e:
        log.error(f"PDF {chemin.name}: {e}")
        return ""

def extraire_docx(chemin: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(chemin))
        paragraphes = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphes)
    except ImportError:
        log.error("Installe python-docx: pip install python-docx")
        return ""

def extraire_html(chemin: Path) -> str:
    from html.parser import HTMLParser
    class Extracteur(HTMLParser):
        def __init__(self):
            super().__init__()
            self.textes = []
            self._skip = False
        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'head', 'nav', 'footer'):
                self._skip = True
        def handle_endtag(self, tag):
            if tag in ('script', 'style', 'head', 'nav', 'footer'):
                self._skip = False
        def handle_data(self, data):
            if not self._skip and data.strip():
                self.textes.append(data.strip())
    ext = Extracteur()
    ext.feed(chemin.read_text(encoding="utf-8", errors="replace"))
    return " ".join(ext.textes)

def extraire_texte(chemin: Path) -> str:
    """Extrait le texte brut selon le format du fichier."""
    ext = chemin.suffix.lower()
    if ext == ".pdf":
        return extraire_pdf(chemin)
    elif ext in (".docx", ".doc"):
        return extraire_docx(chemin)
    elif ext in (".html", ".htm"):
        return extraire_html(chemin)
    elif ext in (".txt", ".md", ".rst", ".text"):
        return chemin.read_text(encoding="utf-8", errors="replace")
    log.warning(f"Format non supporté: {ext}")
    return ""

# ── Nettoyage et chunking ─────────────────────────────────────────────────────

def nettoyer(texte: str) -> str:
    texte = re.sub(r"\r\n", "\n", texte)
    texte = re.sub(r"\n{4,}", "\n\n\n", texte)
    texte = re.sub(r"[ \t]{2,}", " ", texte)
    texte = re.sub(r"[^\S\n]+", " ", texte)
    return texte.strip()

def chunker(texte: str, taille_cible: int = 400, overlap: int = 80) -> list[str]:
    """
    Découpe le texte en chunks en respectant les limites de phrases.
    overlap = nb de caractères partagés entre chunks consécutifs.
    """
    texte = nettoyer(texte)
    if not texte:
        return []

    # Couper par double saut de ligne (paragraphes), puis affiner
    paragraphes = re.split(r"\n{2,}", texte)
    chunks, buf = [], ""

    for para in paragraphes:
        para = para.strip()
        if not para:
            continue

        # Si le paragraphe seul dépasse la cible, le découper en phrases
        if len(para) > taille_cible * 1.5:
            phrases = re.split(r"(?<=[.!?؟।\n])\s+", para)
            for ph in phrases:
                if len(buf) + len(ph) + 1 < taille_cible:
                    buf += (" " if buf else "") + ph
                else:
                    if buf:
                        chunks.append(buf.strip())
                    # Chevauchement
                    buf = buf[-overlap:].lstrip() + " " + ph if buf else ph
        else:
            if len(buf) + len(para) + 2 < taille_cible:
                buf += ("\n\n" if buf else "") + para
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = buf[-overlap:].lstrip() + "\n\n" + para if buf else para

    if buf.strip():
        chunks.append(buf.strip())

    # Filtrer les chunks trop courts (bruit)
    return [c for c in chunks if len(c) > 30]

# ── Index des livres (JSON simple) ────────────────────────────────────────────

def charger_index() -> list[dict]:
    if FICHIER_LIVRES.exists():
        with open(FICHIER_LIVRES, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_index(livres: list[dict]):
    with open(FICHIER_LIVRES, "w", encoding="utf-8") as f:
        json.dump(livres, f, ensure_ascii=False, indent=2)

# ── ChromaDB ──────────────────────────────────────────────────────────────────

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            client = chromadb.PersistentClient(path=str(DOSSIER_CHROMA))

            # paraphrase-multilingual-MiniLM-L12-v2 = 420MB, 50+ langues dont fulfulde
            emb_fn = SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            _collection = client.get_or_create_collection(
                name="corpus_pular",
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )
            log.info(f"ChromaDB prêt — {_collection.count()} chunks indexés")
        except ImportError:
            log.error("Installe chromadb: pip install chromadb sentence-transformers")
            raise
    return _collection

def indexer_livre(
    titre:    str,
    auteur:   str,
    langue:   str,
    texte:    str,
    livre_id: str,
) -> int:
    """Indexe un texte dans ChromaDB. Retourne le nb de chunks ajoutés."""
    collection = get_collection()
    chunks = chunker(texte)

    if not chunks:
        return 0

    ids       = [f"{livre_id}__{i:04d}" for i in range(len(chunks))]
    metadatas = [
        {
            "titre":    titre[:200],
            "auteur":   auteur[:100],
            "langue":   langue,
            "livre_id": livre_id,
            "chunk_no": i,
            "total":    len(chunks),
        }
        for i in range(len(chunks))
    ]

    # Batch de 100 (limite ChromaDB recommandée)
    BATCH = 100
    ajoutes = 0
    for start in range(0, len(ids), BATCH):
        batch_ids   = ids[start:start+BATCH]
        batch_docs  = chunks[start:start+BATCH]
        batch_meta  = metadatas[start:start+BATCH]
        # Vérifier doublons
        existants = set(collection.get(ids=batch_ids)["ids"])
        filtre    = [(i, d, m) for i, d, m in zip(batch_ids, batch_docs, batch_meta)
                     if i not in existants]
        if filtre:
            fi, fd, fm = zip(*filtre)
            collection.add(documents=list(fd), ids=list(fi), metadatas=list(fm))
            ajoutes += len(filtre)

    log.info(f"Indexé '{titre}': {ajoutes}/{len(chunks)} chunks nouveaux")

    # Sauvegarder les mots uniques extraits pour le prompt Whisper
    if chunks:
        import re as _re
        mots_vocab: set[str] = set()
        for chunk in chunks:
            for token in _re.split(r"[\s,.:;!?()\[\]\"']+", chunk):
                t = token.strip().lower()
                if len(t) > 2:
                    mots_vocab.add(t)
        vocab_path = DOSSIER_META / f"{livre_id}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(mots_vocab), f, ensure_ascii=False)

    return ajoutes

def rechercher(
    query:    str,
    n:        int = 5,
    langue:   str = None,
    livre_id: str = None,
) -> list[dict]:
    """Recherche sémantique dans le corpus."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    where = {}
    if langue:
        where["langue"] = langue
    if livre_id:
        where["livre_id"] = livre_id

    results = collection.query(
        query_texts=[query],
        n_results=min(n, total),
        where=where if where else None,
    )

    items = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i] if results.get("distances") else 1.0
        items.append({
            "texte":    doc,
            "titre":    meta.get("titre", "?"),
            "auteur":   meta.get("auteur", "?"),
            "langue":   meta.get("langue", "?"),
            "livre_id": meta.get("livre_id", "?"),
            "chunk_no": meta.get("chunk_no", 0),
            "score":    round(1 - dist, 3),
        })
    return items

def stats_rag() -> dict:
    try:
        collection = get_collection()
        livres = charger_index()
        return {
            "total_chunks":  collection.count(),
            "total_livres":  len(livres),
            "langues":       list({l.get("langue","?") for l in livres}),
        }
    except Exception:
        return {"total_chunks": 0, "total_livres": 0, "langues": []}

# ── Export dataset d'entraînement ─────────────────────────────────────────────

def exporter_dataset(
    dossier_sortie: Path = None,
    format_: str = "jsonl",
) -> Path:
    """
    Exporte tous les chunks du RAG au format JSONL pour l'entraînement LLM.
    Chaque ligne: {"text": "...", "meta": {...}}
    """
    if dossier_sortie is None:
        dossier_sortie = PROJET_ROOT / "corpus-pular" / "dataset" / "livres"
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    collection = get_collection()
    total = collection.count()
    if total == 0:
        log.warning("Aucun chunk dans le RAG")
        return dossier_sortie

    # Récupérer par batch de 500
    sortie = dossier_sortie / "corpus_livres.jsonl"
    compte = 0
    BATCH = 500

    with open(sortie, "w", encoding="utf-8") as f:
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=BATCH,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(batch["documents"], batch["metadatas"]):
                f.write(json.dumps({"text": doc, "meta": meta}, ensure_ascii=False) + "\n")
                compte += 1
            offset += BATCH

    log.info(f"Export dataset: {compte} chunks → {sortie}")
    return sortie
