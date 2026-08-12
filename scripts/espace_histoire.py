"""
espace_histoire.py — Espace Histoire & Patrimoine peul

Gère un corpus séparé du RAG linguistique, dédié à la reconstitution de
l'histoire peule (Macina, Fouta Djallon, Fouta Toro, etc.) :

  - Documents  : manuscrits familiaux, traditions orales transcrites,
                 chroniques, correspondances, thèses, articles, poèmes.
  - Familles   : annuaire des familles/lignages détentrices de sources,
                 avec un niveau de contact plutôt qu'un dépôt de fichier.

Chaque document porte :
  - royaume        : Macina, Fouta Djallon, Fouta Toro, Autre / Général
  - type_source     : manuscrit_familial, tradition_orale, correspondance,
                       chronique, these, article, poeme, autre
  - confidentialite : public | sur_demande | prive
      * public       → fichier + passages visibles par tous
      * sur_demande  → passages (extraits) visibles, fichier non
                        téléchargeable, contact du dépositaire affiché
      * prive        → seule la fiche (titre/royaume/type) est visible,
                        rien du contenu n'est exposé

Réutilise l'extraction de texte de rag_livres.py mais indexe dans une
collection ChromaDB séparée ("corpus_histoire") pour ne pas polluer le
corpus linguistique utilisé pour l'entraînement Whisper/mT5.
"""

import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

from rag_livres import extraire_texte, chunker  # réutilisation de l'extraction/chunking

log = logging.getLogger(__name__)

PROJET_ROOT     = Path(__file__).resolve().parent.parent
DOSSIER_RAW     = PROJET_ROOT / "corpus-pular" / "histoire" / "raw"
DOSSIER_CHROMA  = PROJET_ROOT / "corpus-pular" / "histoire" / "chroma"
FICHIER_DOCS    = PROJET_ROOT / "corpus-pular" / "histoire" / "documents.json"
FICHIER_FAM     = PROJET_ROOT / "corpus-pular" / "histoire" / "familles.json"

for d in [DOSSIER_RAW, DOSSIER_CHROMA]:
    d.mkdir(parents=True, exist_ok=True)

ROYAUMES = ["Macina", "Fouta Djallon", "Fouta Toro", "Bundu", "Adamaoua", "Autre / Général"]
TYPES_SOURCE = {
    "manuscrit_familial": "📜 Manuscrit familial",
    "tradition_orale":    "🗣️ Tradition orale transcrite",
    "correspondance":     "✉️ Correspondance",
    "chronique":          "📖 Chronique historique",
    "these":              "🎓 Thèse / mémoire",
    "article":            "📰 Article académique",
    "poeme":              "🖋️ Poème / création originale",
    "autre":              "📄 Autre",
}
NIVEAUX_CONFIDENTIALITE = ["public", "sur_demande", "prive"]

# ── Index des documents (JSON simple) ──────────────────────────────────────

def charger_documents() -> list[dict]:
    if FICHIER_DOCS.exists():
        with open(FICHIER_DOCS, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_documents(docs: list[dict]):
    with open(FICHIER_DOCS, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

# ── ChromaDB (collection dédiée) ────────────────────────────────────────────

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = chromadb.PersistentClient(path=str(DOSSIER_CHROMA))
        emb_fn = SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        _collection = client.get_or_create_collection(
            name="corpus_histoire",
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB histoire prêt — {_collection.count()} chunks indexés")
    return _collection

def indexer_document(
    titre: str,
    auteur_detenteur: str,
    royaume: str,
    type_source: str,
    confidentialite: str,
    texte: str,
    doc_id: str,
) -> int:
    """Indexe le texte d'un document historique. Retourne le nb de chunks ajoutés."""
    collection = get_collection()
    chunks = chunker(texte)
    if not chunks:
        return 0

    ids = [f"{doc_id}__{i:04d}" for i in range(len(chunks))]
    metadatas = [
        {
            "titre":           titre[:200],
            "auteur_detenteur": auteur_detenteur[:150],
            "royaume":         royaume,
            "type_source":     type_source,
            "confidentialite": confidentialite,
            "doc_id":          doc_id,
            "chunk_no":        i,
            "total":           len(chunks),
        }
        for i in range(len(chunks))
    ]

    BATCH = 100
    ajoutes = 0
    for start in range(0, len(ids), BATCH):
        batch_ids  = ids[start:start+BATCH]
        batch_docs = chunks[start:start+BATCH]
        batch_meta = metadatas[start:start+BATCH]
        existants = set(collection.get(ids=batch_ids)["ids"])
        filtre = [(i, d, m) for i, d, m in zip(batch_ids, batch_docs, batch_meta)
                  if i not in existants]
        if filtre:
            fi, fd, fm = zip(*filtre)
            collection.add(documents=list(fd), ids=list(fi), metadatas=list(fm))
            ajoutes += len(filtre)

    log.info(f"Histoire — indexé '{titre}': {ajoutes}/{len(chunks)} chunks nouveaux")
    return ajoutes

def rechercher(
    query: str,
    n: int = 5,
    royaume: str = None,
    type_source: str = None,
) -> list[dict]:
    """Recherche sémantique dans le corpus historique (documents publics et sur_demande uniquement)."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    where_clauses = []
    if royaume:
        where_clauses.append({"royaume": royaume})
    if type_source:
        where_clauses.append({"type_source": type_source})
    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    results = collection.query(
        query_texts=[query],
        n_results=min(n * 3, total),  # marge, on filtre le privé après coup
        where=where,
    )

    items = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        if meta.get("confidentialite") == "prive":
            continue  # jamais de contenu privé dans les résultats de recherche
        dist = results["distances"][0][i] if results.get("distances") else 1.0
        items.append({
            "texte":            doc if meta.get("confidentialite") == "public"
                                 else doc[:300] + ("…" if len(doc) > 300 else ""),
            "titre":            meta.get("titre", "?"),
            "auteur_detenteur": meta.get("auteur_detenteur", "?"),
            "royaume":          meta.get("royaume", "?"),
            "type_source":      meta.get("type_source", "?"),
            "confidentialite":  meta.get("confidentialite", "?"),
            "doc_id":           meta.get("doc_id", "?"),
            "score":            round(1 - dist, 3),
        })
        if len(items) >= n:
            break
    return items

def stats_histoire() -> dict:
    try:
        collection = get_collection()
        docs = charger_documents()
        return {
            "total_chunks":    collection.count(),
            "total_documents": len(docs),
            "royaumes":        list({d.get("royaume", "?") for d in docs}),
            "par_type":        {t: sum(1 for d in docs if d.get("type_source") == t) for t in TYPES_SOURCE},
        }
    except Exception:
        return {"total_chunks": 0, "total_documents": 0, "royaumes": [], "par_type": {}}

# ── Annuaire des familles détentrices ───────────────────────────────────────

def charger_familles() -> list[dict]:
    if FICHIER_FAM.exists():
        with open(FICHIER_FAM, encoding="utf-8") as f:
            return json.load(f)
    return []

def sauver_familles(familles: list[dict]):
    with open(FICHIER_FAM, "w", encoding="utf-8") as f:
        json.dump(familles, f, ensure_ascii=False, indent=2)

def ajouter_famille(
    nom_famille: str,
    royaume: str,
    lignage: str,
    localisation: str,
    description: str,
    contact: str,
    contact_visible: bool,
) -> dict:
    familles = charger_familles()
    fiche = {
        "id":              str(uuid.uuid4())[:8],
        "nom_famille":     nom_famille,
        "royaume":         royaume,
        "lignage":         lignage,
        "localisation":    localisation,
        "description":     description,
        "contact":         contact,
        "contact_visible": contact_visible,
        "date":            datetime.now().isoformat(),
    }
    familles.append(fiche)
    sauver_familles(familles)
    return fiche
