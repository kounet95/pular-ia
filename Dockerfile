FROM python:3.11-slim

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. PyTorch CPU uniquement (commande séparée — index différent) ────────────
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. Whisper (dépend de torch) ─────────────────────────────────────────────
# setuptools<76 requis : v76+ a supprimé pkg_resources dont dépend openai-whisper
RUN pip install --no-cache-dir "setuptools<76"
RUN pip install --no-cache-dir --timeout 300 --no-build-isolation \
    openai-whisper==20231117

# ── 3. Web + RAG ──────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --timeout 300 \
    fastapi==0.110.3 \
    uvicorn==0.29.0 \
    python-multipart==0.0.9 \
    python-dotenv==1.0.1 \
    pdfplumber==0.10.3 \
    python-docx==1.1.2 \
    tqdm==4.66.4 \
    numpy==1.26.4

# ── 4. ChromaDB + embeddings (lourd, isolé) ───────────────────────────────────
RUN pip install --no-cache-dir --timeout 300 \
    chromadb==0.4.24 \
    sentence-transformers==2.7.0

# ── 5. Forcer numpy 1.x (chromadb/sentence-transformers tirent numpy 2.x) ────
# numpy 2.x casse whisper et torch compilés contre numpy 1.x
RUN pip install --no-cache-dir "numpy==1.26.4"

# ── Code source ───────────────────────────────────────────────────────────────
COPY scripts/ ./scripts/
COPY web/     ./web/

# ── Dossiers de données (montés en volume sur Railway) ───────────────────────
RUN mkdir -p \
        corpus-pular/community/contributions \
        corpus-pular/community/audio \
        corpus-pular/community/corrections \
        corpus-pular/processed/transcriptions \
        corpus-pular/jeu \
        corpus-pular/livres/raw \
        corpus-pular/livres/metadata \
        corpus-pular/rag/chroma \
        corpus-pular/dataset/llm \
        corpus-pular/dataset/translit \
        logs

ENV WEBAPP_PORT=8080
ENV WHISPER_MODEL_BOT=tiny
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

CMD ["python", "scripts/community_webapp.py"]
