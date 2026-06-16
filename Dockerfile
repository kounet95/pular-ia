# ─────────────────────────────────────────────────────────────────────────────
# Pular IA — Image Docker de production
# Build : docker build -t pular-ia .
# Run   : docker run -p 8080:8080 -v $(pwd)/corpus-pular:/app/corpus-pular pular-ia
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Dépendances système (ffmpeg requis par Whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dépendances Python ────────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY scripts/ ./scripts/
COPY web/     ./web/
COPY .env.example .env.example

# ── Dossiers de données (montés en volume en prod) ────────────────────────────
RUN mkdir -p \
        corpus-pular/community/contributions \
        corpus-pular/community/audio \
        corpus-pular/community/corrections \
        corpus-pular/processed/transcriptions \
        corpus-pular/jeu \
        corpus-pular/livres/raw \
        corpus-pular/livres/metadata \
        corpus-pular/rag/chroma \
        logs

# ── Variables d'environnement par défaut ─────────────────────────────────────
ENV WEBAPP_PORT=8080
ENV WHISPER_MODEL_BOT=tiny
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Lancer depuis le répertoire /app (important pour les chemins relatifs)
CMD ["python", "scripts/community_webapp.py"]
