"""
COLAB_TRANSCRIPTION — À coller dans Google Colab (GPU gratuit)
===============================================================
1. Va sur https://colab.research.google.com
2. Nouveau notebook → Runtime → Change runtime type → GPU (T4)
3. Colle ce script cellule par cellule
4. Télécharge le ZIP de résultats à la fin

Gain de vitesse : large-v3 sur GPU T4 ≈ 30-50x plus rapide que CPU
"""

# ── Cellule 1 : Installation ──────────────────────────────────────────────────
"""
!pip install openai-whisper -q
!apt-get install -y ffmpeg -q
"""

# ── Cellule 2 : Upload des fichiers audio ────────────────────────────────────
"""
from google.colab import files
import zipfile, os

# Option A : uploader un ZIP de tes fichiers audio
uploaded = files.upload()  # Sélectionne ton ZIP d'audios
zip_nom  = list(uploaded.keys())[0]

os.makedirs("/content/audio", exist_ok=True)
with zipfile.ZipFile(zip_nom, 'r') as z:
    z.extractall("/content/audio")

fichiers = [f for f in os.listdir("/content/audio")
            if f.endswith(('.mp3', '.ogg', '.wav', '.m4a', '.opus'))]
print(f"✅ {len(fichiers)} fichiers audio trouvés")
"""

# ── Cellule 3 : Transcription Whisper large-v3 ───────────────────────────────
"""
import whisper, json, os
from pathlib import Path
from tqdm import tqdm

os.makedirs("/content/transcriptions", exist_ok=True)

model = whisper.load_model("large-v3")
print("✅ Modèle large-v3 chargé sur GPU")

erreurs = []
for nom in tqdm(fichiers, desc="Transcription"):
    chemin = f"/content/audio/{nom}"
    sortie = f"/content/transcriptions/{Path(nom).stem}.json"

    if os.path.exists(sortie):
        continue  # déjà traité

    try:
        result = model.transcribe(
            chemin,
            task="transcribe",
            no_speech_threshold=0.3,
            initial_prompt="Pular fulfulde fulani langue africaine.",
            condition_on_previous_text=False,
            fp16=True,   # GPU supporte FP16 → 2x plus rapide
        )
        entry = {
            "fichier":       chemin,
            "nom":           nom,
            "texte":         result["text"].strip(),
            "langue_detect": result.get("language", "?"),
            "segments":      result.get("segments", []),
        }
        with open(sortie, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    except Exception as e:
        erreurs.append(f"{nom}: {e}")

print(f"\\n✅ Transcription terminée!")
print(f"❌ Erreurs : {len(erreurs)}")
for e in erreurs[:10]:
    print(" ", e)
"""

# ── Cellule 4 : Télécharger les résultats ────────────────────────────────────
"""
import shutil
from google.colab import files

shutil.make_archive("/content/transcriptions_pular", "zip", "/content/transcriptions")
files.download("/content/transcriptions_pular.zip")
print("📦 ZIP téléchargé — copie dans corpus-pular/processed/transcriptions/")
"""
