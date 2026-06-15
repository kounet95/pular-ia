"""
finetune_mt5.py — Fine-tune mT5-small pour la translitération Latin → Adlam

Ce script fine-tune google/mt5-small sur les paires Latin/Adlam préparées par
prepare_translit_dataset.py.

mT5-small = 300M paramètres, multilingue (101 langues), idéal pour les langues
peu-dotées comme le Pular. Pas besoin de GPU coûteux — fonctionne sur T4 Colab.

Usage local (CPU lent — utilise plutôt Colab):
    python scripts/finetune_mt5.py --epochs 3

Usage recommandé → Google Colab (GPU T4 gratuit):
    python scripts/finetune_mt5.py --colab   # génère le notebook Colab

Sortie:
    models/mt5_adlam/   (modèle fine-tuné, chargeable avec from_pretrained())
"""

import json
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJET_ROOT   = Path(__file__).resolve().parent.parent
DOSSIER_DATA  = PROJET_ROOT / "corpus-pular" / "dataset" / "translit"
DOSSIER_MODEL = PROJET_ROOT / "models" / "mt5_adlam"
MODEL_BASE    = "google/mt5-small"   # 300M params, 2.2GB

# ── Génération notebook Colab ──────────────────────────────────────────────────
COLAB_NOTEBOOK = """
{
 "nbformat": 4,
 "nbformat_minor": 0,
 "metadata": {
  "colab": {"name": "FineTune_mT5_Adlam_Pular.ipynb", "provenance": []},
  "kernelspec": {"name": "python3", "display_name": "Python 3"},
  "accelerator": "GPU"
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Fine-tune mT5-small : Latin → Adlam (Pular)\\n",
               "Runtime → Change runtime type → T4 GPU"]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 1. Install\\n",
    "!pip install -q transformers[torch] datasets accelerate sentencepiece sacrebleu"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 2. Upload dataset\\n",
    "from google.colab import files\\n",
    "uploaded = files.upload()  # upload train.jsonl, val.jsonl"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 3. Charger les données\\n",
    "import json\\n",
    "\\n",
    "def charger_jsonl(path):\\n",
    "    with open(path) as f:\\n",
    "        return [json.loads(l) for l in f if l.strip()]\\n",
    "\\n",
    "train_data = charger_jsonl('train.jsonl')\\n",
    "val_data   = charger_jsonl('val.jsonl')\\n",
    "print(f'Train: {len(train_data)} | Val: {len(val_data)}')\\n",
    "print('Exemple:', train_data[0])"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 4. Fine-tune\\n",
    "from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,\\n",
    "                          Seq2SeqTrainer, Seq2SeqTrainingArguments,\\n",
    "                          DataCollatorForSeq2Seq)\\n",
    "from datasets import Dataset\\n",
    "import torch\\n",
    "\\n",
    "MODEL = 'google/mt5-small'\\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\\n",
    "model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)\\n",
    "\\n",
    "MAX_INPUT  = 128\\n",
    "MAX_TARGET = 128\\n",
    "\\n",
    "def tokeniser(batch):\\n",
    "    inputs  = tokenizer(batch['input'],  max_length=MAX_INPUT,  truncation=True, padding=False)\\n",
    "    targets = tokenizer(batch['target'], max_length=MAX_TARGET, truncation=True, padding=False)\\n",
    "    inputs['labels'] = targets['input_ids']\\n",
    "    return inputs\\n",
    "\\n",
    "ds_train = Dataset.from_list([{'input': d['input'], 'target': d['target']} for d in train_data])\\n",
    "ds_val   = Dataset.from_list([{'input': d['input'], 'target': d['target']} for d in val_data])\\n",
    "ds_train = ds_train.map(tokeniser, batched=True, remove_columns=['input', 'target'])\\n",
    "ds_val   = ds_val.map(tokeniser,   batched=True, remove_columns=['input', 'target'])\\n",
    "\\n",
    "args = Seq2SeqTrainingArguments(\\n",
    "    output_dir='./mt5_adlam',\\n",
    "    num_train_epochs=5,\\n",
    "    per_device_train_batch_size=16,\\n",
    "    per_device_eval_batch_size=16,\\n",
    "    warmup_steps=100,\\n",
    "    weight_decay=0.01,\\n",
    "    learning_rate=5e-4,\\n",
    "    predict_with_generate=True,\\n",
    "    evaluation_strategy='epoch',\\n",
    "    save_strategy='epoch',\\n",
    "    load_best_model_at_end=True,\\n",
    "    metric_for_best_model='eval_loss',\\n",
    "    fp16=True,\\n",
    "    logging_steps=50,\\n",
    "    report_to='none',\\n",
    ")\\n",
    "\\n",
    "collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)\\n",
    "trainer = Seq2SeqTrainer(\\n",
    "    model=model,\\n",
    "    args=args,\\n",
    "    train_dataset=ds_train,\\n",
    "    eval_dataset=ds_val,\\n",
    "    tokenizer=tokenizer,\\n",
    "    data_collator=collator,\\n",
    ")\\n",
    "trainer.train()"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 5. Test rapide\\n",
    "def transliterer(texte_latin):\\n",
    "    inputs = tokenizer(f'translitere en adlam: {texte_latin}', return_tensors='pt').to(model.device)\\n",
    "    with torch.no_grad():\\n",
    "        out = model.generate(**inputs, max_new_tokens=128, num_beams=4)\\n",
    "    return tokenizer.decode(out[0], skip_special_tokens=True)\\n",
    "\\n",
    "tests = ['Jam waali', 'Mi yiɗi ɓiɓɓe am', 'Pulaagu woni ndimaagu', 'A jaaraama walaa']\\n",
    "for t in tests:\\n",
    "    print(f'{t}  →  {transliterer(t)}')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 6. Télécharger le modèle\\n",
    "import shutil\\n",
    "trainer.save_model('./mt5_adlam_final')\\n",
    "tokenizer.save_pretrained('./mt5_adlam_final')\\n",
    "shutil.make_archive('mt5_adlam_final', 'zip', './mt5_adlam_final')\\n",
    "files.download('mt5_adlam_final.zip')"
   ],
   "execution_count": null,
   "outputs": []
  }
 ]
}
"""


def generer_colab():
    """Génère le notebook Colab et prépare le ZIP des données."""
    notebook_path = PROJET_ROOT / "colab_finetune_mt5.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        f.write(COLAB_NOTEBOOK)
    log.info(f"Notebook Colab → {notebook_path}")

    # Créer un ZIP avec train.jsonl et val.jsonl
    import shutil
    zip_data = PROJET_ROOT / "dataset_translit.zip"
    if (DOSSIER_DATA / "train.jsonl").exists():
        import zipfile
        with zipfile.ZipFile(zip_data, "w") as zf:
            zf.write(DOSSIER_DATA / "train.jsonl", "train.jsonl")
            zf.write(DOSSIER_DATA / "val.jsonl",   "val.jsonl")
            if (DOSSIER_DATA / "test.jsonl").exists():
                zf.write(DOSSIER_DATA / "test.jsonl", "test.jsonl")
        log.info(f"ZIP données → {zip_data}")
        log.info("→ Upload dataset_translit.zip dans Colab puis lance le notebook")
    else:
        log.warning("train.jsonl introuvable — lance d'abord prepare_translit_dataset.py")

    log.info(f"\nÉtapes:")
    log.info(f"  1. Ouvre {notebook_path} dans Colab")
    log.info(f"  2. Runtime → Change runtime type → T4 GPU")
    log.info(f"  3. Upload {zip_data}")
    log.info(f"  4. Lance toutes les cellules")
    log.info(f"  5. Télécharge mt5_adlam_final.zip et décompresse dans models/")


# ── Entraînement local (CPU/GPU) ──────────────────────────────────────────────
def charger_jsonl(chemin: Path) -> list[dict]:
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def entrainer_local(epochs: int, batch_size: int, lr: float):
    """Fine-tune mT5 localement."""
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
        import torch
    except ImportError:
        log.error("Installe les dépendances: pip install transformers datasets accelerate sentencepiece")
        sys.exit(1)

    train_fichier = DOSSIER_DATA / "train.jsonl"
    val_fichier   = DOSSIER_DATA / "val.jsonl"

    if not train_fichier.exists():
        log.error(f"train.jsonl introuvable. Lance d'abord prepare_translit_dataset.py")
        sys.exit(1)

    train_data = charger_jsonl(train_fichier)
    val_data   = charger_jsonl(val_fichier)
    log.info(f"Données: {len(train_data)} train | {len(val_data)} val")

    log.info(f"Chargement {MODEL_BASE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_BASE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device.upper()}")

    MAX_LEN = 128

    def tokeniser(batch):
        inputs  = tokenizer(batch["input"],  max_length=MAX_LEN, truncation=True, padding=False)
        targets = tokenizer(batch["target"], max_length=MAX_LEN, truncation=True, padding=False)
        inputs["labels"] = targets["input_ids"]
        return inputs

    ds_train = Dataset.from_list([{"input": d["input"], "target": d["target"]} for d in train_data])
    ds_val   = Dataset.from_list([{"input": d["input"], "target": d["target"]} for d in val_data])
    ds_train = ds_train.map(tokeniser, batched=True, remove_columns=["input", "target"])
    ds_val   = ds_val.map(tokeniser,   batched=True, remove_columns=["input", "target"])

    DOSSIER_MODEL.mkdir(parents=True, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=str(DOSSIER_MODEL),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=lr,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device == "cuda"),
        logging_steps=10,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer  = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    log.info("Démarrage entraînement...")
    trainer.train()

    trainer.save_model(str(DOSSIER_MODEL))
    tokenizer.save_pretrained(str(DOSSIER_MODEL))
    log.info(f"✅ Modèle sauvé → {DOSSIER_MODEL}")

    # Test rapide
    log.info("\nTest du modèle:")
    model.eval()
    tests = ["Jam waali", "Mi yiɗi ɓiɓɓe am", "Pulaagu woni ndimaagu"]
    for t in tests:
        inputs = tokenizer(f"translitere en adlam: {t}", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        log.info(f"  {t}  →  {pred}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune mT5 Latin→Adlam")
    parser.add_argument("--colab",   action="store_true", help="Générer notebook Colab (recommandé)")
    parser.add_argument("--epochs",  type=int,   default=5,    help="Nombre d'époques")
    parser.add_argument("--batch",   type=int,   default=8,    help="Taille de batch")
    parser.add_argument("--lr",      type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    if args.colab:
        generer_colab()
    else:
        log.info("⚠️  Entraînement local sur CPU peut prendre des heures.")
        log.info("   Utilise --colab pour générer un notebook Colab (T4 GPU gratuit).")
        entrainer_local(args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
