"""
correction_vocabulaire.py — Correction post-transcription par dictionnaire
Whisper ne connaît pas le pular : le prompt biaisé (construire_prompt_whisper
dans transcription.py) aide pendant le décodage, mais certains mots sortent
quand même déformés. Ici, chaque mot transcrit qui ne correspond à aucun mot
connu (dictionnaire + vocabulaire coranique) est comparé au vocabulaire pour
le "recoller" au mot pular le plus proche si la ressemblance est forte.

Usage:
    from scripts.correction_vocabulaire import charger_vocabulaire, corriger_transcription

    vocab = charger_vocabulaire()
    texte_corrige, nb_corriges = corriger_transcription(texte_brut, vocab)
"""

import json
import re
import logging
from difflib import get_close_matches
from pathlib import Path

log = logging.getLogger(__name__)

DOSSIER_VOCAB = Path("./corpus-pular/livres/metadata")

_vocab_cache: frozenset[str] | None = None


def charger_vocabulaire() -> frozenset[str]:
    """
    Union de tous les vocabulaires générés par rag_livres.indexer_livre()
    (dictionnaire pular-français + traductions coraniques) — mêmes fichiers
    que ceux utilisés par construire_prompt_whisper(). Mis en cache après le
    premier appel (fichiers volumineux, ne changent pas en cours de run).
    """
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache

    mots: set[str] = set()
    if DOSSIER_VOCAB.exists():
        for fichier in DOSSIER_VOCAB.glob("*_vocab.json"):
            try:
                with open(fichier, encoding="utf-8") as f:
                    mots.update(json.load(f))
            except Exception as e:
                log.warning(f"Vocabulaire illisible ({fichier.name}): {e}")

    _vocab_cache = frozenset(mots)
    log.info(f"Vocabulaire de correction chargé : {len(_vocab_cache):,} mots uniques")
    return _vocab_cache


_MOT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def corriger_transcription(
    texte: str,
    vocab: frozenset[str] | None = None,
    seuil: float = 0.87,
    longueur_min: int = 5,
) -> tuple[str, int]:
    """
    Corrige mot par mot un texte transcrit par Whisper : tout mot absent du
    vocabulaire connu est remplacé par le mot connu le plus proche s'il lui
    ressemble suffisamment (ratio difflib >= seuil). Les mots trop courts
    (< longueur_min) sont laissés tels quels — trop de faux positifs sinon
    (ex: "e", "ko", "no" ressemblent à beaucoup de mots).

    Retourne (texte_corrigé, nombre_de_mots_corrigés).
    """
    if not texte or not texte.strip():
        return texte, 0

    if vocab is None:
        vocab = charger_vocabulaire()
    if not vocab:
        return texte, 0

    nb_corriges = 0
    mots_sortie = []

    for token in texte.split(" "):
        m = _MOT_RE.match(token)
        prefixe, mot, suffixe = m.group(1), m.group(2), m.group(3)

        if not mot or len(mot) < longueur_min:
            mots_sortie.append(token)
            continue

        mot_bas = mot.lower()
        if mot_bas in vocab:
            mots_sortie.append(token)
            continue

        proches = get_close_matches(mot_bas, vocab, n=1, cutoff=seuil)
        if proches:
            correction = proches[0]
            # Réappliquer une majuscule initiale si le mot original en avait une
            if mot[0].isupper():
                correction = correction[0].upper() + correction[1:]
            mots_sortie.append(prefixe + correction + suffixe)
            nb_corriges += 1
        else:
            mots_sortie.append(token)

    return " ".join(mots_sortie), nb_corriges
