"""
adlam.py — Conversion Latin ↔ Adlam + transcription MMS (Meta)

L'Adlam (𞤀𞤣𞤤𞤢𞤥) est l'alphabet natif du Pular/Fulfulde, créé par
Ibrahima et Abdoulaye Barry. Encodé en Unicode U+1E900-U+1E95F.
Écriture de droite à gauche.

Usage:
    from scripts.adlam import latin_vers_adlam, adlam_vers_latin, transcrire_mms

    print(latin_vers_adlam("Jam waali"))      # → 𞤔𞤢𞤥 𞤱𞤢𞤢𞤤𞤭
    print(adlam_vers_latin("𞤔𞤢𞤥 𞤱𞤢𞤢𞤤𞤭"))   # → jam waali

MMS (Meta Massively Multilingual Speech):
    python scripts/adlam.py --audio fichier.wav
"""

import re

# ── Tables de conversion ───────────────────────────────────────────────────────
# Source: Unicode Adlam standard + orthographe officielle guinéenne du Pular

# Digraphes à traiter avant les lettres simples (ordre important)
DIGRAPHES_LATIN_ADLAM = {
    "mb": "𞤃𞤦", "nd": "𞤲𞤣", "nj": "𞤲𞤶", "ng": "𞤲𞤺",
    "nt": "𞤲𞤼", "nc": "𞤲𞤷", "nk": "𞤲𞤳",
    "MB": "𞤃𞤄", "ND": "𞤐𞤁", "NJ": "𞤐𞤔", "NG": "𞤐𞤘",
}

# Lettres simples Latin → Adlam (minuscules)
LATIN_ADLAM = {
    # Voyelles
    "a": "𞤢", "e": "𞤫", "i": "𞤭", "o": "𞤮", "u": "𞤵",
    # Consonnes standard
    "b": "𞤦", "c": "𞤷", "d": "𞤣", "f": "𞤬", "g": "𞤺",
    "h": "𞤸", "j": "𞤶", "k": "𞤳", "l": "𞤤", "m": "𞤥",
    "n": "𞤲", "p": "𞤨", "q": "𞤹", "r": "𞤪", "s": "𞤧",
    "t": "𞤼", "v": "𞤾", "w": "𞤱", "x": "𞤿", "y": "𞤴",
    "z": "𞥁",
    # Lettres spéciales Pular (implosives et nasales)
    "ɓ": "𞤩", "ɗ": "𞤯", "ƴ": "𞤰", "ɲ": "𞤻", "ŋ": "𞤽",
    # Variantes avec apostrophe/tiret parfois utilisées
    "'y": "𞤰", "'b": "𞤩", "'d": "𞤯",
}

# Majuscules Latin → Adlam
LATIN_ADLAM_MAJ = {
    "A": "𞤀", "B": "𞤄", "C": "𞤕", "D": "𞤁", "E": "𞤉",
    "F": "𞤊", "G": "𞤘", "H": "𞤖", "I": "𞤋", "J": "𞤔",
    "K": "𞤑", "L": "𞤂", "M": "𞤃", "N": "𞤐", "O": "𞤌",
    "P": "𞤆", "Q": "𞤗", "R": "𞤈", "S": "𞤅", "T": "𞤚",
    "U": "𞤓", "V": "𞤜", "W": "𞤏", "X": "𞤝", "Y": "𞤒",
    "Z": "𞤟",
    "Ɓ": "𞤇", "Ɗ": "𞤍", "Ƴ": "𞤎", "Ɲ": "𞤙", "Ŋ": "𞤛",
}

# Table inverse : Adlam → Latin (minuscules)
ADLAM_LATIN = {v: k for k, v in LATIN_ADLAM.items() if len(k) == 1}
ADLAM_LATIN.update({v: k.upper() for k, v in LATIN_ADLAM_MAJ.items()})

# ── Conversion Latin → Adlam ──────────────────────────────────────────────────
def latin_vers_adlam(texte: str) -> str:
    """Convertit du Pular romanisé (latin) vers l'alphabet Adlam."""
    if not texte:
        return texte

    resultat = []
    i = 0
    while i < len(texte):
        # Essayer les digraphes en premier (2 caractères)
        bi = texte[i:i+2]
        if bi in DIGRAPHES_LATIN_ADLAM:
            resultat.append(DIGRAPHES_LATIN_ADLAM[bi])
            i += 2
            continue

        c = texte[i]
        if c in LATIN_ADLAM_MAJ:
            resultat.append(LATIN_ADLAM_MAJ[c])
        elif c in LATIN_ADLAM:
            resultat.append(LATIN_ADLAM[c])
        else:
            # Garder les caractères non-Pular (espaces, ponctuation, chiffres...)
            resultat.append(c)
        i += 1

    return "".join(resultat)

# ── Conversion Adlam → Latin ──────────────────────────────────────────────────
def adlam_vers_latin(texte: str) -> str:
    """Convertit de l'alphabet Adlam vers du Pular romanisé (latin)."""
    if not texte:
        return texte

    resultat = []
    for c in texte:
        if c in ADLAM_LATIN:
            resultat.append(ADLAM_LATIN[c])
        else:
            resultat.append(c)
    return "".join(resultat)

# ── Détection de script ───────────────────────────────────────────────────────
def est_adlam(texte: str) -> bool:
    """Retourne True si le texte contient des caractères Adlam."""
    return any("\U0001E900" <= c <= "\U0001E95F" for c in texte)

def est_latin_pular(texte: str) -> bool:
    """Retourne True si le texte contient des lettres spéciales Pular."""
    return any(c in "ɓɗƴɲŋƁƊƳɃ" for c in texte)

# ── Transcription MMS (Meta) ──────────────────────────────────────────────────
_mms_pipeline = None

def get_mms(model_id: str = "facebook/mms-300m"):
    """Charge le modèle MMS de Meta (meilleur que Whisper pour le Pular/Fula)."""
    global _mms_pipeline
    if _mms_pipeline is None:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Chargement MMS {model_id}...")
        _mms_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        print("✅ MMS prêt")
    return _mms_pipeline

def transcrire_mms(audio_path: str, langue: str = "fuf") -> dict:
    """
    Transcrit un audio avec Meta MMS-300M.
    langue: "fuf" = Fulfulde (Pular), "fuf-Adlm" pour sortie Adlam directe si dispo
    """
    asr = get_mms()
    result = asr(
        audio_path,
        generate_kwargs={"language": langue},
        return_timestamps=False,
    )
    texte_latin = result.get("text", "").strip()
    texte_adlam = latin_vers_adlam(texte_latin)
    return {
        "text_latin": texte_latin,
        "text_adlam": texte_adlam,
        "model":      "mms-300m",
        "language":   langue,
    }

# ── Clavier Adlam (JSON pour le frontend) ─────────────────────────────────────
CLAVIER_ADLAM = {
    "voyelles": [
        {"latin": "a", "adlam": "𞤢", "label": "a"},
        {"latin": "e", "adlam": "𞤫", "label": "e"},
        {"latin": "i", "adlam": "𞤭", "label": "i"},
        {"latin": "o", "adlam": "𞤮", "label": "o"},
        {"latin": "u", "adlam": "𞤵", "label": "u"},
    ],
    "consonnes": [
        {"latin": "b", "adlam": "𞤦", "label": "b"},
        {"latin": "c", "adlam": "𞤷", "label": "c"},
        {"latin": "d", "adlam": "𞤣", "label": "d"},
        {"latin": "f", "adlam": "𞤬", "label": "f"},
        {"latin": "g", "adlam": "𞤺", "label": "g"},
        {"latin": "h", "adlam": "𞤸", "label": "h"},
        {"latin": "j", "adlam": "𞤶", "label": "j"},
        {"latin": "k", "adlam": "𞤳", "label": "k"},
        {"latin": "l", "adlam": "𞤤", "label": "l"},
        {"latin": "m", "adlam": "𞤥", "label": "m"},
        {"latin": "n", "adlam": "𞤲", "label": "n"},
        {"latin": "p", "adlam": "𞤨", "label": "p"},
        {"latin": "r", "adlam": "𞤪", "label": "r"},
        {"latin": "s", "adlam": "𞤧", "label": "s"},
        {"latin": "t", "adlam": "𞤼", "label": "t"},
        {"latin": "w", "adlam": "𞤱", "label": "w"},
        {"latin": "y", "adlam": "𞤴", "label": "y"},
    ],
    "speciales": [
        {"latin": "ɓ", "adlam": "𞤩", "label": "ɓ", "info": "b implosif"},
        {"latin": "ɗ", "adlam": "𞤯", "label": "ɗ", "info": "d implosif"},
        {"latin": "ƴ", "adlam": "𞤰", "label": "ƴ", "info": "y implosif"},
        {"latin": "ɲ", "adlam": "𞤻", "label": "ɲ", "info": "ny"},
        {"latin": "ŋ", "adlam": "𞤽", "label": "ŋ", "info": "ng nasal"},
    ],
    "digraphes": [
        {"latin": "mb", "adlam": "𞤃𞤦", "label": "mb"},
        {"latin": "nd", "adlam": "𞤲𞤣", "label": "nd"},
        {"latin": "nj", "adlam": "𞤲𞤶", "label": "nj"},
        {"latin": "ng", "adlam": "𞤲𞤺", "label": "ng"},
    ],
}

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Convertisseur Latin ↔ Adlam + MMS")
    parser.add_argument("--latin",  help="Texte latin à convertir en Adlam")
    parser.add_argument("--adlam",  help="Texte Adlam à convertir en latin")
    parser.add_argument("--audio",  help="Fichier audio à transcrire avec MMS")
    parser.add_argument("--clavier",action="store_true", help="Afficher le clavier Adlam JSON")
    args = parser.parse_args()

    if args.latin:
        adlam = latin_vers_adlam(args.latin)
        print(f"Latin : {args.latin}")
        print(f"Adlam : {adlam}")

    elif args.adlam:
        latin = adlam_vers_latin(args.adlam)
        print(f"Adlam : {args.adlam}")
        print(f"Latin : {latin}")

    elif args.audio:
        result = transcrire_mms(args.audio)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.clavier:
        print(json.dumps(CLAVIER_ADLAM, ensure_ascii=False, indent=2))

    else:
        # Demo
        exemples = [
            "Jam waali",
            "Bismillahi Rahmaani Rahiimi",
            "Mi yiɗi ɓiɓɓe am",
            "Pulaagu woni ndimaagu",
            "A jaaraama walaa",
        ]
        print("=== Démo Latin → Adlam ===\n")
        for ex in exemples:
            print(f"  {ex}")
            print(f"  {latin_vers_adlam(ex)}\n")
