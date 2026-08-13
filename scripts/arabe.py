"""
arabe.py — Clavier arabe virtuel (pour l'écriture en ajami/arabe)

L'ajami peul (transcription du Pular/Fulfulde en écriture arabe) varie selon
les traditions régionales (Fouta Djallon, Macina, Fouta Toro...) : il n'existe
pas de table de correspondance latin↔ajami unique et fiable, contrairement à
l'Adlam qui a un standard Unicode officiel (voir adlam.py). Ce module ne
propose donc PAS de conversion automatique — seulement un clavier arabe
standard pour permettre de composer directement en ajami/arabe à la main,
comme sur un clavier arabe physique.

Usage:
    from scripts.arabe import CLAVIER_ARABE
"""

# ── Clavier arabe (JSON pour le frontend) ───────────────────────────────────
CLAVIER_ARABE = {
    "lettres": [
        {"car": "ا", "label": "alif"},
        {"car": "ب", "label": "bā'"},
        {"car": "ت", "label": "tā'"},
        {"car": "ث", "label": "thā'"},
        {"car": "ج", "label": "jīm"},
        {"car": "ح", "label": "ḥā'"},
        {"car": "خ", "label": "khā'"},
        {"car": "د", "label": "dāl"},
        {"car": "ذ", "label": "dhāl"},
        {"car": "ر", "label": "rā'"},
        {"car": "ز", "label": "zāy"},
        {"car": "س", "label": "sīn"},
        {"car": "ش", "label": "shīn"},
        {"car": "ص", "label": "ṣād"},
        {"car": "ض", "label": "ḍād"},
        {"car": "ط", "label": "ṭā'"},
        {"car": "ظ", "label": "ẓā'"},
        {"car": "ع", "label": "'ayn"},
        {"car": "غ", "label": "ghayn"},
        {"car": "ف", "label": "fā'"},
        {"car": "ق", "label": "qāf"},
        {"car": "ك", "label": "kāf"},
        {"car": "ل", "label": "lām"},
        {"car": "م", "label": "mīm"},
        {"car": "ن", "label": "nūn"},
        {"car": "ه", "label": "hā'"},
        {"car": "و", "label": "wāw"},
        {"car": "ي", "label": "yā'"},
    ],
    "variantes": [
        {"car": "ء", "label": "hamza"},
        {"car": "أ", "label": "alif hamza"},
        {"car": "إ", "label": "alif hamza (bas)"},
        {"car": "آ", "label": "alif madda"},
        {"car": "ؤ", "label": "wāw hamza"},
        {"car": "ئ", "label": "yā' hamza"},
        {"car": "ة", "label": "tā' marbūṭa"},
        {"car": "ى", "label": "alif maqṣūra"},
        {"car": "پ", "label": "p (emprunt)"},
        {"car": "ڤ", "label": "v (emprunt)"},
        {"car": "ڠ", "label": "ng (emprunt)"},
    ],
    "harakat": [
        {"car": "َ", "label": "fatḥa"},
        {"car": "ِ", "label": "kasra"},
        {"car": "ُ", "label": "ḍamma"},
        {"car": "ْ", "label": "sukūn"},
        {"car": "ّ", "label": "shadda"},
        {"car": "ً", "label": "tanwīn fatḥ"},
        {"car": "ٍ", "label": "tanwīn kasr"},
        {"car": "ٌ", "label": "tanwīn ḍamm"},
    ],
}
