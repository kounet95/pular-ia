"""
qa_pular.py — Assistant Q&A en pular/français, appuyé sur le corpus Pular IA
Répond aux questions posées en pular ou en français, dans la même langue,
en s'appuyant sur le dictionnaire, le Coran en pular (traduction + tafsir) et
les livres/documents déjà indexés (RAG) sur la plateforme — pas une réponse
générique de LLM sans ancrage dans les vraies données du projet.

Usage:
    from scripts.qa_pular import repondre_question
    reponse = repondre_question("Kori Alla woodi?")
"""

import os
import logging

log = logging.getLogger(__name__)


def anthropic_configure():
    """Configure le client Anthropic. Retourne le client, ou lève RuntimeError
    avec un message qui distingue clé manquante et paquet non installé
    (même convention que espace_editorial.anthropic_configure)."""
    cle = os.getenv("ANTHROPIC_API_KEY", "")
    if not cle:
        raise RuntimeError("Assistant Q&A non configuré (ANTHROPIC_API_KEY manquante).")
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("Assistant Q&A non disponible sur le serveur (paquet 'anthropic' non installé).")
    return anthropic.Anthropic(api_key=cle)


def _rechercher_contexte(question: str, n: int = 4) -> list[dict]:
    """
    Cherche des passages pertinents dans le RAG (dictionnaire + livres +
    traductions coraniques indexées) et les versets coraniques directement
    liés à la question — le contexte réel qui ancre la réponse du modèle.
    """
    contexte = []
    try:
        from rag_livres import rechercher as rag_rechercher
        for r in rag_rechercher(question, n):
            contexte.append({"origine": r.get("titre", "corpus"), "texte": r.get("texte", "")})
    except Exception as e:
        log.warning(f"Recherche RAG pour Q&A: {e}")
    try:
        from coran_pular import rechercher_versets
        for v in rechercher_versets(question, n=2):
            morceau = (
                f"Sourate {v['sourate']} ({v.get('sourate_nom','')}) verset {v['verset']}: "
                f"{v['traduction']} — {v['explication'][:300]}"
            )
            contexte.append({"origine": "Coran en pular", "texte": morceau})
    except Exception as e:
        log.warning(f"Recherche Coran pour Q&A: {e}")
    return contexte


def repondre_question(question: str, max_tokens: int = 600) -> dict:
    """
    Retourne {"reponse": str, "contexte_utilise": [...]} — le contexte est
    renvoyé aussi pour transparence (afficher les sources dans l'UI si besoin).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Question vide.")

    client = anthropic_configure()
    contexte = _rechercher_contexte(question)

    consigne = f"Question: {question}"
    if contexte:
        consigne += "\n\nEXTRAITS DU CORPUS (dictionnaire, Coran en pular, livres communautaires) :"
        for i, c in enumerate(contexte, 1):
            consigne += f"\n\n[Extrait {i} — {c['origine']}]\n{c['texte']}"

    system = (
        "Tu es l'assistant de la plateforme Pular IA, un corpus communautaire pour la "
        "langue et la culture peules du Fouta Djallon. Réponds à la question posée dans "
        "LA MÊME LANGUE que la question (pular ou français) — si la question mélange les "
        "deux, réponds en pular. Sois clair, concis (3 à 6 phrases sauf si la question "
        "demande plus de détail), et chaleureux. Quand des extraits du corpus (dictionnaire, "
        "Coran en pular, livres) sont fournis, appuie-toi dessus en priorité et reste fidèle "
        "à ce qu'ils disent — n'invente pas de traduction ou de fait religieux non couvert "
        "par les extraits. Si tu n'es pas sûr, dis-le simplement plutôt que d'inventer. "
        "Réponse courte, adaptée à une lecture ou une écoute (message vocal)."
    )

    message = client.messages.create(
        model="claude-opus-5",
        max_tokens=max_tokens,
        output_config={"effort": "medium"},
        system=system,
        messages=[{"role": "user", "content": consigne}],
    )
    reponse = "".join(b.text for b in message.content if b.type == "text").strip()
    return {"reponse": reponse, "contexte_utilise": [c["origine"] for c in contexte]}
