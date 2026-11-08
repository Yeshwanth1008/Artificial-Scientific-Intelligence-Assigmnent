"""Fixed-prompt extraction of research questions, question cards, claims and facets (cached)."""
from .llm import LLM, parse_json
from .prompts import EXTRACT_CARD, EXTRACT_CLAIMS, EXTRACT_RQ, FACETS

CARD_KEYS = ["problem", "gap", "gap_owner", "insight", "why_now", "evidence_plan", "contribution_type", "scope"]
FACET_KEYS = ["purpose", "mechanism", "evaluation", "domain"]


def extractor(seed=0):
    # temperature 0 for extraction: one fixed prompt, deterministic output
    return LLM(temperature=0.0, seed=seed, max_new_tokens=400)


def extract_rq(llm, title, abstract):
    t = llm(EXTRACT_RQ.format(title=title, abstract=abstract), max_new_tokens=80)["text"]
    t = t.strip().split("\n")[0].strip().strip('"')
    return t if t.endswith("?") else t.rstrip(".") + "?"


def extract_card(llm, title, abstract):
    j = parse_json(llm(EXTRACT_CARD.format(title=title, abstract=abstract))["text"]) or {}
    return {k: str(j.get(k, "")) for k in CARD_KEYS}


def card_text(card):
    return " ".join(f"{k.replace('_', ' ').capitalize()}: {card.get(k, '')}." for k in CARD_KEYS if card.get(k))


def extract_claims(llm, title, abstract):
    t = llm(EXTRACT_CLAIMS.format(title=title, abstract=abstract))["text"]
    claims = [l.strip().lstrip("-*0123456789.) ").strip() for l in t.splitlines()]
    claims = [c for c in claims if len(c.split()) >= 4]
    return claims[:10]


def extract_facets(llm, q):
    j = parse_json(llm(FACETS.format(q=q), max_new_tokens=160)["text"]) or {}
    return {k: str(j.get(k, "")) for k in FACET_KEYS}
