"""Fixed-prompt extraction of research questions, question cards, claims and facets (cached).

Every function has a batched twin (`*_many`) that fills the cache in one batched call first; the
single-item functions then read the cache, so both paths give identical outputs.
"""
from .llm import LLM, parse_json
from .prompts import EXTRACT_CARD, EXTRACT_CLAIMS, EXTRACT_RQ, FACETS

CARD_KEYS = ["problem", "gap", "gap_owner", "insight", "why_now", "evidence_plan", "contribution_type", "scope"]
FACET_KEYS = ["purpose", "mechanism", "evaluation", "domain"]


def extractor(seed=0):
    # temperature 0 for extraction: one fixed prompt, deterministic output
    return LLM(temperature=0.0, seed=seed, max_new_tokens=400)


def _rq_from_text(t):
    t = t.strip().split("\n")[0].strip().strip('"')
    return t if t.endswith("?") else t.rstrip(".") + "?"


def extract_rq(llm, title, abstract):
    return _rq_from_text(llm(EXTRACT_RQ.format(title=title, abstract=abstract), max_new_tokens=80)["text"])


def extract_card(llm, title, abstract):
    j = parse_json(llm(EXTRACT_CARD.format(title=title, abstract=abstract))["text"]) or {}
    return {k: str(j.get(k, "")) for k in CARD_KEYS}


def card_text(card):
    return " ".join(f"{k.replace('_', ' ').capitalize()}: {card.get(k, '')}." for k in CARD_KEYS if card.get(k))


def _claims_from_text(t):
    claims = [l.strip().lstrip("-*0123456789.) ").strip() for l in t.splitlines()]
    return [c for c in claims if len(c.split()) >= 4][:10]


def extract_claims(llm, title, abstract):
    return _claims_from_text(llm(EXTRACT_CLAIMS.format(title=title, abstract=abstract))["text"])


def extract_facets(llm, q):
    j = parse_json(llm(FACETS.format(q=q), max_new_tokens=160)["text"]) or {}
    return {k: str(j.get(k, "")) for k in FACET_KEYS}


# ---------------------------------------------------------------- batched
def rq_many(llm, papers):
    llm.batch([EXTRACT_RQ.format(title=p["title"], abstract=p["abstract"]) for p in papers], max_new_tokens=80)
    return [extract_rq(llm, p["title"], p["abstract"]) for p in papers]


def card_many(llm, papers):
    llm.batch([EXTRACT_CARD.format(title=p["title"], abstract=p["abstract"]) for p in papers])
    return [extract_card(llm, p["title"], p["abstract"]) for p in papers]


def claims_many(llm, papers):
    llm.batch([EXTRACT_CLAIMS.format(title=p["title"], abstract=p["abstract"]) for p in papers])
    return [extract_claims(llm, p["title"], p["abstract"]) for p in papers]


def facets_many(llm, qs):
    llm.batch([FACETS.format(q=q) for q in qs], max_new_tokens=160)
    return [extract_facets(llm, q) for q in qs]
