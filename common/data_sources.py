"""Data loaders. Nothing here ships data; it downloads into data/cache (git-ignored).

arXiv     : Kaggle snapshot `Cornell-University/arxiv` (arxiv-metadata-oai-snapshot.json), filtered
            by category; the paper date is the v1 creation date.
OpenReview: official API v2 via `openreview-py`; needs OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD
            (scripted anonymous access is blocked by a challenge page).
Semantic Scholar: Graph API; set S2_API_KEY (anonymous access is rate-limited to near zero).
"""
import json
import os
import re
import time
from datetime import datetime

import requests

from .io import DATA, read_jsonl, write_jsonl

# ------------------------------------------------------------------ arXiv


def _v1_date(rec):
    try:
        return datetime.strptime(rec["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z").strftime("%Y-%m-%d")
    except Exception:
        return rec.get("update_date")


def load_arxiv(categories=("cs.LG", "cs.CL", "cs.CV", "cs.AI"), start="2021-01-01", end="2026-12-31",
               snapshot=None):
    """Return list of {id, title, abstract, date, categories, authors} from the Kaggle snapshot."""
    out_path = DATA / f"arxiv_{'_'.join(categories)}_{start}_{end}.jsonl"
    if out_path.exists():
        return read_jsonl(out_path)
    snapshot = snapshot or os.environ.get("ARXIV_SNAPSHOT", str(DATA / "arxiv-metadata-oai-snapshot.json"))
    cats = set(categories)
    rows = []
    with open(snapshot, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not cats & set(r["categories"].split()):
                continue
            d = _v1_date(r)
            if not d or not (start <= d <= end):
                continue
            rows.append({"id": r["id"], "title": " ".join(r["title"].split()),
                         "abstract": " ".join(r["abstract"].split()), "date": d,
                         "categories": r["categories"], "authors": r.get("authors_parsed", [])})
    write_jsonl(out_path, rows)
    return rows


# ------------------------------------------------------------------ OpenReview

_num = re.compile(r"^\s*(-?\d+(\.\d+)?)")


def _score(v):
    if isinstance(v, dict):
        v = v.get("value")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = _num.match(str(v))
    return float(m.group(1)) if m else None


def load_iclr(year):
    """Submissions with reviews for ICLR <year>: id, title, abstract, decision, reviews[{rating, soundness,
    contribution, presentation, confidence}], cdate."""
    out_path = DATA / f"iclr{year}.jsonl"
    if out_path.exists():
        return read_jsonl(out_path)
    import openreview
    client = openreview.api.OpenReviewClient(baseurl="https://api2.openreview.net",
                                             username=os.environ["OPENREVIEW_USERNAME"],
                                             password=os.environ["OPENREVIEW_PASSWORD"])
    venue = f"ICLR.cc/{year}/Conference"
    subs = client.get_all_notes(invitation=f"{venue}/-/Submission", details="replies")
    rows = []
    for s in subs:
        c = s.content
        reviews, decision = [], None
        for r in s.details.get("replies", []):
            inv = " ".join(r.get("invitations", []))
            rc = r.get("content", {})
            if inv.endswith("Official_Review"):
                reviews.append({k: _score(rc.get(k)) for k in
                                ("rating", "soundness", "contribution", "presentation", "confidence")})
            elif inv.endswith("Decision"):
                decision = (rc.get("decision") or {}).get("value")
        if not reviews:
            continue
        rows.append({"paper_id": s.id, "year": year, "title": c["title"]["value"],
                     "abstract": c["abstract"]["value"],
                     "keywords": (c.get("keywords") or {}).get("value", []),
                     "primary_area": (c.get("primary_area") or {}).get("value"),
                     "decision": decision, "reviews": reviews, "cdate": s.cdate})
    write_jsonl(out_path, rows)
    return rows


# ------------------------------------------------------------------ Semantic Scholar

S2 = "https://api.semanticscholar.org/graph/v1"


def s2_get(path, params=None, retries=6):
    h = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    for i in range(retries):
        r = requests.get(f"{S2}/{path}", params=params, headers=h, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
    raise RuntimeError(f"S2 failed: {path}")


def s2_references(arxiv_id, fields="title,abstract,publicationDate,externalIds,authors"):
    cache = DATA / "s2_refs" / f"{arxiv_id.replace('/', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    j = s2_get(f"paper/arXiv:{arxiv_id}/references", {"fields": fields, "limit": 1000})
    refs = [d["citedPaper"] for d in j.get("data", []) if d.get("citedPaper")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(refs), encoding="utf-8")
    return refs


def s2_citations(paper_id, fields="title,publicationDate,authors"):
    cache = DATA / "s2_cits" / f"{paper_id.replace('/', '_').replace(':', '_')}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out, off = [], 0
    while True:
        j = s2_get(f"paper/{paper_id}/citations", {"fields": fields, "limit": 1000, "offset": off})
        out += [d["citingPaper"] for d in j.get("data", []) if d.get("citingPaper")]
        if "next" not in j:
            break
        off = j["next"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out
