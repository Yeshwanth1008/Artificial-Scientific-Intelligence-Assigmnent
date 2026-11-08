"""Main Problem data.

    python -m main.data index      # literature index: arXiv cs.LG/CL/CV/AI + stat.ML, 2018-2026, embedded once
    python -m main.data iclr       # ICLR 2024/2025/2026 items: q (question + card), R (S2 refs or retrieved), t, scores
    python -m main.data uptake     # arXiv items with 12-month author-disjoint citation counts (S2)
    python -m main.data groups     # O2 groups: papers citing >= 3 of the same seed papers within 12 months
    python -m main.data hashes     # print sha256 of every test list (paste into PREREG.md BEFORE testing)

Splits (Figure M.2): train = outcomes observable by end of 2023, select = 2024, test = from July 2025.
  ICLR    : train ICLR 2024 (submitted Sept 2023), select ICLR 2025, test ICLR 2026 (submitted Sept 2025)
  uptake  : train papers 2021-01..2022-12, select 2023-01..2023-12, test 2025-07-01..2025-10-31
            (12-month window: the latest test paper's window closes Oct 2026, before this run)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from common.data_sources import load_arxiv, load_iclr, s2_citations, s2_get, s2_references
from common.embed import Embedder
from common.extract import extract_facets, extractor
from common.io import DATA, read_jsonl, sha256_list, write_jsonl

HERE = Path(__file__).parent
CATS = ("cs.LG", "cs.CL", "cs.CV", "cs.AI", "stat.ML")
IDX = DATA / "litindex"


class LitIndex:
    """Embeddings of title+abstract for every paper; all searches are restricted to date < t."""

    def __init__(self, meta, X):
        self.meta, self.X = meta, X
        self.dates = np.array([m["date"] for m in meta])
        self.E = Embedder("qwen3-0.6b", use_prompt=False)

    @classmethod
    def build(cls, start="2018-01-01", end="2026-12-31", max_papers=None):
        rows = load_arxiv(CATS, start, end)
        rows.sort(key=lambda r: r["date"])
        if max_papers and len(rows) > max_papers:  # small tier: uniform subsample, documented in README
            rng = random.Random(0)
            rows = sorted(rng.sample(rows, max_papers), key=lambda r: r["date"])
        E = Embedder("qwen3-0.6b", use_prompt=False, batch_size=64)
        X = None
        for i in range(0, len(rows), 4096):
            chunk = E([f"{r['title']}. {r['abstract']}" for r in rows[i:i + 4096]])
            if X is None:
                X = np.zeros((len(rows), chunk.shape[1]), np.float16)
            X[i:i + 4096] = chunk
        IDX.mkdir(parents=True, exist_ok=True)
        np.save(IDX / "X.npy", X)
        write_jsonl(IDX / "meta.jsonl", [{k: r[k] for k in ("id", "date", "title", "abstract", "authors")} for r in rows])
        return cls(rows, X)

    @classmethod
    def load(cls):
        return cls(read_jsonl(IDX / "meta.jsonl"), np.load(IDX / "X.npy", mmap_mode="r"))

    def search_vec(self, v, t, k=50, min_cos=None, t_after=None, exclude=()):
        hi = np.searchsorted(self.dates, t) if t else len(self.dates)          # strictly before t
        lo = np.searchsorted(self.dates, t_after, side="right") if t_after else 0
        if hi <= lo:
            return []
        s = np.asarray(self.X[lo:hi], dtype=np.float32) @ v
        top = np.argsort(-s)[:k + len(exclude)]
        out = []
        for j in top:
            m = self.meta[lo + j]
            if m["id"] in exclude or (min_cos is not None and s[j] < min_cos):
                continue
            out.append({**m, "cos": float(s[j]), "vec": np.asarray(self.X[lo + j], np.float32)})
            if len(out) >= k:
                break
        return out

    def search(self, text, t, **kw):
        return self.search_vec(self.E([text])[0], t, **kw)


# ------------------------------------------------------------------ ICLR items


def s2_match(title):
    try:
        j = s2_get("paper/search/match", {"query": title, "fields": "paperId,title,externalIds,publicationDate"})
        return j["data"][0] if j.get("data") else None
    except Exception:
        return None


def get_R(title, t, index, k_retrieve=10):
    """References with abstracts dated before t (S2). If fewer than 5, retrieve the k nearest papers dated before t
    from the literature index (setting (b) style); r_source records which."""
    m = s2_match(title)
    refs = []
    if m:
        try:
            j = s2_get(f"paper/{m['paperId']}/references", {"fields": "title,abstract,publicationDate", "limit": 500})
            refs = [d["citedPaper"] for d in j.get("data", []) if d.get("citedPaper")]
        except Exception:
            refs = []
    refs = [{"title": r["title"], "abstract": r["abstract"], "date": r["publicationDate"]} for r in refs
            if r.get("abstract") and r.get("publicationDate") and r["publicationDate"] < t]
    if len(refs) >= 5:
        refs = sorted(refs, key=lambda r: r["date"], reverse=True)[:30]
        return refs, "references"
    got = index.search(title, t, k=k_retrieve)
    return [{"title": g["title"], "abstract": g["abstract"], "date": g["date"]} for g in got], "retrieved"


def build_iclr():
    import datetime
    index = LitIndex.load()
    cards = {r["paper_id"]: r for r in read_jsonl(HERE.parent / "E4" / "cards.jsonl")}
    llm = extractor()
    out = HERE / "items_iclr.jsonl"
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = read_jsonl(out) if out.exists() else []
    for y in (2024, 2025, 2026):
        for p in load_iclr(y):
            if p["paper_id"] in done or p["paper_id"] not in cards:
                continue
            t = datetime.datetime.utcfromtimestamp(p["cdate"] / 1000).strftime("%Y-%m-%d")
            c = cards[p["paper_id"]]
            R, src = get_R(p["title"], t, index)
            sc = {k: float(np.nanmean([r[k] for r in p["reviews"] if r.get(k) is not None] or [np.nan]))
                  for k in ("rating", "contribution", "soundness")}
            rows.append({"id": p["paper_id"], "year": y, "split": {2024: "train", 2025: "select", 2026: "test"}[y],
                         "t": t, "q": c["rq"], "card": c["card"], "facets": extract_facets(llm, c["rq"]),
                         "title": p["title"], "abstract": p["abstract"], "topic": p.get("primary_area") or "unknown",
                         "R": R, "r_source": src, "scores": sc, "n_reviews": len(p["reviews"]),
                         "reviews": p["reviews"]})
            if len(rows) % 100 == 0:
                write_jsonl(out, rows)
    write_jsonl(out, rows)


# ------------------------------------------------------------------ uptake items

SPLITS_UPTAKE = {"train": ("2021-01-01", "2022-12-31"), "select": ("2023-01-01", "2023-12-31"),
                 "test": ("2025-07-01", "2025-10-31")}


def _surnames(authors):
    return {a[0].lower() for a in authors if a}


def uptake_12m(arxiv_id, date, authors):
    """Number of citing papers within 12 months of `date` that share no author surname (author-disjoint)."""
    import datetime
    d0 = datetime.date.fromisoformat(date)
    d1 = (d0 + datetime.timedelta(days=365)).isoformat()
    try:
        cits = s2_citations(f"arXiv:{arxiv_id}")
    except Exception:
        return None
    own = _surnames(authors)
    n = 0
    for c in cits:
        pd = c.get("publicationDate")
        if not pd or not (date <= pd <= d1):
            continue
        names = {a.get("name", "").split()[-1].lower() for a in c.get("authors", []) if a.get("name")}
        if names & own:
            continue
        n += 1
    return n


def build_uptake(n_per_split=None):
    n_per_split = n_per_split or {"train": 3000, "select": 1000, "test": 1500}
    index = LitIndex.load()
    llm = extractor()
    rng = random.Random(0)
    from common.extract import extract_rq
    rows = []
    for split, (a, b) in SPLITS_UPTAKE.items():
        cand = [m for m in index.meta if a <= m["date"] <= b]
        for m in rng.sample(cand, min(n_per_split[split], len(cand))):
            u = uptake_12m(m["id"], m["date"], m["authors"])
            if u is None:
                continue
            q = extract_rq(llm, m["title"], m["abstract"])
            try:
                refs = s2_references(m["id"])
            except Exception:
                refs = []
            R = [{"title": r["title"], "abstract": r["abstract"], "date": r["publicationDate"]} for r in refs
                 if r.get("abstract") and r.get("publicationDate") and r["publicationDate"] < m["date"]][:30]
            ref_ids = [r.get("externalIds", {}).get("CorpusId") for r in refs if r.get("externalIds")]
            rows.append({"id": m["id"], "split": split, "t": m["date"], "q": q, "facets": extract_facets(llm, q),
                         "title": m["title"], "abstract": m["abstract"], "n_authors": len(m["authors"]),
                         "R": R, "ref_ids": [x for x in ref_ids if x], "uptake12": u,
                         "log_uptake": float(np.log1p(u))})
    write_jsonl(HERE / "items_uptake.jsonl", rows)


def build_groups(min_shared=3):
    """O2: within the test split, group papers posted within 12 months of each other that cite >= 3 common seeds."""
    rows = [r for r in read_jsonl(HERE / "items_uptake.jsonl") if r["split"] == "test"]
    sets = [set(r["ref_ids"]) for r in rows]
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if len(sets[i] & sets[j]) >= min_shared:
                pairs.append((rows[i]["id"], rows[j]["id"], len(sets[i] & sets[j])))
    write_jsonl(HERE / "o2_pairs.jsonl", [{"a": a, "b": b, "shared": s} for a, b, s in pairs])
    print(len(pairs), "same-seed pairs")


def hashes():
    out = {}
    for f, key in (("items_iclr.jsonl", "iclr"), ("items_uptake.jsonl", "uptake")):
        p = HERE / f
        if p.exists():
            ids = [r["id"] for r in read_jsonl(p) if r["split"] == "test"]
            out[key] = {"n": len(ids), "sha256": sha256_list(ids)}
    for name in ("E1/contrast.jsonl",):
        p = HERE.parent / name
        if p.exists():
            ids = [r["id"] for r in read_jsonl(p)]
            out[name] = {"n": len(ids), "sha256": sha256_list(ids)}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "index":
        LitIndex.build(max_papers=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        {"iclr": build_iclr, "uptake": build_uptake, "groups": build_groups, "hashes": hashes}[cmd]()
