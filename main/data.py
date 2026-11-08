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
        self._gpu = None
        try:
            import torch
            if torch.cuda.is_available():  # whole index on the GPU in fp16: a search is one matrix-vector product
                self._gpu = torch.from_numpy(np.ascontiguousarray(X)).to("cuda", torch.float16)
        except Exception:
            self._gpu = None

    @classmethod
    def build(cls, start="2018-01-01", end="2026-12-31", max_papers=None):
        rows = load_arxiv(CATS, start, end)
        rows.sort(key=lambda r: r["date"])
        if max_papers and len(rows) > max_papers:  # small tier: uniform subsample, documented in README
            rng = random.Random(0)
            rows = sorted(rng.sample(rows, max_papers), key=lambda r: r["date"])
        from common.embed import _load
        model = _load("qwen3-0.6b")       # encode directly: no per-text cache files for 10^5 papers
        model.max_seq_length = 256         # title + first part of the abstract
        X = model.encode([f"{r['title']}. {r['abstract']}" for r in rows], batch_size=128, normalize_embeddings=True,
                         show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
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
        if self._gpu is not None:
            import torch
            s = (self._gpu[lo:hi] @ torch.from_numpy(np.asarray(v, np.float16)).to("cuda")).float().cpu().numpy()
        else:
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


ICLR_T = {2024: "2023-09-28", 2025: "2024-10-01", 2026: "2025-09-24"}   # submission deadlines (mirrors lack cdate)


def build_iclr(k_retrieve=10):
    """ICLR items = the fixed E4 subsample (2000/1000/2000), q = the E4 question (fixed prompt), facets batched.
    R: the public mirrors have no reference lists, so R is retrieved -- the k nearest arXiv papers dated before
    the submission deadline (setting (b) rule of M.A; r_source = 'retrieved')."""
    sys.path.insert(0, str(HERE.parent / "E4"))
    from run_e4 import papers
    from common.extract import facets_many
    index = LitIndex.load()
    cards = {r["paper_id"]: r for r in read_jsonl(HERE.parent / "E4" / "cards.jsonl")}
    P = [p for p in papers() if p["paper_id"] in cards]
    facets = facets_many(extractor(), [cards[p["paper_id"]]["rq"] for p in P])
    rows = []
    for p, fa in zip(P, facets):
        y = p["year"]
        t = ICLR_T[y]
        got = index.search(f"{p['title']}. {p['abstract']}", t, k=k_retrieve)
        R = [{"title": g["title"], "abstract": g["abstract"], "date": g["date"]} for g in got]
        sc = {k: float(np.nanmean([r[k] for r in p["reviews"] if r.get(k) is not None] or [np.nan]))
              for k in ("rating", "contribution", "soundness")}
        rows.append({"id": p["paper_id"], "year": y, "split": {2024: "train", 2025: "select", 2026: "test"}[y],
                     "t": t, "q": cards[p["paper_id"]]["rq"], "card": cards[p["paper_id"]]["card"], "facets": fa,
                     "title": p["title"], "abstract": p["abstract"], "topic": p["topic"], "R": R,
                     "r_source": "retrieved", "scores": sc, "n_reviews": len(p["reviews"])})
    write_jsonl(HERE / "items_iclr.jsonl", rows)
    print(len(rows), "ICLR items")


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
    """arXiv items with 12-month author-disjoint citation counts and references (Semantic Scholar batch, no key)."""
    from common.data_sources import s2_batch
    from common.extract import facets_many, rq_many
    import datetime
    n_per_split = n_per_split or {"train": 3000, "select": 1000, "test": 1500}
    index = LitIndex.load()
    rng = random.Random(0)
    picked = []
    for split, (a, b) in SPLITS_UPTAKE.items():
        cand = [m for m in index.meta if a <= m["date"] <= b]
        picked += [(split, m) for m in rng.sample(cand, min(n_per_split[split], len(cand)))]
    recs = s2_batch([f"arXiv:{m['id']}" for _, m in picked],
                    "references.paperId,references.title,references.abstract,references.publicationDate,"
                    "citations.publicationDate,citations.authors", chunk=20)
    llm = extractor()
    ok = [(s_, m) for s_, m in picked if recs.get(f"arXiv:{m['id']}")]
    qs = rq_many(llm, [m for _, m in ok])
    fas = facets_many(llm, qs)
    rows = []
    for (split, m), q, fa in zip(ok, qs, fas):
        x = recs[f"arXiv:{m['id']}"]
        d1 = (datetime.date.fromisoformat(m["date"]) + datetime.timedelta(days=365)).isoformat()
        own = _surnames(m["authors"])
        up = 0
        for c in x.get("citations") or []:
            pd = c.get("publicationDate")
            if not pd or not (m["date"] <= pd <= d1):
                continue
            names = {a.get("name", "").split()[-1].lower() for a in c.get("authors", []) if a.get("name")}
            if not names & own:
                up += 1
        refs = x.get("references") or []
        R = [{"title": r["title"], "abstract": r["abstract"], "date": r["publicationDate"]} for r in refs
             if r.get("abstract") and r.get("publicationDate") and r["publicationDate"] < m["date"]][:30]
        rows.append({"id": m["id"], "split": split, "t": m["date"], "q": q, "facets": fa, "title": m["title"],
                     "abstract": m["abstract"], "n_authors": len(m["authors"]), "R": R,
                     "r_source": "references" if len(R) >= 5 else "few_references",
                     "ref_ids": [r["paperId"] for r in refs if r.get("paperId")], "uptake12": up,
                     "log_uptake": float(np.log1p(up))})
    write_jsonl(HERE / "items_uptake.jsonl", rows)
    print(len(rows), "uptake items")


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
