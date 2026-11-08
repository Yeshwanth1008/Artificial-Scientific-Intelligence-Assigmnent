"""E2 novelty measures.

    score(idea_text, claims, pool_ids, t) -> {'Y', 'density', 'N', 'yours'}

Only pool papers dated strictly before t are used.

* Y        = 1 - max_p cos(x, p)                                    (distance to nearest paper)
* density  = relative neighbourhood density [R7]: share of x's k nearest papers whose own
             k-NN distance is smaller than x's (1 = x sits in a sparser spot than all its neighbours)
* N        = 1 - mean_i max_p cos(c_i, p)                           (claim coverage, T2)
* yours    = 1 - (C + K)/2  (T2 B.4) with two changes for real text, both motivated by the
             "where it breaks" list in theory/answers.md:
               - claims are matched against *sentences* of pool papers, not whole-paper vectors
                 (a claim is diluted inside a whole-abstract embedding);
               - cosines are calibrated:  s = clip((cos - b) / (1 - b), 0, 1) with b the 90th
                 percentile of claim-to-sentence cosine on the *training* pools, frozen in
                 calib.json (dense topics otherwise give every claim a high cosine without
                 real coverage).
             C = mean_i max_p s(c_i, p)   K = max_p (sum_i s(c_i, p))^2 / m^2
             (K is divided by m^2 rather than m so it stays in [0, 1] for non-orthogonal claims.)
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.embed import Embedder  # noqa: E402
from common.io import read_jsonl  # noqa: E402

HERE = Path(__file__).parent
_EMB = None
_STORE = {}


def emb():
    global _EMB
    if _EMB is None:
        import os
        _EMB = Embedder(os.environ.get("ASI_EMBEDDER", "qwen3-0.6b"), use_prompt=False)
    return _EMB


def register(papers):
    """Add papers (dicts with id, date, title, abstract) to the in-memory store, e.g. planted copies."""
    for p in papers:
        _STORE[p["id"]] = p


def load_store():
    for f in (HERE / "pools").glob("*.jsonl"):
        register(read_jsonl(f))


def sentences(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.split()) >= 5]
    return s or [text]


def _paper_text(p):
    return f"{p['title']}. {p['abstract']}"


def _density(xv, P, k=10):
    sims = P @ xv
    nn = np.argsort(-sims)[:k]
    dx = 1 - sims[nn].mean()
    PP = P[nn] @ P.T
    dn = []
    for r, i in enumerate(nn):
        row = np.delete(PP[r], i)
        dn.append(1 - np.sort(row)[-k:].mean())
    return float(np.mean(np.array(dn) < dx))


def calibrate(claims, papers, q=0.9):
    """Estimate the background level b once, on training pools, and freeze it in calib.json.
    b must NOT be recomputed per pool: then adding papers could raise b and raise novelty,
    breaking the monotonicity proved in T2 B.4."""
    import json
    E = emb()
    sent = [s for p in papers for s in sentences(_paper_text(p))]
    b = float(np.quantile(E(claims) @ E(sent).T, q))
    (HERE / "calib.json").write_text(json.dumps({"b": b, "q": q, "n_claims": len(claims),
                                                 "n_sentences": len(sent)}))
    return b


def _calib_b():
    import json
    f = HERE / "calib.json"
    return json.loads(f.read_text())["b"] if f.exists() else 0.5


def score_papers(idea_text, claims, papers, t, k=10):
    papers = [p for p in papers if p["date"] < t]
    if not papers:
        return {"Y": 1.0, "density": 1.0, "N": 1.0, "yours": 1.0}
    E = emb()
    xv = E([idea_text])[0]
    P = E([_paper_text(p) for p in papers])
    Cv = E(claims)
    out = {"Y": float(1 - (P @ xv).max()),
           "density": _density(xv, P, k=min(k, len(papers) - 1)) if len(papers) > 2 else 1.0,
           "N": float(1 - (Cv @ P.T).max(1).mean())}
    # sentence-level, calibrated version of the T2 B.4 measure
    sent, owner = [], []
    for j, p in enumerate(papers):
        for s in sentences(_paper_text(p)):
            sent.append(s); owner.append(j)
    S = E(sent)
    owner = np.array(owner)
    cs = Cv @ S.T                                       # m x n_sent
    b = _calib_b()                                      # frozen background level (calib.json)
    s = np.clip((cs - b) / (1 - b), 0, 1)
    per_paper = np.zeros((len(claims), len(papers)))    # max over a paper's sentences
    np.maximum.at(per_paper.T, owner, s.T)
    m = len(claims)
    C = per_paper.max(1).mean()
    K = (per_paper.sum(0).max() ** 2) / m ** 2
    out["yours"] = float(1 - 0.5 * (C + K))
    out["_calib_b"] = b
    return out


def score(idea_text: str, claims: list, pool_ids: list, t: str) -> dict:
    """Return {'Y':..., 'density':..., 'N':..., 'yours':...}. Use only pool papers dated before t."""
    if not _STORE:
        load_store()
    papers = [_STORE[i] for i in pool_ids if i in _STORE]
    r = score_papers(idea_text, claims, papers, t)
    return {k: r[k] for k in ("Y", "density", "N", "yours")}
