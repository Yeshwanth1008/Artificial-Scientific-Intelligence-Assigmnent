"""Diversity of a set of research questions (E5-style measures, used by E6 and the Main Problem D readout).

diversity(rqs, facets=True) -> {
    'mean_cos_dist', 'vendi',                       # embedding-only (gameable: see E5 axioms)
    'n_distinct', 'class_entropy',                  # equivalence classes from the same-question cross-encoder
    'facet_vendi': {purpose, mechanism, evaluation, domain},   # per-facet dispersion
    'qd_count'                                      # largest pairwise-distinct subset passing quality gates
}
Quality gates (fixed in advance): a question passes if (a) it ends with '?', (b) it has 8-60 words and
(c) its facet answers name a non-empty mechanism and evaluation that are not generic ("method", "performance").
"""
import numpy as np

from .embed import Embedder

GENERIC = {"", "method", "methods", "approach", "performance", "results", "evaluation", "it", "n/a", "none",
           "unknown", "various"}


def vendi(X):
    """Vendi score of row-normalised embeddings: exp(entropy of eigenvalues of K/n)."""
    if len(X) == 0:
        return 0.0
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = X @ X.T / len(X)
    w = np.clip(np.linalg.eigvalsh(K), 0, None)
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def same_classes(rqs, threshold=0.5):
    """Leader clustering on cross-encoder decisions: a member must be judged 'same' as the class leader
    (T1 A.2: avoids the chaining of connected components and the non-uniqueness of cliques)."""
    from .rerank import same_question_scores
    n = len(rqs)
    if n == 0:
        return [], np.zeros((0, 0))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    S = np.eye(n)
    if pairs:
        s = same_question_scores([(rqs[i], rqs[j]) for i, j in pairs])
        for (i, j), v in zip(pairs, s):
            S[i, j] = S[j, i] = v
    label, leaders = [-1] * n, []
    for i in range(n):
        for c, l in enumerate(leaders):
            if S[i, l] >= threshold:
                label[i] = c
                break
        else:
            label[i] = len(leaders)
            leaders.append(i)
    return label, S


def passes_gate(q, facet):
    w = len(q.split())
    if not q.strip().endswith("?") or not (8 <= w <= 60):
        return False
    return all(facet.get(k, "").strip().lower() not in GENERIC for k in ("mechanism", "evaluation"))


def diversity(rqs, facets=True, facet_answers=None, threshold=0.5):
    rqs = list(rqs)
    E = Embedder("qwen3-0.6b", use_prompt=False)
    X = E(rqs)
    n = len(rqs)
    D = 1 - X @ X.T
    out = {"n": n,
           "mean_cos_dist": float(D[np.triu_indices(n, 1)].mean()) if n > 1 else 0.0,
           "vendi": vendi(X)}
    label, S = same_classes(rqs, threshold)
    counts = np.bincount(label)
    p = counts / counts.sum()
    out["n_distinct"] = int(len(counts))
    out["class_entropy"] = float(-(p * np.log(p)).sum())
    if facets:
        if facet_answers is None:
            from .extract import extract_facets, extractor
            llm = extractor()
            facet_answers = [extract_facets(llm, q) for q in rqs]
        out["facet_vendi"] = {k: vendi(E([f.get(k, "") or "none" for f in facet_answers]))
                              for k in ("purpose", "mechanism", "evaluation", "domain")}
        ok = [i for i in range(n) if passes_gate(rqs[i], facet_answers[i])]
        # greedy largest pairwise-distinct subset among gated questions (exact for the small sets used here)
        chosen = []
        for i in ok:
            if all(S[i, j] < threshold for j in chosen):
                chosen.append(i)
        out["qd_count"] = len(chosen)
        out["facets"] = facet_answers
    return out
