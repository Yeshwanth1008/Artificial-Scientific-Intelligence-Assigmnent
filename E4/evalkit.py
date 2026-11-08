"""E4 evaluation kit -- reused by the Main Problem.

* within_topic_pair_accuracy(pred, target, topics, papers)  -> estimate + paper-cluster bootstrap CI
* split_half_ceiling(papers, score_key, gap)                -> how often half-2 reviewers order a pair
                                                               the same way as half-1 (Task 1)
* explainable(acc, ceiling)                                 -> (acc - .5) / (ceiling - .5)

Pairs share papers, so the bootstrap resamples *papers*: a pair is kept with the multiplicity of
the product of its two papers' draw counts (a standard two-way cluster scheme).
"""
from collections import defaultdict

import numpy as np


def _pairs_within(topics, max_per_topic=400, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, t in enumerate(topics):
        by[t].append(i)
    out = []
    for ids in by.values():
        if len(ids) < 2:
            continue
        cand = [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]
        if len(cand) > max_per_topic:
            cand = [cand[k] for k in rng.choice(len(cand), max_per_topic, replace=False)]
        out += cand
    return out


def _correct(pred, target, pairs, min_gap=0.0):
    keep, ok = [], []
    for i, j in pairs:
        dt = target[i] - target[j]
        if abs(dt) <= min_gap:
            continue
        dp = pred[i] - pred[j]
        keep.append((i, j))
        ok.append(1.0 if dp * dt > 0 else (0.5 if dp == 0 else 0.0))
    return keep, np.array(ok)


def paper_bootstrap(pairs, ok, n_papers, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs])
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_papers, n_papers), minlength=n_papers)
        ww = w[I] * w[J]
        if ww.sum() > 0:
            boots.append(float((ww * ok).sum() / ww.sum()))
    return float(np.quantile(boots, .025)), float(np.quantile(boots, .975))


def within_topic_pair_accuracy(pred, target, topics, min_gap=0.0, seed=0):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    pairs = _pairs_within(topics, seed=seed)
    keep, ok = _correct(pred, target, pairs, min_gap)
    if not keep:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_pairs": 0}
    lo, hi = paper_bootstrap(keep, ok, len(pred), seed=seed)
    return {"est": float(ok.mean()), "lo": lo, "hi": hi, "n_pairs": len(keep)}


def split_half_ceiling(papers, key="rating", gap=1.0, n_splits=20, seed=0):
    """papers: list with 'reviews' (list of dicts) and 'topic'. Returns mean over random splits."""
    rng = np.random.default_rng(seed)
    elig = [p for p in papers if sum(r.get(key) is not None for r in p["reviews"]) >= 2]
    topics = [p["topic"] for p in elig]
    pairs = _pairs_within(topics, seed=seed)
    accs = []
    for _ in range(n_splits):
        h1, h2 = [], []
        for p in elig:
            s = [r[key] for r in p["reviews"] if r.get(key) is not None]
            s = list(rng.permutation(s))
            k = len(s) // 2
            h1.append(np.mean(s[:k])); h2.append(np.mean(s[k:]))
        h1, h2 = np.array(h1), np.array(h2)
        same = [float((h1[i] - h1[j]) * (h2[i] - h2[j]) > 0) + 0.5 * float(h2[i] == h2[j])
                for i, j in pairs if abs(h1[i] - h1[j]) > gap]
        if same:
            accs.append(np.mean(same))
    return {"ceiling": float(np.mean(accs)), "sd_over_splits": float(np.std(accs)),
            "n_papers": len(elig), "gap": gap, "key": key}


def explainable(acc, ceiling):
    return (acc - 0.5) / (ceiling - 0.5) if ceiling > 0.5 else float("nan")


def mean_score(p, key):
    v = [r[key] for r in p["reviews"] if r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")
