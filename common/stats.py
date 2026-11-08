"""Statistics used everywhere: cluster bootstrap, pair accuracy, AUROC, Cohen's kappa.

The rule from the Experiment paper: resample the right unit (paper, input set, system),
never the individual pair when pairs share these.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """95% CI of stat(values) resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    idx = [np.array(groups[k]) for k in keys]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[p] for p in pick])
        boots.append(stat(values[sel]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"est": float(stat(values)), "lo": float(lo), "hi": float(hi), "n": int(len(values)),
            "n_clusters": len(keys)}


def paired_diff_bootstrap(a, b, clusters, n_boot=2000, seed=0):
    """CI of mean(a - b) with cluster resampling (a, b aligned per item)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return cluster_bootstrap(d, clusters, n_boot=n_boot, seed=seed)


def pair_accuracy(scores, targets, pairs):
    """Share of (i, j) pairs ordered the same way by scores and targets (ties in target skipped)."""
    s, t = np.asarray(scores), np.asarray(targets)
    out = []
    for i, j in pairs:
        if t[i] == t[j]:
            continue
        out.append(float((s[i] - s[j]) * (t[i] - t[j]) > 0) + 0.5 * float(s[i] == s[j]))
    return np.array(out)


def within_stratum_pairs(strata, max_pairs_per_stratum=200, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, s in enumerate(strata):
        by[s].append(i)
    pairs = []
    for s, ids in by.items():
        if len(ids) < 2:
            continue
        cand = [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]
        if len(cand) > max_pairs_per_stratum:
            cand = [cand[k] for k in rng.choice(len(cand), max_pairs_per_stratum, replace=False)]
        pairs += cand
    return pairs


def auroc(scores, labels):
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def explainable_share(acc, ceiling):
    """(accuracy - 0.5) / (ceiling - 0.5), the E4 normalisation."""
    return (acc - 0.5) / (ceiling - 0.5) if ceiling > 0.5 else float("nan")
