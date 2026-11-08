"""E1 -- hand-made test of identity and scope.

    python E1/eval_contrast.py --check            # validate contrast.jsonl against the Task-1 constraints
    python E1/eval_contrast.py                    # Task 2 + Task 4, writes E1/results.json

Task-1 constraints (checked by script): Jaccard(A,B) <= 0.3; B_min keeps >= 75% of A's words;
Jaccard(A, C_tilde) <= 0.3; >= 10 items per near-miss type; >= 10 items with W longer than N; >= 60 items.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import read_jsonl, write_json  # noqa: E402
from common.stats import cluster_bootstrap  # noqa: E402

HERE = Path(__file__).parent
TYPES = ["role_swap", "constraint_flip", "goal_swap", "inquiry_change", "direction_flip"]
FIELDS = ["A", "B", "B_min", "C", "C_tilde", "narrower", "broader"]


def words(s):
    return re.findall(r"[a-z0-9\-]+", s.lower())


def jaccard(a, b):
    a, b = set(words(a)), set(words(b))
    return len(a & b) / max(1, len(a | b))


def kept_share(a, b):
    """Share of A's word tokens (multiset) kept in b."""
    ca, cb = Counter(words(a)), Counter(words(b))
    return sum(min(n, cb[w]) for w, n in ca.items()) / max(1, sum(ca.values()))


def check(items):
    errs = []
    for it in items:
        i = it["id"]
        for f in FIELDS:
            if not it.get(f):
                errs.append(f"{i}: missing {f}")
        if it.get("C_type") not in TYPES:
            errs.append(f"{i}: bad C_type {it.get('C_type')}")
        if jaccard(it["A"], it["B"]) > 0.3:
            errs.append(f"{i}: Jaccard(A,B)={jaccard(it['A'], it['B']):.2f} > 0.3")
        if kept_share(it["A"], it["B_min"]) < 0.75:
            errs.append(f"{i}: B_min keeps {kept_share(it['A'], it['B_min']):.0%} < 75%")
        if jaccard(it["A"], it["C_tilde"]) > 0.3:
            errs.append(f"{i}: Jaccard(A,C_tilde)={jaccard(it['A'], it['C_tilde']):.2f} > 0.3")
        lf = len(words(it["broader"])) > len(words(it["narrower"]))
        if bool(it.get("length_flipped")) != lf:
            errs.append(f"{i}: length_flipped flag should be {lf}")
    cnt = Counter(it["C_type"] for it in items)
    for t in TYPES:
        if cnt[t] < 10:
            errs.append(f"only {cnt[t]} items of type {t} (need 10)")
    nlf = sum(len(words(it["broader"])) > len(words(it["narrower"])) for it in items)
    if nlf < 10:
        errs.append(f"only {nlf} length-flipped items (need 10)")
    if len(items) < 60:
        errs.append(f"only {len(items)} items (need 60)")
    return errs, cnt, nlf


# ------------------------------------------------------------------ distances

def embed_distances(items, model_name):
    from common.embed import Embedder
    emb = Embedder(model_name)
    D = {}
    for f in FIELDS:
        D[f] = emb([it[f] for it in items])
    dist = {f: 1 - np.sum(D["A"] * D[f], 1) for f in FIELDS if f != "A"}
    # scope from the embedding: a broader question should sit closer to the corpus mean direction
    # (lower specificity). We use the norm of the projection off the mean as a "specificity" proxy.
    mu = np.mean(np.concatenate([D[f] for f in FIELDS]), 0)
    mu /= np.linalg.norm(mu)
    spec = {f: 1 - D[f] @ mu for f in ("A", "narrower", "broader")}
    return dist, spec


def cross_distances(items):
    from common.rerank import same_question_scores
    dist = {}
    for f in FIELDS:
        if f == "A":
            continue
        s = same_question_scores([(it["A"], it[f]) for it in items])
        dist[f] = 1 - s
    return dist


def metrics(items, dist, spec=None):
    ids = [it["id"] for it in items]
    types = [it["C_type"] for it in items]
    res = {}
    q = {
        "near_miss_error": dist["C"] < dist["B"],
        "overlap_shortcut": dist["C"] < dist["B_min"],
        "low_overlap_shortcut": dist["C_tilde"] < dist["B"],
    }
    for k, v in q.items():
        v = v.astype(float)
        res[k] = {"overall": cluster_bootstrap(v, ids)}
        for t in TYPES:
            m = np.array([x == t for x in types])
            if m.sum():
                res[k][t] = cluster_bootstrap(v[m], np.array(ids)[m])
    if spec is not None:
        order = ((spec["broader"] < spec["A"]) & (spec["A"] < spec["narrower"])).astype(float)
        lf = np.array([bool(it.get("length_flipped")) for it in items])
        res["scope_order"] = {"overall": cluster_bootstrap(order, ids)}
        if lf.sum():
            res["scope_order"]["length_flipped"] = cluster_bootstrap(order[lf], np.array(ids)[lf])
        if (~lf).sum():
            res["scope_order"]["not_flipped"] = cluster_bootstrap(order[~lf], np.array(ids)[~lf])
    return res


def length_baseline(items):
    """Length-only scope model: shorter = broader."""
    n = lambda s: len(words(s))
    order = np.array([float(n(it["broader"]) < n(it["A"]) < n(it["narrower"])) for it in items])
    lf = np.array([bool(it.get("length_flipped")) for it in items])
    ids = np.array([it["id"] for it in items])
    out = {"overall": cluster_bootstrap(order, ids)}
    if lf.sum():
        out["length_flipped"] = cluster_bootstrap(order[lf], ids[lf])
    return out


# ------------------------------------------------------------------ Task 4: dedup

def dedup(items, model_name="qwen3-0.6b", k=10, cross_threshold=0.5):
    from common.embed import Embedder
    from common.rerank import same_question_scores
    texts, owner, role = [], [], []
    for it in items:
        for f in FIELDS:
            texts.append(it[f]); owner.append(it["id"]); role.append(f)
    E = Embedder(model_name)(texts)
    S = E @ E.T
    pos_roles, neg_roles = {"B", "B_min"}, {"C", "C_tilde"}
    a_idx = [i for i, r in enumerate(role) if r == "A"]
    cand = []
    for i in a_idx:
        order = [j for j in np.argsort(-S[i]) if j != i][:k]
        cand += [(i, j) for j in order]
    scores = same_question_scores([(texts[i], texts[j]) for i, j in cand])

    def pr(decide):
        tp = fp = 0
        for (i, j), d in zip(cand, decide):
            same_item = owner[i] == owner[j]
            if d:
                if same_item and role[j] in pos_roles:
                    tp += 1
                else:
                    fp += 1
        total_pos = 2 * len(a_idx)
        return {"precision": tp / max(1, tp + fp), "recall": tp / total_pos,
                "false_accept_C_or_Ctilde": sum(d and owner[i] == owner[j] and role[j] in neg_roles
                                                for (i, j), d in zip(cand, decide)) / max(1, 2 * len(a_idx))}

    cross = pr(scores >= cross_threshold)
    # best cosine threshold chosen with hindsight on the same data (an optimistic reference)
    cos_vals = np.array([S[i, j] for i, j in cand])
    best = max(((t, pr(cos_vals >= t)) for t in np.unique(np.round(cos_vals, 3))),
               key=lambda x: 2 * x[1]["precision"] * x[1]["recall"] / max(1e-9, x[1]["precision"] + x[1]["recall"]))
    return {"cross_encoder": cross, "cross_threshold": cross_threshold,
            "best_cosine_hindsight": {"threshold": float(best[0]), **best[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(HERE / "contrast.jsonl"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--models", default="qwen3-0.6b,specter2,mpnet,bge-small")
    ap.add_argument("--part", default=None, help="run one part in its own process: an embedder name, 'cross' or 'dedup'")
    ap.add_argument("--merge", action="store_true", help="merge the per-part files into results.json")
    a = ap.parse_args()
    items = [it for it in read_jsonl(a.file) if it["id"] != "h000"]  # h000 = the paper's example
    errs, cnt, nlf = check(items)
    print(f"{len(items)} items; types {dict(cnt)}; length-flipped {nlf}")
    for e in errs:
        print("  -", e)
    if a.check:
        sys.exit(1 if errs else 0)
    parts = HERE / "results_parts"
    if a.part:  # one model per process (several models in one process crash torch on some Windows machines)
        parts.mkdir(exist_ok=True)
        if a.part == "cross":
            out = metrics(items, cross_distances(items))
        elif a.part == "dedup":
            out = dedup(items)
        else:
            dist, spec = embed_distances(items, a.part)
            out = metrics(items, dist, spec)
        write_json(parts / f"{a.part}.json", out)
        print(a.part, "done")
        return
    if a.merge:
        from common.io import read_json
        res = {"n_items": len(items), "constraint_errors": errs, "models": {}}
        for m in a.models.split(","):
            res["models"][m] = read_json(parts / f"{m}.json")
        res["models"]["cross-encoder:qwen3-reranker-0.6b"] = read_json(parts / "cross.json")
        res["length_only_scope_baseline"] = length_baseline(items)
        res["dedup"] = read_json(parts / "dedup.json")
        write_json(HERE / "results.json", res)
        print("merged")
        return
    res = {"n_items": len(items), "constraint_errors": errs, "models": {}}
    for m in a.models.split(","):
        dist, spec = embed_distances(items, m)
        res["models"][m] = metrics(items, dist, spec)
    res["models"]["cross-encoder:qwen3-reranker-0.6b"] = metrics(items, cross_distances(items))
    res["length_only_scope_baseline"] = length_baseline(items)
    res["dedup"] = dedup(items)
    write_json(HERE / "results.json", res)
    for m, r in res["models"].items():
        print(m, {k: round(v["overall"]["est"], 3) for k, v in r.items()})


if __name__ == "__main__":
    main()
