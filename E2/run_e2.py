"""E2 -- novelty axioms, rebuilt.

    python E2/run_e2.py pools        # Task 1a: build pools + focal sets, write manifest
    python E2/run_e2.py claims       # Task 1b: extract 4-10 claims per focal paper (fixed prompt)
    python E2/run_e2.py calibrate    # freeze the background level b for 'yours' on the *older* pools
    python E2/run_e2.py axioms       # Task 1c + 2: manipulations, four measures, pass rates
    python E2/run_e2.py human        # Task 3: AUROC on novelty labels vs a quality-only baseline

Pools (arXiv snapshot, primary category):
    for each field F in {cs.CL, cs.CV, cs.RO}: pool_F = 1000 papers posted in 2023,
    older_F = 1000 papers posted in 2021, focal_F = 100 papers posted Jan-Mar 2024 (t = their date).
    'near' pool = own field 2023, 'far' pool = a different field 2023.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_sources import load_arxiv  # noqa: E402
from common.extract import extract_claims, extractor  # noqa: E402
from common.io import read_jsonl, write_json, write_jsonl, read_json  # noqa: E402
from common.llm import LLM  # noqa: E402
from common.prompts import PARAPHRASE  # noqa: E402
from common.stats import auroc, cluster_bootstrap  # noqa: E402
import novelty  # noqa: E402

HERE = Path(__file__).parent
FIELDS = ["cs.CL", "cs.CV", "cs.RO"]
FAR = {"cs.CL": "cs.CV", "cs.CV": "cs.RO", "cs.RO": "cs.CL"}
N_POOL, N_FOCAL = 1000, 100


def primary(p):
    return p["categories"].split()[0]


def cmd_pools():
    rng = random.Random(0)
    allp = load_arxiv(categories=tuple(FIELDS), start="2021-01-01", end="2024-03-31")
    manifest = {}
    for F in FIELDS:
        fp = [p for p in allp if primary(p) == F]
        sets = {
            "pool": [p for p in fp if p["date"].startswith("2023")],
            "older": [p for p in fp if p["date"].startswith("2021")],
            "focal": [p for p in fp if "2024-01-01" <= p["date"] <= "2024-03-31"],
        }
        for name, ps in sets.items():
            n = N_FOCAL if name == "focal" else N_POOL
            ps = rng.sample(ps, min(n, len(ps)))
            rows = [{"id": p["id"], "date": p["date"], "title": p["title"], "abstract": p["abstract"]} for p in ps]
            write_jsonl(HERE / "pools" / f"{F}_{name}.jsonl", rows)
            manifest[f"{F}_{name}"] = [{"id": r["id"], "date": r["date"]} for r in rows]
    write_json(HERE / "pool_manifest.json", manifest)
    print({k: len(v) for k, v in manifest.items()})


def cmd_claims(limit=None):
    """limit: focal papers per field (CPU budget; the paper asks for 100 -- deviation in NOTEBOOK.md)."""
    llm = extractor()
    for F in FIELDS:
        rows = read_jsonl(HERE / "pools" / f"{F}_focal.jsonl")[:limit]
        for r in rows:
            r["claims"] = extract_claims(llm, r["title"], r["abstract"])
        rows = [r for r in rows if len(r["claims"]) >= 4]
        write_jsonl(HERE / "focal" / f"{F}.jsonl", rows)
        print(F, len(rows), "focal papers with >= 4 claims")


def cmd_calibrate():
    claims, papers = [], []
    for F in FIELDS:
        papers += read_jsonl(HERE / "pools" / f"{F}_older.jsonl")[:300]
        claims += [c for r in read_jsonl(HERE / "focal" / f"{F}.jsonl")[:30] for c in r["claims"]]
    print("b =", novelty.calibrate(claims, papers))


# ------------------------------------------------------------------ manipulations

def _plant(paper, claims, tag):
    q = dict(paper)
    q["id"] = f"planted:{paper['id']}:{tag}"
    q["abstract"] = paper["abstract"] + " " + " ".join(claims)
    return q


def _nearest(texts, papers):
    E = novelty.emb()
    P = E([novelty._paper_text(p) for p in papers])
    return [int(np.argmax(P @ v)) for v in E(texts)]


def manipulations(focal, pool, para_llm):
    """Return {name: pool_list}. Expected direction relative to 'original' is in EXPECT."""
    t = focal["date"]
    pool = [p for p in pool if p["id"] != focal["id"]]  # focal paper itself never in pool
    out = {"original": pool}
    copy = {"id": f"copy:{focal['id']}", "date": "2023-12-31", "title": focal["title"], "abstract": focal["abstract"]}
    out["exact_copy"] = pool + [copy]
    para = para_llm(PARAPHRASE.format(text=f"{focal['title']}. {focal['abstract']}"), max_new_tokens=500)["text"]
    out["paraphrase_copy"] = pool + [{"id": f"para:{focal['id']}", "date": "2023-12-31", "title": "", "abstract": para}]
    claims = focal["claims"]
    m = len(claims)
    near = _nearest(claims, pool)
    for frac in (25, 50, 75, 100):
        j = max(1, round(m * frac / 100))
        P = list(pool)
        for i in range(j):  # distributed: each claim into the pool paper nearest to that claim
            P[near[i]] = _plant(P[near[i]], [claims[i]], f"p{frac}_{i}")
        out[f"partial_{frac}"] = P
    j = min(3, m)
    D = list(pool)
    used = set()
    for i in range(j):
        k = near[i]
        while k in used:  # distributed must use j different papers
            k = (k + 1) % len(D)
        used.add(k)
        D[k] = _plant(D[k], [claims[i]], f"dist_{i}")
    out["distributed_3"] = D
    C = list(pool)
    C[near[0]] = _plant(C[near[0]], claims[:j], "conc")
    out["concentrated_3"] = C
    return out


EXPECT = {  # name: (compare_to, sign) -- score(name) must be  < compare_to (sign=-1) or > (sign=+1)
    "exact_copy": ("original", -1), "paraphrase_copy": ("original", -1),
    "partial_25": ("original", -1), "partial_50": ("partial_25", -1),
    "partial_75": ("partial_50", -1), "partial_100": ("partial_75", -1),
    "concentrated_3": ("distributed_3", -1),
    "far_pool": ("original", +1), "older_pool": ("original", +1),
}
MEASURES = ["Y", "density", "N", "yours"]


def cmd_axioms(limit=None):
    para_llm = LLM(temperature=0.0, seed=0, max_new_tokens=500)
    pools = {F: read_jsonl(HERE / "pools" / f"{F}_pool.jsonl") for F in FIELDS}
    older = {F: read_jsonl(HERE / "pools" / f"{F}_older.jsonl") for F in FIELDS}
    raw = []
    for F in FIELDS:
        foc = read_jsonl(HERE / "focal" / f"{F}.jsonl")[:limit]
        for f in foc:
            idea = f"{f['title']}. {f['abstract']}"
            pools_m = manipulations(f, pools[F], para_llm)
            pools_m["far_pool"] = pools[FAR[F]]
            pools_m["older_pool"] = older[F]
            sc = {name: novelty.score_papers(idea, f["claims"], P, f["date"]) for name, P in pools_m.items()}
            raw.append({"field": F, "id": f["id"], "m": len(f["claims"]),
                        "scores": {n: {k: s[k] for k in MEASURES} for n, s in sc.items()}})
    write_json(HERE / "results" / "axioms_raw.json", raw)
    res = {}
    for name, (ref, sign) in EXPECT.items():
        res[name] = {}
        for ms in MEASURES:
            ok = [float(sign * (r["scores"][name][ms] - r["scores"][ref][ms]) > 0) for r in raw]
            res[name][ms] = cluster_bootstrap(ok, [r["id"] for r in raw])
    write_json(HERE / "results" / "axioms_pass_rates.json", res)
    for name in EXPECT:
        print(f"{name:16s}", "  ".join(f"{m}={res[name][m]['est']:.2f}" for m in MEASURES))


# ------------------------------------------------------------------ human check

def cmd_human():
    """Expects data/novelty_judge_bench.jsonl with {id, title, abstract, date, label(1=novel), venue}
    (post-cutoff labels of [R2]) and data/cache/iclr{2024,2025}.jsonl for the quality-only baseline."""
    from sklearn.linear_model import RidgeCV
    from common.data_sources import load_iclr
    from common.embed import Embedder
    import os
    root = HERE.parent
    test = read_jsonl(root / "data" / "novelty_judge_bench.jsonl")
    T_NJB = "2025-09-15"  # ideas are ICLR 2026 submissions (abstract deadline Sept 2025): literature strictly before
    for r in test:
        r.setdefault("date", T_NJB)
    test_titles = {r["title"].lower().strip() for r in test}
    sys.path.insert(0, str(root / "E4"))
    from run_e4 import papers as e4_papers  # same fixed ICLR 2024/2025 subsample as E4 (embeddings cached)
    train = [r for r in e4_papers() if r["year"] in (2024, 2025) and r["title"].lower().strip() not in test_titles]
    E = Embedder("qwen3-0.6b", use_prompt=False)
    Xtr = E([f"{r['title']}. {r['abstract']}" for r in train])
    ytr = [np.mean([v["rating"] for v in r["reviews"] if v["rating"] is not None]) for r in train]
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xtr, ytr)
    Xte = E([f"{r['title']}. {r['abstract']}" for r in test])
    q = ridge.predict(Xte)
    y = np.array([r["label"] for r in test])
    claims_mode = os.environ.get("E2_HUMAN_CLAIMS", "llm")   # "sentences" on CPU (deviation, NOTEBOOK.md)
    llm = extractor() if claims_mode == "llm" else None
    n_pool = int(os.environ.get("E2_HUMAN_POOL", "5000"))
    cand = load_arxiv(categories=("cs.LG", "cs.CL", "cs.CV", "cs.AI"), start="2024-01-01", end=T_NJB)
    pool = random.Random(0).sample(cand, min(n_pool, len(cand)))
    scores = {m: [] for m in MEASURES}
    for r in test:
        claims = (extract_claims(llm, r["title"], r["abstract"]) if llm
                  else novelty.sentences(r["abstract"])[:10])
        s = novelty.score_papers(f"{r['title']}. {r['abstract']}", claims, pool, r["date"])
        for m in MEASURES:
            scores[m].append(s[m])
    ids = [r["id"] for r in test]
    rng = np.random.default_rng(0)
    out = {"quality_only": auroc(q, y), "n_test": len(y), "n_pool": len(pool), "claims_mode": claims_mode,
           "n_train_quality": len(train)}
    for m in MEASURES:
        a = auroc(scores[m], y)
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            boots.append(auroc(np.array(scores[m])[i], y[i]) - auroc(q[i], y[i]))
        out[m] = {"auroc": a, "minus_quality": a - out["quality_only"],
                  "minus_quality_ci": [float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))]}
    write_json(HERE / "results" / "human_check.json", out)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"pools": cmd_pools, "claims": lambda: cmd_claims(lim), "calibrate": cmd_calibrate,
     "axioms": lambda: cmd_axioms(lim), "human": cmd_human}[cmd]()
