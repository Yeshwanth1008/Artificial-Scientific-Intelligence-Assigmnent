"""E2 Tasks 1-2 results (reduced, local CPU; declared deviations).

Real arXiv pools from E2/pools (cs.CL, cs.CV, cs.RO). Deviations from PREREG, all forced by the CPU time budget:
  * encoder bge-small-en-v1.5 instead of Qwen3-Embedding-0.6B;
  * 300 papers per pool instead of 1000; 8 focal papers per field instead of 100;
  * claims = the first 4-8 sentences of the focal abstract (no LM extraction);
  * the paraphrased-copy manipulation is skipped (it needs LM paraphrases);
  * calibration b for `yours` estimated once on the older pools before any manipulation, then frozen.
Manipulations and expected directions are those of E2/run_e2.py.
    python E2/results_axioms.py
"""
import os
import sys
from pathlib import Path

os.environ["ASI_EMBEDDER"] = "bge-small"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "E2"))
import numpy as np  # noqa: E402

import novelty  # noqa: E402
from common.io import read_jsonl, write_json  # noqa: E402
from common.stats import cluster_bootstrap  # noqa: E402

FIELDS = ["cs.CL", "cs.CV", "cs.RO"]
FAR = {"cs.CL": "cs.CV", "cs.CV": "cs.RO", "cs.RO": "cs.CL"}
N_POOL, N_FOCAL = 300, 8
MEASURES = ["Y", "density", "N", "yours"]
EXPECT = {"exact_copy": ("original", -1), "partial_25": ("original", -1), "partial_50": ("partial_25", -1),
          "partial_75": ("partial_50", -1), "partial_100": ("partial_75", -1),
          "concentrated_3": ("distributed_3", -1), "far_pool": ("original", +1), "older_pool": ("original", +1)}

pools = {F: read_jsonl(ROOT / "E2" / "pools" / f"{F}_pool.jsonl")[:N_POOL] for F in FIELDS}
older = {F: read_jsonl(ROOT / "E2" / "pools" / f"{F}_older.jsonl")[:N_POOL] for F in FIELDS}
focal = {F: [f for f in read_jsonl(ROOT / "E2" / "pools" / f"{F}_focal.jsonl")
             if len(novelty.sentences(f["abstract"])) >= 4][:N_FOCAL] for F in FIELDS}
for F in FIELDS:
    for f in focal[F]:
        f["claims"] = novelty.sentences(f["abstract"])[:8]

# frozen calibration (older pools only)
E = novelty.emb()
cl = [c for F in FIELDS for f in focal[F] for c in f["claims"]]
sent = [s for F in FIELDS for p in older[F][:100] for s in novelty.sentences(novelty._paper_text(p))]
B = float(np.quantile(E(cl) @ E(sent).T, 0.9))
novelty._calib_b = lambda: B
print("calibration b =", round(B, 3), flush=True)


def plant(p, claims, tag):
    q = dict(p); q["id"] = f"planted:{p['id']}:{tag}"; q["abstract"] = p["abstract"] + " " + " ".join(claims)
    return q


raw = []
for F in FIELDS:
    pool = pools[F]
    P = E([novelty._paper_text(p) for p in pool])
    for f in focal[F]:
        idea, claims, t = f"{f['title']}. {f['abstract']}", f["claims"], f["date"]
        base = [p for p in pool if p["id"] != f["id"]]
        near = [int(np.argmax(P @ v)) for v in E(claims)]
        man = {"original": base,
               "exact_copy": base + [{"id": "copy", "date": "2023-12-31", "title": f["title"], "abstract": f["abstract"]}]}
        m = len(claims)
        for frac in (25, 50, 75, 100):
            Q = list(pool)
            for i in range(max(1, round(m * frac / 100))):
                Q[near[i]] = plant(Q[near[i]], [claims[i]], f"{frac}_{i}")
            man[f"partial_{frac}"] = [p for p in Q if p["id"] != f["id"]]
        j = min(3, m)
        D, used = list(pool), set()
        for i in range(j):
            k = near[i]
            while k in used:
                k = (k + 1) % len(D)
            used.add(k); D[k] = plant(D[k], [claims[i]], f"d{i}")
        man["distributed_3"] = D
        C = list(pool); C[near[0]] = plant(C[near[0]], claims[:j], "c")
        man["concentrated_3"] = C
        man["far_pool"] = pools[FAR[F]]
        man["older_pool"] = older[F]
        sc = {n: novelty.score_papers(idea, claims, Pl, t) for n, Pl in man.items()}
        raw.append({"field": F, "id": f["id"], "scores": {n: {k: s[k] for k in MEASURES} for n, s in sc.items()}})
        print(F, f["id"], "done", flush=True)

res = {"deviations": __doc__.strip().splitlines()[2:9], "n_focal": len(raw), "calibration_b": B}
for name, (ref, sign) in EXPECT.items():
    res[name] = {}
    for ms in MEASURES:
        ok = [float(sign * (r["scores"][name][ms] - r["scores"][ref][ms]) > 0) for r in raw]
        res[name][ms] = cluster_bootstrap(ok, [r["id"] for r in raw])
write_json(ROOT / "E2" / "results" / "axioms_pass_rates.json", res)
write_json(ROOT / "E2" / "results" / "axioms_raw.json", raw)
print("\npass rates (share of focal papers moving in the correct direction)")
print(f"{'manipulation':16s}" + "".join(f"{m:>9s}" for m in MEASURES))
for name in EXPECT:
    print(f"{name:16s}" + "".join(f"{res[name][m]['est']:9.2f}" for m in MEASURES))
