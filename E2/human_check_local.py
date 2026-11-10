"""E2 Task 3 (reduced, local CPU, declared deviation): novelty-judge-bench human-only labels (299 ICLR 2026 ideas).

* quality-only baseline: ridge on frozen Qwen3 abstract embeddings, trained on the fixed E4 ICLR 2024 subsample,
  alpha chosen on the ICLR 2025 subsample (targets: mean reviewer rating);
* Y (1 - max cosine) and relative neighbourhood density, with the pool = the E4 ICLR 2024 + 2025 subsample
  (3000 submissions, all dated before the ICLR 2026 deadline), instead of the pre-registered 5000 arXiv papers;
* N and the T2 B.4 measure need claim extraction with the LM and are not computed here.
AUROC with a 2000-sample bootstrap over ideas; each novelty measure minus the quality baseline.
    python E2/human_check_local.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "E4"))
from common.embed import Embedder  # noqa: E402
from common.io import read_jsonl, write_json  # noqa: E402
from common.stats import auroc  # noqa: E402
from run_e4 import papers  # noqa: E402

test = read_jsonl(ROOT / "data" / "novelty_judge_bench.jsonl")
titles = {r["title"].lower().strip() for r in test}
P = [p for p in papers() if p["year"] in (2024, 2025) and p["title"].lower().strip() not in titles]
E = Embedder("qwen3-0.6b", use_prompt=False)
X = E([f"{p['title']}. {p['abstract']}" for p in P])            # cached from the E4 run
y = np.array([np.mean([r["rating"] for r in p["reviews"]]) for p in P])
yr = np.array([p["year"] for p in P])
from sklearn.linear_model import Ridge
best = max(np.logspace(-2, 4, 13), key=lambda a: np.corrcoef(Ridge(alpha=a).fit(X[yr == 2024], y[yr == 2024])
                                                             .predict(X[yr == 2025]), y[yr == 2025])[0, 1])
ridge = Ridge(alpha=best).fit(X[yr == 2024], y[yr == 2024])
Xt = E([f"{r['title']}. {r['abstract']}" for r in test])
lab = np.array([r["label"] for r in test])
S = Xt @ X.T                                                    # pool = all 3000 earlier submissions
Y = 1 - S.max(1)
k = 10
dx = 1 - np.sort(S, 1)[:, -k:].mean(1)
PP = X @ X.T
np.fill_diagonal(PP, -np.inf)
dpool = 1 - np.sort(PP, 1)[:, -k:].mean(1)
nn = np.argsort(-S, 1)[:, :k]
density = np.array([(dpool[nn[i]] < dx[i]).mean() for i in range(len(test))])
scores = {"quality_only": ridge.predict(Xt), "Y": Y, "density": density}
rng = np.random.default_rng(0)
out = {"n_test": int(len(lab)), "n_pos": int(lab.sum()), "pool": "E4 ICLR 2024+2025 subsample", "n_pool": len(P),
       "quality_alpha": float(best)}
for m, s in scores.items():
    a = auroc(s, lab)
    boots = [auroc(s[i], lab[i]) for i in (rng.integers(0, len(lab), len(lab)) for _ in range(2000))]
    out[m] = {"auroc": a, "ci": [float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))]}
q = scores["quality_only"]
for m in ("Y", "density"):
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(lab), len(lab))
        d.append(auroc(scores[m][i], lab[i]) - auroc(q[i], lab[i]))
    out[m]["minus_quality"] = out[m]["auroc"] - out["quality_only"]["auroc"]
    out[m]["minus_quality_ci"] = [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]
(ROOT / "E2" / "results").mkdir(exist_ok=True)
write_json(ROOT / "E2" / "results" / "human_check_local.json", out)
for m in scores:
    print(m, round(out[m]["auroc"], 3), [round(x, 3) for x in out[m]["ci"]],
          "" if m == "quality_only" else f"minus quality {out[m]['minus_quality']:+.3f} {[round(x,3) for x in out[m]['minus_quality_ci']]}")
