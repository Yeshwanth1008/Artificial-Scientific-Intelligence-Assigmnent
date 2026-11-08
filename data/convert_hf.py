"""Convert public Hugging Face mirrors to the formats the pipeline expects (outputs go to data/cache/).

    python data/convert_hf.py iclr        # smallari/openreview-iclr-peer-reviews (2024, 2025) + davidheineman/iclr-2026
    python data/convert_hf.py njb         # allenai/novelty-judge-bench human-only pointwise -> data/novelty_judge_bench.jsonl

Sources (public, downloaded with curl into data/cache/hf/):
  https://huggingface.co/datasets/smallari/openreview-iclr-peer-reviews      (ICLR 2024-2025, reviews with ratings)
  https://huggingface.co/datasets/davidheineman/iclr-2026                    (ICLR 2026 submissions + reviews)
  https://huggingface.co/datasets/allenai/novelty-judge-bench                (ODC-BY)
Neither ICLR mirror carries the contribution / soundness sub-scores, so only the overall rating is available
(deviation recorded in NOTEBOOK.md). No copy of these data is committed (rule R7).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import DATA, write_jsonl  # noqa: E402

HF = DATA / "hf"
_num = re.compile(r"^\s*(-?\d+(\.\d+)?)")


def num(v):
    if v is None:
        return None
    m = _num.match(str(v))
    return float(m.group(1)) if m else None


def iclr():
    a = pd.read_parquet(HF / "iclr_reviews.parquet")
    for y in ("2024", "2025"):
        rows = []
        for _, r in a[a.year == y].iterrows():
            revs = [{"rating": num(v.get("rating")), "confidence": num(v.get("confidence")),
                     "contribution": num(v.get("contribution")), "soundness": num(v.get("soundness"))}
                    for v in r["reviews"]]
            revs = [v for v in revs if v["rating"] is not None]
            if not revs:
                continue
            rows.append({"paper_id": r["paper_id"], "year": int(y), "title": r["title"], "abstract": r["abstract"],
                         "decision": r["decision"], "reviews": revs, "primary_area": None,
                         "cdate": None})
        write_jsonl(DATA / f"iclr{y}.jsonl", rows)
        print(y, len(rows))
    b = pd.read_parquet(HF / "iclr2026.parquet")
    rows = []
    for _, r in b.iterrows():
        revs = [{"rating": num(v.get("rating")), "confidence": num(v.get("confidence")),
                 "contribution": num(v.get("contribution")), "soundness": num(v.get("soundness"))}
                for v in r["reviews"]]
        revs = [v for v in revs if v["rating"] is not None]
        if not revs:
            continue
        pid = str(r["url"]).split("id=")[-1]
        rows.append({"paper_id": pid, "year": 2026, "title": r["title"], "abstract": r["abstract"],
                     "decision": "Withdrawn" if "Withdrawn" in str(r["venue"]) else None,
                     "reviews": revs, "primary_area": None, "cdate": None})
    write_jsonl(DATA / "iclr2026.jsonl", rows)
    print(2026, len(rows))


def njb():
    df = pd.read_parquet(HF / "njb_pointwise.parquet")
    rows = [{"id": f"njb{int(r['id'])}", "title": r["title"], "abstract": r["idea"], "label": int(r["label"] == "POSITIVE"),
             "venue": "ICLR", "iclr_area": r["iclr_area"], "forum_id": r["forum_id"],
             "rating": float(r["rating"]), "contribution": float(r["contribution"]), "idea_source": r["idea_source"]}
            for _, r in df.iterrows()]
    write_jsonl(Path(__file__).parent / "novelty_judge_bench.jsonl", rows)
    print(len(rows), {k: sum(r["label"] == k for r in rows) for k in (0, 1)})


def arxiv(cats=("cs.LG", "cs.CL", "cs.CV", "cs.AI", "cs.RO", "stat.ML"), since=2018):
    """librarian-bots/arxiv-metadata-snapshot (same schema as Kaggle's) -> filtered JSONL snapshot.
    https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot (snapshot of 2026-10-27)"""
    import json
    import pyarrow.parquet as pq
    from datetime import datetime
    out = DATA / "arxiv-metadata-oai-snapshot.json"
    keep = set(cats)
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for f in sorted((HF / "arxiv").glob("*.parquet")):
            t = pq.read_table(f, columns=["id", "title", "abstract", "categories", "versions", "authors_parsed"])
            for r in t.to_pylist():
                if not keep & set(r["categories"].split()):
                    continue
                try:
                    y = datetime.strptime(r["versions"][0]["created"], "%a, %d %b %Y %H:%M:%S %Z").year
                except Exception:
                    continue
                if y < since:
                    continue
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(n, "papers ->", out)


if __name__ == "__main__":
    {"iclr": iclr, "njb": njb, "arxiv": arxiv}[sys.argv[1]]()
