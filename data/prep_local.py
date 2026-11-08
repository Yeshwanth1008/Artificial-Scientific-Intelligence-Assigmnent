"""Preparation only -- no evaluation on any test set (that waits for the PREREG.md commit).
Caches embeddings in chunks so an interrupted run loses nothing.
    python data/prep_local.py e4      # embed the fixed E4 subsample (title + abstract)
    python data/prep_local.py e2      # embed E2 pools (papers and sentences)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.embed import Embedder
from common.io import read_jsonl

def chunks(xs, n=64):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

E = Embedder("qwen3-0.6b", use_prompt=False)
if sys.argv[1] == "e4":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E4"))
    from run_e4 import papers
    T = [f"{p['title']}. {p['abstract']}" for p in papers()]
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E2"))
    import novelty
    T = []
    for f in sorted((Path(__file__).resolve().parents[1] / "E2" / "pools").glob("*.jsonl")):
        for p in read_jsonl(f):
            t = novelty._paper_text(p)
            T.append(t)
            T += novelty.sentences(t)
for i, c in enumerate(chunks(T)):
    E(c)
    if i % 20 == 0:
        print(f"{min((i + 1) * 64, len(T))}/{len(T)}", flush=True)
print("done", len(T))
