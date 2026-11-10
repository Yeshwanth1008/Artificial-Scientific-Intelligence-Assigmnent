"""Shared state for the battery: data splits, a cache of encoded Phi objects, baselines and controls."""
import json
import pickle
import random
from pathlib import Path

import numpy as np

from common.io import DATA, read_jsonl
from . import baselines as BL
from .phi import PhiModel

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "configs" / "default.json").read_text())


class Ctx:
    def __init__(self, load_state=True):
        from .data import LitIndex
        self.index = LitIndex.load()
        self.model = PhiModel(self.index)
        if load_state and (HERE / "results" / "phi_state.pkl").exists():
            self.model.load()
        self._items, self._phi, self._bq = {}, {}, {}
        self._cache_file = DATA / "phi_cache.pkl"
        if self._cache_file.exists():
            with open(self._cache_file, "rb") as f:
                self._phi = pickle.load(f)

    def flush(self):
        with open(self._cache_file, "wb") as f:
            pickle.dump(self._phi, f)

    def fresh_model(self):
        return PhiModel(self.index)

    # ---------------------------------------------------------------- data
    def split(self, kind, split):
        key = (kind, split)
        if key not in self._items:
            f = HERE / f"items_{kind}.jsonl"
            rows = [r for r in read_jsonl(f) if r["split"] == split] if f.exists() else []
            n = CONFIG[f"n_{kind}"][split]
            if n and len(rows) > n:
                rows = random.Random(0).sample(sorted(rows, key=lambda r: r["id"]), n)
            self._items[key] = rows
        return self._items[key]

    def e1_items(self):
        return [it for it in read_jsonl(ROOT / "E1" / "contrast.jsonl") if it["id"] != "h000"]

    # ---------------------------------------------------------------- Phi cache
    def phi_item(self, r, with_novelty=True, R=None, tag="true"):
        key = (r["id"], with_novelty, tag)
        if key not in self._phi:
            self._phi[key] = self.model.encode(r["q"], r["R"] if R is None else R, r["t"], facets=r.get("facets"),
                                               with_novelty=with_novelty)
            if len(self._phi) % 500 == 0:
                self.flush()
        return self._phi[key]

    def phi_text(self, text, t, with_novelty=True):
        key = ("text:" + text, t, with_novelty)
        if key not in self._phi:
            self._phi[key] = self.model.encode(text, None, t, with_novelty=with_novelty)
        return self._phi[key]

    def feat_matrix(self, rows):
        return np.stack([self.model.features(self.phi_item(r)) for r in rows])

    # ---------------------------------------------------------------- baselines (fit train, choose on select)
    def _target(self, kind, r, target):
        return r["scores"][target] if kind == "iclr" else r[target]

    def _xy(self, kind, split, target, field="q"):
        rows = [r for r in self.split(kind, split) if not np.isnan(self._target(kind, r, target))]
        return [r[field] if field != "abs" else f"{r['title']}. {r['abstract']}" for r in rows], \
               np.array([self._target(kind, r, target) for r in rows])

    def _test_rows(self, kind, target):
        return [r for r in self.split(kind, "test") if not np.isnan(self._target(kind, r, target))] \
            if kind == "iclr" else self.split(kind, "test")

    def _fit_baseline(self, kind, target, which, field="q"):
        key = (kind, target, which, field)
        if key not in self._bq:
            Xtr, ytr = self._xy(kind, "train", target, field)
            Xse, yse = self._xy(kind, "select", target, field)
            from sklearn.linear_model import Ridge
            if which == "emb":
                E = BL.Embedder("qwen3-0.6b", use_prompt=False)
                m = BL._ridge(E(Xtr), ytr, E(Xse), yse)
                self._bq[key] = lambda texts: m.predict(E(texts))
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tf = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(Xtr)
                m = BL._ridge(tf.transform(Xtr), ytr, tf.transform(Xse), yse)
                self._bq[key] = lambda texts: m.predict(tf.transform(texts))
        return self._bq[key]

    def bq_pred(self, kind, target):
        te = self._test_rows(kind, target)
        return self._fit_baseline(kind, target, "emb")([r["q"] for r in te])

    def bq_apply(self, kind, target, texts):
        return self._fit_baseline(kind, target, "emb")(texts)

    def babs_pred(self, target):
        te = self._test_rows("iclr", target)
        return self._fit_baseline("iclr", target, "emb", "abs")([f"{r['title']}. {r['abstract']}" for r in te])

    def btf_pred(self, kind, target):
        te = self._test_rows(kind, target)
        return self._fit_baseline(kind, target, "tfidf")([r["q"] for r in te])

    # ---------------------------------------------------------------- topics for arXiv uptake items
    def uptake_topics(self, rows, k=30):
        if "km" not in self._bq:
            from sklearn.cluster import KMeans
            tr = self.split("uptake", "train")
            U = np.stack([self.model.E([r["q"]])[0] for r in tr])
            self._bq["km"] = KMeans(k, n_init=10, random_state=0).fit(U)
        U = np.stack([self.model.E([r["q"]])[0] for r in rows])
        return self._bq["km"].predict(U).tolist()

    # ---------------------------------------------------------------- controls
    def controls_quality(self, te, y, topics, head="q_review"):
        from E4 import evalkit as ek
        rng = random.Random(0)
        out = {}
        empty = [self.model.quality(self.phi_item(r, R=[], tag="empty"))[head] for r in te]
        shuf_R = [rng.choice(te)["R"] for _ in te]
        shuf = [self.model.quality(self.phi_item(r, R=R, tag="shuffled"))[head] for r, R in zip(te, shuf_R)]
        by = {}
        for r in te:
            by.setdefault(r["topic"] if "topic" in r else 0, []).append(r)
        decoy = []
        for r in te:
            d = rng.choice(by.get(r.get("topic", 0), te))
            dr = dict(d, id=f"decoy:{r['id']}:{d['id']}", R=r["R"], t=r["t"])
            decoy.append(self.model.quality(self.phi_item(dr, tag="decoy"))[head])
        for name, p in (("empty_inputs", empty), ("shuffled_inputs", shuf), ("decoy_question", decoy)):
            out[name] = ek.within_topic_pair_accuracy(np.array(p), y, topics)
        self.flush()
        return out
