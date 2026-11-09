"""Sentence-embedding wrapper with an on-disk cache.

Small tier: Qwen3-Embedding-0.6B is the default encoder for the whole project.
Other encoders (used in E1) are selected by name.
"""
import os

import numpy as np

from .io import DATA, text_key

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",          # general-purpose, instruction-aware
    "specter2": "allenai/specter2_base",                  # trained on scientific documents
    "mpnet": "sentence-transformers/all-mpnet-base-v2",   # general-purpose
    "bge-small": "BAAI/bge-small-en-v1.5",                # general-purpose, small
}

QUERY_PROMPT = "Instruct: Represent this research question so that the same question asked in other words is close\nQuery: "

_cache_models = {}


def _load(name):
    if name not in _cache_models:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if dev == "cuda" else {}
        _cache_models[name] = SentenceTransformer(MODELS.get(name, name), device=dev, **kw)
    return _cache_models[name]


class Embedder:
    def __init__(self, name="qwen3-0.6b", use_prompt=True, batch_size=32):
        if name == "qwen3-0.6b" and os.environ.get("ASI_EMBEDDER"):  # smoke tests only (e.g. bge-small)
            name = os.environ["ASI_EMBEDDER"]
        self.name = name
        self.use_prompt = use_prompt and name.startswith("qwen3")
        self.batch_size = batch_size
        self.cache_dir = DATA / "emb" / name.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem = {}

    def __call__(self, texts):
        texts = list(texts)
        keys = [text_key(("P|" if self.use_prompt else "") + t) for t in texts]
        disk = os.environ.get("EMB_DISK_CACHE", "1") == "1"   # 0 on Kaggle: keep 10^5+ sentence vectors in memory
        missing = [i for i, k in enumerate(keys) if k not in self._mem and not (disk and (self.cache_dir / f"{k}.npy").exists())]
        if missing:
            model = _load(self.name)
            todo = [texts[i] for i in missing]
            kw = {"prompt": QUERY_PROMPT} if self.use_prompt else {}
            vecs = model.encode(todo, batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=len(todo) > 500, **kw)
            for i, v in zip(missing, vecs):
                v = v.astype(np.float32)
                self._mem[keys[i]] = v
                if disk:
                    np.save(self.cache_dir / f"{keys[i]}.npy", v)
        out = []
        for k in keys:
            if k not in self._mem:
                self._mem[k] = np.load(self.cache_dir / f"{k}.npy")
            out.append(self._mem[k])
        if out:
            return np.stack(out)
        dim = _load(self.name).get_sentence_embedding_dimension()  # empty input: (0, d), not (0, 1)
        return np.zeros((0, dim), np.float32)


def cos_matrix(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T
