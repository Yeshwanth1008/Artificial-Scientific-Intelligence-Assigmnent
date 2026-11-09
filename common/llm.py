"""One generation interface, three back-ends.

* local : a small open model through transformers (one prompt at a time; CPU or GPU).
* vllm  : the same model through vLLM, which batches thousands of prompts (used on Kaggle T4).
* api   : any OpenAI-compatible endpoint (LLM_API_BASE, LLM_API_KEY, LLM_MODEL). Disclosed if used.

Default model Qwen2.5-1.5B-Instruct (released Sept 2024; training cutoff assumed 2024-12-31).
Every call is cached on disk keyed by (model, prompt, temperature, seed, max tokens), so
  * reruns are free and exactly reproducible,
  * `LLM.batch(prompts)` fills the cache for many prompts at once, after which ordinary calls hit the cache,
  * collect mode (`collecting()`): cache misses are recorded instead of generated and return "" -- run a
    pipeline dry, `flush()` the recorded prompts in one batch, repeat until nothing is missing. This lets
    multi-step pipelines (E6) use batched generation without rewriting their logic.
"""
import contextlib
import json
import os
import re

from .io import DATA, text_key

DEFAULT_LOCAL = os.environ.get("LLM_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_local, _vllm = {}, {}
_PENDING = {}          # key -> (llm params, prompt, max_new_tokens)
_COLLECT = [False]


@contextlib.contextmanager
def collecting():
    _COLLECT[0] = True
    try:
        yield
    finally:
        _COLLECT[0] = False


def pending():
    return len(_PENDING)


def flush():
    """Generate every recorded prompt (grouped by model/temperature/seed/max tokens) in batches."""
    groups = {}
    for key, (params, prompt, mnt) in list(_PENDING.items()):
        groups.setdefault((params, mnt), []).append(prompt)
    for (params, mnt), prompts in groups.items():
        backend, model, temperature, seed = params
        LLM(backend=backend, model=model, temperature=temperature, seed=seed).batch(prompts, max_new_tokens=mnt)
    _PENDING.clear()


def run_with_prefill(fn, max_rounds=6):
    """Run fn() dry in collect mode until no prompt is missing (batching each round), then for real."""
    for _ in range(max_rounds):
        with collecting():
            fn(dry=True)
        if not pending():
            break
        print(f"[prefill] generating {pending()} prompts in batch", flush=True)
        flush()
    return fn(dry=False)


class LLM:
    def __init__(self, backend=None, model=None, temperature=0.7, seed=0, max_new_tokens=512):
        self.backend = backend or os.environ.get("LLM_BACKEND", "local")
        self.model = model or (DEFAULT_LOCAL if self.backend in ("local", "vllm") else os.environ["LLM_MODEL"])
        self.temperature, self.seed, self.max_new_tokens = temperature, seed, max_new_tokens
        # local and vllm share one cache (same model, same prompt): the backend is recorded in meta()
        self.cache = DATA / "llm_cache" / self.model.replace("/", "_")
        self.cache.mkdir(parents=True, exist_ok=True)

    def meta(self):
        return {"model": self.model, "backend": self.backend, "temperature": self.temperature, "seed": self.seed}

    def _key(self, prompt, mnt):
        return text_key(json.dumps([self.model, prompt, self.temperature, self.seed, mnt]))

    def __call__(self, prompt, max_new_tokens=None):
        mnt = max_new_tokens or self.max_new_tokens
        key = self._key(prompt, mnt)
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        if _COLLECT[0]:
            _PENDING[key] = ((self.backend, self.model, self.temperature, self.seed), prompt, mnt)
            return {"text": "", "tokens": 0}
        if self.backend == "vllm":
            return self.batch([prompt], mnt)[0]
        out = self._local(prompt, mnt) if self.backend == "local" else self._api(prompt, mnt)
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    def batch(self, prompts, max_new_tokens=None):
        """Outputs for many prompts; only cache misses are generated, in one batched call."""
        mnt = max_new_tokens or self.max_new_tokens
        keys = [self._key(p, mnt) for p in prompts]
        miss = [i for i, k in enumerate(keys) if not (self.cache / f"{k}.json").exists()]
        if miss:
            todo = [prompts[i] for i in miss]
            if self.backend == "vllm":
                outs = self._vllm(todo, mnt)
            elif self.backend == "local":
                outs = self._local_batch(todo, mnt)
            else:
                outs = [self._api(p, mnt) for p in todo]
            for i, o in zip(miss, outs):
                (self.cache / f"{keys[i]}.json").write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
        return [json.loads((self.cache / f"{k}.json").read_text(encoding="utf-8")) for k in keys]

    # ------------------------------------------------------------------ back-ends
    def _load_local(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model not in _local:
            tok = AutoTokenizer.from_pretrained(self.model, padding_side="left")
            if torch.cuda.is_available():
                mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.float16).to("cuda")
            else:  # CPU: float32 (float16 matmuls are slow or unsupported on CPU)
                mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.float32)
            _local[self.model] = (tok, mdl.eval())
        return _local[self.model]

    def _local(self, prompt, mnt):
        return self._local_batch([prompt], mnt)[0]

    def _local_batch(self, prompts, mnt, bs=None):
        """Batched generation. Prompts are sorted by length and packed into batches holding at most LLM_TOKENS
        tokens (prompt + new tokens, summed over the batch) and at most LLM_BS prompts, so long prompts (E6: 15
        abstracts) get small batches and short ones get large batches, without running out of GPU memory."""
        import torch
        tok, mdl = self._load_local()
        gpu = torch.cuda.is_available()
        bs = bs or int(os.environ.get("LLM_BS", "64" if gpu else "4"))
        budget = int(os.environ.get("LLM_TOKENS", "24000" if gpu else "4000"))
        lens = [len(tok(p, add_special_tokens=False)["input_ids"]) + 30 for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lens[i])
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) >= bs or (len(cur) + 1) * (lens[i] + mnt) > budget):
                batches.append(cur); cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        res = [None] * len(prompts)
        for b, idx in enumerate(batches):
            if len(batches) > 1 and b % 20 == 0:
                print(f"[llm] batch {b + 1}/{len(batches)} ({len(idx)} prompts)", flush=True)
            for j, o in zip(idx, self._gen_chunk(tok, mdl, [prompts[k] for k in idx], mnt)):
                res[j] = o
            if gpu:
                torch.cuda.empty_cache()
        return res

    def _gen_chunk(self, tok, mdl, prompts, mnt):
        import torch
        out = []
        if True:
            chunk = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                     for p in prompts]
            enc = tok(chunk, return_tensors="pt", padding=True).to(mdl.device)
            torch.manual_seed(self.seed)
            with torch.no_grad():
                gen = mdl.generate(**enc, max_new_tokens=mnt, do_sample=self.temperature > 0,
                                   temperature=max(self.temperature, 1e-5), top_p=0.95, pad_token_id=tok.eos_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                ids = row[row != tok.pad_token_id] if tok.pad_token_id is not None else row
                out.append({"text": tok.decode(ids, skip_special_tokens=True).strip(), "tokens": int(len(ids))})
        return out

    def _vllm(self, prompts, mnt):
        from vllm import LLM as VLLM, SamplingParams
        if self.model not in _vllm:
            _vllm[self.model] = VLLM(model=self.model, dtype="float16", gpu_memory_utilization=0.85,
                                     max_model_len=4096, seed=0,
                                     tensor_parallel_size=int(os.environ.get("VLLM_TP", "1")))
        eng = _vllm[self.model]
        sp = SamplingParams(temperature=self.temperature, top_p=0.95 if self.temperature > 0 else 1.0,
                            max_tokens=mnt, seed=self.seed)
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        res = eng.chat(msgs, sp, use_tqdm=len(prompts) > 50)
        return [{"text": r.outputs[0].text.strip(), "tokens": len(r.outputs[0].token_ids)} for r in res]

    def _api(self, prompt, mnt):
        import requests
        r = requests.post(os.environ["LLM_API_BASE"].rstrip("/") + "/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"},
                          json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                                "temperature": self.temperature, "seed": self.seed, "max_tokens": mnt},
                          timeout=120)
        r.raise_for_status()
        j = r.json()
        return {"text": j["choices"][0]["message"]["content"].strip(),
                "tokens": int(j.get("usage", {}).get("completion_tokens", 0))}


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_prefixed(text, prefix):
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line.upper().startswith(prefix.upper()):
            out.append(line[len(prefix):].strip())
    return out
