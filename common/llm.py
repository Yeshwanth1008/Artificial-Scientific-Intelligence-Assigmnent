"""One generation interface, two back-ends.

* local  : a small open model through transformers (small tier, default Qwen2.5-1.5B-Instruct,
           training cutoff assumed end of 2024 -- released Sept 2024, so earlier than every
           test window used here).
* api    : any OpenAI-compatible endpoint (set LLM_API_BASE, LLM_API_KEY, LLM_MODEL). Disclosed if used.

Every call returns text and the number of generated tokens, and is cached on disk keyed by
(model, prompt, temperature, seed), so reruns are free and exactly reproducible.
"""
import json
import os
import re

from .io import DATA, text_key

DEFAULT_LOCAL = os.environ.get("LLM_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_local = {}


class LLM:
    def __init__(self, backend=None, model=None, temperature=0.7, seed=0, max_new_tokens=512):
        self.backend = backend or os.environ.get("LLM_BACKEND", "local")
        self.model = model or (DEFAULT_LOCAL if self.backend == "local" else os.environ["LLM_MODEL"])
        self.temperature, self.seed, self.max_new_tokens = temperature, seed, max_new_tokens
        self.cache = DATA / "llm_cache" / self.model.replace("/", "_")
        self.cache.mkdir(parents=True, exist_ok=True)

    def meta(self):
        return {"model": self.model, "backend": self.backend, "temperature": self.temperature, "seed": self.seed}

    def __call__(self, prompt, max_new_tokens=None):
        mnt = max_new_tokens or self.max_new_tokens
        key = text_key(json.dumps([self.model, prompt, self.temperature, self.seed, mnt]))
        f = self.cache / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        out = self._local(prompt, mnt) if self.backend == "local" else self._api(prompt, mnt)
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    def _local(self, prompt, mnt):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.model not in _local:
            tok = AutoTokenizer.from_pretrained(self.model)
            if torch.cuda.is_available():
                mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.float16).to("cuda")
            else:  # CPU: float32 (float16 matmuls are slow or unsupported on CPU)
                mdl = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch.float32)
            mdl.eval()
            _local[self.model] = (tok, mdl)
        tok, mdl = _local[self.model]
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        torch.manual_seed(self.seed)
        gen = mdl.generate(ids, max_new_tokens=mnt, do_sample=self.temperature > 0,
                           temperature=max(self.temperature, 1e-5), top_p=0.95, pad_token_id=tok.eos_token_id)
        new = gen[0, ids.shape[1]:]
        return {"text": tok.decode(new, skip_special_tokens=True).strip(), "tokens": int(new.shape[0])}

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
