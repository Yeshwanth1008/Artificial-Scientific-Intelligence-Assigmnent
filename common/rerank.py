"""Instructed cross-encoder for "do these two texts ask the same research question?".

Uses Qwen3-Reranker-0.6B (a causal LM that scores P(yes) for a query/document pair under an
instruction). Score in [0, 1]; we report d = 1 - score as the cross-encoder distance.
"""
import numpy as np

from .prompts import SAME_QUESTION_INSTRUCTION

_m = {}
NAME = "Qwen/Qwen3-Reranker-0.6B"


def _load():
    if "m" not in _m:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NAME, padding_side="left")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = AutoModelForCausalLM.from_pretrained(
            NAME, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        _m.update(m=mdl, t=tok, yes=tok.convert_tokens_to_ids("yes"), no=tok.convert_tokens_to_ids("no"), dev=dev)
    return _m


PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def same_question_scores(pairs, instruction=SAME_QUESTION_INSTRUCTION, batch_size=16, symmetric=True):
    """pairs: list of (q1, q2). Returns P(same) per pair; symmetric=True averages both orders."""
    import torch
    m = _load()

    def run(ps):
        out = []
        for i in range(0, len(ps), batch_size):
            texts = [PREFIX + f"<Instruct>: {instruction}\n<Query>: {a}\n<Document>: {b}" + SUFFIX
                     for a, b in ps[i:i + batch_size]]
            enc = m["t"](texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(m["dev"])
            with torch.no_grad():
                logits = m["m"](**enc).logits[:, -1, :]
            two = torch.stack([logits[:, m["no"]], logits[:, m["yes"]]], 1).float()
            out += torch.softmax(two, 1)[:, 1].cpu().tolist()
        return np.array(out)

    s = run(list(pairs))
    if symmetric:
        s = 0.5 * (s + run([(b, a) for a, b in pairs]))
    return s
