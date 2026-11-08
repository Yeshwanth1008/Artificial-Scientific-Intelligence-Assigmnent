"""E6 -- should an AI scientist ask its questions first?

    python E6/run_e6.py build_lit      # literature testbed: >=100 input sets (refs of 2025 papers) + targets
    python E6/run_e6.py build_data     # data testbed from HypoBench (data/hypobench/<task>/...)
    python E6/run_e6.py generate       # all arms on both testbeds -> E6/generations.jsonl
    python E6/run_e6.py evaluate       # metrics (a)-(f), paired bootstrap over input sets -> E6/results/
    python E6/run_e6.py sample20       # 20 input sets end to end, for the hand-written error analysis

One model and one decoding setting throughout: Qwen2.5-1.5B-Instruct, temperature 0.7, top_p 0.95.
Generator training cutoff assumed: end of 2024 (model released Sept 2024; we take the later date to be
safe). Literature targets are papers posted from 2025-03-01 onward.
Budget: K = 6 hypotheses, M = 3 questions, cap of 1200 generated tokens per input set and arm.
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.embed import Embedder  # noqa: E402
from common.extract import extract_rq, extractor  # noqa: E402
from common.io import append_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402
from common.llm import LLM, parse_prefixed  # noqa: E402
from common import prompts as P  # noqa: E402
from common.stats import cluster_bootstrap  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
K, M, CAP = 6, 3, 1200
ARMS = ["D", "QH", "QH_sel", "QH_rand", "HQ", "D_empty", "D_shuffled"]
SEED = 0


def gen_llm():
    return LLM(temperature=0.7, seed=SEED, max_new_tokens=400)


# ------------------------------------------------------------------ testbeds

def cmd_build_lit(n_sets=120, min_refs=10, max_refs=15):
    """Targets: arXiv cs.LG/cs.CL papers posted 2025-03..2025-12 with >= 10 references that have abstracts."""
    from common.data_sources import load_arxiv, s2_references
    from common.prompts import EXTRACT_HYP
    rng = random.Random(0)
    cands = [p for p in load_arxiv(("cs.LG", "cs.CL"), "2025-03-01", "2025-12-31")]
    rng.shuffle(cands)
    llm = extractor()
    sets = []
    for p in cands:
        try:
            refs = s2_references(p["id"])
        except Exception:
            continue
        refs = [r for r in refs if r.get("abstract") and r.get("publicationDate") and r["publicationDate"] < p["date"]]
        if len(refs) < min_refs:
            continue
        refs = sorted(refs, key=lambda r: r["publicationDate"], reverse=True)[:max_refs]  # newest refs
        hyp = llm(EXTRACT_HYP.format(title=p["title"], abstract=p["abstract"]), max_new_tokens=80)["text"].split("\n")[0]
        sets.append({"input_set_id": f"lit:{p['id']}", "testbed": "literature", "t": p["date"],
                     "target_rq": extract_rq(llm, p["title"], p["abstract"]), "target_hyp": hyp,
                     "inputs": [{"title": r["title"], "abstract": r["abstract"], "date": r["publicationDate"]}
                                for r in refs], "hand_checked": False})
        if len(sets) >= n_sets:
            break
    write_jsonl(HERE / "testbed_literature.jsonl", sets)
    print(len(sets), "literature input sets; hand-check 20 targets and set hand_checked=true")


def cmd_build_data(n_examples=20):
    """HypoBench layout expected (download from the HypoBench release, Appendix B):
    data/hypobench/<task>/train.jsonl, test.jsonl with {"text", "label"}; optional planted.txt for
    synthetic tasks. The 'input papers' of a data input set are n_examples labelled training examples."""
    sets = []
    for task in sorted((ROOT / "data" / "hypobench").glob("*")):
        tr = read_jsonl(task / "train.jsonl")
        rng = random.Random(0)
        ex = rng.sample(tr, min(n_examples, len(tr)))
        planted = (task / "planted.txt").read_text().strip() if (task / "planted.txt").exists() else None
        sets.append({"input_set_id": f"data:{task.name}", "testbed": "data", "t": None, "task": task.name,
                     "planted": planted, "labels": sorted({str(r["label"]) for r in tr}),
                     "inputs": [{"title": f"Example {i + 1} (label: {r['label']})", "abstract": r["text"]}
                                for i, r in enumerate(ex)]})
    write_jsonl(HERE / "testbed_data.jsonl", sets)
    print(len(sets), "data input sets")


# ------------------------------------------------------------------ arms

def papers_block(inputs):
    return "\n".join(P.PAPERS_BLOCK.format(i=i + 1, title=r["title"], abstract=r["abstract"][:700])
                     for i, r in enumerate(inputs))


class Budget:
    def __init__(self, cap=CAP):
        self.cap, self.used = cap, 0

    def call(self, llm, prompt, want):
        left = self.cap - self.used
        if left <= 20:
            return ""
        out = llm(prompt, max_new_tokens=min(want, left))
        self.used += out["tokens"]
        return out["text"]


def _fill(items, k):
    items = [x for x in items if x][:k]
    return items + [""] * (k - len(items))  # empty string = arm failed to produce (counted in eval)


def arm_D(llm, inputs, b):
    hs = parse_prefixed(b.call(llm, P.E6_DIRECT.format(papers=papers_block(inputs), k=K), 600), "H:")
    return [(None, h) for h in _fill(hs, K)]


def _hyps_for_questions(llm, inputs, qs, b):
    out = []
    per = K // len(qs)
    for q in qs:
        hs = parse_prefixed(b.call(llm, P.E6_HYP_FOR_Q.format(papers=papers_block(inputs), q=q, n=per), 250), "H:")
        out += [(q, h) for h in _fill(hs, per)]
    return out


def arm_QH(llm, inputs, b, m=M):
    qs = parse_prefixed(b.call(llm, P.E6_QUESTIONS.format(papers=papers_block(inputs), m=m), 300), "Q:")
    qs = _fill(qs, m)
    return qs


def select_questions(qs, inputs, rule, rng):
    """QH_sel rule (pre-registered): (1) drop empties; (2) drop duplicates (same-question cross-encoder >= .5,
    keep the first); (3) drop questions already asked by an input (cross-encoder >= .5 against each input's
    title); (4) rank the rest by the mean of two ranks -- groundedness (cosine to the centroid of R, high is
    good) and novelty w.r.t. R (1 - max cosine to any input, high is good); keep the top M. If fewer than M
    survive, refill from the dropped ones in original order. QH_rand keeps M at random."""
    qs = [q for q in qs if q]
    if rule == "rand":
        return rng.sample(qs, min(M, len(qs)))
    from common.rerank import same_question_scores
    keep = []
    for q in qs:
        if keep and max(same_question_scores([(q, k) for k in keep])) >= 0.5:
            continue
        keep.append(q)
    titles = [r["title"] for r in inputs]
    keep2 = [q for q in keep if max(same_question_scores([(q, t) for t in titles])) < 0.5] or keep
    E = Embedder("qwen3-0.6b", use_prompt=False)
    Rv = E([f"{r['title']}. {r['abstract']}" for r in inputs]); c = Rv.mean(0); c /= np.linalg.norm(c)
    Qv = E(keep2)
    g = Qv @ c
    nov = 1 - (Qv @ Rv.T).max(1)
    rank = np.argsort(np.argsort(-g)) + np.argsort(np.argsort(-nov))
    chosen = [keep2[i] for i in np.argsort(rank)[:M]]
    for q in qs:
        if len(chosen) >= M:
            break
        if q not in chosen:
            chosen.append(q)
    return chosen


def run_arm(arm, s, all_sets, rng):
    llm = gen_llm()
    b = Budget()
    inputs = s["inputs"]
    if arm == "D_empty":
        inputs = [{"title": "(no papers given)", "abstract": ""}]
    if arm == "D_shuffled":
        other = rng.choice([x for x in all_sets if x["input_set_id"] != s["input_set_id"]])
        inputs = other["inputs"]
    if arm in ("D", "D_empty", "D_shuffled"):
        pairs = arm_D(llm, inputs, b)
    elif arm == "QH":
        pairs = _hyps_for_questions(llm, inputs, arm_QH(llm, inputs, b), b)
    elif arm in ("QH_sel", "QH_rand"):
        qs = arm_QH(llm, inputs, b, m=2 * M)
        qs = select_questions(qs, inputs, "sel" if arm == "QH_sel" else "rand", rng)
        qs = _fill(qs, M)
        pairs = _hyps_for_questions(llm, inputs, qs, b)
    elif arm == "HQ":
        hs = [h for _, h in arm_D(llm, inputs, b)]
        pairs = []
        for h in hs:
            q = b.call(llm, P.E6_Q_FOR_H.format(h=h), 60).split("\n")[0].strip() if h else ""
            pairs.append((q or None, h))
    pairs = (pairs + [(None, "")] * K)[:K]
    return pairs, b.used


def cmd_generate():
    out = HERE / "generations.jsonl"
    done = {(r["input_set_id"], r["arm"]) for r in read_jsonl(out)} if out.exists() else set()
    for tb in ("literature", "data"):
        f = HERE / f"testbed_{tb}.jsonl"
        if not f.exists():
            print("missing", f); continue
        sets = read_jsonl(f)
        rng = random.Random(SEED)
        for s in sets:
            for arm in ARMS:
                if (s["input_set_id"], arm) in done:
                    continue
                pairs, used = run_arm(arm, s, sets, rng)
                for q, h in pairs:
                    append_jsonl(out, {"input_set_id": s["input_set_id"], "testbed": tb, "arm": arm, "rq": q,
                                       "hypothesis": h, "tokens_generated": used,
                                       "model": gen_llm().model, "seed": SEED})


# ------------------------------------------------------------------ evaluation

CONSTRUCTIVE = re.compile(r"^(how (can|could|should|might) (we|one)|can (we|one)|what .* (design|method|approach)|"
                          r"is it possible to)", re.I)
ANALYTIC = re.compile(r"^(to what extent|how (does|do|is|are)|does|do|is|are|why|what (is|are) the (effect|role|"
                      r"relationship|impact)|which)", re.I)


def q_type(q):
    if not q:
        return "none"
    if CONSTRUCTIVE.search(q.strip()):
        return "constructive"
    if re.search(r"compar|versus|vs\.?|than", q, re.I):
        return "comparative"
    if ANALYTIC.search(q.strip()):
        return "analytic"
    return "other"


def data_score(h, s, n_test=60):
    """HypoBench-style: accuracy of hypothesis-guided label prediction on held-out examples."""
    if not h:
        return 0.0
    te = read_jsonl(ROOT / "data" / "hypobench" / s["task"] / "test.jsonl")[:n_test]
    llm = LLM(temperature=0.0, max_new_tokens=10)
    ok = 0
    for r in te:
        a = llm(P.HYPO_PREDICT.format(h=h, x=r["text"][:1500], labels=", ".join(s["labels"])))["text"]
        ok += a.strip().lower().startswith(str(r["label"]).lower())
    return ok / len(te)


def already_asked(qs, t, index, k=5, cos_thr=0.75):
    """Share of questions with an earlier (date < t) paper asking the same question: retrieve up to k
    candidates above a fixed cosine threshold, extract their RQ with the fixed prompt, judge with the
    same-question cross-encoder (>= .5)."""
    from common.rerank import same_question_scores
    llm = extractor()
    res = []
    for q in qs:
        if not q:
            continue
        cands = index.search(q, t, k=k, min_cos=cos_thr)
        if not cands:
            res.append(0.0); continue
        rqs = [extract_rq(llm, c["title"], c["abstract"]) for c in cands]
        res.append(float(max(same_question_scores([(q, r) for r in rqs])) >= 0.5))
    return float(np.mean(res)) if res else float("nan")


def cmd_evaluate():
    from common.diversity import diversity
    from common.rerank import same_question_scores
    gens = read_jsonl(HERE / "generations.jsonl")
    sets = {s["input_set_id"]: s for tb in ("literature", "data") if (HERE / f"testbed_{tb}.jsonl").exists()
            for s in read_jsonl(HERE / f"testbed_{tb}.jsonl")}
    E = Embedder("qwen3-0.6b", use_prompt=False)
    index = None
    try:
        sys.path.insert(0, str(ROOT))
        from main.data import LitIndex
        index = LitIndex.load()
    except Exception as e:
        print("no literature index; already-asked rate skipped:", e)
    by = {}
    for g in gens:
        by.setdefault((g["input_set_id"], g["arm"]), []).append(g)
    rows = []
    for (sid, arm), G in by.items():
        s = sets[sid]
        hs = [g["hypothesis"] for g in G]
        qs = list(dict.fromkeys(g["rq"] for g in G if g["rq"]))
        r = {"input_set_id": sid, "testbed": s["testbed"], "arm": arm, "tokens": G[0]["tokens_generated"],
             "n_empty": sum(not h for h in hs)}
        good = [h for h in hs if h]
        if s["testbed"] == "literature" and good:
            Hv = E(good)
            r["recon_hyp_cos"] = float((Hv @ E([s["target_hyp"]])[0]).max())
            r["recon_rq_cos"] = float((E(qs) @ E([s["target_rq"]])[0]).max()) if qs else None
            r["recon_rq_same"] = float(max(same_question_scores([(q, s["target_rq"]) for q in qs]))) if qs else None
            if index is not None and qs:
                r["already_asked"] = already_asked(qs, s["t"], index)
        if s["testbed"] == "data" and good:
            sc = [data_score(h, s) for h in good]
            r["data_best"], r["data_mean"] = max(sc), float(np.mean(sc))
            if s.get("planted"):
                r["planted_recovery_cos"] = float((E(good) @ E([s["planted"]])[0]).max())
        if len(good) >= 2:
            d = diversity(good, facets=False)
            r.update({"div_mean_cos_dist": d["mean_cos_dist"], "div_vendi": d["vendi"], "div_distinct": d["n_distinct"]})
        types = [q_type(q) for q in qs]
        r["qtype"] = {t: types.count(t) / len(types) for t in set(types)} if types else {}
        rows.append(r)
    write_jsonl(HERE / "results" / "per_set.jsonl", rows)
    rows += judge_rows(by, sets)
    summarise(rows)


def judge_rows(by, sets, max_ratio=1.2):
    """(f) one LM judge, both orders, on length-matched D vs QH / QH_sel hypothesis pairs. A win counts only if
    the judge prefers the same hypothesis in both orders; otherwise a tie (0.5)."""
    llm = LLM(temperature=0.0, max_new_tokens=3)
    out = []
    for (sid, arm), G in by.items():
        if arm not in ("QH", "QH_sel", "HQ"):
            continue
        D = [g["hypothesis"] for g in by.get((sid, "D"), []) if g["hypothesis"]]
        X = [g["hypothesis"] for g in G if g["hypothesis"]]
        res = []
        for x in X:
            match = [d for d in D if max(len(x), len(d)) / max(1, min(len(x), len(d))) <= max_ratio]
            if not match:
                continue
            d = min(match, key=lambda d: abs(len(d) - len(x)))
            a1 = llm(P.JUDGE_PAIR.format(a=x, b=d))["text"].strip()[:1].upper()
            a2 = llm(P.JUDGE_PAIR.format(a=d, b=x))["text"].strip()[:1].upper()
            res.append(1.0 if (a1, a2) == ("A", "B") else 0.0 if (a1, a2) == ("B", "A") else 0.5)
        if res:
            out.append({"input_set_id": sid, "testbed": sets[sid]["testbed"], "arm": arm,
                        "judge_win_vs_D": float(np.mean(res)), "judge_n": len(res)})
    return out


METRICS = ["recon_hyp_cos", "recon_rq_cos", "recon_rq_same", "already_asked", "data_best", "data_mean",
           "planted_recovery_cos", "div_mean_cos_dist", "div_vendi", "div_distinct", "tokens", "n_empty",
           "judge_win_vs_D"]


def summarise(rows):
    res = {}
    for tb in ("literature", "data"):
        R = [r for r in rows if r["testbed"] == tb]
        if not R:
            continue
        res[tb] = {}
        for m in METRICS:
            per_arm = {}
            for arm in ARMS:
                v = {r["input_set_id"]: r[m] for r in R if r["arm"] == arm and r.get(m) is not None}
                if v:
                    per_arm[arm] = cluster_bootstrap(list(v.values()), list(v.keys()))
            # paired differences against D over input sets
            base = {r["input_set_id"]: r[m] for r in R if r["arm"] == "D" and r.get(m) is not None}
            for arm in ARMS:
                if arm == "D":
                    continue
                v = {r["input_set_id"]: r[m] for r in R if r["arm"] == arm and r.get(m) is not None}
                common_ids = sorted(set(v) & set(base))
                if len(common_ids) >= 5:
                    per_arm[f"{arm}-D"] = cluster_bootstrap([v[i] - base[i] for i in common_ids], common_ids)
            v1 = {r["input_set_id"]: r[m] for r in R if r["arm"] == "QH_sel" and r.get(m) is not None}
            v2 = {r["input_set_id"]: r[m] for r in R if r["arm"] == "QH_rand" and r.get(m) is not None}
            ids = sorted(set(v1) & set(v2))
            if len(ids) >= 5:
                per_arm["QH_sel-QH_rand"] = cluster_bootstrap([v1[i] - v2[i] for i in ids], ids)
            if per_arm:
                res[tb][m] = per_arm
        qt = {}
        for arm in ARMS:
            T = [r["qtype"] for r in R if r["arm"] == arm and r.get("qtype")]
            if T:
                keys = set().union(*T)
                qt[arm] = {k: float(np.mean([t.get(k, 0) for t in T])) for k in keys}
        res[tb]["question_type"] = qt
    write_json(HERE / "results" / "summary.json", res)
    print(json.dumps({tb: {m: {a: round(v["est"], 3) for a, v in d.items()} for m, d in x.items()
                           if m != "question_type"} for tb, x in res.items()}, indent=1)[:4000])


def cmd_sample20():
    gens = read_jsonl(HERE / "generations.jsonl")
    ids = sorted({g["input_set_id"] for g in gens if g["testbed"] == "literature"})
    pick = random.Random(1).sample(ids, min(20, len(ids)))
    sets = {s["input_set_id"]: s for s in read_jsonl(HERE / "testbed_literature.jsonl")}
    lines = []
    for sid in pick:
        s = sets[sid]
        lines.append(f"## {sid}\n\n**Target RQ:** {s['target_rq']}\n\n**Target hypothesis:** {s['target_hyp']}\n")
        lines.append("**Inputs:** " + "; ".join(r["title"] for r in s["inputs"]) + "\n")
        for arm in ARMS:
            G = [g for g in gens if g["input_set_id"] == sid and g["arm"] == arm]
            lines.append(f"### {arm}\n" + "\n".join(f"- Q: {g['rq']}  \n  H: {g['hypothesis']}" for g in G) + "\n")
    (HERE / "error_analysis_samples.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote E6/error_analysis_samples.md -- read these and write E6/error_analysis.md")


if __name__ == "__main__":
    {"build_lit": cmd_build_lit, "build_data": cmd_build_data, "generate": cmd_generate,
     "evaluate": cmd_evaluate, "sample20": cmd_sample20}[sys.argv[1]]()
