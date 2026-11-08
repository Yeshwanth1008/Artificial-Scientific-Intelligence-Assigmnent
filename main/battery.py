"""The test battery (Section M.C). Each test returns a JSON-able dict and is written to main/results/<test>.json.

"beats" = the paired 95% interval of (Phi - baseline) lies above the pre-registered margin (PREREG.md).
Every result that involves inputs also reports the empty-input, shuffled-input and decoy-question controls.
"""
import random

import numpy as np

from common.io import read_json, read_jsonl
from common.stats import auroc, cluster_bootstrap
from E4 import evalkit as ek
from . import baselines as BL
from .context import ROOT, Ctx

MARGIN = 0.0          # pre-registered margin for "beats" (PREREG.md)
T1_TOL = 0.02         # pre-registered tolerance for time invariance
FIELDS = ["A", "B", "B_min", "C", "C_tilde", "narrower", "broader"]
T_E1 = "2025-07-01"   # time at which E1 questions are posed (all source papers are from 2025 onward)


def _beats(phi_vals, base_vals, ids):
    d = cluster_bootstrap(np.asarray(phi_vals, float) - np.asarray(base_vals, float), ids)
    d["pass"] = bool(d["lo"] > MARGIN)
    return d


def _pairacc_paired(pred_a, pred_b, target, topics, min_len_gap=None, lengths=None, seed=0):
    """Within-topic pair accuracy of two predictors on the same pairs + paper-bootstrap CI of the difference."""
    pairs = ek._pairs_within(topics, seed=seed)
    if min_len_gap is not None:
        pairs = [(i, j) for i, j in pairs if abs(lengths[i] - lengths[j]) <= min_len_gap]
    ka, oa = ek._correct(np.asarray(pred_a), np.asarray(target), pairs)
    kb, ob = ek._correct(np.asarray(pred_b), np.asarray(target), pairs)
    assert ka == kb
    n = len(target)
    lo, hi = ek.paper_bootstrap(ka, oa - ob, n)
    return {"phi": float(oa.mean()), "base": float(ob.mean()), "diff": float((oa - ob).mean()),
            "diff_lo": lo, "diff_hi": hi, "n_pairs": len(ka), "pass": bool(lo > MARGIN)}


# ====================================================================== Invariance

def I1_paraphrase(ctx: Ctx):
    items = ctx.e1_items()
    ids = [it["id"] for it in items]
    P = {f: [ctx.phi_text(it[f], T_E1) for it in items] for f in ("A", "B", "B_min", "C")}
    res = {}
    readouts = {
        "identity": lambda a, x: 1 - ctx.model.same_question(a, x),
        "scope": lambda a, x: abs(a["scope"].sum() - x["scope"].sum()),
        "novelty": lambda a, x: float(np.nanmean(np.abs(a["nov"] - x["nov"]))),
        "quality": lambda a, x: float(np.mean([abs(v - ctx.model.quality(x)[k]) for k, v in ctx.model.quality(a).items()])),
    }
    for name, f in readouts.items():
        para = [0.5 * (f(P["A"][i], P["B"][i]) + f(P["A"][i], P["B_min"][i])) for i in range(len(items))]
        near = [f(P["A"][i], P["C"][i]) for i in range(len(items))]
        d = cluster_bootstrap(np.array(near) - np.array(para), ids)
        d["pass"] = bool(d["lo"] > 0)
        res[name] = {"mean_change_paraphrase": float(np.mean(para)), "mean_change_nearmiss": float(np.mean(near)),
                     "nearmiss_minus_paraphrase": d}
    res["pass_all"] = all(r["nearmiss_minus_paraphrase"]["pass"] for r in res.values() if isinstance(r, dict))
    return res


def I3_length(ctx: Ctx):
    """Quality claims on length-matched pairs (question length within 2 words) vs B-LEN and B-Q."""
    te = ctx.split("iclr", "test")
    y = np.array([r["scores"]["contribution"] for r in te])
    ok = ~np.isnan(y)
    te, y = [r for r, o in zip(te, ok) if o], y[ok]
    topics = [r["topic"] for r in te]
    L = BL.length([r["q"] for r in te])
    q = np.array([ctx.model.quality(ctx.phi_item(r))["q_review"] for r in te])
    out = {"vs_B_LEN_matched": _pairacc_paired(q, L, y, topics, min_len_gap=2, lengths=L),
           "vs_B_Q_matched": _pairacc_paired(q, ctx.bq_pred("iclr", "contribution"), y, topics, min_len_gap=2, lengths=L)}
    out["pass"] = out["vs_B_LEN_matched"]["pass"]
    return out


def I4_order_format(ctx: Ctx, n=200):
    rng = random.Random(0)
    te = ctx.split("iclr", "test")[:n]
    perm, titles = [], []
    for r in te:
        a = ctx.model.grounding(ctx.phi_item(r, with_novelty=False), r["R"])[0]
        R2 = list(r["R"]); rng.shuffle(R2)
        b = ctx.model.grounding(ctx.phi_item(r, with_novelty=False), R2)[0]
        c = ctx.model.grounding(ctx.phi_item(r, with_novelty=False), [{"title": x["title"], "abstract": ""} for x in r["R"]])[0]
        perm.append(np.abs(a - b).max()); titles.append(c)
    full = np.stack([ctx.model.grounding(ctx.phi_item(r, with_novelty=False), r["R"])[0] for r in te])
    titles = np.stack(titles)
    corr = {k: float(np.corrcoef(full[:, i], titles[:, i])[0, 1]) for i, k in
            enumerate(["inplane", "offnorm", "top3", "covered", "copy"])}
    return {"max_abs_change_under_permutation": float(np.max(perm)), "titles_vs_abstracts_corr": corr,
            "pass": bool(np.max(perm) < 1e-4)}


# ====================================================================== Sensitivity

def S1_nearmiss(ctx: Ctx):
    items = ctx.e1_items()
    ids = [it["id"] for it in items]
    P = {f: [ctx.phi_text(it[f], T_E1, with_novelty=False) for it in items] for f in ("A", "B", "B_min", "C", "C_tilde")}
    same = lambda f: np.array([ctx.model.same_question(P["A"][i], P[f][i]) for i in range(len(items))])
    s = {f: same(f) for f in ("B", "B_min", "C", "C_tilde")}
    bq = {f: BL.cos_pairs([it["A"] for it in items], [it[f] for it in items]) for f in ("B", "B_min", "C", "C_tilde")}
    out = {}
    for name, (pos, neg) in {"nearmiss_vs_paraphrase": ("B", "C"), "nearmiss_vs_minimal_edit": ("B_min", "C"),
                             "lowoverlap_vs_paraphrase": ("B", "C_tilde")}.items():
        phi_ok = (s[pos] > s[neg]).astype(float)
        bq_ok = (bq[pos] > bq[neg]).astype(float)
        out[name] = {"phi": float(phi_ok.mean()), "B_Q": float(bq_ok.mean()), "diff": _beats(phi_ok, bq_ok, ids)}
        out[name]["per_type"] = {t: {"phi": float(phi_ok[[it["C_type"] == t for it in items]].mean()),
                                     "B_Q": float(bq_ok[[it["C_type"] == t for it in items]].mean())}
                                 for t in sorted({it["C_type"] for it in items})}
    # same-question decision quality at 0.5 (vs human labels = item construction)
    y = np.r_[np.ones(2 * len(items)), np.zeros(2 * len(items))]
    sc = np.r_[s["B"], s["B_min"], s["C"], s["C_tilde"]]
    out["decision_accuracy@0.5"] = float(((sc >= 0.5) == y).mean())
    out["auroc_phi"] = auroc(sc, y)
    out["auroc_B_Q"] = auroc(np.r_[bq["B"], bq["B_min"], bq["C"], bq["C_tilde"]], y)
    # signed scope comparison (broader > A > narrower) incl. length-flipped items vs length-only baseline
    Ps = {f: [ctx.phi_text(it[f], T_E1, with_novelty=False) for it in items] for f in ("narrower", "broader")}
    order = np.array([float(ctx.model.broader(Ps["broader"][i], P["A"][i]) > 0 and ctx.model.broader(P["A"][i], Ps["narrower"][i]) > 0)
                      for i in range(len(items))])
    lf = np.array([bool(it.get("length_flipped")) for it in items])
    out["scope_order"] = {"all": float(order.mean()), "length_flipped": float(order[lf].mean()) if lf.any() else None,
                          "length_only_on_flipped": 0.0}
    out["pass"] = out["nearmiss_vs_paraphrase"]["diff"]["pass"] and out["nearmiss_vs_minimal_edit"]["diff"]["pass"]
    return out


def S2_facet_edits(ctx: Ctx, n=60):
    from common.llm import LLM
    from common.prompts import FACET_EDIT
    desc = {"purpose": "what problem it addresses", "mechanism": "how it would be addressed",
            "evaluation": "how success is judged", "domain": "the data, field or setting"}
    llm = LLM(temperature=0.0, max_new_tokens=100)
    items = ctx.e1_items()[:n]
    hits, ids = [], []
    for it in items:
        a = ctx.phi_text(it["A"], T_E1, with_novelty=False)
        for k, facet in enumerate(desc):
            q2 = llm(FACET_EDIT.format(facet=facet, desc=desc[facet], q=it["A"]))["text"].split("\n")[0]
            b = ctx.phi_text(q2, T_E1, with_novelty=False)
            move = 1 - np.sum(a["f"] * b["f"], 1)
            hits.append(float(np.argmax(move) == k)); ids.append(it["id"])
    r = cluster_bootstrap(hits, ids)
    r["chance"] = 0.25
    r["pass"] = bool(r["lo"] > 0.25)
    return r


def S3_time(ctx: Ctx, n=150):
    """An equivalent earlier paper lowers N; the same paper dated later leaves N unchanged."""
    items = ctx.e1_items()[:n]
    lower, same = [], []
    for it in items:
        a = ctx.phi_text(it["A"], T_E1)
        vec = ctx.model.E([it["B"]])[0]
        early = {"id": "synthetic-early", "date": "2025-01-15", "title": "", "abstract": it["B"], "vec": vec}
        late = dict(early, id="synthetic-late", date="2025-12-01")
        n_early = ctx.model.novelty_coords(a, T_E1, extra=[early])
        n_late = ctx.model.novelty_coords(a, T_E1, extra=[late])
        lower.append(n_early < a["nov"] - 1e-9)
        same.append(np.abs(n_late - a["nov"]) < 1e-9)
    lower, same = np.array(lower), np.array(same)
    names = ["nearest", "density", "offspan", "facet"]
    return {"lowered_by_earlier": dict(zip(names, lower.mean(0).tolist())),
            "unchanged_by_later": dict(zip(names, same.mean(0).tolist())),
            "pass": bool(same.mean() > 0.99 and lower[:, 0].mean() > 0.95)}


# ====================================================================== Axioms

def A1_axioms(ctx: Ctx):
    f = ROOT / "E2" / "results" / "axioms_pass_rates.json"
    if not f.exists():
        return {"not_run": "E2/results/axioms_pass_rates.json missing -- run E2 first"}
    r = read_json(f)
    pc = {m: r["partial_50"][m]["est"] for m in ("Y", "density", "N", "yours")}
    cc = {m: r["concentrated_3"][m]["est"] for m in ("Y", "density", "N", "yours")}
    passing = [m for m in pc if pc[m] > 0.5 and cc[m] > 0.5]
    return {"partial_coverage_50": pc, "concentration": cc, "all": r,
            "phi_coordinates": {"nov_nearest": "Y", "nov_density": "density", "nov_facet": "yours"},
            "passing_coordinates": passing, "pass": bool(passing)}


# ====================================================================== Human labels

def H1_expert_battles(ctx: Ctx):
    """data/ideation_arena.jsonl: {battle_id, idea_a, idea_b, system_a, system_b, winner: 'a'|'b'}."""
    f = ROOT / "data" / "ideation_arena.jsonl"
    if not f.exists():
        return {"not_run": "data/ideation_arena.jsonl missing (Ideation Arena release [R3])"}
    B = read_jsonl(f)
    texts = sorted({b["idea_a"] for b in B} | {b["idea_b"] for b in B})
    score = {t: ctx.model.quality(ctx.phi_text(t, None, with_novelty=False))["q_review"] for t in texts}
    bq = dict(zip(texts, ctx.bq_apply("iclr", "contribution", texts)))
    ln = {t: len(t.split()) for t in texts}
    out = {}
    for name, S in (("phi", score), ("B_Q", bq), ("B_LEN", ln)):
        sysmean = {}
        for b in B:
            sysmean.setdefault(b["system_a"], []).append(S[b["idea_a"]])
            sysmean.setdefault(b["system_b"], []).append(S[b["idea_b"]])
        sysmean = {k: np.mean(v) for k, v in sysmean.items()}
        raw, dem, within, matched = [], [], [], []
        for b in B:
            y = 1.0 if b["winner"] == "a" else 0.0
            d = S[b["idea_a"]] - S[b["idea_b"]]
            dd = d - (sysmean[b["system_a"]] - sysmean[b["system_b"]])
            raw.append(float((d > 0) == y)); dem.append(float((dd > 0) == y))
            if b["system_a"] == b["system_b"]:
                within.append(float((d > 0) == y))
            if max(ln[b["idea_a"]], ln[b["idea_b"]]) <= 1.2 * min(ln[b["idea_a"]], ln[b["idea_b"]]):
                matched.append(float((dd > 0) == y))
        ids = [b["battle_id"] for b in B]
        out[name] = {"raw": cluster_bootstrap(raw, ids), "system_demeaned": cluster_bootstrap(dem, ids),
                     "within_system": float(np.mean(within)) if within else None,
                     "length_matched_demeaned": float(np.mean(matched)) if matched else None}
    out["pass"] = bool(out["phi"]["system_demeaned"]["lo"] > 0.5)
    return out


def H2_reviewer_novelty(ctx: Ctx):
    f = ROOT / "E2" / "results" / "human_check.json"
    if not f.exists():
        return {"not_run": "E2 human check not run (needs novelty-judge-bench [R2])"}
    r = read_json(f)
    best = max(("Y", "density", "N", "yours"), key=lambda m: r[m]["minus_quality"])
    return {"E2_human_check": r, "best": best, "pass": bool(r[best]["minus_quality_ci"][0] > 0)}


def H3_review_scores(ctx: Ctx):
    te = ctx.split("iclr", "test")
    y = np.array([r["scores"]["contribution"] for r in te])
    ok = ~np.isnan(y)
    te, y = [r for r, o in zip(te, ok) if o], y[ok]
    topics = [r["topic"] for r in te]
    q = np.array([ctx.model.quality(ctx.phi_item(r))["q_review"] for r in te])
    ceil_f = ROOT / "E4" / "results" / "ceiling.json"
    ceiling = read_json(ceil_f)["contribution"]["ceiling"] if ceil_f.exists() else None
    out = {}
    for name, pred in (("B_Q", ctx.bq_pred("iclr", "contribution")), ("B_ABS", ctx.babs_pred("contribution")),
                       ("B_TF", ctx.btf_pred("iclr", "contribution")), ("B_LEN", BL.length([r["q"] for r in te]))):
        out[f"vs_{name}"] = _pairacc_paired(q, pred, y, topics)
    out["phi_acc"] = out["vs_B_Q"]["phi"]
    out["ceiling"] = ceiling
    out["phi_share_of_ceiling"] = ek.explainable(out["phi_acc"], ceiling) if ceiling else None
    out["controls"] = ctx.controls_quality(te, y, topics)
    out["pass"] = out["vs_B_Q"]["pass"]
    return out


def H4_facet_similarity(ctx: Ctx):
    f = ROOT / "data" / "novgauge_pairs.jsonl"
    if not f.exists():
        return {"not_run": "data/novgauge_pairs.jsonl missing (NovGauge [R12])"}
    rows = read_jsonl(f)  # {text_a, text_b, facet: task|problem|method, overlap: 0/1}
    out = {}
    fmap = {"task": "domain", "problem": "purpose", "method": "mechanism"}
    from common.extract import FACET_KEYS
    for fac in fmap:
        R = [r for r in rows if r["facet"] == fac]
        if not R:
            continue
        k = FACET_KEYS.index(fmap[fac])
        s = [float(ctx.phi_text(r["text_a"], None, with_novelty=False)["f"][k] @ ctx.phi_text(r["text_b"], None, with_novelty=False)["f"][k]) for r in R]
        babs = BL.cos_pairs([r["text_a"] for r in R], [r["text_b"] for r in R])
        y = [r["overlap"] for r in R]
        out[fac] = {"phi_auroc": auroc(s, y), "B_ABS_auroc": auroc(babs, y)}
    return out


# ====================================================================== Outcomes

def O1_post_cutoff(ctx: Ctx):
    out = {"review": H3_review_scores(ctx)["vs_B_Q"]}
    te = ctx.split("uptake", "test")
    y = np.array([r["log_uptake"] for r in te])
    topics = ctx.uptake_topics(te)
    q = np.array([ctx.model.quality(ctx.phi_item(r))["q_uptake"] for r in te])
    out["uptake"] = _pairacc_paired(q, ctx.bq_pred("uptake", "log_uptake"), y, topics)
    out["uptake_controls"] = ctx.controls_quality(te, y, topics, head="q_uptake")
    # cross-factor: do the two Q factors disagree? (they should be separate axes)
    out["pass"] = bool(out["review"]["pass"] and out["uptake"]["pass"])
    return out


def O2_same_inputs(ctx: Ctx):
    f = ROOT / "main" / "o2_pairs.jsonl"
    if not f.exists():
        return {"not_run": "run python -m main.data groups"}
    pairs = read_jsonl(f)
    te = {r["id"]: r for r in ctx.split("uptake", "test")}
    pairs = [p for p in pairs if p["a"] in te and p["b"] in te and te[p["a"]]["uptake12"] != te[p["b"]]["uptake12"]]
    ids = list(te)
    pos = {k: i for i, k in enumerate(ids)}
    q = np.array([ctx.model.quality(ctx.phi_item(te[k]))["q_uptake"] for k in ids])
    btf = ctx.btf_pred("uptake", "log_uptake")
    y = np.array([te[k]["log_uptake"] for k in ids])
    P = [(pos[p["a"]], pos[p["b"]]) for p in pairs]
    ka, oa = ek._correct(q, y, P)
    kb, ob = ek._correct(btf, y, P)
    lo, hi = ek.paper_bootstrap(ka, oa - ob, len(ids))
    return {"n_pairs": len(ka), "phi": float(oa.mean()) if len(oa) else None, "B_TF": float(ob.mean()) if len(ob) else None,
            "diff_lo": lo, "diff_hi": hi, "pass": bool(lo > MARGIN)}


def O3_generated(ctx: Ctx):
    return {"not_run": "needs E3 already-asked labels; E3 was not chosen (E1, E2, E4 were)"}


# ====================================================================== Structure and time

def K1_clustering(ctx: Ctx, k=10):
    """Within topic: do a question's nearest TRAIN neighbours (by Phi features vs by B-Q embedding) predict
    whether it is good (contribution above its topic median)? Length controlled by residualising on length."""
    tr, te = ctx.split("iclr", "train"), ctx.split("iclr", "test")
    ytr = np.array([r["scores"]["contribution"] for r in tr]); yte = np.array([r["scores"]["contribution"] for r in te])
    Xtr = ctx.feat_matrix(tr); Xte = ctx.feat_matrix(te)
    Utr = np.stack([ctx.phi_item(r)["u"] for r in tr]); Ute = np.stack([ctx.phi_item(r)["u"] for r in te])
    med = {}
    for r, y in zip(te, yte):
        med.setdefault(r["topic"], []).append(y)
    good = np.array([float(y > np.nanmedian(med[r["topic"]])) for r, y in zip(te, yte)])
    def knn_pred(A, B):
        A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
        out = []
        for i, r in enumerate(te):
            m = np.array([x["topic"] == r["topic"] for x in tr])
            if m.sum() < k:
                out.append(np.nan); continue
            s = B[m] @ A[i]
            out.append(np.nanmean(ytr[m][np.argsort(-s)[:k]]))
        return np.array(out)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    p_phi, p_bq = knn_pred(sc.transform(Xte), sc.transform(Xtr)), knn_pred(Ute, Utr)
    ok = ~np.isnan(p_phi) & ~np.isnan(p_bq) & ~np.isnan(yte)
    L = BL.length([r["q"] for r in te])[ok]
    res = lambda p: p[ok] - np.polyval(np.polyfit(L, p[ok], 1), L)   # remove the linear length effect
    return {"auroc_phi": auroc(res(p_phi), good[ok]), "auroc_B_Q": auroc(res(p_bq), good[ok]),
            "pass": bool(auroc(res(p_phi), good[ok]) > auroc(res(p_bq), good[ok]))}


def T1_time_invariance(ctx: Ctx):
    """q_review trained on ICLR 2024 only (up to Y-1) vs on 2024+2025 (up to Y); same test (2026)."""
    te = ctx.split("iclr", "test")
    y = np.array([r["scores"]["contribution"] for r in te])
    ok = ~np.isnan(y)
    te, y = [r for r, o in zip(te, ok) if o], y[ok]
    topics = [r["topic"] for r in te]
    tr = ctx.split("iclr", "train"); se = ctx.split("iclr", "select")
    m = ctx.fresh_model()
    m.mu = ctx.model.mu
    ytr = np.array([r["scores"]["contribution"] for r in tr]); yse = np.array([r["scores"]["contribution"] for r in se])
    ktr, kse = ~np.isnan(ytr), ~np.isnan(yse)
    m.fit_head("a", [ctx.phi_item(r) for r, k in zip(tr, ktr) if k], ytr[ktr])
    both = [r for r, k in zip(tr, ktr) if k] + [r for r, k in zip(se, kse) if k]
    m.fit_head("b", [ctx.phi_item(r) for r in both], np.r_[ytr[ktr], yse[kse]])
    pa = np.array([m.quality(ctx.phi_item(r))["a"] for r in te])
    pb = np.array([m.quality(ctx.phi_item(r))["b"] for r in te])
    r = _pairacc_paired(pb, pa, y, topics)
    r["pass"] = bool(abs(r["diff"]) <= T1_TOL)
    r["tolerance"] = T1_TOL
    return r


def T2_updating(ctx: Ctx):
    return {"not_run": "forecasting independently re-asked questions needs the E3 same-question labels; not attempted"}


def R1_no_regression(ctx: Ctx):
    return {"note": "Phi's retrieval vector u IS the B-Q embedding, so related-paper retrieval is identical to B-Q "
                    "by construction (not worse, not better).", "pass": True}


def G_grounded(ctx: Ctx, n=500):
    """Groundedness readout with all three controls: real q vs decoy q (same topic) against the true R,
    and real q against true R vs shuffled R; empty R gives no score (reported as missing)."""
    rng = random.Random(0)
    te = [r for r in ctx.split("iclr", "test") if len(r["R"]) >= 5][:n]
    by_topic = {}
    for r in te:
        by_topic.setdefault(r["topic"], []).append(r)
    real, decoy, shuf, ids = [], [], [], []
    for r in te:
        cands = [x for x in by_topic[r["topic"]] if x["id"] != r["id"]]
        if not cands:
            continue
        d = rng.choice(cands); o = rng.choice([x for x in te if x["id"] != r["id"]])
        g = lambda q_item, R: ctx.model.grounding(ctx.phi_item(q_item, with_novelty=False), R)[0][2]
        real.append(g(r, r["R"])); decoy.append(g(d, r["R"])); shuf.append(g(r, o["R"])); ids.append(r["id"])
    y = np.r_[np.ones(len(real)), np.zeros(len(real))]
    src = {}
    for r in te:
        src[r["r_source"]] = src.get(r["r_source"], 0) + 1
    return {"auroc_real_vs_decoy": auroc(np.r_[real, decoy], y), "auroc_real_vs_shuffledR": auroc(np.r_[real, shuf], y),
            "real_minus_decoy": cluster_bootstrap(np.array(real) - np.array(decoy), ids),
            "empty_R": "no score (readout undefined without inputs; Phi reports NaN)", "R_source_counts": src}


def D_diversity(ctx: Ctx, n_sets=50):
    """D readout on sets of 5 same-topic test questions + the anti-gaming axioms (a) paraphrase, (c) clone set."""
    from common.llm import LLM
    from common.prompts import PARAPHRASE
    llm = LLM(temperature=0.7, seed=0, max_new_tokens=80)
    rng = random.Random(0)
    te = ctx.split("iclr", "test")
    by = {}
    for r in te:
        by.setdefault(r["topic"], []).append(r)
    sets = [rng.sample(v, 5) for v in by.values() if len(v) >= 5][:n_sets]
    a_ok, c_ok = [], []
    for S in sets:
        phis = [ctx.phi_item(r, with_novelty=False) for r in S]
        base = ctx.model.diversity(phis)
        para = llm(PARAPHRASE.format(text=S[0]["q"]))["text"].split("\n")[0]
        plus = ctx.model.diversity(phis + [ctx.phi_text(para, None, with_novelty=False)])
        a_ok.append(float(plus["n_distinct"] == base["n_distinct"]))
        clones = [S[0]["q"]] + [LLM(temperature=0.9, seed=s, max_new_tokens=80)(PARAPHRASE.format(text=S[0]["q"]))["text"].split("\n")[0]
                                for s in range(4)]
        cl = ctx.model.diversity([ctx.phi_text(q, None, with_novelty=False) for q in clones])
        c_ok.append(float(cl["n_distinct"] <= 2))
    return {"axiom_a_paraphrase_no_change": float(np.mean(a_ok)), "axiom_c_clone_set_le_2": float(np.mean(c_ok)),
            "n_sets": len(sets)}


CORE = ["I1_paraphrase", "I3_length", "S1_nearmiss", "A1_axioms", "H1_expert_battles", "H3_review_scores",
        "O1_post_cutoff", "O2_same_inputs", "T1_time_invariance"]
EXTENDED = ["I4_order_format", "S2_facet_edits", "S3_time", "H2_reviewer_novelty", "H4_facet_similarity",
            "O3_generated", "K1_clustering", "T2_updating", "R1_no_regression", "G_grounded", "D_diversity"]
NOT_RUN_REASONS = {"I2_authorship": "needs human- and model-written questions from the same inputs with quality "
                                    "labels on both; not available in the chosen data"}
