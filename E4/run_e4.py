"""E4 -- where does quality live?

    python E4/run_e4.py data        # ICLR 2024/2025/2026 via OpenReview (needs login env vars)
    python E4/run_e4.py cards       # (i) question and (ii) card for every paper -> cards.jsonl
    python E4/run_e4.py ceiling     # Task 1
    python E4/run_e4.py views       # Task 2  (train 2024, choose alpha on 2025, test once on 2026)
    python E4/run_e4.py reading     # Task 3  reading-state probe + contamination checks
    python E4/run_e4.py summary     # Task 4  share of explainable signal, view (i)->(iii) gap

Topic stratum = ICLR 'primary_area'. Gap for the ceiling fixed in advance in PREREG.md:
rating 1.0 (1-10 scale), contribution / soundness 0.5 (1-4 scale).
Language model for extraction and reading state: Qwen2.5-1.5B-Instruct (released Sept 2024, before the
ICLR 2026 reviews of Nov 2025).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_sources import load_iclr  # noqa: E402
from common.embed import Embedder  # noqa: E402
from common.extract import card_text, extract_card, extract_rq, extractor  # noqa: E402
from common.io import read_jsonl, write_json, write_jsonl, read_json  # noqa: E402
import evalkit as ek  # noqa: E402

HERE = Path(__file__).parent
YEARS = (2024, 2025, 2026)
GAPS = {"rating": 1.0, "contribution": 0.5, "soundness": 0.5}
TARGET = "rating"   # the public mirrors carry no contribution/soundness sub-scores (deviation, NOTEBOOK.md)
N_SUB = {2024: 2000, 2025: 1000, 2026: 2000}   # CPU budget: fixed random subsample (seed 0) per year
N_TOPICS = 20


def papers(subsample=True):
    """ICLR papers with a topic stratum. primary_area is missing from the public mirrors, so the stratum is a
    KMeans cluster (k=20) of TF-IDF abstract vectors, fit on 2024 only and applied to every year."""
    import random
    rows = []
    for y in YEARS:
        P = load_iclr(y)
        if subsample and len(P) > N_SUB[y]:
            P = random.Random(0).sample(sorted(P, key=lambda p: p["paper_id"]), N_SUB[y])
        rows += P
    _assign_topics(rows)
    return rows


def _assign_topics(rows):
    import pickle
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    f = HERE / "results" / "topic_model.pkl"
    if f.exists():
        tf, km = pickle.load(open(f, "rb"))
    else:
        tr = [p for p in load_iclr(2024)]
        tf = TfidfVectorizer(min_df=5, max_features=30000, stop_words="english", sublinear_tf=True)
        X = tf.fit_transform([p["title"] + ". " + p["abstract"] for p in tr])
        km = KMeans(N_TOPICS, n_init=5, random_state=0).fit(X)
        f.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((tf, km), open(f, "wb"))
    lab = km.predict(tf.transform([p["title"] + ". " + p["abstract"] for p in rows]))
    for p, l in zip(rows, lab):
        p["topic"] = p.get("primary_area") or f"k{int(l)}"


def cmd_data():
    for y in YEARS:
        print(y, len(load_iclr(y)))


def cmd_cards():
    from common.io import append_jsonl
    llm = extractor()
    f = HERE / "cards.jsonl"
    done = {r["paper_id"] for r in read_jsonl(f)} if f.exists() else set()
    for p in papers():  # resumable: one line appended per paper
        if p["paper_id"] in done:
            continue
        append_jsonl(f, {"paper_id": p["paper_id"], "year": p["year"],
                         "rq": extract_rq(llm, p["title"], p["abstract"]),
                         "card": extract_card(llm, p["title"], p["abstract"])})


def _views(P, cards):
    if not cards:  # no LLM extraction yet: only view (iii) is available
        return P, {"abstract": [f"{p['title']}. {p['abstract']}" for p in P]}
    c = {r["paper_id"]: r for r in cards}
    P = [p for p in P if p["paper_id"] in c]
    v = {"question": [c[p["paper_id"]]["rq"] for p in P],
         "card": [card_text(c[p["paper_id"]]["card"]) for p in P],
         "abstract": [f"{p['title']}. {p['abstract']}" for p in P]}
    return P, v


def _cards():
    f = HERE / "cards.jsonl"
    return read_jsonl(f) if f.exists() else []


def _fit_eval(Xtr, ytr, Xse, yse, Xte):
    from sklearn.linear_model import Ridge
    best = None
    for a in np.logspace(-2, 4, 13):  # choose alpha on 2025 (selection year) only
        m = Ridge(alpha=a).fit(Xtr, ytr)
        r = np.corrcoef(m.predict(Xse), yse)[0, 1]
        if best is None or r > best[0]:
            best = (r, a)
    m = Ridge(alpha=best[1]).fit(Xtr, ytr)
    return m.predict(Xte), float(best[1])


def cmd_ceiling():
    P = [p for p in papers(subsample=False) if p["year"] == 2026]
    res = {k: ek.split_half_ceiling(P, k, g) for k, g in GAPS.items()
           if sum(r.get(k) is not None for p in P for r in p["reviews"]) > 0}
    write_json(HERE / "results" / "ceiling.json", res)
    print(res)


def cmd_views():
    from sklearn.feature_extraction.text import TfidfVectorizer
    P, V = _views(papers(), _cards())
    yrs = np.array([p["year"] for p in P])
    y = np.array([ek.mean_score(p, TARGET) for p in P])
    ok = ~np.isnan(y)
    tr, se, te = ok & (yrs == 2024), ok & (yrs == 2025), ok & (yrs == 2026)
    topics = [P[i]["topic"] for i in np.where(te)[0]]
    E = Embedder("qwen3-0.6b", use_prompt=False)
    res = {}
    for name, texts in V.items():
        X = E(texts)
        pred, a = _fit_eval(X[tr], y[tr], X[se], y[se], X[te])
        res[f"emb:{name}"] = {**ek.within_topic_pair_accuracy(pred, y[te], topics), "alpha": a}
        tf = TfidfVectorizer(min_df=3, max_features=50000, sublinear_tf=True).fit([texts[i] for i in np.where(tr)[0]])
        Xt = tf.transform(texts)
        pred, a = _fit_eval(Xt[tr], y[tr], Xt[se], y[se], Xt[te])
        res[f"tfidf:{name}"] = {**ek.within_topic_pair_accuracy(pred, y[te], topics), "alpha": a}
        L = np.array([[len(t.split())] for t in texts], float)
        res[f"length:{name}"] = ek.within_topic_pair_accuracy(L[te, 0], y[te], topics)
    write_json(HERE / "results" / "views.json", res)
    for k, v in res.items():
        print(f"{k:22s} {v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")


READ_Q = "\n\nQuestion: How significant and novel is the contribution of this research? Answer:"


def reading_states(texts, model="Qwen/Qwen2.5-1.5B-Instruct", layer=-8, cache_name="reading"):
    """Mean of the last 8 tokens' hidden state at a middle-upper layer after reading text + fixed question."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.io import DATA, text_key
    cdir = DATA / cache_name; cdir.mkdir(parents=True, exist_ok=True)
    tok = mdl = None
    out = []
    for t in texts:
        f = cdir / f"{text_key(model + t)}.npy"
        if f.exists():
            out.append(np.load(f)); continue
        if mdl is None:
            tok = AutoTokenizer.from_pretrained(model)
            mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        ids = tok(t[:6000] + READ_Q, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            hs = mdl(**ids, output_hidden_states=True).hidden_states[layer][0, -8:].float().mean(0).cpu().numpy()
        np.save(f, hs); out.append(hs)
    return np.stack(out)


def cmd_reading():
    P, V = _views(papers(), _cards())
    yrs = np.array([p["year"] for p in P])
    y = np.array([ek.mean_score(p, TARGET) for p in P])
    ok = ~np.isnan(y)
    E = Embedder("qwen3-0.6b", use_prompt=False)
    res = {}
    # Main comparison: train 2024, select 2025, test 2026 (after the model's release)
    # Contamination comparison: train 2024 -> test 2025 (a year the model may have read about? Qwen2.5
    # was released Sept 2024; ICLR 2025 submissions appeared Oct 2024 -> also unseen. So we add a
    # *seen* year: ICLR 2024 itself, evaluated by 5-fold cross-fitting within 2024.)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    for name, texts in V.items():
        R = reading_states(texts)
        X = E(texts)
        tr, se, te = ok & (yrs == 2024), ok & (yrs == 2025), ok & (yrs == 2026)
        topics_te = [P[i]["topic"] for i in np.where(te)[0]]
        pr, _ = _fit_eval(R[tr], y[tr], R[se], y[se], R[te])
        pe, _ = _fit_eval(X[tr], y[tr], X[se], y[se], X[te])
        res[f"reading:{name}:2026"] = ek.within_topic_pair_accuracy(pr, y[te], topics_te)
        res[f"emb:{name}:2026"] = ek.within_topic_pair_accuracy(pe, y[te], topics_te)
        # seen year (2024), cross-fitted
        idx = np.where(tr)[0]
        pr24, pe24 = np.zeros(len(idx)), np.zeros(len(idx))
        for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
            pr24[b] = Ridge(alpha=100).fit(R[idx[a]], y[idx[a]]).predict(R[idx[b]])
            pe24[b] = Ridge(alpha=10).fit(X[idx[a]], y[idx[a]]).predict(X[idx[b]])
        t24 = [P[i]["topic"] for i in idx]
        res[f"reading:{name}:2024cv"] = ek.within_topic_pair_accuracy(pr24, y[idx], t24)
        res[f"emb:{name}:2024cv"] = ek.within_topic_pair_accuracy(pe24, y[idx], t24)
    res["title_recall"] = title_recall(P)
    write_json(HERE / "results" / "reading.json", res)
    for k, v in res.items():
        print(k, v if k == "title_recall" else f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")


def title_recall(P, n=100):
    """Can the model name the title from the abstract's first sentence? Fuzzy match ratio >= 0.8."""
    from difflib import SequenceMatcher
    from common.llm import LLM
    llm = LLM(temperature=0.0, max_new_tokens=40)
    rng = np.random.default_rng(0)
    out = {}
    for y in YEARS:
        S = [p for p in P if p["year"] == y]
        S = [S[i] for i in rng.choice(len(S), min(n, len(S)), replace=False)]
        hits = 0
        for p in S:
            first = p["abstract"].split(". ")[0] + "."
            g = llm(f"Which paper begins its abstract with this sentence? Give only the paper's title.\n\n{first}\n\nTitle:")["text"]
            hits += SequenceMatcher(None, g.lower(), p["title"].lower()).ratio() >= 0.8
        out[str(y)] = hits / len(S)
    return out


def cmd_summary():
    ceil = read_json(HERE / "results" / "ceiling.json")[TARGET]["ceiling"]
    views = read_json(HERE / "results" / "views.json")
    out = {"ceiling": ceil}
    for k, v in views.items():
        out[k] = {"acc": v["est"], "explainable_share": ek.explainable(v["est"], ceil)}
    q, c, a = (views[f"emb:{n}"]["est"] for n in ("question", "card", "abstract"))
    out["card_recovers_share_of_q_to_abs_gap"] = (c - q) / (a - q) if a != q else float("nan")
    write_json(HERE / "results" / "summary.json", out)
    print(out)


if __name__ == "__main__":
    {"data": cmd_data, "cards": cmd_cards, "ceiling": cmd_ceiling, "views": cmd_views,
     "reading": cmd_reading, "summary": cmd_summary}[sys.argv[1]]()
