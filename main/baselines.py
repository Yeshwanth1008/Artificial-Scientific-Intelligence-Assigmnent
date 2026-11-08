"""Baselines computed on the same data as every test (Section M.B).

B-Q    frozen sentence embedding of the question; linear probe (ridge) for scores, cosine for relations
B-ABS  the same on title + abstract (setting (a) only; a reference point, not a target)
B-TF   TF-IDF on the same text, ridge for scores, cosine for relations
B-LEN  length alone (number of words)
B-JUDGE one language-model judge, both orders (pairwise; optional)
"""
import numpy as np

from common.embed import Embedder


def _ridge(Xtr, ytr, Xse, yse):
    from sklearn.linear_model import Ridge
    alphas = np.logspace(-2, 4, 13)
    best = max(alphas, key=lambda a: np.nan_to_num(np.corrcoef(Ridge(alpha=a).fit(Xtr, ytr).predict(Xse), yse)[0, 1]))
    return Ridge(alpha=best).fit(Xtr, ytr)


def emb_probe(texts_tr, y_tr, texts_se, y_se, texts_te):
    E = Embedder("qwen3-0.6b", use_prompt=False)
    m = _ridge(E(texts_tr), y_tr, E(texts_se), y_se)
    return m.predict(E(texts_te))


def tfidf_probe(texts_tr, y_tr, texts_se, y_se, texts_te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True).fit(texts_tr)
    m = _ridge(tf.transform(texts_tr), y_tr, tf.transform(texts_se), y_se)
    return m.predict(tf.transform(texts_te))


def length(texts):
    return np.array([len(t.split()) for t in texts], float)


def cos_pairs(a, b, kind="emb"):
    """Similarity of aligned text pairs: B-Q cosine (kind='emb') or B-TF cosine (kind='tfidf')."""
    if kind == "emb":
        E = Embedder("qwen3-0.6b", use_prompt=False)
        return np.sum(E(a) * E(b), 1)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer().fit(list(a) + list(b))
    A, B = tf.transform(a), tf.transform(b)
    num = np.asarray(A.multiply(B).sum(1)).ravel()
    den = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel() * np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-12
    return num / den


def judge_pairs(a, b, prompt):
    """B-JUDGE: returns P(a better) in {0, .5, 1} from both orders."""
    from common.llm import LLM
    llm = LLM(temperature=0.0, max_new_tokens=3)
    out = []
    for x, y in zip(a, b):
        r1 = llm(prompt.format(a=x, b=y))["text"].strip()[:1].upper()
        r2 = llm(prompt.format(a=y, b=x))["text"].strip()[:1].upper()
        out.append(1.0 if (r1, r2) == ("A", "B") else 0.0 if (r1, r2) == ("B", "A") else 0.5)
    return np.array(out)
