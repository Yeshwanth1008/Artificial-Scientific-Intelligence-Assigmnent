"""Phi(q, R, t): one shared representation of a research question, its inputs and a time.

Every readout is a head on the same object. One encoder (Qwen3-Embedding-0.6B) produces every vector.

    phi = model.encode(q, R, t, facets=None)

Components (and the problem that motivated each):
  u          topic direction: embedding of q, no instruction                         (T1: topic needs its own vector)
  v          identity direction: same encoder, same-question instruction prompt      (T1 B.2/C.3: a second head)
  f[4]       facet vectors: purpose, mechanism, evaluation, domain                    (T1 C.3, E5, T2 'facets')
  scope[4]   per-facet specificity = 1 - cos(f_k, mu_k), mu_k = mean facet-k vector  (T1 C.1: scope not from length)
  nov        novelty coordinates at t, each answering ONE T2 group, against literature dated < t:
               nov_nearest   (G1) 1 - max cos to an earlier paper
               nov_density   (G2) relative neighbourhood density among the 50 nearest earlier papers
               nov_offspan   (G3) share of (u - local mean) outside the top-20 PCs of the 50 nearest papers
               nov_facet     (G4) T2 B.4 measure (sentence-level, calibrated) with q's facets as the claims
  grd        groundedness in R (T4): in-plane share of q - centroid(R), off-plane norm, top-3 mean cos to R,
             coverage of q's facets by R ("addressed by the inputs"), copy score (max cos to an input),
             and for each facet the input it rests on
  qual       two quality heads (ridge on the features below), trained separately:
               q_review  -> ICLR reviewer contribution score      (who judges: peer review)
               q_uptake  -> log 12-month author-disjoint citations (who judges: later community use)
Feature vector for the heads: [u, f_purpose, f_mechanism, f_evaluation, f_domain, scope(4), nov(4), grd(5)].
Time: nov is recomputed against literature < t + Delta; quality heads are refit on data up to t + Delta with
frozen hyper-parameters (update rule, readout T).
"""
import json
import pickle
from pathlib import Path

import numpy as np

from common.embed import Embedder
from common.extract import FACET_KEYS, extract_facets, extractor

HERE = Path(__file__).parent
STATE = HERE / "results" / "phi_state.pkl"


def _unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class PhiModel:
    def __init__(self, index=None):
        self.E = Embedder("qwen3-0.6b", use_prompt=False)
        self.Eid = Embedder("qwen3-0.6b", use_prompt=True)
        self.index = index
        self.mu = None          # facet means, fit on training questions only
        self.heads = {}         # name -> (ridge, scaler)
        self._llm = None
        self.calib_b = None

    # ---------------------------------------------------------------- encoding
    def facets(self, q):
        if self._llm is None:
            self._llm = extractor()
        return extract_facets(self._llm, q)

    def fit_facet_means(self, facet_dicts):
        F = {k: self.E([f.get(k) or "none" for f in facet_dicts]) for k in FACET_KEYS}
        self.mu = {k: _unit(F[k].mean(0)) for k in FACET_KEYS}

    def encode(self, q, R=None, t=None, facets=None, with_novelty=True):
        facets = facets or self.facets(q)
        u = self.E([q])[0]
        phi = {"q": q, "t": t, "u": u, "v": self.Eid([q])[0], "facets": facets}
        fv = self.E([facets.get(k) or "none" for k in FACET_KEYS])
        phi["f"] = fv
        phi["scope"] = np.array([1 - fv[i] @ self.mu[k] for i, k in enumerate(FACET_KEYS)]) if self.mu else np.zeros(4)
        phi["nov"] = self.novelty_coords(phi, t) if (with_novelty and self.index is not None and t) else np.full(4, np.nan)
        phi["grd"], phi["rests_on"] = self.grounding(phi, R)
        return phi

    # ---------------------------------------------------------------- novelty (N)
    def novelty_coords(self, phi, t, k=50, extra=()):
        """extra: synthetic papers {id, date, title, abstract, vec} added to the literature (test S3)."""
        nn = self.index.search_vec(phi["u"], t, k=k)
        ex = [e for e in extra if e["date"] < t]
        if ex:
            nn = sorted(nn + [dict(e, cos=float(e["vec"] @ phi["u"])) for e in ex], key=lambda n: -n["cos"])[:k]
        if len(nn) < 5:
            return np.array([1.0, 1.0, 1.0, 1.0])
        P = np.stack([n["vec"] for n in nn])
        sims = P @ phi["u"]
        nearest = 1 - sims.max()
        # G2 density: x's mean distance to its 10 NN vs each neighbour's mean distance to its 10 NN (within pool)
        kk = min(10, len(nn) - 1)
        dx = 1 - np.sort(sims)[-kk:].mean()
        PP = P @ P.T
        np.fill_diagonal(PP, -np.inf)
        dn = 1 - np.sort(PP, 1)[:, -kk:].mean(1)
        density = float(np.mean(dn < dx))
        # G3 off-span share
        C = P - P.mean(0)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)
        B = Vt[:20]
        d = phi["u"] - P.mean(0)
        offspan = float(1 - np.sum((B @ d) ** 2) / max(1e-12, d @ d))
        # G4 facet coverage (T2 B.4, sentence level, calibrated) -- reuse E2 implementation
        facet_claims = [f"{k}: {phi['facets'].get(k, '')}" for k in FACET_KEYS if phi["facets"].get(k)]
        import sys
        sys.path.insert(0, str(HERE.parent / "E2"))
        import novelty as nv  # the 5 nearest earlier papers, sentence level (cost: ~35 sentences per question)
        papers = [{"id": n["id"], "date": n["date"], "title": n["title"], "abstract": n["abstract"]} for n in nn[:5]]
        fac = nv.score_papers(phi["q"], facet_claims or [phi["q"]], papers, t)["yours"]
        return np.array([nearest, density, offspan, fac], dtype=float)

    # ---------------------------------------------------------------- groundedness (G)
    def grounding(self, phi, R):
        if not R:
            return np.array([np.nan] * 5), []
        Rv = self.E([f"{r['title']}. {r.get('abstract') or ''}" for r in R])
        c = Rv.mean(0)
        delta = phi["u"] - c
        if len(R) >= 2:
            D = (Rv[1:] - Rv[0]).T
            Qd, _ = np.linalg.qr(D)
            inplane = float(np.sum((Qd.T @ delta) ** 2) / max(1e-12, delta @ delta))
        else:
            inplane = 0.0
        offnorm = float(np.sqrt(max(0.0, (delta @ delta) * (1 - inplane))))
        sims = Rv @ phi["u"]
        top3 = float(np.sort(sims)[-3:].mean())
        copy = float(sims.max())
        F = phi["f"] @ Rv.T                                         # facet x input
        covered = float(np.clip((F.max(1) - 0.5) / 0.5, 0, 1).mean())   # 'addressed by inputs' proxy
        rests = [{"facet": k, "input": R[int(F[i].argmax())]["title"], "cos": float(F[i].max())}
                 for i, k in enumerate(FACET_KEYS)]
        return np.array([inplane, offnorm, top3, covered, copy]), rests

    # ---------------------------------------------------------------- features + quality (Q)
    def features(self, phi):
        nov = np.nan_to_num(phi["nov"], nan=0.5)
        grd = np.nan_to_num(phi["grd"], nan=0.0)
        return np.concatenate([phi["u"], phi["f"].ravel(), phi["scope"], nov, grd])

    def fit_head(self, name, phis, y, phis_sel=None, y_sel=None):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        X = np.stack([self.features(p) for p in phis])
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        alphas = np.logspace(0, 5, 11)
        if phis_sel is not None:
            Xv = sc.transform(np.stack([self.features(p) for p in phis_sel]))
            best = max(alphas, key=lambda a: np.corrcoef(Ridge(alpha=a).fit(Xs, y).predict(Xv), y_sel)[0, 1])
        else:
            best = 1e3
        self.heads[name] = (Ridge(alpha=best).fit(Xs, y), sc, float(best))

    def quality(self, phi):
        x = self.features(phi)[None]
        return {n: float(m.predict(sc.transform(x))[0]) for n, (m, sc, _) in self.heads.items()}

    # ---------------------------------------------------------------- identity and scope (I)
    def same_question(self, p1, p2, use_cross=True, tau=0.80):
        """P(same question). Identity-head cosine pre-filters; the cross-encoder decides (T1 A.2)."""
        s = float(p1["v"] @ p2["v"])
        if not use_cross:
            return s
        if s < tau - 0.2:
            return 0.0
        from common.rerank import same_question_scores
        return float(same_question_scores([(p1["q"], p2["q"])])[0])

    def broader(self, p1, p2):
        """Signed scope comparison: > 0 means p1 is broader than p2 (summed per-facet specificity)."""
        return float(p2["scope"].sum() - p1["scope"].sum())

    # ---------------------------------------------------------------- diversity (D)
    def diversity(self, phis):
        from common.diversity import diversity
        return diversity([p["q"] for p in phis], facets=True, facet_answers=[p["facets"] for p in phis])

    # ---------------------------------------------------------------- time (T)
    def update(self, phi, t_new):
        """Phi at t -> Phi at t_new: novelty recomputed with literature < t_new, plus later matches in (t, t_new]."""
        new = dict(phi)
        new["t"] = t_new
        new["nov"] = self.novelty_coords(phi, t_new)
        later = self.index.search_vec(phi["u"], t_new, k=20, t_after=phi["t"], min_cos=0.85)
        new["later_close_papers"] = [(m["id"], m["date"], m["cos"]) for m in later]
        new["delta_nov"] = new["nov"] - phi["nov"]
        return new

    def readouts(self, phi):
        return {"novelty": dict(zip(["nearest", "density", "offspan", "facet"], map(float, phi["nov"]))),
                "grounded": dict(zip(["inplane_share", "offplane_norm", "top3_cos", "facet_covered_by_R",
                                      "copy_score"], map(float, phi["grd"]))),
                "rests_on": phi.get("rests_on"),
                "scope": dict(zip(FACET_KEYS, map(float, phi["scope"]))),
                "quality": self.quality(phi) if self.heads else {}}

    # ---------------------------------------------------------------- persistence
    def save(self):
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE, "wb") as f:
            pickle.dump({"mu": self.mu, "heads": self.heads}, f)

    def load(self):
        with open(STATE, "rb") as f:
            s = pickle.load(f)
        self.mu, self.heads = s["mu"], s["heads"]
        return self
