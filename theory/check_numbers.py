"""Numerical checks for every number on the Theory answer sheets (T1-T5).

Run:  python theory/check_numbers.py
Every check asserts; the script prints a table of values and exits non-zero on any mismatch.
"""
import math
import numpy as np
from scipy.stats import norm

out = []


def rec(name, value, expected=None, tol=1e-3):
    if expected is not None:
        assert abs(value - expected) < tol, f"{name}: got {value}, expected {expected}"
    out.append((name, value))


def h(p):
    return 0.0 if p in (0, 1) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# ---------------- T1 ----------------
ang = np.deg2rad([0, 30, 60])
u, v, w = [np.array([math.cos(a), math.sin(a)]) for a in ang]
rec("T1 A.2 cos(u,v)", u @ v, 0.866)
rec("T1 A.2 cos(v,w)", v @ w, 0.866)
rec("T1 A.2 cos(u,w)", u @ w, 0.5)
assert u @ v >= 0.8 and v @ w >= 0.8 and u @ w < 0.8

# T1 C.2 (i): ratio identity on random boxes
rng = np.random.default_rng(0)
for _ in range(1000):
    lo1, lo2 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
    hi1, hi2 = lo1 + rng.uniform(0.5, 2, 3), lo2 + rng.uniform(0.5, 2, 3)
    inter = np.prod(np.clip(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0, None))
    if inter == 0:
        continue
    vA, vB = np.prod(hi1 - lo1), np.prod(hi2 - lo2)
    assert abs((inter / vB) / (inter / vA) - vA / vB) < 1e-9

# T1 C.2 (iii): A=[0,10]x[0,1], B=[9.9,10.9]x[0,1]
vA, vB, vI = 10 * 1, 1 * 1, 0.1 * 1
rec("T1 C.2 Vol(A)/Vol(B)", vA / vB, 10)
rec("T1 C.2 P(B in A)", vI / vB, 0.1)
rec("T1 C.2 P(A in B)", vI / vA, 0.01)

# ---------------- T2 ----------------
for m in range(2, 9):
    c = np.eye(m)
    x = c.sum(0) / math.sqrt(m)

    def Y(P):
        return 1 - max(x @ p for p in P)

    def N(P):
        return 1 - np.mean([max(ci @ p for p in P) for ci in c])

    def M(P):  # our measure (B.4)
        C = np.mean([max(max(ci @ p, 0) for p in P) for ci in c])
        K = max(sum(max(ci @ p, 0) for ci in c) ** 2 for p in P) / m
        return 1 - 0.5 * (C + K)

    prev = None
    for j in range(1, m + 1):
        dist = [c[i] for i in range(j)]
        conc = [c[:j].sum(0) / math.sqrt(j)]
        assert abs(Y(dist) - (1 - 1 / math.sqrt(m))) < 1e-9
        assert abs(N(dist) - (1 - j / m)) < 1e-9
        assert abs(Y(conc) - (1 - math.sqrt(j / m))) < 1e-9
        assert abs(N(conc) - (1 - math.sqrt(j) / m)) < 1e-9
        assert abs(M(dist) - (1 - (j + 1) / (2 * m))) < 1e-9
        assert abs(M(conc) - (1 - (j + math.sqrt(j)) / (2 * m))) < 1e-9
        if j > 1:
            assert Y(conc) < Y(dist)          # Y satisfies concentration
            assert N(conc) > N(dist)          # N violates it
            assert M(conc) < M(dist)          # ours satisfies it
            assert M(dist) < prev             # ours satisfies gradient
        prev = M(dist)
    # monotonicity: adding random papers never increases M
    P = [rng.normal(size=m) for _ in range(3)]
    P = [p / np.linalg.norm(p) for p in P]
    for _ in range(20):
        q = rng.normal(size=m); q /= np.linalg.norm(q)
        assert M(P + [q]) <= M(P) + 1e-12
        P = P + [q]
rec("T2 B.* toy-model formulas, m=2..8", 1.0, 1.0)

r_half = 0.36
rel = 2 * r_half / (1 + r_half)
rec("T2 C.1 reliability (Spearman-Brown)", rel, 0.529)
rec("T2 C.1 max correlation sqrt(rel)", math.sqrt(rel), 0.727)
rpn, rpr, rnr = 0.30, 0.25, 0.70
pc = (rpn - rpr * rnr) / math.sqrt((1 - rpr ** 2) * (1 - rnr ** 2))
rec("T2 C.2 partial correlation", pc, 0.181)

# ---------------- T3 ----------------
# A: simulate S = L + eps
n = 200_000
L1, L2 = rng.normal(size=n), rng.normal(size=n)
longer_is_1 = L1 > L2
expert_picks_1 = np.where(rng.uniform(size=n) < 0.65, longer_is_1, ~longer_is_1)
S1, S2 = L1 + rng.normal(size=n), L2 + rng.normal(size=n)
rec("T3 A simulated agreement (>0.5)", np.mean((S1 > S2) == expert_picks_1))
assert np.mean((S1 > S2) == expert_picks_1) > 0.5

# B
rec("T3 B overall", 0.6 * 0.8 + 0.4 * 0.5, 0.68)
nA, nB = rng.uniform(-0.4, 0.4, n), rng.uniform(-0.4, 0.4, n)
assert np.all(1 + nA > nB)  # cross-system always picks A

# C
pr = 0.3 * 0.10 + 0.7 * 0.004
rec("T3 C(i) P(realised)", pr, 0.0328)
rec("T3 C(i) share already asked", 0.03 / pr, 0.915)
rec("T3 C(ii) G2 realised rate (5% already asked)", 0.05 * 0.10 + 0.95 * 0.004, 0.0088, 1e-5)

# E(ii)
p0, p1 = 0.5, 0.54
za, zb = norm.ppf(0.975), norm.ppf(0.8)
nn = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
rec("T3 E(ii) pairs needed (raw)", nn, 1224.3, 1.0)
rec("T3 E(ii) pairs needed (ceil)", math.ceil(nn), 1225, 0.5)
nn0 = ((za + zb) * math.sqrt(p0 * (1 - p0)) / (p1 - p0)) ** 2
rec("T3 E(ii) pairs needed, null variance only", math.ceil(nn0), 1227, 0.5)
rec("T3 E(ii) half-width n=400", za * math.sqrt(p1 * (1 - p1) / 400), 0.049)

# ---------------- T4 ----------------
d, k = 768, 30
rec("T4 A.1 (k-1)/d", (k - 1) / d, 0.0378)
# Monte-Carlo check of the in-plane share
Q, _ = np.linalg.qr(rng.normal(size=(d, k - 1)))
deltas = rng.normal(size=(4000, d))
share = np.mean(np.sum((deltas @ Q) ** 2, 1)) / np.mean(np.sum(deltas ** 2, 1))
rec("T4 A.1 Monte-Carlo in-plane share", share, 0.0378, 2e-3)

c_obs, r_e, r_f = [math.cos(math.radians(a)) for a in (50, 28, 35)]
ct = c_obs / math.sqrt(r_e * r_f)
rec("T4 B cos(theta_true)", ct, 0.7558)
rec("T4 B theta_true (deg)", math.degrees(math.acos(ct)), 40.9, 0.05)

# ---------------- T5 ----------------
def value(a, b, pi):
    return h(pi * a + (1 - pi) * b) - pi * h(a) - (1 - pi) * h(b)


for name, (a, b, pi), ev, eh in [("Q1", (0.9, 0.1, 0.5), 0.531, 1.0),
                                 ("Q2", (0.5, 0.5, 0.5), 0.0, 1.0),
                                 ("Q3", (0.9, 0.1, 0.99), 0.025, 0.494)]:
    rec(f"T5 A.2 {name} v", value(a, b, pi), ev)
    rec(f"T5 A.2 {name} H(Y)", h(pi * a + (1 - pi) * b), eh)
rec("T5 A.3 Q3 H(H)", h(0.99), 0.081)

# T5 A.1 identity check against direct mutual information
for _ in range(200):
    a, b, pi = rng.uniform(size=3)
    joint = np.array([[pi * a, pi * (1 - a)], [(1 - pi) * b, (1 - pi) * (1 - b)]])
    py = joint.sum(0); ph = joint.sum(1)
    mi = sum(joint[i, j] * math.log2(joint[i, j] / (ph[i] * py[j]))
             for i in range(2) for j in range(2) if joint[i, j] > 0)
    assert abs(mi - value(a, b, pi)) < 1e-9

# T5 B.1 pair (see answers): Q_near open, Q_far with a = b
rec("T5 B.1 Q_near v (pi=.5,a=.9,b=.1)", value(0.9, 0.1, 0.5), 0.531)
rec("T5 B.1 Q_far v (pi=.5,a=.6,b=.6)", value(0.6, 0.6, 0.5), 0.0)

w = max(len(n_) for n_, _ in out)
for n_, v_ in out:
    print(f"{n_:<{w}}  {v_:.4f}")
print(f"\nAll {len(out)} recorded values and all assertions passed.")
