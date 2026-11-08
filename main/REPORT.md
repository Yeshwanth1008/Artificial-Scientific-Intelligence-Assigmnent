# Main Problem — report

> Results sections are filled from `main/results/test/*.json` after the single test run.

## 1. Design: one representation
Φ(q, R, t) is defined in `main/phi.py`. One encoder produces every vector. Each component and the problem that
motivated it:

| Component | Readout | Motivated by |
|---|---|---|
| u (topic), v (identity prompt) + cross-encoder | I: same question | T1 A.2, B.2: one cosine cannot be both topic and identity; cosine groups chain |
| per-facet specificity (scope[4]) | I: signed scope | T1 C.1: scope must not be read from length; C.3 facet geometry |
| nov_nearest, nov_density, nov_offspan, nov_facet | N: one coordinate per T2 group (G1–G4) | T2 A.1; E2 axioms |
| in-plane share, off-plane norm, top-3 cos, facet coverage, copy, rests-on | G | T4 A: off-plane carries the signal; D: proximity can be gamed |
| q_review, q_uptake (separate heads) | Q: two factors | T2 C, T3, E4; T4 D(iii): separate axes |
| identity classes, facet Vendi, QD count | D | E5 axioms; T1 A.2 |
| novelty recomputed at t+Δ, heads refit | T | Figure M.2 splits |

## 2. Battery: Table 1 (core) and Table 2 (extended)
`python -m main.run test core` / `extended`. Tests that could not run are listed with the reason.

## 3. Results
## 4. Insight: what the space captures and what it misses
## 5. Error analysis with real examples
## 6. Limitations and deviations from PREREG
