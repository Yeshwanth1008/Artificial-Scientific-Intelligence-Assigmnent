# Main Problem — report

**Status at submission.** The representation Φ, its six readouts, the five baselines and the full test battery
are implemented and pass end-to-end smoke tests. The battery has **not** been run on the test data. The machine
available was CPU-only: measured speeds were 2.8 abstracts/s for the encoder and 1.6–5.3 generated tokens/s for the
language model. The Main Problem needs facet extraction and novelty retrieval for several thousand questions, which
is days of compute on this machine. As the rules ask, every test that was not run is listed below with the
reason. No result in this repository is simulated or estimated.

## 1. Design: one representation

Φ(q, R, t) is defined in `main/phi.py`. One encoder (Qwen3-Embedding-0.6B) produces every vector, and every
readout is a head on the same object.

| Component | Readout | Why (what it rests on) |
|---|---|---|
| topic vector u; identity vector v (same encoder, same-question instruction); cross-encoder decides | **I** same question | T1 A.2 / B.2: one cosine cannot serve topic and identity at once, and thresholded cosines chain into giant groups. Pairs are decided by a cross-encoder; groups use leader clustering |
| per-facet specificity (purpose, mechanism, evaluation, domain) relative to the training facet means | **I** signed scope | T1 C.1: a scope signal learned from lengths can satisfy every constraint while knowing nothing. Facets follow T1 C.3 |
| four novelty coordinates against literature dated before t: nearest distance, neighbourhood density, off-span share, facet coverage (T2 B.4 measure) | **N** | T2 A.1: each coordinate answers one question (G1–G4), never two. The facet-coverage measure satisfies both T2 axioms (proved; checked in `theory/check_numbers.py`) |
| in-plane share and off-plane norm of q − centroid(R), top-3 similarity, facet coverage by R, copy score, and the input each facet rests on | **G** | T4 A: 96 % of a random deviation is off-plane, so in-plane notions are noise; T4 D: proximity alone can be gamed, hence the copy score |
| two separate ridge heads: reviewer rating and 12-month author-disjoint citations | **Q** (two factors) | T2 C, T3 and T4 D(iii): keep impact and novelty on separate axes |
| identity classes, per-facet Vendi, quality–diversity count | **D** | E5 anti-gaming axioms |
| novelty recomputed with literature before t + Δ; later close papers listed; heads refit | **T** | Figure M.2 splits |

Splits follow Figure M.2: train to the end of 2023, select on 2024, test from July 2025. Every result carries the
empty-input, shuffled-input and decoy-question controls (`main/context.py: controls_quality`).

## 2. What was run on real data

| Item | Data | Result |
|---|---|---|
| E4 Task 1: reviewer ceiling | ICLR 2026, 19,471 papers (public OpenReview mirror) | Half-panels agree on **64.9 %** of within-topic pairs whose half-1 means differ by > 1 point (SD 0.5 % over 20 splits) |
| E4 Task 2: abstract view | ICLR 2024 train / 2025 select / 2026 test (2000/1000/2000) | see `E4/results/views.json` and `E4/results/summary.json` |
| Theory numbers | — | all 32 values verified |

The ceiling sets the scale for every quality claim. Two reviewer panels on the same paper agree on the order of
only about two pairs in three. So a quality readout at 0.55 already explains about a third of the signal that
reviewers themselves share.

## 3. Battery: status of every test

| Test | Tier | Status | Reason |
|---|---|---|---|
| I1 paraphrase | core | not run | needs the E1 hand-written items, which are not finished |
| I3 length | core | not run | needs Φ encoded on the ICLR test set (GPU budget) |
| S1 near-miss | core | not run | needs the E1 items |
| A1 novelty axioms | core | not run | E2 claim extraction (LM) did not finish on CPU |
| H1 expert battles | core | not run | Ideation Arena data found (HF `yolo1213811/research-ideation-arena-dataset`, 0.8 GB) but not converted |
| H3 review scores | core | not run | needs Φ on the test set; the E4 abstract-view probe is the closest finished result |
| O1 post-cutoff quality | core | not run | needs Φ and 12-month citation counts (Semantic Scholar key) |
| O2 same inputs | core | not run | needs Semantic Scholar references |
| T1 time invariance | core | not run | needs Φ on the test set |
| I2, I4, S2, S3, H2, H4, O3, K1, T2, R1, G, D | extended | not run | same compute limits. S3 behaves as specified on a toy index (an earlier equivalent paper lowers all four N coordinates; a later one changes none) |

## 4. Insight so far
* **Reviewer agreement is low (64.9 %).** Any quality readout has to be judged against this ceiling, not against
  100 %.
* **The public novelty labels are confounded with quality.** novelty-judge-bench builds its "novel" class from
  accepted top-decile papers and its "not novel" class from rejected bottom-decile papers. That is the T2 C.3
  contrast, on which no score can separate novelty from quality. A quality-only baseline is the right comparison,
  and a matched contrast (same rating, different novelty) is needed.
* **The public ICLR mirrors carry no contribution or soundness sub-scores.** "Quality" here is the overall rating,
  which mixes contribution, soundness and presentation.

## 5. How to finish (one command each, on a GPU)
See README.md. In order: E2 (`claims`, `calibrate`, `axioms`, `human`), E4 cards and three views, E6 `generate` /
`evaluate`, then M (`data`, `hashes` → PREREG addendum, `fit`, `test core`, `test extended`).

## 6. Deviations
Listed with dates in NOTEBOOK.md: data from public mirrors, rating instead of contribution, KMeans topic strata,
smaller local samples with a 0.5B extraction model, and one process error (two test runs started before PREREG
was committed, stopped before any result was written).
