# Pre-registration

This file is committed before any test-set evaluation; the git history is the timestamp.
Deviations go in the **Deviations** section at the end, dated, and earlier text is never edited.

> **What this commit covers (2026-11-08).** E2 and E4 are final here and are run only after this commit.
> E1, E6 and the Main Problem are stated here, but their test sets do not exist yet. Each gets a dated
> **addendum** at the end of this file (test-list hashes plus any final details) committed before its test run.
> Earlier text is never edited.

## Common
* Encoder Qwen3-Embedding-0.6B; cross-encoder Qwen3-Reranker-0.6B with the same-question instruction in
  `common/prompts.py`; LM Qwen2.5-1.5B-Instruct on GPU, Qwen2.5-0.5B-Instruct in the local CPU runs (extraction at temperature 0;
  generation at 0.7, top-p 0.95, seed 0).
* Assumed LM training cutoff 2024-12-31.
* 95 % bootstrap intervals, 2000 resamples. The unit resampled is the E1 item, the E2 focal paper, the E4/M paper
  (two-way paper bootstrap for pairs) or the E6 input set.
* "Beats" = the paired 95 % interval of (Φ − baseline) lies above the margin **0.00**.

## E1 — identity and scope
* Test set: `E1/contrast.jsonl` (hash: `<sha256 from main.run hashes>`), ≥ 60 hand-written items.
* Metrics: near-miss error, overlap shortcut, low-overlap shortcut and scope order (overall, per type, length-flipped).
* Dedup: top-10 retrieval with Qwen3-Embedding, then the cross-encoder at threshold 0.5 (fixed).
* **Predictions.** I expect plain embedders to fall for word overlap a lot. On roughly half the items or more, they
  will put the near-miss C closer to A than the paraphrase B: Qwen3 about 0.55, SPECTER2 about 0.70 (it is trained
  on citations, so topic is all it sees), MPNet and BGE-small about 0.65. The instructed cross-encoder should do
  much better, around 0.25. Role swaps will be the hardest type (above 0.8 for every embedder), since they change
  no words. On scope, embedders will mostly track length, so on length-flipped items they will get the order
  right less than 40 % of the time (the length-only baseline gets 0 %).

## E2 — novelty axioms
* Pools: cs.CL, cs.CV and cs.RO, each with 1000 papers from 2023 (near), the other field's pool (far) and 1000
  papers from 2021 (older). Focal papers posted Jan–Mar 2024: **20 per field** in the local CPU run (100 on GPU).
  Claims: Qwen2.5-0.5B-Instruct, fixed prompt `EXTRACT_CLAIMS`.
* Human check: novelty-judge-bench human-only pointwise (299 ideas). Claims = abstract sentences. Pool = 5000
  random arXiv cs.LG/CL/CV/AI papers dated 2024-01-01..2025-09-15. Quality-only baseline = ridge on frozen
  abstract embeddings of the E4 ICLR 2024/2025 subsample (test titles excluded).
* Calibration b for `yours` is frozen in `E2/calib.json`, computed on the 2021 pools before any axiom run.
* Correct direction: copy, paraphrased copy and partial coverage lower novelty, and each further 25 % step lowers
  it again; concentrated < distributed; far > near; older > newer.
* **Predictions (pass rates).** Y (nearest-paper distance) should catch an exact copy almost always (~1.0), but
  it barely moves when only some claims are covered, so about 0.5 on the partial steps and on concentration.
  N (claim coverage) should handle partial coverage well (above 0.8) but get concentration backwards, as T2 B.3
  shows (below 0.3). Density should catch copies but be noisy elsewhere (around 0.6). My measure should pass
  partial coverage (above 0.8) and concentration (above 0.7).
  Human check: the labels mix novelty with quality (accepted top-decile vs rejected bottom-decile papers), so the
  quality-only baseline will score high, and no novelty measure will beat it by more than 0.05 AUROC.

## E4 — where quality lives
* Train ICLR 2024, choose alpha on ICLR 2025, test once on ICLR 2026. Target: **mean overall rating** (the public
  mirrors have no contribution sub-score). Fixed random subsample (seed 0): 2000 / 1000 / 2000 papers.
* Strata: KMeans(20) topic clusters of TF-IDF abstracts, fit on 2024. The ceiling gap is fixed at rating 1.0.
* Already computed before this file was committed (reviewer agreement only, no model): ceiling = 0.649.
* **Predictions.** More text should help, so abstract > card > question, with within-topic pair accuracy of about
  0.58 / 0.56 / 0.54. That is small next to the 0.649 ceiling. TF-IDF should be close to the embedder on the
  abstract, and length alone should be a little above 0.5. I expect the card to recover about 40 % of the gap
  between the question and the abstract. The reading-state probe should add almost nothing on 2026 (gain below
  0.01) and more on 2024, a year the model may have seen, which would point to memory rather than judgement.

## E6 — questions first?
* K = 6, M = 3, cap of 1200 generated tokens per input set and arm. Arms D, QH, QH-sel, QH-rand, HQ, D-empty
  and D-shuffled.
* Selection rule for QH-sel: as stated in `E6/run_e6.py: select_questions` (dedup, drop already-asked-by-input,
  rank by the mean of groundedness rank and novelty rank, keep 3).
* Primary metric: literature testbed `recon_hyp_cos` (best cosine to the target hypothesis); data testbed
  `data_best`.
* Analyses run whatever the outcome: every metric (a)–(f) for every arm, paired differences vs D and QH-sel
  vs QH-rand, the controls, and the error analysis on 20 sets.
* **Predictions.** Asking questions first will not change how close the hypotheses get to the real one
  (QH − D within ±0.01), but it will make them more varied (+0.03 mean cosine distance), because each question
  pushes the hypotheses in a different direction. Selection will not beat random choice (interval includes 0):
  with a small model the questions are similar, so there is little to choose between. Writing hypotheses first
  and then the question (HQ) will give the most analytic, "does X affect Y" questions. Both controls (no inputs,
  wrong inputs) will be clearly worse than D, by more than 0.05.

## Main Problem
* Splits: train up to the end of 2023 (ICLR 2024; arXiv 2021–2022 for uptake), select 2024 (ICLR 2025; arXiv
  2023), test from July 2025 (ICLR 2026; arXiv 2025-07-01..2025-10-31). Sizes in `main/configs/default.json`.
* Test-list hashes:
  * `iclr_test`: `<sha256>` (n = `<n>`)
  * `uptake_test`: `<sha256>` (n = `<n>`)
* Controls on every result: empty inputs, shuffled inputs, decoy question.
* Pass criteria: as in the paper's battery table, with margin 0.00 and T1 tolerance 0.02.
* **Hypotheses / predictions.**
  1. I1: the identity and scope readouts will change less under a paraphrase than under a near-miss edit.
     Novelty and quality should too, but by a smaller margin.
  2. S1: with the cross-encoder deciding, Φ will beat the plain embedding on near-miss vs paraphrase by more than
     0.2.
  3. H3/O1: the review head will beat the plain question embedding only slightly (about +0.01), probably not enough
     to clear the interval. The citation head will not beat it. I expect a negative or borderline result here.
  4. T1: training on one more year will change test accuracy by at most 0.02 (pass).
  5. A1: the facet novelty coordinate will pass partial coverage and concentration.
  6. H1: raw accuracy on expert battles will be higher than system-demeaned accuracy, and demeaned will be close to
     chance (about 0.52).

## Deviations
(none yet)

### Addendum 2026-11-08: E1 test set (before any E1 evaluation)
* `E1/contrast.jsonl`: n = 60 items (h001-h060), sha256 of ids `113d813578e17fe9cd42a5d26816b20b67642e1714f0bbb36e0fd00b849da3aa`.
* **Deviation:** the items were written by Claude (an AI assistant), not by hand, at the student's request; this is
  declared in AI_LOG.md and NOTEBOOK.md. This makes E1 partly circular (a language model wrote the items used to
  test language models), so E1 results are reported with that caveat and are not presented as a hand-made test.
* The 24-hour re-check was not done by a human. No second reader.
* Predictions in the E1 section above are unchanged.
