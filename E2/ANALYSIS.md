# E2: status and results

**Done on Kaggle (preparation):** pools for cs.CL, cs.CV and cs.RO (1000 papers from 2023, 1000 from 2021, 100 focal
papers from Jan–Mar 2024), 4–10 LM claims for 299 of the 300 focal papers, and the frozen calibration level
b = 0.504 for the T2 B.4 measure. **The axiom evaluation (Tasks 1–2) was not run before the deadline.**

## Task 3: human check (reduced, run on the laptop CPU; declared deviation)
Labels: novelty-judge-bench, human-only pointwise, 299 ICLR 2026 ideas (154 validated novel, 145 validated lower
novelty). Deviations from PREREG: the pool is the 3000 ICLR 2024 + 2025 submissions of the fixed E4 subsample (all
dated before the ICLR 2026 deadline) instead of 5000 arXiv papers. N and the T2 B.4 measure were not computed
(they need LM claims for the 299 ideas). Reproduce with `python E2/human_check_local.py`.

| Score | AUROC [95 % CI] | Minus quality-only [95 % CI] |
|---|---|---|
| Quality-only baseline (ridge on abstract embeddings, trained on ICLR 2024 ratings) | **0.652** [0.590, 0.712] | — |
| Y = 1 − max cosine to an earlier submission | 0.505 [0.435, 0.569] | **−0.147** [−0.252, −0.052] |
| Relative neighbourhood density | 0.503 [0.438, 0.568] | **−0.149** [−0.246, −0.047] |

**Against the prediction.** PREREG predicted that the quality-only baseline would score high and that no novelty
measure would beat it by more than 0.05. **Confirmed, and more strongly than predicted:** both distance-based novelty
measures are at chance (≈ 0.50), and both are significantly *below* the quality baseline.

**Interpretation (T2 C.3).** These labels contrast accepted, top-decile papers that reviewers praised for
originality with rejected, bottom-decile papers they faulted for it. Novelty and quality are confounded, so a
score that knows only quality, and nothing about novelty, separates the classes (0.65). Meanwhile, "how far is
this from the nearest earlier paper" (T2 group G1) and "how crowded is its neighbourhood" (G2) carry no
information about this kind of reviewer novelty (G5). This is the T2 A.2 point measured: distance-from-prior-work
and reviewer novelty are different questions.

## Task 4: what it means when a measure passes the axioms but fails the human check (or the reverse)
Passing the axioms shows a measure moves the right way when the literature is manipulated in a known way. It
answers its own question correctly (for Y, "how close is the nearest earlier text?"). Failing the human check
then means that question is not the one reviewers answer when they call a paper novel: reviewers judge whether the
contribution is new *in kind*, and their judgement is entangled with quality. The reverse, failing the axioms but
passing the human check, would be a warning sign: the measure is probably tracking quality, style or memory rather
than novelty, as the quality-only baseline does here. That is why the human check has to be reported against a
quality-only baseline and, ideally, on a contrast matched for quality.
