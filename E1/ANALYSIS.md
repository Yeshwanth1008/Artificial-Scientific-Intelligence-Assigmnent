# E1: results and comparison with predictions (Task 3)

**Caveat first.** The 60 items were written by Claude, not by hand (a declared deviation; see the PREREG addendum).
The paraphrases B and C̃ were written to share at most 30 % of A's words, and a language model writing such
paraphrases may push them further from A's wording than a person would. That could make the overlap effects
below look stronger than on a hand-made set.

Setup: 60 items (12 per near-miss type, 12 length-flipped), four sentence embedders (Qwen3-Embedding-0.6B,
SPECTER2 for scientific documents, all-mpnet-base-v2, bge-small-en-v1.5) and one instructed cross-encoder
(Qwen3-Reranker-0.6B). 95 % intervals resample items. Full numbers are in `E1/results.json`; reproduce with
`python E1/eval_contrast.py --part <model>` for each model, then `--merge`.

## Task 2: results (share of items; lower is better except scope order)

| Model | Near-miss error d(A,C) < d(A,B) | Overlap shortcut d(A,C) < d(A,B_min) | Low-overlap shortcut d(A,C̃) < d(A,B) | Scope order (all / length-flipped) |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B | **1.00** [1.00, 1.00] | 0.32 [0.20, 0.43] | 0.22 [0.12, 0.33] | 0.48 / 0.58 |
| SPECTER2 (scientific) | **1.00** [1.00, 1.00] | 0.52 [0.38, 0.65] | 0.47 [0.35, 0.58] | 0.42 / 0.25 |
| all-mpnet-base-v2 | **1.00** [1.00, 1.00] | 0.35 [0.23, 0.48] | 0.23 [0.13, 0.35] | 0.35 / 0.42 |
| bge-small-en-v1.5 | **1.00** [1.00, 1.00] | 0.47 [0.35, 0.58] | 0.38 [0.27, 0.50] | 0.18 / 0.42 |
| Cross-encoder (Qwen3-Reranker) | **0.48** [0.35, 0.60] | **0.02** [0.00, 0.05] | 0.22 [0.12, 0.33] | — |
| Length-only scope baseline | | | | 0.55 / 0.00 |

Per near-miss type, the overlap shortcut is lowest for **goal swaps** (0.00–0.08 for every embedder) and highest for
**constraint flips, inquiry changes and direction flips** (up to 0.67). Changing the goal word (accuracy → speed)
moves an embedding a lot. Flipping "with"/"without", "how"/"why" or "improve"/"harm" barely moves it.

## Task 3: predictions against results

| Prediction (PREREG.md) | Result | Verdict |
|---|---|---|
| Near-miss error about 0.55 (Qwen3), 0.70 (SPECTER2), 0.65 (MPNet, BGE) | 1.00 for all four | **Wrong, and badly.** I expected embedders to fall for word overlap often; they fall for it every time |
| Cross-encoder much better, about 0.25 | 0.48 [0.35, 0.60] | **Direction right, size wrong.** It halves the error (1.00 → 0.48) but is far from 0.25. Per type: constraint flips 0.00, direction flips 0.50, role swaps 0.58, **inquiry changes 0.92** |
| Role swaps worst (above 0.8) | Every type is at 1.00, so role swaps are not special | Wrong in its premise: there is no ranking left to observe |
| Scope order on length-flipped items below 0.40 for every embedder | 0.25–0.58 | Mixed: SPECTER2 is below, Qwen3 and the others are above. Small samples (12 items) give wide intervals |

**What this teaches about these models.**
1. **To a sentence embedder, "same question" means "same words".** A one-word near-miss is always closer than a
   faithful paraphrase in other words. This is the T1 B.2 argument made concrete: mean-pooled vectors are
   dominated by content tokens, and contrastive training almost never shows role-swapped or flipped negatives.
2. **The scientific embedder is the most overlap-driven.** SPECTER2 has the highest overlap and low-overlap shortcut
   rates. Trained on citation links, it learns *topic*, which is exactly what a near-miss keeps.
3. **Goal words carry weight, function words do not.** "Accuracy" → "speed" moves the vector, while "with" →
   "without" or "improve" → "harm" barely does, even though they reverse the question.
4. **A cross-encoder reads some meaning, but not the kind of question.** It almost never confuses a one-word
   paraphrase with a near-miss (overlap shortcut 0.02), and it never misses a constraint flip (0.00). But it
   treats "how does X…" and "why does X…" as the same question 92 % of the time. Inquiry changes are its blind spot,
   so the same-question decision in Φ cannot rely on this cross-encoder for that type.
5. **Scope is not read reliably from a single vector.** On all items, counting words (0.55) orders
   broader / A / narrower better than any embedder's distance from the mean direction (0.18–0.48). The length rule
   gets 0 % on the length-flipped items by construction, while the embedders get 0.25–0.58 there, so the embedders
   are not simply tracking length either. They carry a weak scope signal that is noisier than length on ordinary
   items. That supports T1 C.1 and C.3: scope needs its own component, and it must be tested on length-flipped
   items.

## Task 4: deduplication
All 420 texts were pooled. For each A, its 10 nearest texts (Qwen3-Embedding) were judged by the cross-encoder at
the fixed threshold 0.5. The comparison is the best cosine threshold chosen *with hindsight* on the same data.

| Decision rule | Precision | Recall (finds B and B_min) | Near-misses wrongly accepted (C, C̃) |
|---|---|---|---|
| Retrieve top-10, then cross-encoder ≥ 0.5 | 0.28 | **0.98** | **93 %** |
| Best cosine threshold, chosen with hindsight (0.931) | 0.51 | 0.50 | 44 % |

**Why they differ.** The cross-encoder *ranks* well (in Task 2 it scores the paraphrase above the near-miss on about
half the items, and almost always above the minimal edit), but its *absolute* scores are badly calibrated. At 0.5
it says "same question" to nearly every same-topic text, including 93 % of the near-misses. So recall is almost
perfect and precision collapses. The hindsight cosine threshold looks more balanced only because it was tuned on the
answers. Even so, it still accepts 44 % of the near-misses and misses half the true paraphrases, because cosine
cannot separate them (Task 2: near-miss error 1.00).

**Consequence for the Main Problem.** A fixed 0.5 cut on this cross-encoder is not a usable same-question decision.
The threshold has to be calibrated on held-out labelled pairs. Better still, the decision should be *relative*:
"is q′ closer to q than q's own paraphrases are?". The leader clustering in `common/diversity.py` and
`PhiModel.same_question` should take a calibrated threshold, which is listed as a limitation in `main/REPORT.md`.

