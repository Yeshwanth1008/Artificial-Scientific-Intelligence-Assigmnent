# Notebook

Dated log of decisions. Every abandoned idea gets one line saying why it was dropped.

* 2026-11-07 — Chose E1, E2 and E4 (plus E6). They feed the core battery directly: I1/S1 come from E1, A1 from E2,
  and H3/O1 from E4. Dropped E3, because 200 hand labels plus two judges only feed the extended O3 test. Dropped E5,
  because its measures are rebuilt inside `common/diversity.py` for the D readout anyway.
* 2026-11-07 — Dropped LaTeX for the theory answers: no TeX installation on this machine. The answers are
  Markdown with KaTeX maths, printed to PDF by headless Edge (`theory/make_pdf.py`).
* 2026-11-07 — Dropped anonymous OpenReview scraping: the API returns a challenge page to scripted anonymous
  requests. We use the official `openreview-py` client with a login instead.
* 2026-11-07 — Dropped per-pool calibration of the background cosine b in the E2 `yours` measure. It breaks the
  monotonicity proved in T2 B.4 (adding papers can raise b and so raise novelty). b is now frozen once on the
  2021 pools.
* 2026-11-07 — Dropped connected components and cliques for grouping "same question" decisions (T1 A.2). They are
  replaced by leader clustering on cross-encoder decisions (`common/diversity.py`).
* 2026-11-07 — Dropped whole-paper vectors for claim coverage: a claim is diluted inside a whole-abstract
  embedding. The `yours` measure matches claims against sentences.
* 2026-11-07 — Considered hyperbolic or box embeddings for scope (T1 C.2/C.3). Not built: training boxes needs
  scope-labelled pairs we do not have at scale. Scope is per-facet specificity relative to the training facet
  mean, which is cheap and testable on E1's length-flipped items.
* 2026-11-07 — Considered a learned identity projection head. Not trained (only ~60 hand items, all reserved
  for testing). The identity head is the same encoder under a same-question instruction prompt, with the
  cross-encoder deciding.
* 2026-11-07 — Subsampling M data (3000/1500/3000 ICLR, 3000/1000/1500 arXiv): the full sets at ~1 s/question
  exceed the free-GPU budget. Sizes are fixed in `main/configs/default.json` before hashing.
* 2026-11-08 — Data sourced from public Hugging Face mirrors, not from logged-in APIs (`data/convert_hf.py`, sources in
  its docstring): ICLR 2024/2025 (smallari/openreview-iclr-peer-reviews), ICLR 2026 (davidheineman/iclr-2026),
  novelty-judge-bench (allenai, human-only pointwise, 299 ideas), arXiv metadata snapshot of 2026-10-27
  (librarian-bots; 609,211 cs.LG/CL/CV/AI/RO + stat.ML papers since 2018).
* 2026-11-08 — **Deviation:** the ICLR mirrors have no contribution or soundness sub-scores, so the E4/M quality
  target is the overall **rating**. E4 Task 1 reports the ceiling for rating only.
* 2026-11-08 — **Deviation:** `primary_area` is missing from the mirrors, so topic strata are KMeans(20) clusters of
  TF-IDF abstract vectors, fit on ICLR 2024 only (`E4/run_e4.py:_assign_topics`).
* 2026-11-08 — Local machine is CPU-only (torch CPU build; MX450 unused). Measured: Qwen3-Embedding 2.8 abstracts/s,
  Qwen2.5-1.5B 1.6 tok/s, Qwen2.5-0.5B 5.3 tok/s. **Deviation for local runs:** extraction LM is Qwen2.5-0.5B-Instruct
  (same family, small tier). E4 subsample: 2000/1000/2000 papers (2024/2025/2026). E2: 20 focal papers per field
  (paper asks for 100). E2 human check: claims = abstract sentences, pool = 5000 random arXiv papers before
  2025-09-15. The full-scale runs use the same commands on a GPU.
* 2026-11-08 — **Process error, corrected:** I started E4 Task 2 (scored on ICLR 2026) and the E2 human check before
  PREREG.md was committed. Both were stopped before writing any result (E4/results/ and E2/results/ contained no
  evaluation output). The E4 ceiling (reviewer-vs-reviewer agreement on 2026, no model involved) had already been
  computed: 0.649. It is reported, and flagged as computed before pre-registration.
* 2026-11-08 — Noted for E2 Task 3: novelty-judge-bench labels are built from accepted top-decile vs rejected
  bottom-decile papers, which is exactly the novelty/quality confound of T2 C.3. Prediction: the quality-only baseline
  will score a high AUROC.
* 2026-11-08 — **Deviation (E1):** the 60 E1 items (h001–h060) were written by Claude, not by hand, at my request,
  and this is declared to the organisers. Consequence: the test is partly circular, and the human 24-hour re-check
  and second reader were not done. Every E1 result is reported with this caveat.
