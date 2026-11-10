# ASI Lab selection round — the space of research questions

Submission repository: Theory (T1–T5), Experiments E1, E2, E4 and E6, and the Main Problem M.

**Compute tier: small** (rule R5). Three open models, no paid API:
* encoder **Qwen3-Embedding-0.6B**: every vector (similarity, novelty, the Φ representation);
* cross-encoder **Qwen3-Reranker-0.6B**: "same question?" judgements (identity, deduplication, diversity);
* language model **Qwen2.5-1.5B-Instruct**: extraction of questions, cards and claims, and E6 generation.
  **Qwen2.5-0.5B-Instruct** was used for the runs on the CPU laptop, for speed (declared in NOTEBOOK.md).

**Where things ran.** On a CPU-only laptop: the Theory checks, E1 (all tasks), the E4 ceiling and the E4 abstract-view
probe. On a free Kaggle notebook with 2× T4 GPUs (`kaggle/run_all.py`): E2, E4 (question cards, three views,
reading state), E6 and the Main Problem.

**Assumed LM training cutoff: 2024-12-31** (rule R4). Qwen2.5 was released in September 2024; we take a later
date to be safe. Every claim that rests on the language model is tested on papers or reviews dated after it:
E4 tests on ICLR 2026, E6 on papers from March 2025, and the Main Problem on items from July 2025. This guards
against the model *remembering* outcomes rather than *judging* them (Theory T3 E).

## Status at submission
| Part | Status |
|---|---|
| Theory T1–T5 | **complete**: `theory/working.pdf`, `theory/answer_sheets.pdf`; all 32 numbers verified by `theory/check_numbers.py` |
| E1 | **complete** (Tasks 1–4) on 60 items written by Claude (declared deviation): `E1/ANALYSIS.md`, `E1/results.json` |
| E2 | Pools, LM claims and calibration produced on Kaggle. **Task 3 results**: quality-only AUROC 0.652, Y 0.505, density 0.503, both significantly below quality (`E2/ANALYSIS.md`). Task 4 written. The axiom evaluation (Tasks 1–2) was not run |
| E4 | **Task 1** (ceiling 0.649) and **Task 2, abstract view** (embedding 0.548, TF-IDF 0.553, length 0.504) are complete. Question cards for 5000 papers were produced on Kaggle; views (i)/(ii), Task 3 and Task 4 were not evaluated |
| E6 | 120 literature + 35 data input sets built and **6,510 generations** produced for all 7 arms on Kaggle. **Not evaluated** before the deadline (`E6/memo.md`) |
| Main Problem | Φ, readouts, baselines and the full battery are implemented and smoke-tested; 5000 ICLR items built on Kaggle. **The battery was not run** (`main/REPORT.md` lists every test with its reason) |

No number in this repository is simulated. Every test that was not run is listed with its reason. The Kaggle pipeline
(`kaggle/run_all.py`) resumes from the saved outputs and finishes all remaining evaluations in one more GPU session.

## Layout
```
theory/      working.pdf, answer_sheets.pdf (+ Markdown sources, check_numbers.py, make_pdf.py)
common/      encoder, cross-encoder, LM wrapper, fixed prompts, data loaders, statistics, diversity
E1/          contrast.jsonl (hand-written), eval_contrast.py
E2/          pool manifest, run_e2.py (manipulations), novelty.py, results/*.json
E4/          cards.jsonl, evalkit.py, run_e4.py, results/*.json
E6/          testbeds, generations.jsonl, run_e6.py, results/, memo.md
main/        phi.py (the representation), data.py, baselines.py, battery.py, run.py, configs/, results/, REPORT.md
data/        schemas and download instructions only (no licensed data is redistributed)
```

## Setup
```bash
pip install -r requirements.txt
export OPENREVIEW_USERNAME=...  OPENREVIEW_PASSWORD=...     # ICLR reviews (API v2)
export S2_API_KEY=...                                        # Semantic Scholar
# arXiv: download Kaggle "Cornell-University/arxiv" and point to it:
export ARXIV_SNAPSHOT=/path/arxiv-metadata-oai-snapshot.json
```
Downloaded data and every model output are cached in `data/cache/`, which git ignores, so re-running is free.

## Reproduce every table (one command each)

| Table | Command |
|---|---|
| Theory: all numbers | `python theory/check_numbers.py` |
| Theory: PDFs | `python theory/make_pdf.py` |
| E1 constraints | `python E1/eval_contrast.py --check` |
| E1 Tasks 2 + 4 | `python E1/eval_contrast.py` |
| E2 pools / claims / calibration | `python E2/run_e2.py pools && python E2/run_e2.py claims && python E2/run_e2.py calibrate` |
| E2 Task 2 (axiom pass rates) | `python E2/run_e2.py axioms` |
| E2 Task 3 (human check) | `python E2/run_e2.py human` |
| E4 data + cards | `python E4/run_e4.py data && python E4/run_e4.py cards` |
| E4 Task 1 (ceiling) | `python E4/run_e4.py ceiling` |
| E4 Task 2 (three views) | `python E4/run_e4.py views` |
| E4 Task 3 (reading state) | `python E4/run_e4.py reading` |
| E4 Task 4 (summary) | `python E4/run_e4.py summary` |
| E6 testbeds | `python E6/run_e6.py build_lit && python E6/run_e6.py build_data` |
| E6 generations | `python E6/run_e6.py generate` |
| E6 Task 2 | `python E6/run_e6.py evaluate` |
| M data | `python -m main.data index && python -m main.data iclr && python -m main.data uptake && python -m main.data groups` |
| M test-list hashes (for PREREG) | `python -m main.run hashes` |
| M fit | `python -m main.run fit` |
| M Table 1 (core battery) | `python -m main.run test core` |
| M Table 2 (extended battery) | `python -m main.run test extended` |
| M timing | `python -m main.run timing` |
