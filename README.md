# ASI Lab selection round — the space of research questions

Submission repository: Theory (T1–T5), Experiments E1, E2, E4 and E6, and the Main Problem M.

**Compute tier: small.** One encoder (Qwen3-Embedding-0.6B), one cross-encoder (Qwen3-Reranker-0.6B) and one
language model (Qwen2.5-1.5B-Instruct; 0.5B in the local runs). Everything reported here ran on a CPU-only laptop
(designed for one free T4 GPU). No paid API was used.
**Assumed LM training cutoff:** 2024-12-31. Qwen2.5 was released in September 2024; we take a later date to be safe.
Every claim that rests on the language model is tested on papers or reviews dated after it: E4 tests on ICLR 2026, E6
on papers from March 2025, and M on items from July 2025.

## Status at submission
| Part | Status |
|---|---|
| Theory T1–T5 | complete: `theory/working.pdf`, `theory/answer_sheets.pdf`; all numbers verified by script |
| E1 | checker and evaluation implemented; hand-writing workbook of 75 real 2025 papers built; items not finished |
| E2 | measures, manipulations and real arXiv pools built; claim extraction not finished on CPU, so no pass rates yet |
| E4 | **Task 1 run on real data** (ceiling 0.649 on ICLR 2026); Task 2 abstract view run on real data (see `E4/results/`) |
| E6 | implemented, not run (compute); see `E6/memo.md` |
| Main Problem | Φ, readouts, baselines and battery implemented and smoke-tested; battery not run (compute); see `main/REPORT.md` |

No number in this repository is simulated. Every test that was not run is listed with its reason.

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

## Order of work (and why)
1. Theory, then E1 items (by hand), then E2/E4 (they produce the novelty and quality tools M reuses).
2. Build M data. Run `python -m main.run select` freely on the 2024 selection year.
3. Freeze every choice, run `hashes`, paste into PREREG.md and **commit PREREG.md**.
4. Only then run `python -m main.run test core` / `extended`, once.
