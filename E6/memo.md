# E6 — status

**Generations produced, evaluation not run.** On Kaggle (2× T4), 120 literature input sets (references of arXiv papers
posted from March 2025) and 35 data input sets (HypoGeniC tasks, 5 example draws each) were built, and all 7 arms
produced their K = 6 hypotheses under the 1200-token cap: 6,510 generations in total. The evaluation
(`python E6/run_e6.py evaluate`) was not reached before the deadline, so no E6 result is reported.

Implemented: the seven arms (D, QH, QH-sel, QH-rand, HQ, D-empty, D-shuffled), the budget rule (K = 6, M = 3,
1200 generated tokens per input set and arm), both testbeds and the six metrics are implemented in
`E6/run_e6.py`, and the selection rule is pre-registered.

To finish: `python E6/run_e6.py evaluate` (the generations are in the Kaggle output). No result is estimated or simulated.
