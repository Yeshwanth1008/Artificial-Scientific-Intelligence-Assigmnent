# E6 — status

**Generations produced, evaluation not run.** On Kaggle (2× T4), 120 literature input sets (references of arXiv papers
posted from March 2025) and 35 data input sets (HypoGeniC tasks, 5 example draws each) were built, and all 7 arms
produced their K = 6 hypotheses under the 1200-token cap: 6,510 generations in total. The evaluation
(`python E6/run_e6.py evaluate`) was not reached before the deadline, so no E6 result is reported.

Earlier status: The seven arms (D, QH, QH-sel, QH-rand, HQ, D-empty, D-shuffled), the budget rule (K = 6, M = 3,
1200 generated tokens per input set and arm), both testbeds and the six metrics are implemented in
`E6/run_e6.py`, and the selection rule is pre-registered.

The run needs about 100 input sets × 7 arms × up to 1200 tokens, roughly 840k generated tokens. At the measured
1.6–5.3 tokens/s on the available CPU that is several days, so no E6 numbers are reported. No result is estimated
or simulated.

To run on a GPU: `python E6/run_e6.py build_lit && python E6/run_e6.py generate && python E6/run_e6.py evaluate`.
