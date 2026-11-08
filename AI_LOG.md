# AI log

Every substantive use of an AI assistant, as rule R1 requires.

| Date | Tool | Task | What I checked |
|---|---|---|---|
| 2026-10/11 | another AI assistant (`<name it>`) | First draft of the Theory answers (answer sheets and working) | Every number re-derived by script (`theory/check_numbers.py`); proofs read line by line; three problems found (below) |
| 2026-11-07 | Claude Code (Claude Opus 5.5) | Planning: read the paper, proposed choosing E1/E2/E4 and the repo layout | Choice of experiments checked against the core battery table in M.C |
| 2026-11-07 | Claude Code | Rewrote the Theory answers with the corrections below; wrote `check_numbers.py` and the PDF build | Ran the checks (all 32 values pass); read both PDFs |
| 2026-11-07 | Claude Code | Wrote the code in `common/`, `E1/`, `E2/`, `E4/`, `E6/` and `main/` (Φ, readouts, baselines, battery, runner) | Smoke tests on toy data: E1 validator on the paper's example item; E2 measures on toy pools (time cut-off, concentration, copy); E4 evalkit on synthetic reviews; Φ end to end on a toy index (all readouts, update rule, S3) |
| 2026-11-07/08 | Claude Code | Drafted README, NOTEBOOK and PREREG, including the predictions (drafted by Claude, read and accepted by me before the commit) | Removed the E4 ceiling prediction because the ceiling had already been computed |
| 2026-11-08 | Claude Code | Found public data mirrors, wrote `data/convert_hf.py`, ran the E4 ceiling and the preparation steps | Sources listed in convert_hf.py; deviations in NOTEBOOK.md |
| | | **E1 items: written by hand, no AI** | |

## AI errors caught
1. **T5 B.1 (other assistant's draft).** The "far" question was a Mars-microbe question given prior π = 0.99
   ("settled"), but the existing reports were about contrastive learning, so nothing in them could settle it.
   The example was internally inconsistent. It was replaced by a question whose measurement does not depend on
   H (a = b), which gives v = 0 by construction (`check_numbers.py` verifies v = 0).
2. **Theory draft claimed "all values checked by script"**, but no script existed. One was written, and it
   confirmed every value.
3. **T2 A.1 grouping (draft)** put measure 7 (later papers do something similar) and 10 (uptake) in one group,
   though they answer different questions: "ahead of its time?" versus "did others build on it?". Split into G6/G7.
4. **Claude, E2 `novelty.py` first version:** the calibration level b was recomputed from each pool. Planting a
   claim could then raise b and so *raise* novelty, which contradicts the monotonicity proof in T2 B.4. Caught
   while reviewing the code against the proof; b is now frozen in `calib.json`.
5. **Claude, `PhiModel.same_question` first version** returned a rescaled cosine on one branch that was not a
   probability, so it was not comparable with the cross-encoder output. Rewritten to pre-filter on cosine and let
   the cross-encoder decide.
