# AI log

Every substantive use of an AI assistant (rule R1), with what I checked.

## Uses

| Tool | Used for | How I checked it |
|---|---|---|
| `<name the other AI>` | First draft of the Theory answers (answer sheets and working for T1–T5) | Re-derived every number with a script; read every proof; found 3 errors (below) |
| Claude Code (Claude Opus 5.5) | Reading the paper with me and planning: choosing E1/E2/E4 (+E6) and the repo layout | Checked that the three experiments cover the core tests of the Main Problem battery (I1, S1, A1, H3, O1) |
| Claude Code | Corrected and rewrote the Theory answers; wrote `theory/check_numbers.py` and the PDF build | All 32 numerical answers pass the script; read both PDFs |
| Claude Code | Wrote the code: shared library (`common/`), E1 checker and evaluation, E2 novelty measures and pool manipulations, E4 evaluation kit, E6 pipeline arms, and the Main Problem (Φ, readouts, baselines, test battery, runner) | Smoke tests on toy data: E1 checker on the paper's example; E2 time cut-off, concentration and copy on toy pools; E4 kit on synthetic reviews; Φ end to end on a toy index |
| Claude Code | Drafted README, NOTEBOOK and PREREG, including the predictions | I read and accepted the predictions before committing. The E4 ceiling prediction was removed because the ceiling had already been computed |
| Claude Code | Found public data mirrors (ICLR 2024–2026 reviews, novelty-judge-bench, arXiv snapshot), wrote `data/convert_hf.py`, measured CPU speed, ran the E4 ceiling and the preparation steps | Sources are listed in `convert_hf.py`; every deviation (rating instead of contribution, topic clusters, smaller local samples) is in NOTEBOOK.md |
| Claude Code + fixed prompt `EXTRACT_RQ` | E1 workbook: selected 75 real 2025 arXiv papers and extracted question A with the one fixed prompt | Used only to choose the papers |
| Claude Code | **Wrote all 60 E1 items (every field, A included) at my request**, instead of writing them by hand as Task 1 asks (`E1/build_contrast.py`) | Declared deviation: E1 is partly circular (an LM wrote the items that test LMs). Checker passes; no human 24-hour re-check |

## AI errors caught
1. **T5 B.1 (other AI).** The "far" question was given prior π = 0.99 ("settled"), but the reports were about
   contrastive learning, so nothing in them could settle it. Replaced with a question whose measurement does not
   depend on H (a = b, so v = 0).
2. **Theory draft (other AI)** said all numbers were "checked by script", but no script existed. I wrote
   `check_numbers.py`; all values pass.
3. **T2 A.1 (other AI)** put measures 7 (later papers do something similar) and 10 (uptake) in one group, though
   they answer different questions. Split them into G6 and G7.
4. **E2 `novelty.py` (Claude).** The first version recalibrated the background cosine for each pool, so adding a
   paper could *raise* novelty. That contradicts the monotonicity proof in T2 B.4. The calibration is now
   computed once and frozen.
5. **`PhiModel.same_question` (Claude).** One branch returned a rescaled cosine that was not a probability. It was
   rewritten so the cross-encoder makes the decision.
6. **Process (Claude).** Two test-set evaluations were started before PREREG.md was committed. They were stopped
   before writing any result, and this is recorded in NOTEBOOK.md.
