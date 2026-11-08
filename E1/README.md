# E1 — hand-made test of identity and scope

* `contrast.jsonl` — one item per line, schema exactly as in the paper. **Written by hand** (see the
  item-writing protocol below). The first line, `h000`, is the worked example from the paper and is
  excluded from scoring.
* `eval_contrast.py --check` — validates every Task-1 constraint (Jaccard, B_min overlap, type counts,
  length-flipped count).
* `eval_contrast.py` — Task 2 (4 embedders + 1 instructed cross-encoder, per near-miss type, 95 % CI
  resampling items) and Task 4 (deduplication with top-10 retrieval + cross-encoder vs best cosine
  threshold with hindsight). Writes `results.json`.

Models: `Qwen3-Embedding-0.6B` (general, instruction), `specter2_base` (scientific documents),
`all-mpnet-base-v2` (general), `bge-small-en-v1.5` (general, small), cross-encoder
`Qwen3-Reranker-0.6B` with the same-question instruction in `common/prompts.py`. Small tier.

## Item-writing protocol
1. Pick a paper posted on arXiv from 2025 onward (cs.LG / cs.CL / cs.CV).
2. Run the fixed extraction prompt (`common/prompts.py: EXTRACT_RQ`) on its title + abstract; edit by
   hand until the question is faithful to the paper → `A`.
3. Write `B` (same question, few shared words), `B_min` (≥75 % of A's words), `C` (one near-miss of the
   assigned type), `C_tilde` (C's meaning, few shared words with A), `narrower`, `broader`. In at least
   10 items make the broader question longer than the narrower one.
4. Run `--check` and fix what it flags.
5. Re-read every item at least 24 hours later; record the date in NOTEBOOK.md.
