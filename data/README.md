# data/

No licensed data is stored here (rule R7). Downloads go to `data/cache/` (git-ignored).

| File expected | Source | Used by |
|---|---|---|
| `cache/arxiv-metadata-oai-snapshot.json` | Kaggle `Cornell-University/arxiv` | E2, E6, M |
| `cache/iclr{2024,2025,2026}.jsonl` | built by `common/data_sources.load_iclr` (OpenReview API v2, login) | E4, M |
| `novelty_judge_bench.jsonl` `{id,title,abstract,date,label,venue}` | novelty-judge-bench [R2] | E2 Task 3, M H2 |
| `ideation_arena.jsonl` `{battle_id,idea_a,idea_b,system_a,system_b,winner}` | Ideation Arena [R3] | M H1 |
| `novgauge_pairs.jsonl` `{text_a,text_b,facet,overlap}` | NovGauge [R12] | M H4 |
| `hypobench/<task>/{train,test}.jsonl` `{text,label}` (+ `planted.txt`) | HypoBench [R44] | E6 data testbed |

The three benchmark files must be converted to these schemas from their releases; the converters depend on
the release format and are added once the files are downloaded.
