# Research run 002

The executable recipe is `pretrain_20b.json`: fresh initialization, 24 layers, 768 hidden width, 175,611,648 parameters, tied embeddings, and 2,048-token context. The original 32,000-token OpenSML tokenizer is reused explicitly; model weights and optimizer state are new.

| File | Purpose |
| --- | --- |
| `pretrain_20b.json` | Five-Mac 20B-token training recipe |
| `corpus_20b.json` | Pinned educational prose/Python sources and token shares |
| `sample_prompts.json` | Live qualitative prompts; editable during training |
| `pretrain_20b_baseline.json` | Optional original 20-layer geometry for comparison |
| `research_plan.json` | Summary of selected settings |

See [run instructions](../docs/RESEARCH_PLAN.md). Training has not started.
