# OpenSML-150M-V2 — Initial Technical Report

**Author:** William Zebrowski  
**Status:** Ongoing pretraining experiment; initial report dated October 2, 2026.  
**GitHub:** https://github.com/williamzebrowskI/OpenSML-150M-V2  
**Model repository:** https://huggingface.co/wzebrowski/OpenSML-150M-V2

## 1. Scope and objectives

OpenSML-150M-V2 is the project name. The selected architecture has **175,611,648 parameters (175.61M)**, verified by instantiating the implemented MLX model on CPU. It is pretrained from random initialization on a five-Mac cluster with a target of **20 billion processed training token positions**. It reuses the frozen 32,000-token byte-level BPE tokenizer trained for OpenSML-150M; it does not initialize from that model's weights.

The experiment tests whether additional model depth, a larger token budget and Python exposure improve held-out prediction and downstream capabilities relative to the previous 150.44M base. These changes occur together, so a final comparison cannot isolate the contribution of any individual change. Better reasoning or benchmark performance is an objective, not an established result.

No supervised fine-tuning or preference optimization is configured for this run. The previous release and its post-training results remain separate.

## 2. Model architecture

| Setting | Previous OpenSML base | Selected V2 architecture |
| --- | ---: | ---: |
| Implemented parameters | 150,439,168 | 175,611,648 |
| Transformer layers | 20 | 24 |
| Hidden width | 768 | 768 |
| Query attention heads | 12 | 12 |
| Key/value heads | 4 | 4 |
| Head dimension | 64 | 64 |
| SwiGLU hidden width | 2,048 | 2,048 |
| Vocabulary | 32,000 | 32,000 |
| Context length | 2,048 | 2,048 |
| Pretraining token positions | Approximately 7.8B completed | 20B target |

The decoder uses grouped-query causal attention, rotary position embeddings, RMS normalization including query/key normalization, SwiGLU blocks, bias-free projections and tied token embeddings/output weights. The configured feed-forward settings are `mlp_ratio=4.0` and `mlp_multiple_of=256`; the implementation's SwiGLU sizing yields width 2,048.

The implementation selects MLX fast attention, packed Metal feed-forward computation and a Metal cross-entropy implementation with float32 loss reductions. The repository retains internal `sml_v1` module names for import compatibility; this is inherited implementation code, not initialization from an earlier model.

## 3. Tokenizer and data

The frozen tokenizer is stored at `reference/opensml-150m/tokenizer/bytebpe32k_v1`. Its artifacts and manifest are included. No new tokenizer training has been performed for this experiment. The `reuse-frozen` policy explicitly permits the changed training corpus, while checkpoint compatibility records the tokenizer and corpus fingerprints.

### 3.1 Selected token mixture

| Source | Token share | Target token positions | Selection |
| --- | ---: | ---: | --- |
| SmolLM corpus / FineWeb-Edu deduplicated | 50% | 10B | Text subset |
| DCLM-Edu | 18% | 3.6B | `edu_int_score >= 3` |
| FineWiki English | 13% | 2.6B | English configuration |
| SmolLM corpus / Cosmopedia v2 | 9% | 1.8B | Textbook, tutorial, blog or educational formats |
| Python-Edu text mirror | 10% | 2B | `int_score >= 4` |

These are intended shares of processed token positions, not percentages of unique documents. The Python mirror is `Avelina/python-edu-cleaned`: the upstream Python-Edu subset contains file identifiers rather than directly usable code text. A bounded compatibility check accepted 91 of 100 sampled rows; all accepted snippets round-tripped through the tokenizer. This does not establish corpus-wide quality or available unique token counts.

### 3.2 Source versions

| Repository | Configuration | Revision |
| --- | --- | --- |
| `HuggingFaceTB/smollm-corpus` | `fineweb-edu-dedup`, `cosmopedia-v2` | `3ba9d605774198c5868892d7a8deda78031a781f` |
| `HuggingFaceTB/dclm-edu` | Repository default | `dbad8ad71224482740cd9c9d353591adbf62fe04` |
| `HuggingFaceFW/finewiki` | `en` | `8bd13e72e6a002407649b3e898535f42ceb1aeb9` |
| `Avelina/python-edu-cleaned` | `default` | `3fedf55e02e72dd1fd68963f991f5e91adcb0f8d` |

The executable corpus specification preserves source/license metadata. Dataset notices and original code licenses remain applicable; the code license is not a replacement for dataset or model-weight licensing review.

### 3.3 Streaming and partitioning

Data streams from pinned Hugging Face datasets. Training text and code indentation remain intact; NFC and whitespace normalization are used for document identity and partitioning only. Content hashes allocate 97% of buckets to training and 1% each to validation, final evaluation and tokenizer development. The final partition is reserved; final evaluation results have not yet been produced.

The selected streaming settings use a 256-document shuffle buffer, a bounded 100,000-document deduplication window, four prefetched batches and up to five read attempts for retryable failures. This is bounded deduplication, not a claim of globally unique training documents. Source exhaustion fails explicitly rather than silently recycling data. Document filters require at least 200 characters and at most 200,000 UTF-8 bytes. Exact settings are in `configs/corpus_20b.json` and `configs/pretrain_20b.json`.

## 4. Distributed training

| Physical Mac | Hardware | Local microbatch |
| --- | --- | ---: |
| Mac 1 | Apple M5 Ultra | 4 |
| Mac 2 | Apple M3 Ultra | 4 |
| Mac 3 | Apple M4 Max | 2 |
| Mac 4 | Apple M4 Max | 2 |
| Mac 5 | Apple M4 Max | 2 |

The physical ring is **1 → 2 → 3 → 5 → 4 → 1**. Recipe batches `[4, 4, 2, 2, 2]` use that ring order. All ranks use 2,048-token sequences and four gradient accumulation microsteps:

`(4 + 4 + 2 + 2 + 2) × 4 × 2,048 = 114,688 tokens per optimizer update`.

The implementation weights unequal-rank contributions by their token counts. Ethernet provides SSH orchestration; verified Thunderbolt/RDMA links provide the training ring. Launch preflight checks identities, package consistency, idle workers, routes and all ten directed link pings. Prior five-rank tests covered collectives, unequal-batch gradient comparison, replica agreement and checkpoint round trips with the current worker modules. These checks validate distributed mechanics, not eventual model quality.

### 4.1 Optimization schedule

| Argument | Selected value |
| --- | --- |
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Warmup | Linear through 200M tokens (1% of target) |
| Plateau | Through 16B tokens (80% of target) |
| Decay | Cosine to `3e-5` at 20B tokens |
| Betas | `(0.9, 0.95)` |
| Epsilon | `1e-8` |
| Weight decay | `0.1` |
| Gradient clipping | `1.0` |
| Seed | `7337` |

The schedule follows processed tokens. Complete updates can exceed the target by less than one update. At this batch size, 174,387 updates would process 20,000,096,256 positions. No final training duration has been established.

The active MLX/frontend and Metal backend are a locally built `0.32.3.dev20260920+7241f12` snapshot from upstream commit `7241f12e631440387a2156ec1018d1c9fe8e56b9`, including the native ring all-gather correction. Other packages include NumPy 2.4.2, datasets 4.5.0, huggingface-hub 1.4.0 and tokenizers 0.22.2. The source repository documents custom-wheel provenance and hashes; these development wheels are not distributed through its initial `requirements.txt`.

## 5. Validation, diagnostics and checkpoints

Fixed held-out validation batches are rebuilt on the coordinator and fingerprinted for resume compatibility: 32 batches per source, with batch size two and the selected context. Validation evaluates loss/perplexity by source and the selected aggregate mixture every **50M tokens** and at completion. This validation sample is smaller than the complete reserved partition.

Qualitative prompts are sampled at the same token interval. The coordinator rereads `configs/sample_prompts.json` at each event, allowing edits during training. Generated text and prompt hashes are recorded. Prompt edits do not alter held-out validation or checkpoint selection.

Training loss, exponential moving average loss, learning rate, gradient norm, token counts and throughput are logged at step one and every ten updates. This is sampled logging, not a persisted raw-loss measurement for every optimizer update. Validation events remain in the job log; `last_eval.json` contains the latest event. Loss curves will be derived from these records after sufficient progress.

Regular checkpoint saves occur every **100M tokens**, with additional best-validation saves and final/coordinated-stop saves. Retention keeps three recent checkpoints and protects the best checkpoint where needed. At regular save points and successful completion, the lowest-validation-loss checkpoint is copied into an independent inference snapshot for the playground. Checkpoint selection does not use benchmark scores or editable prompt responses.

## 6. Current evidence and planned results

The selected run's job is `20261002_153324_e3b204bd`, under `pretraining/runs/research_20b_176m_b44222`. A log read during initial report preparation confirmed **step 60, 6,881,280 processed tokens**, training loss **8.8928**, EMA **9.7252**, interval throughput **4,697 tokens/s**, and run throughput **4,518 tokens/s**. This is an early observation during warmup; it is neither a final result nor a stable throughput estimate. The run may have progressed since that observation. No scheduled 50M-token validation was observed in that snapshot.

Earlier exploratory 237.91M and initial-batch 175.61M runs were canceled and are not combined with this run's token counts. Their weights do not initialize the selected run.

| Planned result | Status |
| --- | --- |
| Training and validation loss curves | Pending sufficient training history |
| ARC-Easy / ARC-Challenge `acc` and `acc_norm` | Pending |
| PIQA `acc` and `acc_norm` | Pending |
| HellaSwag `acc` and `acc_norm` | Pending |
| Python capability evaluation | Protocol and results pending |
| Final pretrained checkpoint | Pending |

Before comparisons, record harness/version, task settings, shots, splits and scoring conventions. Base-model likelihood results should be compared with the previous base under matching settings. Instruction benchmarks require a separately defined prompting/post-training protocol. No earlier OpenSML or external benchmark result is attributed to this model.

## 7. Reproducibility and next report updates

The initial private GitHub repository contains the implementation, selected recipes, frozen tokenizer, setup instructions, notices, tests and a file-hash manifest with the source commit. The private Hugging Face repository initially contains the model card, report, selected recipes and tokenizer artifacts. Initial repository creation does not upload trained model weights.

The machine-specific network/correctness receipts are required to launch on the existing cluster and are retained in its workspace. A new cluster must regenerate those receipts and configure its own identities/network; the source snapshot alone is not a ready-configured five-Mac network.

Next updates should record validation milestones, loss curves, completed token counts, actual throughput and resource observations, selected checkpoint hashes, and evaluation protocols/results. The private repositories will remain private until the owner explicitly requests publication. No automatic remote checkpoint upload or report refresh is enabled by this document.
