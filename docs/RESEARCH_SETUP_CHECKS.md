# Setup checks — October 2, 2026

No full research-model training was started.

- New setup tests: 6 passed. They cover prompt edits, missing/invalid files, individual context errors, the plan-only launcher, explicit tokenizer reuse, corpus compatibility fingerprints and tamper-resistant read-only network adoption.
- Existing network tests: 8 passed.
- Focused corpus/streaming regression tests: 8 passed, including deterministic held-out batches and checkpointed prefetch position.
- Tiny CPU streaming worker/checkpoint/resume test: passed. Resumed weights matched the uninterrupted fixture exactly; the fixture has 7,928 parameters and 256 training token positions, rather than the research architecture.
- Actual MLX architecture inspection: 150,286,976 parameters; SwiGLU hidden dimension 1,600.
- Tokenizer inspection: 32,000 IDs, integrity manifest verified; all 91 accepted Python snippets in a bounded 100-row schema sample round-tripped exactly. The sample averaged 2.73 characters/token. This is not a corpus-wide audit.
- Fresh launcher plan: passed; no data download, SSH, model initialization or training.
- Python compilation and Git whitespace check: passed.

The complete historical `test_pipeline.py` suite did not pass in the initial sandbox run: fixture streaming encountered cache permission errors, and mixed MLX import/mocking tests aborted with a duplicate native enum registration. The relevant streaming tests were rerun separately with cache access, and the tiny worker test ran in its own process; both passed. No full-suite pass is claimed.

## Distributed readiness

Read-only, identity-pinned inventory succeeded on all five Macs, but ring validation failed. Reciprocal links observed:

- Mac 1 `en3` ↔ Mac 2 `en5`
- Mac 2 `en4` ↔ Mac 3 `en5`
- Mac 3 `en4` ↔ Mac 5 `en5`
- Mac 4 `en4` ↔ Mac 1 `en4`

Mac 4 ↔ Mac 5 was absent from the fresh Thunderbolt inventories. No network settings were changed, no successful network receipt was created, and no five-rank correctness or research-model smoke run was attempted. Restore that connection and use the checks in `RESEARCH_PLAN.md` before launch. Throughput, memory at 2,048 context and distributed correctness for the new run remain unverified.

## Follow-up: ring restored and automatic playground publication

The repaired ring passed all ten directed route/ping checks, package and identity verification, five-rank native gather/sum, unequal-batch weighted gradients, replica equality and BF16/FP32 checkpoint roundtrips. The full research model has not been trained or profiled.

Playground publication tests: 4 passed, including independent snapshots surviving training checkpoint deletion, reusing an unchanged best, preserving the prior snapshot after corruption, server discovery without restart and nonfatal publication errors. A GPU HTTP streaming integration check loaded a tiny temporary random model and generated two tokens successfully; it did not train a model.

The worker code changed to publish the selected best at regular save points. The five-rank correctness proof is refreshed against this worker code before the training command is reused. The local server waits for the first 100M-token publication.

## Selected larger architecture — 237.91M

The production recipe now has 24 layers, width 896, 14 query / 2 KV heads and SwiGLU hidden width 2,560, for exactly 237,909,888 implemented parameters. The 32K tokenizer, 2,048 context, 20B target, 90% prose / 10% Python mixture, learning schedule and five-Mac microbatches remain selected. Production output is `pretraining/runs/research_20b_238m/` and the playground follows its published snapshots.

A separate synthetic-data smoke test completed three real distributed optimizer updates at the full production architecture and batch geometry: `[4,3,2,2,2]`, accumulation 4, 2,048 context. All five replicas agreed and stopped cleanly at 319,488 token positions. The saved checkpoint's file sizes and SHA-256 hashes verified. Observed update throughput was approximately 4,205–4,644 tokens/second over only three steps; this is not a sustained production-data estimate. Full-model validation on the pinned real corpus and longer-run memory/throughput remain to be measured.

Smoke logs and metadata remain under `pretraining/diagnostics/smoke_238m/`; disposable synthetic model weights were removed. Production training has not started, and the smoke weights will not be resumed by the production launcher. Playground publication tests and setup tests passed after the configuration change. The existing five-rank correctness proof matches the unchanged worker modules.

## Fresh 175.61M configuration

The 237.91M run was canceled at step 35; its checkpoint and STOP marker remain separate. The new recipe uses 24 layers, width 768, 12 query / 4 KV heads and SwiGLU width 2,048: 175,611,648 parameters. Compared with the released 150.44M base, it adds four layers without narrowing the blocks. Output and playground snapshots use `pretraining/runs/research_20b_176m/`. The 20B target, streaming mixture, optimizer schedule, five-Mac batches, live editable prompts and validation/checkpoint intervals are unchanged.

The actual MLX model instantiated on CPU has exactly 175,611,648 parameters. The plan-only launcher and six setup tests passed. Worker modules are unchanged, so the existing five-rank correctness receipt remains applicable. No new production training was started during setup.

## Revised five-Mac batches

The selected recipe now uses `[4, 4, 2, 2, 2]` in ring order (Macs 1, 2, 3, 5, 4), with accumulation 4 and context 2,048: 114,688 tokens per optimizer update. The previous 175.61M run stopped at step 13 and is preserved separately. The revised default output is `pretraining/runs/research_20b_176m_b44222/`; its playground follows this new run.
