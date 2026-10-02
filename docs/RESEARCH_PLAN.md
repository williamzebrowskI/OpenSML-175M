# Research run 002 — 20B tokens

A fresh 24-layer model with **175,611,648 parameters**, 768 hidden width, 12 query heads / 4 key-value heads, 2,048-wide SwiGLU, tied embeddings and 2,048-token sequences. Reuse the verified OpenSML 32,000-token byte-level BPE tokenizer. No previous pretrained or SFT weights initialize this run.

## Training settings

| Setting | Selected value |
| --- | --- |
| Target | 20B training token positions |
| Physical ring | Mac 1 → Mac 2 → Mac 3 → Mac 5 → Mac 4 → Mac 1 |
| Per-rank microbatches, in ring order | `[4, 4, 2, 2, 2]` |
| Accumulation | 4 |
| Tokens per update | 114,688 |
| Context on every rank | 2,048 |
| AdamW | β₁ 0.9, β₂ 0.95, ε 1e-8; weight decay 0.1; gradient clipping 1.0 |
| Learning rate | Linear warmup through 200M tokens to 3e-4; constant through 16B; cosine decay to 3e-5 at 20B |
| Validation / qualitative samples | Every 50M tokens and at completion |
| Regular checkpoints | Every 100M tokens; retention 3 plus protected best/latest |

The budget counts processed token positions, rather than unique documents. Complete optimizer updates can exceed the target by less than one update. There is no automatic recycling when a source exhausts. Time and throughput must be measured on this architecture before estimating the full run.

## Data mixture

Shares apply to training token positions. Versions and filters are pinned in `configs/corpus_20b.json`.

| Source | Share | Target token positions |
| --- | --- | --- |
| SmolLM corpus FineWeb-Edu deduplicated | 50% | 10B |
| DCLM-Edu, education score ≥3 | 18% | 3.6B |
| FineWiki English | 13% | 2.6B |
| Cosmopedia v2, educational formats | 9% | 1.8B |
| Python-Edu text mirror, education score ≥4 | 10% | 2B |

The Python source is `Avelina/python-edu-cleaned`, pinned to `3fedf55e02e72dd1fd68963f991f5e91adcb0f8d`. The original SmolLM Python-Edu subset contains file identifiers rather than text; this mirror includes the retrieved code. Original code licenses and provenance still apply. Text and indentation are preserved; whitespace normalization is used only for content split/deduplication keys. A bounded 100-row schema sample had 91 accepted snippets; all 91 round-tripped exactly through the retained tokenizer. This checks compatibility, not corpus-wide quality or available unique tokens.

The `reuse-frozen` tokenizer policy allows the changed corpus explicitly. Its actual mixture, pinned versions, stream settings and tokenizer fingerprint remain part of checkpoint compatibility checks. Validation uses fixed content-hash-held-out data from the new corpus, with the same source weights as training. It is separate from the editable prompts. No dedicated math source or post-training stage is configured.

## Edit prompts during training

Edit **`configs/sample_prompts.json` on Mac 1**. The coordinator opens and parses it again before every sampling event. Prompts are loaded for that event only; a change during generation takes effect at the next event. No restart, peer synchronization or training-recipe change is needed. The launcher passes the original editable path, rather than a staged copy.

```json
{
  "prompts": [
    "Plants need sunlight because",
    "def count_words(text):\n    "
  ]
}
```

Use 1–32 nonempty strings. If the file is missing, temporarily invalid while being saved, or exceeds its limits, that sample event is skipped and training continues. A prompt that exceeds model context is skipped individually. The next event rereads the file. Actual prompt text, file hash, generated answers and skipped-event reasons are saved under `pretraining/runs/research_20b_176m_b44222/samples/`. These are qualitative checks; edits do not change validation loss or benchmark scores.

## Inspect and launch

From the workspace root, inspect the plan without downloading data, initializing weights, contacting peers or starting training:

```bash
.venv/bin/python pretraining/scripts/launch_research.py
```

The Mac 4 ↔ Mac 5 connection was restored, and Mac 1’s training interfaces were detached from its bridge and assigned their existing /30 addresses. All ten directed training links and five-rank distributed correctness checks passed. The verified network plan is `pretraining/diagnostics/m5_migration_20260922/network_20261002_142104`. Live preflight is repeated at launch.

Record a verified existing network without modifying interfaces:

```bash
.venv/bin/python pretraining/scripts/configure_five_mac_network.py --adopt-existing
```

Use the printed plan directory as `NETWORK_DIR`. If the ring needs configuration after reboot, use the existing `--prepare` / reviewed `--plan ... --apply` workflow instead; adoption accepts only matching existing addresses and interfaces.

```bash
.venv/bin/python pretraining/scripts/check_five_mac.py --network "$NETWORK_DIR" --run
.venv/bin/python pretraining/scripts/launch_research.py --network "$NETWORK_DIR" --run --stop-after-steps 10
```

The first command runs bounded distributed correctness checks on tiny disposable models. The second is an explicit full-architecture smoke run. Review its memory, throughput, loss and checkpoints before extending it. The distributed correctness check has passed. The prior 237.91M architecture completed a bounded smoke run before its production run was canceled; throughput for this 175.61M architecture has not yet been measured. The launcher refuses production work without current five-rank correctness receipts, then checks identities, packages, all training routes and cable pings again.

Resume only this new run using the explicit checkpoint path recorded in its `latest.json`:

```bash
.venv/bin/python pretraining/scripts/launch_research.py --network "$NETWORK_DIR" --run --resume "$CHECKPOINT_DIR"
```

Without `--resume`, initialization is fresh and existing checkpoint output is refused. The selected recipe remains a 20B-token schedule even for bounded pilots. An optional architecture comparison uses `--config configs/pretrain_20b_baseline.json --output pretraining/runs/research_20b_baseline`; keep its output separate.

Preserve the final pretrained weights and a selected intermediate checkpoint before any future cleanup. The published first model is a reference, not a backup of this new run.

## Best checkpoint in the playground

The new recipes enable `playground_publish_best`. After each regular 100M-token checkpoint save, and at full training completion, the coordinator publishes the current **lowest validation-loss pretrained checkpoint** to `pretraining/runs/research_20b_176m_b44222/playground/`. The selected model may be older than the save event. Best-loss saves between regular events do not update the playground until the next regular event.

Each publication copies verified model weights and metadata into an independent inference-only snapshot. It excludes optimizer state and stream buffers, survives training checkpoint pruning, and atomically updates the playground pointer. Failed publication keeps the prior snapshot and reports the error without invalidating the saved training checkpoint. The best checkpoint in the training run remains protected by normal retention.

Open `http://127.0.0.1:8767/` on Mac 1. The page checks for snapshot changes every 15 seconds; generation loads the selected snapshot on the next request. A response already generating finishes on its current model. Before the first publication, the page displays a waiting message. The playground does not run full benchmarks or save chat history. Inference shares Mac 1’s GPU with training and can temporarily reduce training throughput.

To start the server again after it stops or after reboot:

```bash
.venv/bin/python pretraining/scripts/completion_playground.py
```

For the optional baseline run, pass `--run-dir pretraining/runs/research_20b_baseline/playground`. The run’s best selection is always its own fixed held-out validation loss.
