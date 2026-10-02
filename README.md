# OpenSML-150M-V2

An ongoing small language model pretraining experiment on a five-Mac Apple Silicon cluster using MLX.

The project name is **OpenSML-150M-V2**; the selected model has **175,611,648 parameters (175.61M)**: 24 layers, width 768, grouped-query attention, a frozen 32K OpenSML tokenizer, and 2,048-token context. Training starts from random weights and targets 20B processed tokens with 90% educational prose and 10% Python.

The selected run is underway. No final weights or benchmark results have been published for this model.

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Training recipe](configs/pretrain_20b.json)
- [Pinned dataset mixture](configs/corpus_20b.json)
- [Cluster launch and editable prompts](docs/RESEARCH_PLAN.md)
- [Private model card](https://huggingface.co/wzebrowski/OpenSML-150M-V2)
- [Five-Mac activity monitor](pretraining/dashboard/README.md)

## Setup

On compatible Apple Silicon/macOS hardware, first follow [runtime provenance](docs/RUNTIME.md). Custom MLX wheels are required and are not on PyPI:

```bash
python3.11 -m venv .venv
# Install the verified custom MLX pair first; see docs/RUNTIME.md
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python pretraining/scripts/launch_research.py
```

The last command prints the plan without training. Five-Mac execution requires machine-specific SSH identities, configured links and regenerated network/correctness receipts; these are intentionally not bundled as credentials or ready-to-use receipts. The existing author's cluster can use its verified network directory. Follow the research plan before executing `--run`.

The source reuses OpenSML implementation modules under the internal name `sml_v1`, with no dependency on older weights. The file manifest identifies the source snapshot and hashes. There is no automatic synchronization between this research snapshot, the working cluster repository and Hugging Face.

## Initial checks

```bash
.venv/bin/python -m unittest discover -s pretraining/tests -p test_research_setup.py
PYTHONPATH=pretraining:pretraining/scripts .venv/bin/python -m unittest discover -s pretraining/scripts -p test_rack_sensors.py
```

See the technical report for validation intervals, checkpoint retention, current evidence and limitations. Model weights and large training records belong in the model repository when selected and uploaded; this initial source repository contains neither.
