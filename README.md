# OpenSML-175M

An ongoing small language model pretraining experiment on a five-Mac Apple Silicon cluster using MLX.

The project name is **OpenSML-175M**; the selected model has **175,611,648 parameters (175.61M)**: 24 layers, width 768, grouped-query attention, a frozen 32K OpenSML tokenizer, and 2,048-token context. Training starts from random weights and targets 20B processed tokens with 90% educational prose and 10% Python.

The selected run is underway. No final weights or benchmark results have been published for this model.

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Training recipe](configs/pretrain_20b.json)
- [Pinned dataset mixture](configs/corpus_20b.json)
- [Cluster launch and editable prompts](docs/RESEARCH_PLAN.md)
- [Private model card](https://huggingface.co/wzebrowski/OpenSML-175M)
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


## JSONLite descendant

The JSONLite descendant is intended for short English extraction into a flat object with string, nullable-string or string-array fields. It can omit, merge or misassign values even when its JSON is valid. Nested/record schemas, numeric/Boolean outputs, optional-property logic and advanced schema rules are outside that released runner's verified scope. Its reported scores are specific to the documented evaluation panels. See the descendant model card's **Intended use and limitations** section before choosing it for an application; these statements do not describe new capabilities of the pretrained base.

The main [OpenSML-JSONLite-175M-Instruct](https://huggingface.co/OpenSML/OpenSML-JSONLite-175M) includes the latest joint source/schema/JSON extraction SFT. Current descendant complete-request correctness is 88.02% on the observed 1,444-request panel. [Training and evaluation update](docs/JSONLITE_DESCENDANT.md). The pretrained base weights and base benchmark claims are unchanged.
