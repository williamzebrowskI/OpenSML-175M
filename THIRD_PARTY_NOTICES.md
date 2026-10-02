# Third-party and data notices

The source depends on MLX (including a custom upstream snapshot), NumPy, Hugging Face datasets/hub/tokenizers, FastAPI, Pydantic and Uvicorn. Consult their upstream distributions for licenses and notices; pinned package versions and custom runtime provenance are included. The optional native display uses Apple AppKit/WebKit, and hardware telemetry uses macmon. No third-party model weights are bundled.

The frozen tokenizer was trained for the prior OpenSML release and retains its manifest and corpus provenance. Reuse does not mean this run consumed every tokenizer-training source.

Training datasets are pinned in `configs/corpus_20b.json`. Recorded source declarations include ODC-BY for SmolLM corpus subsets, CC-BY-4.0 for DCLM-Edu, CC-BY-SA-4.0/GFDL for FineWiki, and ODC-BY mirror metadata plus original upstream repository/code terms for Python-Edu. These are provenance records, not a determination that every individual document has identical rights. Preserve original attribution and provenance and review applicable source terms before redistribution.

- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/HuggingFaceTB/dclm-edu
- https://huggingface.co/datasets/HuggingFaceFW/finewiki
- https://huggingface.co/datasets/Avelina/python-edu-cleaned
- https://github.com/ml-explore/mlx
- https://github.com/vladkens/macmon
