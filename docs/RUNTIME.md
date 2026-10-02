# Runtime provenance

The active cluster uses a locally built MLX development snapshot, not wheels published on PyPI:

- MLX and mlx-metal: `0.32.3.dev20260920+7241f12`.
- Upstream source: https://github.com/ml-explore/mlx/commit/7241f12e631440387a2156ec1018d1c9fe8e56b9
- Includes the native ring all-gather correction discussed at https://github.com/ml-explore/mlx/pull/4443.
- Built for CPython 3.11 / arm64, with a macOS 26 deployment target. Exact artifact hashes are in `runtime_wheels.json`.

The author's existing wheels are retained at `~/.cache/opensml/mlx-pr4443/wheels/`. Install the compatible frontend/backend pair into a new environment before installing `requirements.txt`. Do not modify the active training environment while workers are running.

```bash
.venv/bin/python -m pip install --no-deps /path/to/verified/wheels/*.whl
.venv/bin/python -m pip install -r requirements.txt
```

These custom wheels are not included in this initial source snapshot. Other users must build the pinned source with a compatible Xcode/Metal toolchain or obtain and verify the matching artifacts. The historical build used backend-first `MLX_BUILD_BACKEND_PACKAGE=1`, then frontend `MLX_BUILD_FRONTEND_PACKAGE=1`, with `uv build --wheel`, `MACOSX_DEPLOYMENT_TARGET=26.2`, `CMAKE_ARGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2` and build parallelism 8. Regenerate five-rank correctness receipts against the installed runtime and current worker modules before launching.
