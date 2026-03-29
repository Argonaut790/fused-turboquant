# Contributing to fused-turboquant

Thanks for your interest in contributing! This project welcomes contributions of all kinds — bug reports, feature requests, documentation improvements, and code.

## Getting Started

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
uv sync --extra dev
```

**Requirements**: Python 3.10+, NVIDIA GPU with CUDA support, Linux (Triton requires Linux for GPU kernels).

## Development Workflow

1. **Fork** the repo and create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes** and ensure they pass checks:
   ```bash
   uv run ruff check src/ tests/           # lint
   uv run ruff format --check src/ tests/   # format check
   uv run pytest                            # tests
   ```

3. **Commit** with a clear message describing the change.

4. **Open a Pull Request** against `main`.

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (config in `pyproject.toml`).
- Line length: 100 characters.
- Run `uv run ruff format src/ tests/` to auto-format before committing.

## Triton Kernel Guidelines

- All kernels live in `src/fused_turboquant/kernels/`.
- Include a PyTorch reference implementation or fallback for testing correctness.
- Add benchmarks in `benchmarks/` if your change affects performance.
- Test across multiple `head_dim` values (64, 128, 256) and bit-widths (2, 4).

## Testing

```bash
uv run pytest                    # full test suite
uv run pytest tests/test_foo.py  # single file
uv run pytest -k "test_name"    # single test
```

GPU tests require an NVIDIA GPU. Tests that need a GPU are skipped automatically on CPU-only machines.

## Reporting Bugs

Please [open an issue](https://github.com/Argonaut790/fused-turboquant/issues/new?template=bug_report.yml) with:
- Your GPU model and driver version (`nvidia-smi`)
- Python, PyTorch, and Triton versions
- Minimal reproduction script
- Full error traceback

## Feature Requests

[Open a feature request](https://github.com/Argonaut790/fused-turboquant/issues/new?template=feature_request.yml) describing the use case and expected behavior.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
