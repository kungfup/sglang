# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `python/sglang` (runtime, frontend language, configs). Tests live in `python/sglang/test/`. 
- GPU kernels: `sgl-kernel/` (C++/CUDA, CMake, Python wrapper). 
- Router & load balancer: `sgl-router/` (Rust crate with Python wrapper). 
- Additional dirs: `examples/` (usage), `benchmark/` (perf), `docs/` (Sphinx), `scripts/`, `docker/`, `.github/`.

## Build, Test, and Development Commands
- Python (editable dev install): `pip install -e "python[dev]"`  (or select extras: `dev_cpu`, `dev_hip`, `dev_hpu`, `dev_xpu`).
- Format changed Python files: `make format` (runs isort + black on modified files).
- Run pre-commit checks locally: `pre-commit install && pre-commit run -a`.
- Python tests (unit/smoke): `pytest -q python/sglang/test` (or `python -m pytest ...`). Many tests require GPUs/models; prefer small tests first.
- Kernel tests: `cd sgl-kernel && make test` (build helpers in `sgl-kernel/Makefile`).
- Router tests: `cd sgl-router && cargo test`.
- Version bump helper: `make update 0.x.y` (updates coordinated version strings).

## Coding Style & Naming Conventions
- Indentation: spaces, 4-wide (`.editorconfig`). JSON/YAML use 2. 
- Python: black + isort (profile=black), ruff on selected dirs. Use snake_case for funcs/vars, PascalCase for classes, module-private helpers prefixed with `_`. Group imports (stdlib, third-party, first-party).
- C++/CUDA: clang-format with repo style. 
- Rust: `cargo fmt` and `cargo clippy -D warnings` before committing.

## Testing Guidelines
- Place tests under the corresponding component (`python/sglang/test`, `sgl-kernel/tests`, `sgl-router/tests`). Use `test_*.py`/`*_test.rs` naming.
- Keep GPU-heavy tests optional; add small CPU smoke tests where possible. Useful envs: `SGLANG_IS_IN_CI=1`, `CUDA_VISIBLE_DEVICES=...`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (<=72 chars), explain “what/why”. Group logically; avoid noisy reformat-only commits unless intentional.
- PRs: include a clear description, linked issues, how to test (commands), expected perf impact, and screenshots/logs when UI/metrics change. Update docs/examples when behavior changes.

