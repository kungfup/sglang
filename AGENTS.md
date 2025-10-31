# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `python/sglang/`, including the `semi_pd` pipeline, API entrypoints, and CLI helpers; unit tests for those modules sit in `python/sglang/test/`. Kernel operators are in `sgl-kernel/`, while request routing and load-balancing logic are in `sgl-router/` and `sgl-pdlb/`. Integration assets such as IPC shims (`semi-pd-ipc/`), benchmarking harnesses (`benchmark/`), runnable examples (`examples/`), and repo-level smoke tests (`test/`) keep parity across environments. Visual assets, diagrams, and long-form docs belong in `assets/` and `docs/`.

## Build, Test, and Development Commands
Set up a dev environment with `pip install -e "python[dev]"` (swap `dev_cpu`, `dev_hip`, etc. to match hardware). Format staged Python edits using `make format`, rebuild GEMM kernels through `python/sglang/compile_deep_gemm.py`, and sync version metadata via `make update <version>`. Run focused runtime checks with `pytest -q python/sglang/test`, repo smoke tests with `pytest -q test`, GPU kernel suites inside `sgl-kernel` using `make test`, and router validation from `sgl-router` via `cargo test`.

## Coding Style & Naming Conventions
Python code follows `black` + `isort --profile=black`, 4-space indentation, and descriptive snake_case identifiers; reserve PascalCase for classes and prefix internal helpers with `_`. Ensure CUDA/C++ sources pass `clang-format`, Rust crates run `cargo fmt` and `cargo clippy -D warnings`, and keep comments focused on intent rather than mechanics.

## Testing Guidelines
Place new tests beside the code they guard: Python in `python/sglang/test`, cross-component scenarios in `test/`, kernel specs in `sgl-kernel/tests`, and routing checks in `sgl-router/tests`. Use `test_*` naming, parametrize boundary cases, gate heavyweight GPU suites with markers, and record expected logs in `test/lang` when extending language features.

## Commit & Pull Request Guidelines
Write commits around a single change and use imperative, present-tense subjects (e.g., `Tighten semi_pd admission control`); add bodies that state motivation or fallbacks. PRs should list test commands, performance or accuracy impact, linked issues, and screenshots or logs whenever behavior shifts; request reviewers noted in `docs/` for shared kernels or router updates.

## Security & Configuration Tips
Avoid checking in credentials or large model artifacts—existing `.gitignore` patterns cover common cases; add new exclusions when needed. Validate CUDA/ROCm prerequisites with `python/sglang/check_env.py`, document newly required environment variables in `docs/start/install.md`, and pin port ranges (`SGLANG_PORT`, `PORT_BASE`) during distributed experiments to keep shared runners stable.
