# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Install in editable mode with CPU torch wheels, as CI does (never bare pip install)
pip install uv
uv pip install --system -e . torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Tests: none exist. CI (.github/workflows/ci.yml) only verifies the editable install
# succeeds; it installs pytest but defines no test step, and there is no coverage setup.

# Format/lint: applied automatically to PRs by Ultralytics Actions (.github/workflows/format.yml
# is the source of truth: Ruff + docformatter for Python, Prettier for YAML/JSON/Markdown,
# codespell for spelling). Approximate locally with:
uvx ruff format . && uvx ruff check --fix .
uvx codespell # reads [tool.codespell] in pyproject.toml
```

CI (`ci.yml`) runs on push and PRs to `main` with a matrix of Python {3.11, 3.13} × torch {2.5.0, 2.8.0}; the package itself declares `requires-python >=3.8` in `pyproject.toml`.

## Architecture

Ultralytics fork of Apple's [ml-mobileclip](https://github.com/apple/ml-mobileclip), repackaged as a `pyproject.toml`-based pip package named `mobileclip` (version `0.1.0`). There is no release or publish workflow — the package is not on PyPI and installs are from source, so PRs never trigger a release.

- Public API is two functions in `mobileclip/__init__.py`: `create_model_and_transforms()` (returns `(model, None, preprocess)`) and `get_tokenizer()`. Both resolve a per-model JSON config from `mobileclip/configs/` (`mobileclip_s0/s1/s2/b.json`) — supporting a new variant means adding a config there, not new code.
- `mobileclip/clip.py` `CLIP` composes the two towers: `MCi` image encoder (`image_encoder.py`, backed by `models/mci.py` and `models/vit.py` on top of timm) and `TextTransformer` (`text_encoder.py`). Reusable blocks live in `mobileclip/modules/`; `modules/common/mobileone.py:reparameterize_model()` folds re-parameterizable branches for inference and is part of the documented API.
- Only `mobileclip*` packages ship (`[tool.setuptools]` in `pyproject.toml`). `training/` (OpenCLIP patch + DataCompDR configs), `eval/zeroshot_imagenet.py` (needs the `full` extra for clip-benchmark), `results/*.jsonl` (eval metrics), and `ios_app/` (Swift/Xcode demo, untouched by Python CI) sit outside the package.
- Pretrained checkpoints are not in the repo; `get_pretrained_models.sh` downloads them from Apple's CDN into `checkpoints/`.

## Conventions

- Every file carries the `# Ultralytics 🚀 AGPL-3.0 License` header — Ultralytics Actions adds them automatically; don't add or revert them manually. Preserve the original Apple copyright notices that follow in files that have them.
- Google-style docstrings (Args/Returns sections); Ruff/docformatter/Prettier formatting is enforced by the `format.yml` auto-commit, so don't hand-fight its output.
- There are no tests to run; `hf_dataset_example.py` and `training/` scripts stream data from Hugging Face over the live network, so don't invoke them for validation.
- No version-bump or release process: `version = "0.1.0"` in `pyproject.toml` is static and PRs don't change it. Dependabot updates pip and GitHub Actions dependencies weekly under the `dependencies` label.
