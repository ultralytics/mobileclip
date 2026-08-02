# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

Ultralytics fork of Apple's [ml-mobileclip](https://github.com/apple/ml-mobileclip) (AGPL-3.0; the released weights and data keep Apple's terms in `LICENSE_weights_data`). It packages MobileCLIP — fast image-text models for zero-shot inference on mobile hardware — as the `mobileclip` pip package, alongside DataCompDR training recipes, zero-shot ImageNet evaluation, and a SwiftUI iOS demo app.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Install in editable mode with CPU torch wheels (never bare pip install);
# CI does the same with torch pinned to the matrix version plus pytest
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

- Code files (Python, shell, Swift, YAML, `pyproject.toml` — not Markdown or JSON) carry an `Ultralytics 🚀 AGPL-3.0 License` header comment at the top, after any shebang — Ultralytics Actions adds them automatically; don't add or revert them manually. Preserve the original Apple copyright notices that follow in files that have them.
- Google-style docstrings (Args/Returns sections); Ruff/docformatter/Prettier formatting is enforced by the `format.yml` auto-commit, so don't hand-fight its output.
- There are no tests to run; `hf_dataset_example.py` streams data from Hugging Face over the live network, and `training/` run scripts expect local DataCompDR shards plus multi-node `torchrun`, so don't invoke either for validation.
- No version-bump or release process: `version = "0.1.0"` in `pyproject.toml` is static and PRs don't change it. Dependabot updates pip and GitHub Actions dependencies weekly under the `dependencies` label.
