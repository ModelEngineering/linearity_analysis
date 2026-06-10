# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Most important

1. Don’t assume. Don’t hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

## Environment Setup

The project uses a local venv at `mla/`. Always activate it before running anything:

```bash
source activate.sh
```

`activate.sh` also adds `src/` to `PYTHONPATH`, so imports like `from linear_analyzer import LinearAnalyzer` work in tests without a package install.

## Commands

```bash
# Run all tests
source activate.sh && python3 -m pytest tests/ -v

# Run a single test file
source activate.sh && python3 -m pytest tests/test_linear_analyzer.py -v

# Run a single test by name
source activate.sh && python3 -m pytest tests/test_linear_analyzer.py::TestCollectJacobians::test_shape -v

# Lint
source activate.sh && pylint src/

# Run with coverage
source activate.sh && python3 -m pytest tests/ --cov=src
```

## Architecture

All code lives in `src/`, all tests in `tests/`. 

**[src/constants.py](src/constants.py)** — Project-wide paths: `PROJECT_DIR` (repo root) and `DATA_DIR` (`<repo>/data/`).

## BioModels Data

SBML models are stored in `/Users/jlheller/home/Technical/repos/temp-biomodels/final/`. Each subdirectory (e.g. `BIOMD0000000001/`) contains `<ID>_url.xml` (the SBML file) and `manifest.xml` (skip this).

## Coding Style

Delegate all coding style to ``python-coder.md``.

## Tests

Delegate all coding style to ``test-builder.md``.