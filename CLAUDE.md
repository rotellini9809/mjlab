# Developer Guide

## Commands

Use `make` for common workflows:

```bash
make format     # Format and lint
make type       # Format, lint, and type-check
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

You can also run individual commands directly with `uv`:

```bash
uv run pytest tests/<test_file>.py  # Run a specific test file
uv run ty check                     # Run the ty type checker (faster)
uv run pyright                      # Run the pyright type checker (slower)
```

## General Guidelines

- Run `make check` frequently to format, lint, and type-check your code. This catches many issues before tests are executed.
- Before finalizing changes, run `make test-fast` to ensure nothing is broken.

## Style Guidelines

- General
  - Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
