"""Module entrypoint shim."""

from __future__ import annotations

from dataset_generator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
