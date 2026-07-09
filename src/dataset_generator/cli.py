"""Command-line entry point for the dataset generator."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, load_config
from .generator import generate_all
from .llm import OpenAICompatibleClient
from .writer import write_jsonl, write_metadata, write_rejects


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file if present.

    Existing environment variables are not overridden. Missing files are silent.
    """
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        value = value.strip('"')
        value = value.rstrip("'")
        _ = os.environ.setdefault(key, value)


def _load_env_files(start: Path) -> None:
    """Walk upward from ``start`` looking for a .env to load."""
    candidates = [start.parent] if start.is_file() else [start]
    cwd = Path.cwd()
    if cwd not in candidates:
        candidates.append(cwd)
    for directory in candidates:
        _load_env_file(directory / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset-generator",
        description="Generate ChatML datasets from a YAML config using an OpenAI-compatible LLM.",
    )
    _ = parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config file.",
    )
    return parser


def _run(config_path: Path) -> int:
    _load_env_files(config_path.resolve().parent)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if not config.provider.api_key:
        env_var = "OPENAI_API_KEY"
        hint = (
            "provider.api_key_env points to an empty environment variable; "
            f"export {env_var}=... or copy .env.example to .env and edit it"
        )
        print(hint, file=sys.stderr)
        return 3

    client = OpenAICompatibleClient(config.provider)
    try:
        result = generate_all(
            config.groups,
            client,
            max_attempts=config.max_attempts,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"generation failed: {exc}", file=sys.stderr)
        return 4

    write_jsonl(config.output.combined, result.rows)
    write_rejects(config.output.rejects, result.rejected)
    write_metadata(config.output.metadata, result)

    summary = (
        f"wrote {len(result.rows)} rows to {config.output.combined} "
        f"({len(result.rejected)} rejected)"
    )
    print(summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
