"""Command-line entry point for the dataset generator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ConfigError, load_config
from .generator import generate_all
from .llm import OpenAICompatibleClient
from .writer import write_jsonl, write_metadata, write_rejects


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
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if not config.provider.api_key:
        print(
            "provider.api_key_env points to an empty environment variable; set it before running",
            file=sys.stderr,
        )
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
