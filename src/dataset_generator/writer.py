"""Write combined JSONL datasets and metadata files."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    DatasetRow,
    GenerationResult,
    RejectedRow,
    rejected_dict,
    row_dict,
)


def write_jsonl(path: str, rows: list[DatasetRow]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _ = handle.write(json.dumps(row_dict(row), ensure_ascii=False))
            _ = handle.write("\n")


def write_rejects(path: str, rows: list[RejectedRow]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _ = handle.write(json.dumps(rejected_dict(row), ensure_ascii=False))
            _ = handle.write("\n")


def write_metadata(path: str, result: GenerationResult) -> None:
    if not path:
        return
    payload = {
        "total_rows": len(result.rows),
        "total_rejected": len(result.rejected),
        "total_attempts": result.total_attempts,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
