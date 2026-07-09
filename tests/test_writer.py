"""Tests for combined JSONL writer and metadata output."""

from __future__ import annotations

import json
from pathlib import Path

from dataset_generator.models import DatasetRow, GenerationResult, Message, RejectedRow
from dataset_generator.writer import write_jsonl, write_metadata, write_rejects


def _sample_rows() -> list[DatasetRow]:
    return [
        DatasetRow(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ],
            group="thai",
            type="thai_general_qa",
        ),
        DatasetRow(
            messages=[
                Message(role="user", content="sum?"),
                Message(role="assistant", content="SELECT SUM(x) FROM t;"),
            ],
            group="tsql",
            type="text_to_tsql_readonly",
        ),
    ]


def test_write_jsonl_combined(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_jsonl(str(path), _sample_rows())
    contents = path.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    first = json.loads(contents[0])
    assert first["group"] == "thai"
    assert first["messages"][0]["role"] == "user"
    second = json.loads(contents[1])
    assert second["type"] == "text_to_tsql_readonly"


def test_write_rejects_skips_when_path_empty(tmp_path: Path) -> None:
    write_rejects("", [RejectedRow(group="g", reason="x", raw="y")])
    assert list(tmp_path.iterdir()) == []


def test_write_metadata_payload(tmp_path: Path) -> None:
    result = GenerationResult(
        rows=_sample_rows(),
        rejected=[RejectedRow(group="g", reason="x", raw="y")],
        total_attempts=4,
    )
    path = tmp_path / "meta.json"
    write_metadata(str(path), result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "total_rows": 2,
        "total_rejected": 1,
        "total_attempts": 4,
    }


def test_write_metadata_skips_when_path_empty(tmp_path: Path) -> None:
    write_metadata("", GenerationResult(rows=[], rejected=[], total_attempts=0))
    assert list(tmp_path.iterdir()) == []
