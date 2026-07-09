"""Tests for the generator using fake LLM responses."""

from __future__ import annotations

import json
from collections.abc import Iterable

from dataset_generator.generator import generate_group
from dataset_generator.models import ColumnSpec, GroupConfig, SchemaSource, TableSpec


class FakeClient:
    def __init__(self, responses: Iterable[str]) -> None:
        self._responses = list(responses)
        self.calls: int = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self._responses[self.calls - 1]


def _tsql_group(count: int) -> GroupConfig:
    return GroupConfig(
        name="tsql",
        type="text_to_tsql_readonly",
        count=count,
        language="th",
        schema_source=SchemaSource(
            tables=[TableSpec(name="dbo.Orders", columns=[ColumnSpec(name="TotalAmount")])]
        ),
    )


def _qa_group(count: int) -> GroupConfig:
    return GroupConfig(
        name="thai",
        type="thai_general_qa",
        count=count,
        language="th",
        schema=SchemaSource(),
    )


def test_generate_group_writes_valid_rows() -> None:
    payload = {
        "rows": [
            {
                "messages": [
                    {"role": "user", "content": "ยอดขายรวมเป็นเท่าไหร่?"},
                    {"role": "assistant", "content": "SELECT SUM(TotalAmount) FROM dbo.Orders;"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "มีออเดอร์กี่รายการ?"},
                    {"role": "assistant", "content": "SELECT COUNT(*) FROM dbo.Orders;"},
                ]
            },
        ]
    }
    client = FakeClient([json.dumps(payload)])
    rows, rejected, attempts = generate_group(_tsql_group(2), client, max_attempts=2)
    assert len(rows) == 2
    assert rows[0].type == "text_to_tsql_readonly"
    assert rows[0].group == "tsql"
    assert rejected == []
    assert attempts == 1
    assert client.calls == 1


def test_generate_group_rejects_non_readonly_sql() -> None:
    payload = {
        "rows": [
            {
                "messages": [
                    {"role": "user", "content": "ลบข้อมูล"},
                    {"role": "assistant", "content": "DELETE FROM dbo.Orders;"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "ยอดขาย"},
                    {"role": "assistant", "content": "SELECT SUM(TotalAmount) FROM dbo.Orders;"},
                ]
            },
        ]
    }
    client = FakeClient([json.dumps(payload)])
    rows, rejected, _ = generate_group(_tsql_group(2), client, max_attempts=1)
    assert len(rows) == 1
    assert rows[0].messages[1].content.startswith("SELECT")
    assert len(rejected) == 1
    assert rejected[0].reason is not None and "non-readonly" in rejected[0].reason


def test_generate_group_rejects_invalid_json() -> None:
    client = FakeClient(["not-json"])
    rows, rejected, _ = generate_group(_qa_group(1), client, max_attempts=1)
    assert rows == []
    assert len(rejected) == 1
    assert rejected[0].reason is not None and "invalid JSON" in rejected[0].reason


def test_generate_group_rejects_payload_without_rows() -> None:
    client = FakeClient([json.dumps({"data": []}), json.dumps({"data": []})])
    rows, rejected, _attempts = generate_group(_qa_group(1), client, max_attempts=2)
    assert rows == []
    assert all(entry.reason is not None and "rows" in entry.reason for entry in rejected)
    assert len(rejected) == 2


def test_generate_group_records_llm_error() -> None:
    class ErroringClient:
        def chat(self, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("boom")

    rows, rejected, attempts = generate_group(
        _qa_group(1), ErroringClient(), max_attempts=1
    )
    assert rows == []
    assert len(rejected) == 1
    assert rejected[0].reason is not None and "llm_error" in rejected[0].reason
    assert attempts == 1


def test_generate_group_retries_until_count_met() -> None:
    payloads = [
        json.dumps(
            {
                "rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "SELECT 1;"},
                        ]
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rows": [
                    {
                        "messages": [
                            {"role": "user", "content": "q2"},
                            {"role": "assistant", "content": "SELECT 2;"},
                        ]
                    }
                ]
            }
        ),
    ]
    client = FakeClient(payloads)
    rows, rejected, attempts = generate_group(_tsql_group(2), client, max_attempts=3)
    assert len(rows) == 2
    assert rejected == []
    assert attempts == 2
    assert client.calls == 2


def test_generate_group_stops_after_max_attempts() -> None:
    client = FakeClient([json.dumps({"rows": []})])
    rows, _rejected, attempts = generate_group(_tsql_group(5), client, max_attempts=2)
    assert rows == []
    assert attempts == 2
    assert client.calls == 2


def test_generate_group_requires_user_and_assistant() -> None:
    payload = {
        "rows": [
            {"messages": [{"role": "system", "content": "no user"}]}
        ]
    }
    client = FakeClient([json.dumps(payload)])
    rows, rejected, _ = generate_group(_qa_group(1), client, max_attempts=1)
    assert rows == []
    assert rejected[0].reason is not None and "user and assistant" in rejected[0].reason
