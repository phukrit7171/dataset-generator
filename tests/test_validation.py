"""Tests for ChatML and readonly T-SQL validation."""

from __future__ import annotations

import pytest

from dataset_generator.models import Message
from dataset_generator.validation import (
    RowValidationError,
    coerce_row,
    parse_model_json,
    validate_messages,
    validate_readonly_tsql,
)


def test_coerce_row_strict_messages() -> None:
    raw = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }
    row = coerce_row(raw, expected_type="thai_general_qa")
    assert row.messages[0].role == "user"
    assert row.messages[0].content == "Q"
    assert row.type == "thai_general_qa"


def test_coerce_row_rejects_non_string_role() -> None:
    raw = {"messages": [{"role": 1, "content": "Q"}]}
    with pytest.raises(RowValidationError):
        coerce_row(raw, expected_type="x")


def test_coerce_row_rejects_unknown_role() -> None:
    raw = {"messages": [{"role": "tool", "content": "x"}]}
    with pytest.raises(RowValidationError):
        coerce_row(raw, expected_type="x")


def test_coerce_row_rejects_empty_content() -> None:
    raw = {"messages": [{"role": "user", "content": "   "}]}
    with pytest.raises(RowValidationError):
        coerce_row(raw, expected_type="x")


def test_validate_messages_requires_user_and_assistant() -> None:
    with pytest.raises(RowValidationError):
        validate_messages([Message(role="system", content="s")])


def test_validate_readonly_tsql_accepts_select() -> None:
    validate_readonly_tsql(
        "SELECT FullName FROM dbo.Customers WHERE Province = N'กรุงเทพ';"
    )


def test_validate_readonly_tsql_rejects_insert() -> None:
    with pytest.raises(RowValidationError):
        validate_readonly_tsql("INSERT INTO dbo.Customers VALUES (1, 'x');")


def test_validate_readonly_tsql_rejects_drop() -> None:
    with pytest.raises(RowValidationError):
        validate_readonly_tsql("DROP TABLE dbo.Customers;")


def test_validate_readonly_tsql_rejects_exec() -> None:
    with pytest.raises(RowValidationError):
        validate_readonly_tsql("EXEC sp_helpdb;")


def test_validate_readonly_tsql_rejects_missing_select() -> None:
    with pytest.raises(RowValidationError):
        validate_readonly_tsql("PRINT 'hello';")


def test_validate_readonly_tsql_strips_line_comments() -> None:
    validate_readonly_tsql(
        "-- DROP TABLE x\nSELECT 1 AS x;"
    )


def test_parse_model_json_strips_fences() -> None:
    payload = parse_model_json("```json\n{\"rows\": []}\n```")
    assert payload == {"rows": []}


def test_parse_model_json_rejects_invalid() -> None:
    with pytest.raises(RowValidationError):
        parse_model_json("not json")


def test_parse_model_json_extracts_embedded_json_object() -> None:
    payload = "<think>reasoning</think>\n{\"rows\": []}"
    assert parse_model_json(payload) == {"rows": []}


def test_parse_model_json_extracts_embedded_fenced_object() -> None:
    payload = "noise before\n```json\n{\"rows\": []}\n```\nnoise after"
    assert parse_model_json(payload) == {"rows": []}
