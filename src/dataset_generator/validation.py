"""Validation helpers for ChatML rows and readonly T-SQL."""

from __future__ import annotations

import json
import re
from typing import Any, Final, cast

from pydantic import ValidationError

from .models import DatasetRow, Message, Role

_VALID_ROLES: Final[frozenset[Role]] = frozenset({"system", "user", "assistant"})

_FORBIDDEN_SQL_KEYWORDS: Final[tuple[str, ...]] = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "MERGE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
    "DENY",
    "BULK",
    "OPENROWSET",
    "OPENDATASOURCE",
    "OPENQUERY",
    "xp_",
    "sp_",
)

_TSQL_REQUIRED_TOKEN: Final[str] = "SELECT"


class RowValidationError(ValueError):
    """Raised when a generated row fails validation."""


def coerce_row(raw: Any, *, expected_type: str) -> DatasetRow:
    """Coerce a raw mapping into a typed DatasetRow."""
    if not isinstance(raw, dict):
        raise RowValidationError("row must be a mapping")
    payload = cast(dict[str, Any], raw).copy()
    payload.setdefault("type", expected_type)
    try:
        return DatasetRow.model_validate(payload)
    except ValidationError as exc:
        raise RowValidationError(str(exc)) from exc


def validate_messages(messages: list[Message]) -> None:
    """Validate basic ChatML structure of a list of messages."""
    if not messages:
        raise RowValidationError("messages must not be empty")
    has_user = any(message.role == "user" for message in messages)
    has_assistant = any(message.role == "assistant" for message in messages)
    if not has_user or not has_assistant:
        raise RowValidationError("messages must include user and assistant turns")


_READONLY_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _strip_sql_comments(statement: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", " ", statement, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return re.sub(r"'[^']*'", "''", no_line)


def validate_readonly_tsql(statement: str) -> None:
    """Reject any non-readonly T-SQL keywords."""
    cleaned = _strip_sql_comments(statement)
    tokens: list[str] = [
        token.upper() for token in _READONLY_TOKEN_PATTERN.findall(cleaned)
    ]
    for token in tokens:
        if token in _FORBIDDEN_SQL_KEYWORDS:
            raise RowValidationError(f"non-readonly keyword not allowed: {token}")
    if _TSQL_REQUIRED_TOKEN not in tokens:
        raise RowValidationError("T-SQL must contain a SELECT statement")


def parse_model_json(payload: str) -> Any:
    """Parse a strict JSON payload, tolerating markdown code fences."""
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RowValidationError(f"invalid JSON: {exc}") from exc
