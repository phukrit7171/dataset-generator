"""Coordinate LLM calls, validation, and retry/reject handling."""

from __future__ import annotations

from typing import Any, cast

from .llm import ChatClient
from .models import DatasetRow, GenerationResult, GroupConfig, Message, RejectedRow
from .prompts import build_prompts
from .validation import (
    RowValidationError,
    coerce_row,
    parse_model_json,
    validate_messages,
    validate_readonly_tsql,
)


class GenerationError(RuntimeError):
    """Raised when a generation step cannot be completed."""


def _candidates(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        raise RowValidationError("payload must be a JSON object")
    payload_dict = cast(dict[str, Any], payload)
    rows_value: Any = payload_dict.get("rows")
    if not isinstance(rows_value, list):
        raise RowValidationError("payload must contain a 'rows' list")
    return cast(list[Any], rows_value)


def _assistant_text(messages: list[Message]) -> str:
    for message in messages:
        if message.role == "assistant":
            return message.content
    return ""


def _collect_from_payload(
    payload: Any,
    group: GroupConfig,
    rows: list[DatasetRow],
    rejected: list[RejectedRow],
    remaining: int,
) -> None:
    try:
        candidates = _candidates(payload)
    except RowValidationError as exc:
        rejected.append(
            RejectedRow(group=group.name, type=group.type, reason=str(exc), raw=str(payload))
        )
        return
    for raw in candidates:
        if remaining <= 0:
            break
        try:
            row = coerce_row(raw, expected_type=group.type)
            validate_messages(row.messages)
            if group.type == "text_to_tsql_readonly":
                validate_readonly_tsql(_assistant_text(row.messages))
            final_row = DatasetRow(
                messages=row.messages,
                type=group.type,
                group=group.name,
            )
            rows.append(final_row)
            remaining -= 1
        except RowValidationError as exc:
            rejected.append(
                RejectedRow(group=group.name, type=group.type, reason=str(exc), raw=str(raw))
            )


def _remaining_needed(group: GroupConfig, produced: int) -> int:
    return max(group.count - produced, 0)


def generate_group(
    group: GroupConfig,
    client: ChatClient,
    *,
    max_attempts: int,
) -> tuple[list[DatasetRow], list[RejectedRow], int]:
    """Generate and validate rows for a single dataset group."""
    prompts = build_prompts(group)
    rows: list[DatasetRow] = []
    rejected: list[RejectedRow] = []
    attempts = 0
    no_progress_attempts = 0

    while _remaining_needed(group, len(rows)) > 0 and attempts < max_attempts:
        attempts += 1
        try:
            raw = client.chat(prompts.system, prompts.user)
        except Exception as exc:
            rejected.append(
                RejectedRow(
                    group=group.name,
                    type=group.type,
                    reason=f"llm_error: {exc}",
                    raw="",
                )
            )
            no_progress_attempts += 1
            if no_progress_attempts >= max_attempts:
                break
            continue
        try:
            payload = parse_model_json(raw)
        except RowValidationError as exc:
            rejected.append(
                RejectedRow(
                    group=group.name,
                    type=group.type,
                    reason=str(exc),
                    raw=raw,
                )
            )
            no_progress_attempts += 1
            if no_progress_attempts >= max_attempts:
                break
            continue
        before = len(rows)
        _collect_from_payload(
            payload, group, rows, rejected, _remaining_needed(group, len(rows))
        )
        if len(rows) == before:
            no_progress_attempts += 1
            if no_progress_attempts >= max_attempts:
                break
        else:
            no_progress_attempts = 0

    return rows, rejected, attempts


def generate_all(
    groups: list[GroupConfig],
    client: ChatClient,
    *,
    max_attempts: int,
) -> GenerationResult:
    rows: list[DatasetRow] = []
    rejected: list[RejectedRow] = []
    total_attempts = 0
    for group in groups:
        generated_rows, generated_rejected, attempts = generate_group(
            group,
            client,
            max_attempts=max_attempts,
        )
        total_attempts += attempts
        rows.extend(generated_rows)
        rejected.extend(generated_rejected)
    return GenerationResult(
        rows=rows,
        rejected=rejected,
        total_attempts=total_attempts,
    )
