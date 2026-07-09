"""Typed data models for the dataset generator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must be a non-empty string")
        return value


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    group: str | None = None
    type: str | None = None


class RejectedRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: str | None = None
    type: str | None = None
    reason: str | None = None
    raw: str | None = None


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.7
    request_timeout: float = 60.0


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str | None = None
    description: str | None = None


class TableSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    columns: list[ColumnSpec] = Field(default_factory=list)


class SchemaSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: list[TableSpec] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    extra: str = ""


class GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str
    type: str
    count: int
    language: str = "th"
    schema_source: SchemaSource = Field(default_factory=SchemaSource, alias="schema")
    custom_system: str | None = None
    custom_user: str | None = None

    @field_validator("count")
    @classmethod
    def _validate_count(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("count must be a positive integer")
        return value


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    combined: str
    rejects: str = ""
    metadata: str = ""


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig
    output: OutputConfig
    groups: list[GroupConfig]
    max_attempts: int = 3

    @field_validator("max_attempts")
    @classmethod
    def _validate_max_attempts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_attempts must be a positive integer")
        return value


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[DatasetRow]
    rejected: list[RejectedRow]
    total_attempts: int


def message_dict(message: Message) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def row_dict(row: DatasetRow) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": [message_dict(m) for m in row.messages]}
    if row.group is not None:
        payload["group"] = row.group
    if row.type is not None:
        payload["type"] = row.type
    return payload


def rejected_dict(row: RejectedRow) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if row.group is not None:
        payload["group"] = row.group
    if row.type is not None:
        payload["type"] = row.type
    if row.reason is not None:
        payload["reason"] = row.reason
    if row.raw is not None:
        payload["raw"] = row.raw
    return payload
