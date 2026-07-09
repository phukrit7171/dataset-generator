"""Tests for preset prompt builders."""

from __future__ import annotations

from dataset_generator.models import ColumnSpec, GroupConfig, SchemaSource, TableSpec
from dataset_generator.prompts import build_prompts


def _schema() -> SchemaSource:
    return SchemaSource(
        tables=[
            TableSpec(
                name="dbo.Orders",
                description="Sales orders.",
                columns=[
                    ColumnSpec(name="OrderId", type="BIGINT"),
                    ColumnSpec(name="TotalAmount", type="DECIMAL(12,2)"),
                ],
            )
        ],
        rules=["Filter by OrderDate"],
        extra="Only Thai province names are allowed.",
    )


def test_tsql_prompt_includes_schema_and_count() -> None:
    group = GroupConfig(
        name="tsql",
        type="text_to_tsql_readonly",
        count=7,
        language="th",
        schema=_schema(),
    )
    prompts = build_prompts(group)
    assert "readonly" in prompts.system.lower()
    assert "Generate 7" in prompts.user
    assert "dbo.Orders" in prompts.user
    assert "Filter by OrderDate" in prompts.user
    assert "Only Thai province names" in prompts.user


def test_thai_qa_prompt_uses_language() -> None:
    group = GroupConfig(
        name="thai_qa",
        type="thai_general_qa",
        count=4,
        language="th",
        schema=SchemaSource(),
    )
    prompts = build_prompts(group)
    assert "thai content writer" in prompts.system.lower()
    assert "Language: th" in prompts.user


def test_custom_prompt_uses_user_template_when_missing() -> None:
    group = GroupConfig(
        name="custom",
        type="custom",
        count=2,
        language="en",
        schema=SchemaSource(),
        custom_system="You are a JSON emitter.",
        custom_user=None,
    )
    prompts = build_prompts(group)
    assert prompts.system == "You are a JSON emitter."
    assert "Generate 2" in prompts.user


def test_custom_prompt_prefers_user_supplied_body() -> None:
    group = GroupConfig(
        name="custom",
        type="custom",
        count=1,
        language="en",
        schema=SchemaSource(),
        custom_system=None,
        custom_user="Provide exactly 1 pair about coding.",
    )
    prompts = build_prompts(group)
    assert prompts.user == "Provide exactly 1 pair about coding."
