"""Prompt presets for the built-in dataset groups."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from .models import GroupConfig, SchemaSource

_TSQL_SYSTEM: str = "You are a senior data engineer who writes readonly Microsoft SQL Server (T-SQL) queries. You only respond with strict JSON."  # noqa: E501

_TSQL_USER_TEMPLATE: Template = Template("Generate $count distinct high-quality training pairs.\nEach pair is one natural-language business question and one readonly T-SQL SELECT statement that answers it using the schema below.\nLanguage: $language.\nSchema (read this carefully):\n$schema_text\nRules:\n- Only SELECT, WITH, or CTE-based SELECT statements.\n- No INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/MERGE/EXEC/GRANT/REVOKE.\n- Use fully qualified column names when useful.\n- Questions must reference the schema columns or tables explicitly.\n- Return strict JSON with this shape:\n{\"rows\": [{\"messages\": [{\"role\": \"user\", \"content\": \"<question>\"}, {\"role\": \"assistant\", \"content\": \"<t-sql>\"}]}, ...]}\n- Output ONLY the JSON object, no prose or markdown fences.")  # noqa: E501

_THAI_GENERAL_SYSTEM: str = "You are a careful Thai content writer creating a supervised fine-tuning dataset. You only respond with strict JSON."  # noqa: E501

_THAI_GENERAL_USER_TEMPLATE: Template = Template("Generate $count distinct Thai general-purpose instruction pairs.\nEach pair is a user instruction and an assistant response in Thai.\nLanguage: $language.\nRules:\n- Cover diverse topics: everyday life, knowledge, reasoning, writing, math, advice.\n- Keep answers concise (1-3 short paragraphs).\n- Do not include any harmful, illegal, or hateful content.\n- Return strict JSON with this shape:\n{\"rows\": [{\"messages\": [{\"role\": \"user\", \"content\": \"<instruction>\"}, {\"role\": \"assistant\", \"content\": \"<response>\"}]}, ...]}\n- Output ONLY the JSON object, no prose or markdown fences.")  # noqa: E501

_CUSTOM_SYSTEM_PLACEHOLDER: str = (
    "You are a dataset generation assistant. You only respond with strict JSON."
)

_CUSTOM_USER_TEMPLATE: Template = Template("Generate $count distinct instruction pairs.\nLanguage: $language.\nFollow the rules from the system prompt.\nReturn strict JSON with this shape:\n{\"rows\": [{\"messages\": [{\"role\": \"user\", \"content\": \"<instruction>\"}, {\"role\": \"assistant\", \"content\": \"<response>\"}]}, ...]}\n- Output ONLY the JSON object, no prose or markdown fences.")  # noqa: E501


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


def _render_schema_text(schema: SchemaSource) -> str:
    sections: list[str] = []
    for table in schema.tables:
        header = f"Table: {table.name}"
        if table.description:
            header = f"{header} - {table.description}"
        column_lines: list[str] = []
        for column in table.columns:
            pieces = [column.name]
            if column.type:
                pieces.append(column.type)
            line = " - ".join(pieces)
            if column.description:
                line = f"{line} ({column.description})"
            column_lines.append(line)
        sections.append("\n".join([header, *column_lines]) if column_lines else header)
    if schema.rules:
        joined_rules = "\n".join(f"- {rule}" for rule in schema.rules)
        sections.append(f"Extra rules:\n{joined_rules}")
    if schema.extra:
        sections.append(schema.extra)
    if not sections:
        return "(no schema provided)"
    return "\n\n".join(sections)


def build_prompts(group: GroupConfig) -> PromptBundle:
    """Return the system/user prompts for a group config."""
    if group.type == "text_to_tsql_readonly":
        system = _TSQL_SYSTEM
        user = _TSQL_USER_TEMPLATE.safe_substitute(
            count=group.count,
            language=group.language,
            schema_text=_render_schema_text(group.schema_source),
        )
        return PromptBundle(system=system, user=user)
    if group.type == "thai_general_qa":
        system = _THAI_GENERAL_SYSTEM
        user = _THAI_GENERAL_USER_TEMPLATE.safe_substitute(
            count=group.count,
            language=group.language,
        )
        return PromptBundle(system=system, user=user)
    system = group.custom_system or _CUSTOM_SYSTEM_PLACEHOLDER
    template_body = group.custom_user or _CUSTOM_USER_TEMPLATE.safe_substitute(
        count=group.count,
        language=group.language,
    )
    return PromptBundle(system=system, user=template_body)
