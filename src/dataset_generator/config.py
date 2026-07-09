"""YAML config loading with strict typing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from .models import AppConfig, GroupConfig, OutputConfig, ProviderConfig

_SUPPORTED_GROUP_TYPES: frozenset[str] = frozenset(
    {"text_to_tsql_readonly", "thai_general_qa", "custom"}
)


class ConfigError(ValueError):
    """Raised when the YAML config is missing fields or invalid."""


def _yaml_load(file_path: Path) -> dict[str, Any]:
    """Load YAML as a strict dict, hiding PyYAML's loose return type."""
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {file_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping")
    return cast(dict[str, Any], raw)


def _ensure_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"'{path}' must be a mapping")
    return cast(dict[str, Any], value)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _load_provider(raw: dict[str, Any]) -> ProviderConfig:
    api_key_env_value = raw.get("api_key_env", "OPENAI_API_KEY")
    api_key_env = api_key_env_value if isinstance(api_key_env_value, str) else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    model = _first_env("DATASET_GENERATOR_MODEL", "OPENAI_MODEL") or raw.get("model")
    base_url = _first_env("OPENAI_BASE_URL") or raw.get("base_url")
    payload = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": raw.get("temperature", 0.7),
        "request_timeout": raw.get("request_timeout", 60.0),
    }
    try:
        return ProviderConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"invalid 'provider' config: {exc}") from exc


def _load_output(raw: dict[str, Any]) -> OutputConfig:
    if not isinstance(raw.get("combined"), str):
        raise ConfigError("'output.combined' must be a string path")
    try:
        return OutputConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid 'output' config: {exc}") from exc


def _load_group(raw: dict[str, Any], index: int) -> GroupConfig:
    path = f"groups[{index}]"
    group_type = raw.get("type")
    if not isinstance(group_type, str):
        raise ConfigError(f"'{path}.type' must be a string")
    if group_type not in _SUPPORTED_GROUP_TYPES:
        raise ConfigError(
            f"'{path}.type' must be one of {sorted(_SUPPORTED_GROUP_TYPES)}"
        )
    try:
        return GroupConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid '{path}' config: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the YAML config from ``path``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"Config file not found: {file_path}")
    raw = _yaml_load(file_path)
    provider_raw = _ensure_dict(raw.get("provider"), "provider")
    output_raw = _ensure_dict(raw.get("output"), "output")
    provider = _load_provider(provider_raw)
    output = _load_output(output_raw)
    groups_raw_value: Any = raw.get("groups")
    if not isinstance(groups_raw_value, list) or not groups_raw_value:
        raise ConfigError("'groups' must be a non-empty list")
    groups_raw = cast(list[Any], groups_raw_value)
    groups: list[GroupConfig] = []
    for index, item in enumerate(groups_raw):
        groups.append(_load_group(_ensure_dict(item, f"groups[{index}]"), index))
    payload: dict[str, Any] = {
        "provider": provider,
        "output": output,
        "groups": groups,
        "max_attempts": raw.get("max_attempts", 3),
    }
    try:
        return AppConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"invalid top-level config: {exc}") from exc
