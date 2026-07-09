"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_generator.config import ConfigError, load_config


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_minimal(tmp_path: Path) -> None:
    body = (
        "provider:\n"
        "  model: gpt-4o-mini\n"
        "output:\n"
        "  combined: ./out.jsonl\n"
        "groups:\n"
        "  - name: thai_qa\n"
        "    type: thai_general_qa\n"
        "    count: 3\n"
    )
    config = load_config(_write_config(tmp_path, body))
    assert config.provider.model == "gpt-4o-mini"
    assert config.output.combined == "./out.jsonl"
    assert len(config.groups) == 1
    assert config.groups[0].type == "thai_general_qa"
    assert config.groups[0].count == 3


def test_load_config_supports_env_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_KEY", "abc123")
    body = (
        "provider:\n"
        "  model: m\n"
        "  api_key_env: MY_KEY\n"
        "output:\n"
        "  combined: out.jsonl\n"
        "groups:\n"
        "  - name: g\n"
        "    type: custom\n"
        "    count: 1\n"
    )
    config = load_config(_write_config(tmp_path, body))
    assert config.provider.api_key == "abc123"


def test_load_config_rejects_unknown_type(tmp_path: Path) -> None:
    body = (
        "provider:\n"
        "  model: m\n"
        "output:\n"
        "  combined: o.jsonl\n"
        "groups:\n"
        "  - name: g\n"
        "    type: wrong\n"
        "    count: 1\n"
    )
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, body))


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_bad_schema_tables(tmp_path: Path) -> None:
    body = (
        "provider:\n"
        "  model: m\n"
        "output:\n"
        "  combined: o.jsonl\n"
        "groups:\n"
        "  - name: g\n"
        "    type: thai_general_qa\n"
        "    count: 1\n"
        "    schema:\n"
        "      tables: 'not-a-list'\n"
    )
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, body))


def test_load_config_rejects_non_positive_count(tmp_path: Path) -> None:
    body = (
        "provider:\n"
        "  model: m\n"
        "output:\n"
        "  combined: o.jsonl\n"
        "groups:\n"
        "  - name: g\n"
        "    type: thai_general_qa\n"
        "    count: 0\n"
    )
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, body))
