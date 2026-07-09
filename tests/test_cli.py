"""Tests for the CLI entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dataset_generator import cli


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: int = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self._content


def _write_config(tmp_path: Path) -> Path:
    body = (
        "provider:\n"
        "  model: m\n"
        "  api_key_env: DUMMY_KEY\n"
        "output:\n"
        "  combined: out.jsonl\n"
        "  rejects: out.rejects.jsonl\n"
        "  metadata: out.metadata.json\n"
        "groups:\n"
        "  - name: qa\n"
        "    type: thai_general_qa\n"
        "    count: 1\n"
        "  - name: sql\n"
        "    type: text_to_tsql_readonly\n"
        "    count: 1\n"
        "    schema:\n"
        "      tables:\n"
        "        - name: dbo.Orders\n"
        "          columns:\n"
        "            - name: TotalAmount\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_cli_writes_combined_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DUMMY_KEY", "key")
    monkeypatch.chdir(tmp_path)
    payload = {
        "rows": [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "sum?"},
                    {"role": "assistant", "content": "SELECT SUM(TotalAmount) FROM dbo.Orders;"},
                ]
            },
        ]
    }
    fake = _FakeClient(json.dumps(payload))
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda provider: fake)

    config_path = _write_config(tmp_path)
    exit_code = cli.main([str(config_path)])

    assert exit_code == 0
    combined = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(combined) == 2
    metadata = json.loads((tmp_path / "out.metadata.json").read_text(encoding="utf-8"))
    assert metadata["total_rows"] == 2
    assert fake.calls == 2


def test_cli_rejects_invalid_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    exit_code = cli.main([str(missing)])
    assert exit_code == 2


def test_cli_rejects_missing_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DUMMY_KEY", raising=False)
    config_path = _write_config(tmp_path)
    exit_code = cli.main([str(config_path)])
    assert exit_code == 3


def test_load_env_file_loads_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOAD_TEST_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LOAD_TEST_KEY=loaded-value\n# comment\n", encoding="utf-8")
    cli._load_env_file(env_file)
    assert os.environ["LOAD_TEST_KEY"] == "loaded-value"


def test_load_env_file_does_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEEP_ME", "preset")
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=from-file\n", encoding="utf-8")
    cli._load_env_file(env_file)
    assert os.environ["KEEP_ME"] == "preset"


def test_load_env_file_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('QUOTED_KEY="quoted value"\n', encoding="utf-8")
    cli._load_env_file(env_file)
    assert os.environ["QUOTED_KEY"] == "quoted value"
