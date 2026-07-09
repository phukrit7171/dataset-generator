---
name: llm-dataset-cli-impl
description: Implement a Pydantic-based LLM-dataset-generation CLI (uv/Python) end-to-end after planning approval, including strict typed models, combined ChatML JSONL output, readonly T-SQL validation, fake-llm pytest tests, and a ruff + basedpyright code gate.
source: auto-skill
extracted_at: '2026-07-09T13:17:33.177Z'
---

# Implement the LLM-dataset CLI

Use this procedure after the plan in `auto-skill-llm-dataset-cli-planning` has been approved. It converts the plan into a working, strictly-typed `uv` Python CLI with a passing test suite and code gate.

## Step 1 — Lock the toolchain early

In `pyproject.toml`, before writing any code, set:

- Runtime deps: `openai>=2.44.0` (already pinned), `pyyaml>=6.0.3` (PyYAML is NOT stdlib).
- Dev deps: `pytest>=9.0.2`, `ruff>=0.14.10`, `basedpyright>=1.31.6`.
- Script entry: `[project.scripts] dataset-generator = "dataset_generator.cli:main"`.
- Python: `requires-python = ">=3.14"` (matches `.python-version`).

If the tool blocks `uv add`, ask the user for explicit approval before manually editing `pyproject.toml` dependency lists. The package code does not need network access — `uv run pytest/basedpyright/ruff` is sufficient.

## Step 2 — Package layout

```
src/
  main.py                     # thin shim that calls dataset_generator.cli:main
  dataset_generator/
    __init__.py
    models.py                 # Pydantic models (NOT dataclasses, NOT TypedDicts)
    config.py                 # YAML loader → AppConfig, raises ConfigError
    prompts.py                # Preset prompt builders returning (system, user)
    validation.py             # ChatML + readonly T-SQL validators
    llm.py                    # OpenAICompatibleClient wraps openai.OpenAI
    generator.py              # Retry loop, per-group row collection
    writer.py                 # JSONL + metadata writers
    cli.py                    # argparse + main()
tests/
  conftest.py                 # adds src/ to sys.path
  test_config.py
  test_prompts.py
  test_validation.py
  test_llm.py
  test_generator.py
  test_writer.py
  test_cli.py
examples/
  thailand_sales.yaml
```

Why Pydantic over dataclass/TypedDict for this project: gives strict extra="forbid" validation, automatic ValidationError → friendly messages, clean attribute access (`row.messages`), and plays nicely with basedpyright strict mode.

## Step 3 — Pydantic model conventions

Use `ConfigDict(extra="forbid", frozen=True)` on every model. Validate positive ints with `field_validator`. Use `populate_by_name=True` plus `alias="schema"` if you must keep a YAML key that collides with a Pydantic parent attribute (BaseModel already has a `model_config` etc. — pick internal names that don't shadow).

Critical pattern: rename internal fields to avoid shadowing parent attributes (e.g. `schema_source: SchemaSource = Field(default_factory=SchemaSource, alias="schema")` lets YAML keep `schema:` while Python uses `group.schema_source`).

Always include a `row_dict(row)` and `rejected_dict(row)` helper that serializes Pydantic models to dicts for `json.dumps`, omitting `None` values. This is what the JSONL writer calls.

## Step 4 — ChatML + readonly T-SQL validation

Required-field validation goes through Pydantic (`Message.model_validate`). Domain rules are added in `validation.py`:

- Non-empty `content` (after `.strip()`) — put this in a `@field_validator("content")` on `Message`.
- `messages` must contain at least one `user` and one `assistant` turn (check in `validate_messages`).
- Readonly T-SQL: strip `/* */` and `--` comments and `'...'` literals via `re.sub`, then tokenize with `re.compile(r"[A-Za-z_][A-Za-z0-9_]*")`, uppercase, and reject any token in `_FORBIDDEN_SQL_KEYWORDS` (`INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/MERGE/EXEC/EXECUTE/GRANT/REVOKE/DENY/BULK/OPENROWSET/OPENDATASOURCE/OPENQUERY/xp_/sp_`) BEFORE checking that `SELECT` is present. Checking forbidden keywords first gives the right error message for `DELETE FROM t;`.

Reject path: catch `RowValidationError` in the generator and append a `RejectedRow` with `group`, `type`, `reason`, `raw`.

## Step 5 — Generator retry loop pattern

Per group, loop until `len(rows) == group.count` OR `attempts >= max_attempts` OR `no_progress_attempts >= max_attempts`:

1. Call `client.chat(system, user)` — on Exception, append a rejected row with reason `f"llm_error: {exc}"` and continue.
2. `parse_model_json(raw)` — on `RowValidationError`, append rejected with the raw text.
3. `_collect_from_payload(payload, group, rows, rejected, remaining)` — cap by `remaining` so a 1-row request that returns 5 rows only keeps 1.
4. If `len(rows) did not grow`, increment `no_progress_attempts`; else reset to 0.

The `remaining` cap is essential — without it a single over-eager LLM response will exceed `count` and break downstream tests that assume exact counts.

`_collect_from_payload` must also wrap `_candidates(payload)` in try/except so a malformed top-level payload becomes a rejected row rather than crashing the run.

## Step 6 — OpenAI-compatible client

The `OpenAICompatibleClient` must:

- Cache the `openai.OpenAI(...)` instance on first `chat()` call (frozen dataclass + `_client: Any = field(default=None, init=False)`). Otherwise tests that mutate the fake client between calls see the wrong state.
- Honor `provider.base_url` and `provider.api_key` from config.
- Raise `LLMClientError` on empty choices and on empty content.

For tests, monkeypatch `dataset_generator.llm.OpenAI` with a fake class that records `(api_key, base_url, timeout, model, temperature, messages)` and returns a configurable `SimpleNamespace(choices=[...])`. Store constructed instances in a `ClassVar[list[_FakeOpenAI]]` and reset via an autouse fixture so tests don't see stale state.

## Step 7 — Pytest fixtures

Use the existing `tests/conftest.py` that prepends `src/` to `sys.path` — no need to install the package. Use `pytest.MonkeyPatch` to set env vars (`api_key_env`), chdir into `tmp_path` for relative output paths, and patch the OpenAI client.

Critical: when a test only expects generation to fail (e.g. `{"data": []}`), give the fake client at least two responses so the `IndexError` from a list-out-of-range doesn't pollute the `rejected` list.

## Step 8 — Ruff + basedpyright code gate

`pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E","F","W","I","B","UP","SIM","RET","PL","RUF"]
ignore = ["PLR0913"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["PLR2004"]
```

`pyrightconfig.json` (at repo root) — strict mode on `src/` only, exclude tests so test fixtures stay terse:

```json
{
    "include": ["src"],
    "exclude": ["tests"],
    "typeCheckingMode": "strict",
    "reportMissingTypeStubs": "error",
    "reportImplicitStringConcatenation": "error",
    "reportUnusedCallResult": "error",
    "reportUnnecessaryTypeIgnoreComment": "error"
}
```

## Step 9 — Defeat basedpyright's `Unknown` complaints about PyYAML

PyYAML has no stubs, so `yaml.safe_load(...)` returns `Any` that basedpyright treats as `Unknown`. To clear the strict errors:

- Wrap the YAML read in `_yaml_load(file_path) -> dict[str, Any]` that returns `cast(dict[str, Any], raw)` after `isinstance(raw, dict)`.
- For any nested dict, cast through `cast(dict[str, Any], value)` after `isinstance(value, dict)` checks.
- For list iteration, do `groups_raw_value: Any = raw.get("groups"); ...; groups_raw = cast(list[Any], groups_raw_value)`.

These casts are necessary even when the source value is `Any` — basedpyright's narrowing of `Any` through `.get()` on `dict[str, Any]` is not transitive in strict mode.

For prompt templates that mix `'` and `"` strings, collapse to a single string per `Template(...)` call — implicit string concatenation is an error in strict mode. Long lines need `# noqa: E501` since prompt bodies are inherently long.

Unused `Action` from `parser.add_argument(...)` requires `_ = parser.add_argument(...)`.

## Step 10 — Verification checklist

Run these in order and require all three to be green before reporting done:

1. `uv run pytest -q` → all tests pass (target ≥40 tests).
2. `uv run ruff check src tests` → "All checks passed!".
3. `uv run basedpyright` → "0 errors, 0 warnings, 0 notes".

If any tool blocks `uv add` or `uv run` commands due to safety classifier, stop and ask for explicit approval — never try to bypass through shell indirection.

## Common gotchas (learned the hard way)

- **Ruff `--fix` strips seemingly unnecessary `cast()` calls** that basedpyright then complains about. Re-apply the casts after any ruff auto-fix.
- **Implicit string concatenation across mixed quotes** — basedpyright flags a Template body that has `'"...'` followed by `"-..."`. Collapse to one literal.
- **`SchemaSource(tables=[{...}])`** in tests fails basedpyright — use `SchemaSource(tables=[TableSpec(name=..., columns=[ColumnSpec(name=...)])])` instead. Schema tests should use typed constructors, not raw dicts.
- **Frozen dataclass + lazy init**: use `field(default=None, init=False, repr=False, compare=False)` and `object.__setattr__` if you mutate, OR switch to a non-frozen dataclass and use `self._client = cached` directly.
- **Mock LLM with `index error`**: provide enough fake responses for the retry loop or it will add a spurious `llm_error` rejected row.
- **`monkeypatch.chdir(tmp_path)`** is required for tests that rely on the CLI writing relative `out.jsonl` paths.
- **OutputConfig field names**: keep YAML-friendly names (`combined`, `rejects`, `metadata`), not the renamed `combined_path` etc., to avoid breaking the loader.