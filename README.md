# dataset-generator

Generate ChatML datasets from a YAML config using an OpenAI-compatible LLM.

## Highlights

- Declarative YAML config with multiple dataset groups in one run.
- Built-in presets:
  - `text_to_tsql_readonly` — Thai/English natural-language questions paired with readonly T-SQL `SELECT` queries.
  - `thai_general_qa` — general-purpose Thai instruction pairs.
  - `custom` — bring your own system/user prompt.
- Strict ChatML row validation plus readonly T-SQL enforcement (`INSERT`, `UPDATE`, `DELETE`, etc. are rejected).
- Combined JSONL output plus rejects and metadata files.
- Strict, explicit Python type annotations across the package.

## Install

```bash
uv sync
```

## Configure

See [`examples/thailand_sales.yaml`](examples/thailand_sales.yaml) for a complete config covering both preset dataset groups.

Required environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

## Run

```bash
uv run dataset-generator examples/thailand_sales.yaml
```

Outputs:

- `out/dataset.jsonl` — combined ChatML JSONL with all groups merged.
- `out/dataset.rejects.jsonl` — rejected rows with reason.
- `out/dataset.metadata.json` — counts and attempts summary.

Each row uses the ChatML shape:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "group": "...", "type": "..."}
```

## Tests

```bash
uv run pytest
```

Tests use fake LLM responses; no API keys are required.

## Code gate

```bash
uv run ruff check src tests
uv run basedpyright
```

Strict typing + lint are enforced on the `src/` tree (tests are intentionally excluded from basedpyright to keep fixtures terse).