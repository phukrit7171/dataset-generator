---
name: llm-dataset-cli-planning
description: Plan an LLM-dataset-generation CLI tool (uv/Python project) from a short brief, using a two-round clarifying-question workflow and a default tech stack.
source: auto-skill
extracted_at: '2026-07-09T12:42:08.178Z'
---

# Plan an LLM-dataset-generation CLI

Use this procedure when the user asks to build a tool that uses an LLM to generate training datasets from a config file, with support for multiple dataset "groups" (e.g. text-to-SQL, Thai general QA).

## Step 1 — Inspect the existing project skeleton (read-only)

Before asking anything, scan these files (in parallel) to learn the language, dependency manager, Python version, and what already exists:

- `pyproject.toml` (deps, project metadata)
- `uv.lock` (if present — confirms pinned versions and actually-installed transitive deps)
- `.python-version`
- `src/main.py` (current entry point)
- `README.md`
- `.gitignore`

Look for already-declared deps (e.g. `openai`) — do not re-add them. Note Python version constraints (e.g. `>=3.14`) — they limit what libraries can be used.

## Step 2 — Round 1 questions (interface + format)

Ask in **one** `ask_user_question` call with 4 multi-select questions:

1. **Interface** — `CLI config` (recommended, e.g. `uv run <tool> config.yaml`), `Library first`, or `Both`.
2. **Config format** — `YAML` (recommended, supports nested group definitions), `JSON`, or `TOML`.
3. **LLM provider** — `OpenAI-compatible` (recommended, env-driven so it works with gateways), `OpenAI only`, or `Provider plugins`.
4. **Output** — For datasets, default to `JSONL combined (one file, all groups)` rather than per-group files unless the user asks otherwise.

Why combined JSONL: simpler downstream pipelines, easier to shuffle/split, one metadata sidecar instead of N.

## Step 3 — Round 2 questions (schema + quality)

Ask in a second `ask_user_question` call with 4 more questions:

1. **Row schema** — `ChatML messages` (recommended for chat fine-tuning), `instruction/input/output` (Alpaca-style), or both.
2. **SQL source** (only if SQL groups are needed) — `Schema in YAML` (recommended), external `.sql` files, or both.
3. **Quality control** — `Validate + retry` (recommended: parse JSON, validate required fields, enforce domain rules like `readonly T-SQL`, retry bad rows, write rejects), `Simple generation`, or `LLM judge`.
4. **Presets** — `Built-in presets + custom` (recommended: ship `text_to_tsql_readonly` and `thai_general_qa` as named templates, allow user-defined group prompts).

## Step 4 — Finalize the plan and present it for approval

The plan should explicitly cover:

- **Project shape**: uv/Python CLI entry, `pyproject.toml` script entry point, no extra runtime deps beyond what's already pinned unless required (e.g. `pyyaml`).
- **Config schema (YAML)**: top-level `provider` (api_base, model, api_key_env, temperature), top-level `output` (path, format=jsonl), and a list of `groups` each with `name`, `type` (preset id or `custom`), `count`, and type-specific fields (e.g. `schema` for SQL groups, `topics` for Thai QA).
- **Built-in presets**:
  - `text_to_tsql_readonly`: emits `system` (Thai SQL expert, SELECT-only) + `user` (natural-language question) + `assistant` (T-SQL using only `SELECT`/`WITH`, no DDL/DML).
  - `thai_general_qa`: emits Thai general-knowledge chat with `system` + `user` + `assistant`.
  - `custom`: user supplies `system_prompt` + `user_prompt_template` with `{topic}` placeholder.
- **Row schema**: `{"messages": [{"role": ..., "content": ...}, ...], "group": <name>, "id": <uuid>}`.
- **Validation + retry**: JSON-parse → required-fields check → domain rules (regex `^\s*(SELECT|WITH)\b`, no `;` followed by other statements, no DDL/DML keywords for SQL groups) → on failure, retry up to N times with the error fed back to the LLM → after N failures, append to `rejects.jsonl`.
- **Output**: combined `output.jsonl` + run metadata sidecar (`run.json` with counts, rejects path, provider, model).
- **Verification**: `uv run python -m unittest discover -s tests` with a fake OpenAI-compatible client that returns canned completions.

**Stop here and present the plan via `exit_plan_mode`. Do not edit files until the user confirms.**

## Defaults if the user doesn't answer

- Interface: CLI config
- Config: YAML
- Provider: OpenAI-compatible (env-driven)
- Output: combined JSONL
- Row schema: ChatML `messages`
- Quality: validate + retry, 3 retries, separate rejects file
- Presets: built-in `text_to_tsql_readonly`, `thai_general_qa`, plus `custom`

## Reusable checks / patterns discovered

- `pyyaml` is not stdlib; must be added to `pyproject.toml` if YAML is chosen.
- The existing repo already pins `openai>=2.44.0` and Python `>=3.14` — use the `openai.OpenAI(base_url=..., api_key=...)` constructor for compatibility with non-OpenAI gateways.
- For readonly T-SQL validation, a simple regex on the assistant content is enough for v1; full SQL parsing is out of scope.