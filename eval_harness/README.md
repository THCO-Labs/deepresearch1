# Deep Research Agent Evaluation Harness

This package is the in-house evaluation harness for running `deep_research_bench` against `dra/agent.py` (and other agents) in a reproducible, portable way.

The goal is:

- keep benchmark logic isolated from agent internals,
- standardize how all agents are invoked,
- run on Windows/macOS/Linux without Bash assumptions,
- keep each run fully reproducible inside `eval_runs/<run_id>/`.

## Table of contents

1. [Architecture in plain terms](#architecture-in-plain-terms)
2. [Quick start](#quick-start)
3. [How questions are loaded and passed to an agent](#how-questions-are-loaded-and-passed-to-an-agent)
4. [What “agent adapters” are and how to use them](#what-agent-adapters-are-and-how-to-use-them)
5. [Swap in another agent](#swap-in-another-agent)
6. [Run and output locations](#run-and-output-locations)
7. [CLI usage reference](#cli-usage-reference)
8. [Judge configuration and comparison discipline](#judge-configuration-and-comparison-discipline)
9. [Resume and rerun behavior](#resume-and-rerun-behavior)
10. [Troubleshooting checklist](#troubleshooting-checklist)

## Architecture in plain terms

The harness has four layers:

1. `datasets`: load and filter benchmark tasks from `query.jsonl`.
2. `agents`: normalize all possible agent implementations behind one interface.
3. `judges`: prepare environment variables used by DRB scoring scripts.
4. `engine`: orchestrates run folder creation, generation, RACE, FACT, and summaries.

Execution flow for `python -m eval_harness run`:

- read config
- build run folder in `eval_runs/<run_id>/`
- copy benchmark workspace needed by DRB into `eval_runs/<run_id>/bench`
- load and filter tasks
- run one task at a time through selected adapter
- persist task outputs
- optionally run RACE
- optionally run FACT
- write `manifest.json`, `summary.txt`, and stage outputs

## Quick start

The quick start is the fastest safe path:

1. Install dependencies from `requirements.txt`.
2. Ensure API keys are available in environment (or `.env`).
3. Run a 1-task English smoke test.
4. Run a small batch before full benchmarking.

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Set environment variables

Minimum for smoke run:

```bash
TAVILY_API_KEY=...
OPENROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
```

If FACT is enabled, you also need:

```bash
JINA_API_KEY=...
```

### 3) Smoke run (recommended first command)

```bash
python -m eval_harness run --config configs/dra.example.json --limit 1 --only-en --no-fact
```

What this does:

- loads only 1 English benchmark question,
- runs agent generation,
- runs RACE (enabled by default),
- skips FACT (`--no-fact`),
- writes to `eval_runs/run_<timestamp>/...`.

### 4) Small scaled run

```bash
python -m eval_harness run --config configs/dra.example.json --limit 2 --only-en --no-fact
```

If this looks good, drop `--limit` for more tasks and add FACT if needed:

```bash
python -m eval_harness run --config configs/dra.example.json --limit 2 --only-en
```

## How questions are loaded and passed to an agent

The harness does not ask for single ad-hoc questions. It reads tasks from:

`<data.source_root>/<data.query_file>`

Defaults in `configs/dra.example.json` resolve to:

- `source_root`: `C:/Users/Dell/deep_research_bench`
- `query_file`: `data/prompt_data/query.jsonl`

Each line in JSONL should include:

- `id` (int)
- `prompt` (string)
- `language` (`en` or `zh`)
- `topic` (string)

For each selected task, the harness calls the adapter with:

- `question`: task `prompt`
- `output_dir`: isolated directory for that task: `eval_runs/<run_id>/tasks/<id>/`
- `fresh=True` (always on by default in current adapters)

That means each task is isolated and cannot clobber another task’s files.

## What agent adapters are and how to use them

An adapter is a shim that turns different ways of calling agents into one common output format: **a string article/report**.

Every adapter must provide:

- input: `task` dict (`id`, `prompt`, `language`, `topic`)
- output: string containing report text

### Built-in adapter types

- `dra_agent`: call current `agent.py` API (`run_research(question, output_dir, fresh)`).
- `python_callable`: call any importable Python function.
- `command`: execute any shell command with placeholders.
- `raw_jsonl`: reuse precomputed outputs from a JSONL file.

## Swap in another agent

Use one agent block in config. Replace only the `agent` section unless your environment also differs.

### A) Default `dra_agent` (current `dra/agent.py`)

```json
{
  "agent": {
    "type": "dra_agent",
    "name": "dra-agent",
    "module_path": "C:/Users/Dell/dra/agent.py",
    "function": "run_research",
    "command": null,
    "jsonl_path": null,
    "extra": {}
  }
}
```

Meaning:

- `type: dra_agent` tells harness to load a module and call a function by name.
- `module_path` is the absolute/relative Python file path.
- `function` must exist and return article markdown/text.

When to edit:

- only if module path or function name changes, for example if your file moves or you rename the entry function.

### B) Python callable adapter

Use this when teammates have a reusable package module.

```json
{
  "agent": {
    "type": "python_callable",
    "name": "team-agent",
    "module_path": "my_team_agents.research_agent",
    "function": "run_research"
  }
}
```

What happens at runtime:

- imports `my_team_agents.research_agent`
- invokes `run_research(question=<prompt>, output_dir=<task_dir>, fresh=True)`
- expects a string return

Equivalent colon syntax is supported by the harness:

```json
"function": "my_team_agents.research_agent:run_research"
```

When to use this:

- code-based teams running inside same runtime / virtual env,
- no shell wrapper needed,
- easiest for unit-testable agent logic.

### C) Command adapter

Use this for standalone CLIs, compiled binaries, or wrappers.

```json
{
  "agent": {
    "type": "command",
    "name": "cli-agent",
    "command": "python -m team_agent_runner --question \"{prompt}\" --output-file \"{output_file}\"",
    "extra": {
      "workdir": "C:/path/to/team/agent/repo"
    }
  }
}
```

Placeholder keys:

- `{prompt}`: raw question text
- `{task_id}`: task id
- `{run_dir}`: task folder (`eval_runs/<run_id>/tasks/<id>`)
- `{prompt_file}`: file containing prompt text
- `{output_file}`: file to read output from if stdout is empty

How output is consumed:

- primary: parser reads command stdout,
- fallback: reads `{output_file}` if output file exists.

What to change:

- set `command` to your CLI format,
- set `extra.workdir` if the command depends on repo cwd.

### D) Raw JSONL adapter

Use this when you already have outputs and only want scoring.

```json
{
  "agent": {
    "type": "raw_jsonl",
    "name": "baseline",
    "jsonl_path": "C:/path/to/precomputed.jsonl",
    "jsonl_key": "id"
  }
}
```

Row format expected:

- `id` or `prompt` identifies row,
- `article` (or `text`) contains text content.

## Run and output locations

Default output root is `eval_runs/`.

Each run creates:

- `eval_runs/<run_id>/`
- `eval_runs/<run_id>/raw/<agent>.jsonl`
- `eval_runs/<run_id>/race/<agent>/`
- `eval_runs/<run_id>/fact/`
- `eval_runs/<run_id>/tasks/<task_id>/`
- `eval_runs/<run_id>/bench/` (copied benchmark workspace)
- `eval_runs/<run_id>/manifest.json`
- `eval_runs/<run_id>/summary.txt`

You can change output location without changing commands by editing:

```json
"run": {
  "output_root": "eval_runs"
}
```

Or override `run_id` for named runs:

```bash
python -m eval_harness run --config configs/dra.example.json --run-id teamA_2026_06_10
```

Important: each run is isolated. Do not reuse `run_id` unless you intend to inspect or append that exact run.

## CLI usage reference

### Common full command

```bash
python -m eval_harness run --config <path> [options]
```

Common options:

- `--limit N`
- `--only-en` / `--only-zh`
- `--force`
- `--no-race`
- `--no-fact`
- `--skip-cleaning`
- `--dry-run`
- `--run-id <name>`
- `--max-workers N`

### Command variants

```bash
python -m eval_harness generate --config <path> [--limit N] [--only-en|--only-zh]
python -m eval_harness race --config <path> [--limit N]
python -m eval_harness fact --config <path> [--limit N]
python -m eval_harness summarize <run_dir>
```

`generate` only writes `raw/<agent>.jsonl`.

`run` runs generation + enabled scoring stages.

`summarize` prints manifest summary from an existing run.

## Judge configuration and comparison discipline

RACE and FACT are configured separately:

```json
{
  "judge": {
    "race": {
      "provider": "openai_compatible",
      "backend": "openrouter",
      "model": "openai/gpt-5.5",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    "fact": {
      "provider": "openai_compatible",
      "backend": "openrouter",
      "model": "openai/gpt-5.4-mini",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "fact_model": "openai/gpt-5.4-mini"
    }
  }
}
```

Changing judge settings changes score baselines. For fair model/agent comparisons, keep judge config identical and only change agent block.

## Resume and rerun behavior

- Existing task rows are reused by task id unless `--force` is used.
- `--force` re-runs selected tasks and updates outputs.
- Existing per-run scoring artifacts are reused/extended safely.
- `--dry-run` creates structure and manifest but skips generation.

This enables:

- interrupted-run recovery,
- incremental expansion (new run ids),
- fixed historical comparisons (pinned run config + judge model).

## Run artifact meanings (what your team will read)

The harness writes:

- `manifest.json`: config snapshot + environment-independent metadata + run summary,
- `summary.txt`: human-readable summary,
- `raw/<agent>.jsonl`: normalized raw outputs,
- `race/<agent>/race_result.txt`: RACE metrics when enabled,
- `fact/*`: FACT intermediate and final score files when enabled.

## Troubleshooting checklist

- If agent output is `"ERROR: ..."` in raw JSONL, install missing dependencies in your environment first.
- If DRB scripts fail due to missing key, ensure judge env variables are set as configured.
- If benchmark copy feels slow or flaky, run with `--limit 1` first to validate connectivity.
- If FACT fails with missing endpoint key, run with `--no-fact` or provide `JINA_API_KEY`.

## Suggested team workflow

1. Freeze judge config for the comparison window.
2. Create a per-team named `run_id`.
3. Run smoke tests and then controlled batches.
4. Store run IDs + notes externally (or in your lab tracker).
5. Compare using manifest metadata + `summary.txt`, and keep score interpretations tied to the exact judge model.
