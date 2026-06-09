# Deep Research Agent Evaluation Harness

This directory contains a modular harness for running the `deep_research_bench` evaluation on `dra/agent.py` or any compatible agent implementation.

## Architecture

The harness is split into four loosely coupled layers:

- `eval_harness.datasets` loads and filters task prompts.
- `eval_harness.agents` is the adapter layer that normalizes all agent interfaces.
- `eval_harness.judges` builds judge environment variables used by DRB scripts.
- `eval_harness.pipelines` (future extension point) will hold dedicated scoring/validation flows.

The Python CLI in `eval_harness.cli` wires those layers together through `eval_harness.engine.EvaluationHarness`.

### Adapter contract

All adapters return an article string:

- Input: `task` dict with at least `id`, `prompt`, `language`, `topic`
- Output: Markdown/text article string for that prompt

Built-in adapters:

- `dra_agent`: imports a module and calls `run_research(question=..., output_dir=..., fresh=...)`
- `python_callable`: imports `module:function` and calls it with `(question, output_dir=..., fresh=...)`
- `command`: executes a shell command with templated placeholders
- `raw_jsonl`: reads `id`/`prompt` rows from a precomputed JSONL file

### Isolation design

`generate` runs each task in its own directory:

- `eval_runs/<run_id>/tasks/<task_id>/`

This prevents cross-task leakage of `agent.py` state such as `final_report.md` or local cache files.

### Run lifecycle

1. `prepare()` creates a unique `run_id` and initializes `eval_runs/<run_id>/`.
2. The benchmark workspace is copied from `data_root` into `eval_runs/<run_id>/bench`.
3. Tasks are loaded and filtered from `query.jsonl`.
4. Agents produce one article per task and write raw JSONL.
5. Optional scoring stages run:
6. RACE stage (`deepresearch_bench_race.py`) when enabled.
7. FACT stage (`utils.extract`, `utils.deduplicate`, `utils.scrape`, `utils.validate`, `utils.stat`) when enabled.
8. `manifest.json` is finalized and a compact summary is produced.

## What the harness evaluates

- RACE scoring (default enabled): criterion alignment + structure from DRB.
- FACT scoring (default disabled because scraping is optional): citation extraction and validation.
- Output cleaning pass is done by DRB unless `--skip-cleaning` is used.

RACE does not require `JINA_API_KEY`.  
FACT requires web-capture support and will fail clearly if `JINA_API_KEY` is missing unless FACT is disabled.

## Folder layout created per run

Each run writes artifacts under `eval_runs/<run_id>/`:

- `manifest.json`
- `raw/<agent-id>.jsonl`
- `cleaned/<agent-id>.jsonl`
- `race/<agent-id>/`
  - `race_result.txt`
- `fact/extracted.jsonl`
- `fact/deduplicated.jsonl`
- `fact/scraped.jsonl`
- `fact/validated.jsonl`
- `fact/fact_result.txt`
- `summary.txt`
- `logs/agent/*.log`, `logs/judge/*.log` (when available)
- `tasks/<task_id>/` (per-task agent output directory)
- `bench/` (copied benchmark workspace)

`summary.txt` contains a human-readable high-level result, while `manifest.json` stores machine-readable metadata and run summary.

## Configuration

Primary configuration file: `configs/dra.example.json`.

Key sections:

- `data`: benchmark data location and dataset filenames
- `agent`: adapter setup
- `judge.race`: judge model/provider settings for RACE
- `judge.fact`: judge model/provider settings for FACT
- `run`: output root, filters, concurrency, resume controls
- `features`: enable/disable pipeline stages

## Quick start for `dra/agent.py`

1. Set up environment variables:

```bash
TAVILY_API_KEY=...
OPENROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
```

2. Ensure dependencies are installed from `requirements.txt`.

3. Run a smoke test:

```bash
python -m eval_harness run --config configs/dra.example.json --limit 1 --only-en --no-fact
```

4. Run a larger batch (example):

```bash
python -m eval_harness run --config configs/dra.example.json --limit 2 --only-en --no-fact
```

If needed, skip FACT for quick turnaround with `--no-fact` in the command or `"run_fact": false` in config.

## CLI

```bash
python -m eval_harness run --config <config> [--limit N] [--only-en|--only-zh] [--force] [--no-race] [--no-fact] [--skip-cleaning] [--dry-run]
python -m eval_harness generate --config <config> [--limit N] [--only-en|--only-zh]
python -m eval_harness race --config <config> [--run-dir if implemented later]
python -m eval_harness fact --config <config> [--run-dir if implemented later]
python -m eval_harness summarize <run_dir>
```

## Swap in another agent

### Python callable

```json
{
  "agent": {
    "type": "python_callable",
    "module_path": "my_pkg.my_agent",
    "function": "run_research",
    "name": "my-agent"
  }
}
```

### Command adapter

```json
{
  "agent": {
    "type": "command",
    "name": "cmd-agent",
    "command": "python -m my_agent_runner --question \"{prompt}\" --output-file \"{output_file}\""
  }
}
```

### Raw JSONL adapter

```json
{
  "agent": {
    "type": "raw_jsonl",
    "name": "baseline",
    "jsonl_path": "/path/to/precomputed.jsonl"
  }
}
```

## Swap judge providers and keys

RACE and FACT are configured separately:

- Set `judge.race` and `judge.fact` blocks independently.
- Example with OpenRouter-compatible endpoint:

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

To compare across agents, keep judge config identical so scores remain comparable.

## Run control and resume behavior

- `--limit` and `only_en`/`only_zh` reduce the task set.
- `--force` re-runs completed task ids in the filtered set.
- `--no-race` / `--no-fact` run only the selected stages.
- If RACE/FACT artifacts already exist and `--force` is not set, existing task rows are reused by task id.

## Expected outputs by stage

- `generate`: only `raw/<agent-id>.jsonl` is guaranteed.
- `run`: generate + enabled scoring stages.
- `summarize`: reads and prints `manifest.json["summary"]` if present.

## Notes from design intent

- Cross-platform (Windows/macOS/Linux) by using Python orchestration.
- No Bash-only assumptions.
- Resumable JSONL writes with task-id keys.
- All generated benchmark artifacts stay under `eval_runs/<run_id>/`.
- Copying benchmark data/scripts into each run folder makes runs reproducible and avoids hard dependency on external working directories.
- `.gitignore` should include `eval_runs/` to keep large run artifacts out of version control.
