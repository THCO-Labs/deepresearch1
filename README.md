# Deep Research Agent — Architecture & Design Document

A multi-agent research system that orchestrates web research, synthesizes findings, and produces cited reports. Built with LangChain, OpenRouter, Anthropic, and Tavily. Designed to run on free/low-cost API tiers with aggressive call optimization.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Provider Split](#provider-split)
3. [Architecture](#architecture)
4. [Code Walkthrough](#code-walkthrough)
5. [Prompt Design](#prompt-design)
6. [Persistence & Resumability](#persistence--resumability)
7. [Issues Encountered & Solutions](#issues-encountered--solutions)
8. [Configuration & Scaling](#configuration--scaling)
9. [Running the Agent](#running-the-agent)

---

## System Overview

The agent follows an **orchestrator/worker pattern**: a top-level orchestrator LLM plans the research, delegates to specialized sub-agent workers that perform web searches, then synthesizes their findings into a final cited report.

```
User Question
     │
     ▼
┌──────────────────────────────────┐
│  Orchestrator                     │
│  OpenRouter / DeepSeek V3 (free) │  ← plans, delegates, synthesizes, writes report
└──────────────┬───────────────────┘
               │ task() tool call
               ▼
┌──────────────────────────────────┐
│  Sub-Agent                        │
│  Anthropic / Claude Haiku 4.5    │  ← searches the web, returns structured findings
└──────────────┬───────────────────┘
               │ tavily_search() tool call (max 2 per sub-agent)
               ▼
┌──────────────────────────────────┐
│  Tavily + Web Fetch               │  ← fetches results and full page content
└──────────────────────────────────┘
```

Every sub-agent's findings are saved to disk as `findings_N.md` the moment they complete. If the process crashes or hits a rate limit, re-running picks up from the saved findings.

---

## Provider Split

### Why Two Providers?

Different parts of the pipeline have different requirements:

| Role | Needs | Provider Chosen |
|------|-------|-----------------|
| **Orchestrator** | Strong reasoning, planning, long context for synthesis | OpenRouter / DeepSeek V3 `:free` |
| **Sub-agents** | Reliable tool calling, low latency | Anthropic / Claude Haiku 4.5 |

### Orchestrator: OpenRouter + DeepSeek V3 (free)

OpenRouter's free tier gives 50 req/day (1,000/day with $10 credit loaded). The orchestrator makes ~8-10 calls per run — well within budget. DeepSeek V3 was chosen because it handles long-context synthesis well and is available as a free model on OpenRouter (`:free` suffix).

OpenRouter is accessed via the OpenAI-compatible API using `langchain-openai`'s `ChatOpenAI` with a custom base URL. No new SDK needed.

### Sub-agents: Anthropic / Claude Haiku 4.5

Claude Haiku was chosen for sub-agents for one overriding reason: **it has the most reliable structured tool calling of any available model**. The entire debugging history of this project (see Issues below) was caused by models that either output tool calls as raw text or don't register tools properly. Haiku eliminates that class of problem entirely.

Rate limits: Anthropic Tier 1 (requires $5 minimum deposit) gives **50 RPM** across all Claude models with 50,000 ITPM for Haiku. At 2 searches per sub-agent, each sub-agent consumes ~4 API calls. A full run of 1-2 sub-agents uses ~12-16 Haiku calls — far below the rate limit.

Cost: Haiku 4.5 is $1.00/$5.00 per million tokens input/output. A full research run costs well under $0.01.

---

## Architecture

### Why Not `deepagents`?

The original codebase used `deepagents` (`create_deep_agent`). It has a fundamental incompatibility with any provider except Anthropic and OpenAI: it injects a `task()` tool into the system prompt as text but never registers it in the API request's `tools` parameter. Groq (and OpenRouter) strictly validate that every tool the model invokes was declared upfront. The fix was to replace `deepagents` entirely with a manual ~60-line implementation that registers `task()` as a proper `@tool`.

### Component Breakdown

| Component | Role | Implementation |
|-----------|------|----------------|
| **Orchestrator** | Plans, delegates, synthesizes, writes report | `_run_agent()` — tool-calling loop, bound to 4 tools |
| **Sub-agent** | Executes web searches, returns structured findings | `_run_subagent()` — tool-calling loop, bound to `tavily_search` |
| **`task()` tool** | Bridge between orchestrator and sub-agent | `@tool` that calls `_run_subagent()` internally |
| **`tavily_search()` tool** | Web search + page fetch | Tavily API + `fetch_webpage_content()`, truncated |
| **`write_file()` / `read_file()`** | Filesystem persistence | Scoped to `output/` directory |
| **`invoke_with_retry()`** | Resilient LLM calls | Handles 429s with exponential backoff + wait-time parsing |
| **Progress logger** | Observability | Timestamps every step to `output/progress.md` and stdout |
| **Resume loader** | Crash recovery | On startup, injects saved `findings_*.md` into orchestrator prompt |

---

## Code Walkthrough

### Environment Setup

Loads three API keys from `.env`: `TAVILY_API_KEY`, `OPENROUTER_API_KEY`, and `ANTHROPIC_API_KEY`. Fails fast with a clear error message if any are missing. `OUTPUT_DIR` is created immediately so all persistence functions can assume it exists.

### `invoke_with_retry(model, messages, provider_name, max_retries=4)`

Wraps every LLM call with rate-limit handling. On a 429:
1. Defaults to exponential backoff (15s, 30s, 60s, 120s)
2. Parses the actual wait time from the error message using regex (`"try again in 19m10s"`)
3. Sleeps for the parsed duration + 5s buffer
4. Retries up to 4 times before raising

The `provider_name` parameter is used for logging so you can tell at a glance which provider is being rate-limited.

### `fetch_webpage_content(url)`

Fetches a URL and converts HTML to markdown. Truncated to **3,000 characters** (reduced from 4,000). Each Tavily search fetches 1 URL, so a sub-agent doing 2 searches adds at most ~6,000 chars of webpage content to its context — well within Haiku's limits and the orchestrator's context window.

### `tavily_search(query)`

Calls Tavily with `max_results=1` (injected, not visible to model). The `InjectedToolArg` annotation hides `max_results` and `topic` from the LLM's tool schema — the model only decides the query, not configuration details.

### Sub-Agent Runner (`_run_subagent`)

Key design decisions:

- **`MAX_SUBAGENT_SEARCHES = 2`**: Hard cap on searches. After the budget is exhausted, a `HumanMessage` is injected telling the model to stop calling tools and write its findings. This prevents the model from ignoring the budget instruction in the system prompt (models often do).
- **Separate model instance per run**: Each `_run_subagent` call gets a fresh `bind_tools()` call. This is fine for Haiku since tool binding is cheap.
- **`save_findings()` on every exit path**: Findings are saved whether the sub-agent finishes normally, hits the iteration limit, or exits early. The original code only saved on clean exits, which caused zero findings files to be written when sub-agents failed.

### `task()` Bridge Tool

```python
@tool
def task(description: str) -> str:
```

The orchestrator's system prompt (from the original design) tells it to call `task()` to delegate. By implementing it as a proper `@tool` registered via `bind_tools()`, OpenRouter sees it in the API request and accepts calls to it. The original `subagent_type` parameter was removed since there's only one sub-agent type.

### Orchestrator Runner (`_run_agent`)

1. **Resume check**: Loads `findings_*.md`. If found, appends as `## RESUME CONTEXT` in the system prompt.
2. **Completion check**: Returns `final_report.md` immediately if it exists.
3. **Main loop**: Up to 20 orchestrator iterations. Each iteration: invoke → execute tool calls → repeat. Saves latest orchestrator text to `orchestrator_latest.md` every iteration.
4. Tool results truncated to 12,000 chars at the orchestrator level.

### Model Initialization

```python
# Orchestrator: OpenRouter via OpenAI-compatible API
orchestrator_model = ChatOpenAI(
    model=ORCHESTRATOR_MODEL,
    openai_api_key=_openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    ...
)

# Sub-agents: Anthropic directly
subagent_model = ChatAnthropic(
    model=SUBAGENT_MODEL,
    anthropic_api_key=_anthropic_api_key,
    ...
)
```

OpenRouter uses the OpenAI-compatible endpoint — `langchain-openai`'s `ChatOpenAI` works with a custom `openai_api_base`. The `HTTP-Referer` and `X-Title` headers are required by OpenRouter's API. Anthropic sub-agents use `langchain-anthropic`'s `ChatAnthropic` directly.

---

## Prompt Design

Three prompt blocks, preserved from original design with one addition to `RESEARCHER_INSTRUCTIONS`:

### `RESEARCH_WORKFLOW_INSTRUCTIONS` (Orchestrator)
Defines the 6-step workflow, report structure templates, and citation format. Unchanged from original.

### `RESEARCHER_INSTRUCTIONS` (Sub-agent)
Significantly tightened from original. Key changes:
- Explicit `{max_searches}` budget variable injected at runtime
- Instruction to search **once** with a precise query before assessing
- Explicit stop conditions
- Removed the verbose "Tool Logs" requirement to reduce output tokens

### `SUBAGENT_DELEGATION_INSTRUCTIONS` (Orchestrator)
Guides when to use 1 vs multiple sub-agents. Default is always 1. Unchanged from original.

---

## Persistence & Resumability

| File | Written by | Purpose |
|------|-----------|---------|
| `progress.md` | `log_progress()` | Timestamped log of every action |
| `findings_N.md` | `save_findings()` | Sub-agent results, one per delegation |
| `orchestrator_latest.md` | Orchestrator loop | Last orchestrator output |
| `research_request.md` | Orchestrator via `write_file` | The original question |
| `final_report.md` | Orchestrator via `write_file` | The synthesized report |

**Resume flow:**
```
py agent.py
  ├─ final_report.md exists?  →  return immediately
  ├─ findings_*.md exist?     →  inject as RESUME CONTEXT, skip to synthesis
  └─ nothing                  →  full run from scratch
```

To force a fresh run: delete the `output/` folder.

---

## Issues Encountered & Solutions

### Issue 1: `groq-1` Model Not Found
**Error:** `groq.NotFoundError: The model 'groq-1' does not exist`
**Cause:** Invalid Groq model identifier in original code.
**Fix:** Changed to a valid model ID.

### Issue 2: `deepagents` Tool Registration Incompatibility with Groq
**Error:** `tool call validation failed: attempted to call tool 'task' which was not in request.tools`
**Cause:** `deepagents` describes `task()` in the system prompt text but never registers it in the API's `tools` parameter. Groq (unlike Anthropic/OpenAI) strictly validates this.
**Fix:** Replaced `deepagents` with a manual orchestrator/sub-agent implementation where `task()` is a proper registered `@tool`.

### Issue 3: Missing `write_file` / `read_file` Tools
**Error:** `attempted to call tool 'write_file' which was not in request.tools`
**Cause:** `deepagents` provided these tools internally. After removing it, they weren't defined or registered.
**Fix:** Added explicit `@tool` definitions and registered them in `bind_tools()`.

### Issue 4: Llama 70B Broken Tool-Calling Format
**Error:** `Failed to call a function... failed_generation: '<function=tavily_search{...}>'`
**Cause:** Llama 3.3 70B on Groq emits tool calls as raw text (`<function=...>`) instead of structured JSON. Groq's API rejects this.
**Fix:** Switched to `openai/gpt-oss-120b` on Groq, which has reliable structured tool calling.

### Issue 5: Context Window Overflow
**Error:** `Please reduce the length of the messages or completion.`
**Cause:** `fetch_webpage_content()` returned full web pages (50,000+ chars). After 2-3 searches, conversation history exceeded the context window.
**Fix:** Truncated pages to 3,000 chars, sub-agent results to 6,000, orchestrator results to 12,000.

### Issue 6: Groq Daily Token Limit (200k TPD)
**Error:** `Rate limit reached... Limit 200000, Used 199306`
**Cause:** Groq's free tier has a 200k tokens/day limit. A multi-search agent run consumes most of it, and the retry waits stretched a single run to 6+ hours.
**Fix:** Moved sub-agents to Anthropic Haiku (50 RPM, ~$0.01/run) and orchestrator to OpenRouter free (50 req/day, sufficient for orchestration). Added `invoke_with_retry()` with exponential backoff.

### Issue 7: Sub-agents Ignored Search Budget
**Cause:** The researcher prompt said "use 2-3 searches maximum" but the model often ignored this instruction and continued searching.
**Fix:** Added a hard programmatic check: after `MAX_SUBAGENT_SEARCHES` searches are counted in the Python loop, a `HumanMessage` is injected forcing the model to write its final answer. The model cannot bypass an injected message the way it can ignore a system prompt instruction.

### Issue 8: No Findings Files Saved on Sub-agent Failure
**Cause:** `save_findings()` was only called on clean exits (normal completion or iteration limit). When a sub-agent failed due to a rate-limit `RuntimeError`, it exited via the exception path and never wrote a findings file — so resuming a failed run had nothing to resume from.
**Fix:** Ensured `save_findings()` is called on every exit path in `_run_subagent`.

---

## Configuration & Scaling

### Key Config Variables (top of file)

| Variable | Default | Notes |
|----------|---------|-------|
| `ORCHESTRATOR_MODEL` | `deepseek/deepseek-chat-v3-0324:free` | Any OpenRouter free model |
| `SUBAGENT_MODEL` | `claude-haiku-4-5` | Switch to `claude-sonnet-4-6` for higher quality |
| `MAX_SUBAGENT_SEARCHES` | `2` | Increase for more thorough research |
| `max_researcher_iterations` | `3` | Max sub-agent delegation rounds |
| `max_concurrent_research_units` | `3` | Max parallel sub-agents (prompt guidance only) |

### Switching Models

To use Sonnet for sub-agents (higher quality, ~10x cost):
```python
SUBAGENT_MODEL = "claude-sonnet-4-6"
```

To use a different OpenRouter free model for the orchestrator:
```python
ORCHESTRATOR_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
```

---

## Running the Agent

### Install Dependencies

```bash
pip install langchain langchain-openai langchain-anthropic langchain-core \
            tavily-python httpx markdownify python-dotenv
```

### Environment (`.env` file)

```env
TAVILY_API_KEY=tvly-your_key_here
OPENROUTER_API_KEY=sk-or-your_key_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

Get keys from:
- Tavily: https://app.tavily.com
- OpenRouter: https://openrouter.ai/keys
- Anthropic: https://console.anthropic.com/keys (requires $5 minimum deposit for Tier 1)

### Run

```bash
py agent.py
```

### Output Files

```
output/
├── progress.md              # Timestamped execution log
├── findings_1.md            # Sub-agent #1 results
├── findings_2.md            # Sub-agent #2 results (if applicable)
├── orchestrator_latest.md   # Orchestrator's last output
├── research_request.md      # Saved question (for self-verification)
└── final_report.md          # The final synthesized report
```

### Resuming After a Crash

Just run `py agent.py` again. Existing `findings_*.md` files are detected and the orchestrator skips to synthesis.

### Starting Fresh

Delete the `output/` folder and re-run.