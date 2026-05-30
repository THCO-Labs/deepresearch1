# Deep Research Agent — Architecture & Design Document

A multi-agent research system that orchestrates web research, synthesizes findings, and produces cited reports. Built with LangChain, Groq inference, and Tavily search. Designed to run on consumer hardware using free-tier APIs.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Code Walkthrough](#code-walkthrough)
4. [Prompt Design](#prompt-design)
5. [Persistence & Resumability](#persistence--resumability)
6. [Issues Encountered & Solutions](#issues-encountered--solutions)
7. [Configuration & Scaling](#configuration--scaling)
8. [Running the Agent](#running-the-agent)

---

## System Overview

The agent follows an **orchestrator/worker pattern**: a top-level orchestrator LLM plans the research, delegates to specialized sub-agent workers that perform web searches, then synthesizes their findings into a final cited report.

```
User Question
     │
     ▼
┌─────────────┐
│ Orchestrator │ ── plans research, delegates, synthesizes, writes report
└──────┬──────┘
       │ task() tool call
       ▼
┌─────────────┐
│  Sub-Agent   │ ── searches the web via Tavily, returns structured findings
└──────┬──────┘
       │ tavily_search() tool call
       ▼
┌─────────────┐
│ Tavily + Web │ ── fetches search results and full page content
└─────────────┘
```

Every sub-agent's findings are persisted to disk as `findings_N.md`. If the process crashes or hits a rate limit, re-running the script picks up from the saved findings rather than starting from scratch.

---

## Architecture

### Why Not `deepagents`?

The original codebase used the `deepagents` library (`create_deep_agent`), which provides a high-level orchestrator/sub-agent abstraction. However, `deepagents` has a fundamental incompatibility with Groq's API:

**The problem:** `deepagents` injects a `task()` tool into the LLM's system prompt as text, telling the model "you can call `task()` to delegate." But it never registers `task()` in the API request's `tools` parameter. This works with Anthropic and OpenAI because they're more lenient with tool-calling formats. Groq strictly validates that every tool the model invokes was declared upfront in `tools` — so when the model tried to call `task()`, Groq returned a 400 error:

```
tool call validation failed: attempted to call tool 'task' which was not in request.tools
```

**The solution:** Replace `deepagents` with a lightweight manual implementation (~60 lines) that does the same thing — orchestrator delegates to sub-agents — but with `task()` properly registered as a LangChain `@tool`. This gives us full control over tool registration and makes the system Groq-compatible.

### Why Groq over Anthropic?

The agent makes many sequential LLM calls (orchestrator iterations + sub-agent iterations + tool calls). Anthropic's API enforces a 5-request-per-minute rate limit on lower tiers, which an agent loop burns through almost instantly. Groq's free tier has higher request-rate limits and much faster inference (~280-500 tokens/sec), making it practical for agentic workloads.

### Component Breakdown

| Component | Role | Implementation |
|-----------|------|----------------|
| **Orchestrator** | Plans research, delegates to sub-agents, synthesizes, writes report | `_run_agent()` — a tool-calling loop with the model bound to 4 tools |
| **Sub-agent** | Executes web searches on a specific topic, returns findings | `_run_subagent()` — a tool-calling loop with the model bound to `tavily_search` |
| **`task()` tool** | Bridge between orchestrator and sub-agent | A registered `@tool` that the orchestrator calls; internally runs `_run_subagent()` |
| **`tavily_search()` tool** | Web search | Calls Tavily API, fetches full page content, returns truncated markdown |
| **`write_file()` / `read_file()` tools** | Filesystem persistence | Write/read files in the `output/` directory |
| **`groq_invoke_with_retry()`** | Resilient LLM calls | Wraps every Groq call with rate-limit detection, wait-time parsing, and exponential backoff |
| **Progress logger** | Observability | Timestamps every step to `output/progress.md` and stdout |
| **Resume loader** | Crash recovery | On startup, loads any `findings_*.md` files and injects them into the orchestrator prompt |

---

## Code Walkthrough

### Environment Setup (Lines 1–31)

```python
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
```

Loads API keys from a `.env` file adjacent to the script. Fails fast with a clear error message if keys are missing — no silent failures deep in the Groq/Tavily stack.

`OUTPUT_DIR` is created immediately so all persistence functions can assume it exists.

### Helper Functions (Lines 36–83)

**`log_progress(message)`** — Dual-writes to `output/progress.md` and stdout. Every significant action (sub-agent start, search query, orchestrator iteration, errors) goes through this. The timestamp format is `HH:MM:SS` for quick scanning.

**`save_findings(index, topic, content)`** — Writes sub-agent results to `output/findings_N.md` the moment they're available. This is the key persistence primitive: if the process dies after a sub-agent finishes but before the orchestrator synthesizes, the findings survive on disk.

**`load_existing_findings()`** — Scans `output/` for `findings_*.md` files using glob, parses the index from the filename, and returns a `{index: content}` dict. Called once at the start of `_run_agent()` to detect a resumable state.

**`groq_invoke_with_retry(model, messages, max_retries=3)`** — The rate-limit handler. On a 429 error, it:
1. Defaults to progressive backoff (30s, 60s, 90s)
2. Parses the actual wait time from Groq's error message (`"try again in 19m10s"`) using regex
3. Sleeps for the parsed duration + 10s buffer
4. Retries up to `max_retries` times before raising

This handles both per-minute rate limits (short waits) and daily token limits (longer waits).

### Tools (Lines 90–166)

**`fetch_webpage_content(url)`** — Fetches a URL and converts HTML to markdown using `markdownify`. The critical addition is **truncation to 4,000 characters**. Without this, a single webpage could consume 50k+ characters of context, and after 2-3 searches the conversation history would exceed Groq's 131k context window.

**`tavily_search(query)`** — Calls the Tavily search API, then fetches full content for each result URL. Uses `InjectedToolArg` for `max_results` and `topic` parameters — these are set programmatically and hidden from the LLM's tool schema, so the model only sees `query` as a parameter. This prevents the model from wasting tokens deciding between topic categories.

**`write_file(filename, content)` / `read_file(filename)`** — Simple filesystem tools scoped to `OUTPUT_DIR`. The orchestrator uses these to save `research_request.md` and `final_report.md` as instructed by its system prompt. These had to be explicitly registered as tools (the original `deepagents` provided them internally).

### Sub-Agent Runner (Lines 368–406)

```python
def _run_subagent(topic: str) -> str:
```

A standalone tool-calling loop that:

1. Creates a fresh model instance with `tavily_search` bound as its only tool
2. Seeds the conversation with the researcher system prompt + the topic
3. Loops: invoke model → if no tool calls, return content as findings → if tool calls, execute them and append results
4. Caps at `max_researcher_iterations * 3` iterations (default 9) as a safety valve
5. Saves findings to disk via `save_findings()` before returning

Each sub-agent gets its own message history, so context doesn't leak between topics.

**Why `bind_tools([tavily_search])` instead of all tools?** The sub-agent should only search — it shouldn't write files or delegate further. Limiting the tool set prevents the sub-agent from going off-script.

### The `task()` Bridge (Lines 409–417)

```python
@tool
def task(subagent_type: str, description: str) -> str:
```

This is the architectural linchpin. The orchestrator's system prompt (inherited from the original `deepagents` design) tells it to call `task()` to delegate research. By implementing `task()` as a proper `@tool`, Groq sees it in the API request's `tools` list and accepts the call. Internally it just calls `_run_subagent(description)`.

The `subagent_type` parameter exists for compatibility with the original prompt (which references "research-agent"), but the current implementation ignores it since there's only one sub-agent type.

### Orchestrator Runner (Lines 424–498)

```python
def _run_agent(question: str) -> dict:
```

The main execution loop:

1. **Resume check:** Loads any existing `findings_*.md` files. If found, appends a `## RESUME CONTEXT` section to the system prompt containing the findings, with instructions to skip re-researching and go straight to synthesis.

2. **Completion check:** If `final_report.md` already exists, returns it immediately without calling the LLM at all.

3. **Main loop:** Up to 20 iterations of: invoke orchestrator → if no tool calls, break → execute tool calls → append results. The orchestrator has access to all 4 tools (`tavily_search`, `task`, `write_file`, `read_file`).

4. **Observability:** Every iteration saves the orchestrator's latest text output to `orchestrator_latest.md`, so even if the process crashes mid-loop, you can see what the orchestrator was thinking.

Tool results are truncated to 12,000 characters at the orchestrator level (vs 8,000 at the sub-agent level) because the orchestrator needs to see more of the sub-agent's synthesized findings than the sub-agent needs to see of raw webpage content.

### Agent Shim (Line 501)

```python
agent = type("Agent", (), {"invoke": staticmethod(lambda inp: _run_agent(inp["messages"][0].content))})()
```

A one-liner that creates an object with an `.invoke()` method matching the interface the `__main__` block expects (same signature as `deepagents`' agent). This keeps the entry point code unchanged from the original.

---

## Prompt Design

The system uses three prompt blocks, all preserved from the original `deepagents` design:

### `RESEARCH_WORKFLOW_INSTRUCTIONS`

The orchestrator's primary directive. Defines a 6-step workflow:
1. Plan → 2. Save request → 3. Delegate research → 4. Synthesize → 5. Write report → 6. Verify

Also specifies report structure templates (comparison, list, overview) and citation format (`[1]`, `[2]` inline with a `### Sources` section). The citation consolidation instruction ("each unique URL gets one number across ALL sub-agent findings") is important for multi-source reports.

### `RESEARCHER_INSTRUCTIONS`

The sub-agent's directive. Emphasizes efficiency: tool-call budgets (2-3 for simple, 5 max for complex), explicit stop conditions ("3+ sources found", "last 2 searches returned similar info"), and a structured output format with inline citations and a Sources section.

The "think like a human researcher with limited time" framing is a deliberate prompt engineering choice — it discourages the model from exhaustively searching when adequate information is already available, which reduces API calls and context consumption.

### `SUBAGENT_DELEGATION_INSTRUCTIONS`

Guides the orchestrator on when to use 1 vs multiple sub-agents. The strong default is 1 sub-agent for most queries, with parallelization only for explicit comparisons. This is a token-efficiency optimization: on Groq's free tier, each sub-agent loop costs 3-6 API calls, so unnecessary parallelization burns through rate limits fast.

---

## Persistence & Resumability

The agent writes several files to the `output/` directory during execution:

| File | Written by | Purpose |
|------|-----------|---------|
| `progress.md` | `log_progress()` | Timestamped log of every action — searchable post-mortem |
| `findings_N.md` | `save_findings()` | Sub-agent results, one per delegation. Survives crashes. |
| `orchestrator_latest.md` | Orchestrator loop | Last orchestrator output — see what it was thinking if it crashes |
| `research_request.md` | Orchestrator (via `write_file`) | The original question, saved for self-verification |
| `final_report.md` | Orchestrator (via `write_file`) | The final synthesized report |

### Resume Flow

```
py agent.py
  │
  ├─ output/final_report.md exists?
  │   └─ YES → return it immediately, done
  │
  ├─ output/findings_*.md exist?
  │   └─ YES → inject into system prompt as RESUME CONTEXT
  │            → orchestrator skips research, goes to synthesis
  │
  └─ NO existing files → full run from scratch
```

To force a completely fresh run, delete the `output/` folder.

---

## Issues Encountered & Solutions

### Issue 1: `groq-1` Model Not Found

**Error:** `groq.NotFoundError: The model 'groq-1' does not exist`

**Cause:** The original code used `model="groq:groq-1"` which isn't a valid Groq model identifier.

**Fix:** Changed to `groq:llama-3.3-70b-versatile`, then ultimately to `groq:openai/gpt-oss-120b`.

### Issue 2: `deepagents` Tool Registration Incompatibility

**Error:** `tool call validation failed: attempted to call tool 'task' which was not in request.tools`

**Cause:** `deepagents` describes the `task()` tool in the system prompt text but doesn't include it in the Groq API's `tools` parameter. Groq strictly validates that any tool the model invokes was declared in the request. Anthropic/OpenAI are more lenient here.

**Fix:** Replaced `deepagents` entirely with a manual orchestrator/sub-agent implementation where `task()` is a proper `@tool` registered via `bind_tools()`.

### Issue 3: Missing `write_file` / `read_file` Tools

**Error:** `attempted to call tool 'write_file' which was not in request.tools`

**Cause:** The orchestrator's system prompt tells it to use `write_file()` and `read_file()`, but after removing `deepagents` (which provided them internally), they weren't defined or registered.

**Fix:** Added explicit `@tool` definitions for both, plus `OUTPUT_DIR` setup. Registered them in the orchestrator's `bind_tools()` and `tool_map`.

### Issue 4: Llama 70B Broken Tool-Calling Format

**Error:** `Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.`

**Cause:** Llama 3.3 70B on Groq emits tool calls in a raw text format (`<function=tavily_search{...}></function>`) instead of structured `tool_calls` JSON. Groq's API rejects this malformed format.

**Fix:** Switched from `llama-3.3-70b-versatile` to `openai/gpt-oss-120b`, which has reliable structured tool-calling on Groq.

### Issue 5: Context Window Overflow

**Error:** `Please reduce the length of the messages or completion.`

**Cause:** `fetch_webpage_content()` was returning entire web pages (often 50,000+ characters) as markdown. After 2-3 Tavily searches, the conversation history exceeded Groq's 131k context window.

**Fix:** Three truncation layers:
- `fetch_webpage_content()` truncates each page to 4,000 characters
- Sub-agent tool results capped at 8,000 characters per message
- Orchestrator tool results capped at 12,000 characters per message

### Issue 6: Groq Daily Token Limit (200k TPD)

**Error:** `Rate limit reached... Limit 200000, Used 199306`

**Cause:** Groq's free tier allows only 200,000 tokens per day. A single research run with multiple sub-agent delegations and webpage fetches can consume most of this budget.

**Fix:** Added `groq_invoke_with_retry()` with automatic backoff and wait-time parsing. Added persistence so interrupted runs can resume from saved findings instead of starting over.

---

## Configuration & Scaling

### Key Config Values

| Variable | Default | Location | Notes |
|----------|---------|----------|-------|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Line 361 + 379 | Change in both places |
| `max_researcher_iterations` | `3` | Line 339 | Sub-agent search rounds |
| `max_concurrent_research_units` | `3` | Line 338 | Max parallel sub-agents |
| Orchestrator max iterations | `20` | Line 470 | Hard cap on orchestrator loop |
| Webpage truncation | `4000` chars | Line 99 | Per-page content limit |
| Sub-agent result truncation | `8000` chars | Line 402 | Per-tool-call limit |
| Orchestrator result truncation | `12000` chars | Line 495 | Per-tool-call limit |

### Multi-Key Rotation (For Higher Throughput)

Rate limits on Groq's free tier are per-organization. To increase throughput, create multiple Groq accounts and rotate keys:

```env
GROQ_API_KEY_1=gsk_abc...
GROQ_API_KEY_2=gsk_def...
GROQ_API_KEY_3=gsk_ghi...
```

```python
import itertools
_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 4)]
_key_cycle = itertools.cycle([k for k in _keys if k])

def get_next_model():
    return init_chat_model(
        model="groq:openai/gpt-oss-120b",
        temperature=0.0,
        api_key=next(_key_cycle),
    )
```

---

## Running the Agent

### Prerequisites

```bash
pip install langchain langchain-groq langchain-core tavily-python httpx markdownify python-dotenv
```

### Environment

Create a `.env` file in the same directory as `agent.py`:

```env
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly-your_key_here
```

### Execution

```bash
py agent.py
```

### Output

All files are written to `output/` adjacent to the script:

```
output/
├── progress.md              # Timestamped execution log
├── findings_1.md            # Sub-agent #1 results
├── findings_2.md            # Sub-agent #2 results (if applicable)
├── orchestrator_latest.md   # Orchestrator's last output
├── research_request.md      # Saved question (for verification)
└── final_report.md          # The final synthesized report
```

### Resuming After a Crash

Just run `py agent.py` again. The agent will detect existing findings and skip to synthesis.

### Starting Fresh

Delete the `output/` folder and re-run.
