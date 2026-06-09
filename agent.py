import os
import time
import json
import re
from typing import Annotated, Literal
from datetime import datetime

import httpx
from langchain.tools import InjectedToolArg, tool
from markdownify import markdownify
import openai
from tavily import TavilyClient
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the repository root if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

_tavily_api_key = os.getenv("TAVILY_API_KEY")
if not _tavily_api_key:
    raise RuntimeError(
        "Environment variable TAVILY_API_KEY is not set.\n"
        "Set it in PowerShell with: $env:TAVILY_API_KEY=\"your_key\"\n"
        "Or persistently with: setx TAVILY_API_KEY \"your_key\"\n"
    )

_openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not _openrouter_api_key:
    raise RuntimeError(
        "Environment variable OPENROUTER_API_KEY is not set.\n"
        "Set it in PowerShell with: $env:OPENROUTER_API_KEY=\"your_key\"\n"
    )

_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _anthropic_api_key:
    raise RuntimeError(
        "Environment variable ANTHROPIC_API_KEY is not set.\n"
        "Set it in PowerShell with: $env:ANTHROPIC_API_KEY=\"your_key\"\n"
    )

tavily_client = TavilyClient(api_key=_tavily_api_key)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

# Orchestrator: OpenRouter free tier (DeepSeek V3 — strong reasoning, free)
#ORCHESTRATOR_MODEL  = "deepseek/deepseek-chat-v3-0324:free"
#ORCHESTRATOR_MODEL = "openai/gpt-oss-120b:free"
#ORCHESTRATOR_MODEL = "openrouter/free"
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

# Sub-agents: Claude Haiku 4.5 — fast, cheap, best-in-class tool calling
# Switch to "claude-sonnet-4-6" for higher quality at higher cost
SUBAGENT_MODEL = "claude-sonnet-4-6"

# Max searches a sub-agent may perform — keep low to save API calls
MAX_SUBAGENT_SEARCHES = 2

max_concurrent_research_units = 3
max_researcher_iterations = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_progress(message: str):
    """Append a timestamped line to output/progress.md and print it."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"- [{ts}] {message}"
    print(line)
    with open(OUTPUT_DIR / "progress.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_findings(index: int, topic: str, content: str):
    """Save sub-agent findings to a numbered file."""
    path = OUTPUT_DIR / f"findings_{index}.md"
    header = f"# Findings #{index}: {topic}\n\n"
    path.write_text(header + content, encoding="utf-8")
    log_progress(f"Saved findings to output/findings_{index}.md ({len(content)} chars)")


def load_existing_findings() -> dict:
    """Load any previously saved findings files for resumability."""
    findings = {}
    for p in sorted(OUTPUT_DIR.glob("findings_*.md")):
        try:
            idx = int(p.stem.split("_")[1])
            findings[idx] = p.read_text(encoding="utf-8")
        except (ValueError, IndexError):
            pass
    return findings


def invoke_with_retry(model, messages, provider_name: str, max_retries: int = 4):
    """Invoke a model with automatic retry + backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "rate_limit" in err_str.lower()
                or "rate limit" in err_str.lower()
                or "overloaded" in err_str.lower()
            )
            if is_rate_limit:
                # Try to parse an explicit wait time from the error message
                wait = 15 * (2 ** attempt)   # 15s, 30s, 60s, 120s
                m = re.search(r"try again in (\d+)m(?:(\d+)s)?", err_str)
                if m:
                    wait = int(m.group(1)) * 60 + int(m.group(2) or 0) + 5
                log_progress(f"[{provider_name}] Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                log_progress(f"[{provider_name}] Error: {err_str[:200]}")
                raise
    raise RuntimeError(f"[{provider_name}] Rate limit exceeded after {max_retries} retries")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch webpage and convert HTML to markdown, truncated to 3000 chars."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        md = markdownify(response.text)
        if len(md) > 3000:
            md = md[:3000] + "\n\n[... content truncated ...]"
        return md
    except Exception as e:
        return f"Error fetching {url}: {e!s}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute

    Returns:
        Formatted search results with full webpage content
    """
    search_results = tavily_client.search(query, max_results=max_results, topic=topic)
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = fetch_webpage_content(url)
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(result_texts)


@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the output directory.

    Args:
        filename: Name of the file to write (e.g. 'final_report.md').
        content: Content to write to the file.
    """
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to output/{filename}"


@tool
def read_file(filename: str) -> str:
    """Read content from a file in the output directory.

    Args:
        filename: Name of the file to read.
    """
    path = OUTPUT_DIR / filename
    if not path.exists():
        return f"File output/{filename} does not exist."
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompts (unchanged from original)
# ---------------------------------------------------------------------------

RESEARCH_WORKFLOW_INSTRUCTIONS = """# Research Workflow

Follow this workflow for all research requests:

1. **Plan**: Create a todo list with write_file to break down the research into focused tasks
2. **Save the request**: Use write_file() to save the user's research question to `/research_request.md`
3. **Research**: Delegate research tasks to sub-agents using the task() tool - ALWAYS use sub-agents for research, never conduct research yourself
4. **Synthesize**: Review all sub-agent findings and consolidate citations (each unique URL gets one number across all findings)
5. **Write Report**: Write a comprehensive final report to `/final_report.md` (see Report Writing Guidelines below)
6. **Verify**: Read `/research_request.md` and confirm you've addressed all aspects with proper citations and structure

## Research Planning Guidelines
- Batch similar research tasks into a single TODO to minimize overhead
- For simple fact-finding questions, use 1 sub-agent
- For comparisons or multi-faceted topics, delegate to multiple parallel sub-agents
- Each sub-agent should research one specific aspect and return findings

## Report Writing Guidelines

When writing the final report to `/final_report.md`, follow these structure patterns:

**For comparisons:**
1. Introduction
2. Overview of topic A
3. Overview of topic B
4. Detailed comparison
5. Conclusion

**For lists/rankings:**
Simply list items with details - no introduction needed:
1. Item 1 with explanation
2. Item 2 with explanation
3. Item 3 with explanation

**For summaries/overviews:**
1. Overview of topic
2. Key concept 1
3. Key concept 2
4. Key concept 3
5. Conclusion

**General guidelines:**
- Use clear section headings (## for sections, ### for subsections)
- Write in paragraph form by default - be text-heavy, not just bullet points
- Do NOT use self-referential language ("I found...", "I researched...")
- Write as a professional report without meta-commentary
- Each section should be comprehensive and detailed
- Use bullet points only when listing is more appropriate than prose

**Citation format:**
- Cite sources inline using [1], [2], [3] format
- Assign each unique URL a single citation number across ALL sub-agent findings
- End report with ### Sources section listing each numbered source
- Number sources sequentially without gaps (1,2,3,4...)
- Format: [1] Source Title: URL (each on separate line for proper list rendering)
- Example:

 Some important finding [1]. Another key insight [2].

 ### Sources
 [1] AI Research Paper: https://example.com/paper
 [2] An Industry Analysis: https://example.com/analysis

## Reasoning & Trace Guidance
- Do NOT expose internal chain-of-thought in final report. Instead, produce a concise, numbered rationale for actions and conclusions when requested, and log your chain of thought in the final entry into progress.md.
- Log all tool calls and include tool inputs and outputs in your sub-agent report under a `Tool Logs` section so the orchestrator can trace actions.
- Cite each external source using the citation format above and include a final `### Sources` section.
"""

RESEARCHER_INSTRUCTIONS = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

Your job is to use the tavily_search tool to gather information. Be highly efficient — you have a strict budget of {max_searches} search calls maximum. Make them count.

Think like a human researcher with limited time:

1. **Read the question carefully** — identify the most specific query that will return useful results
2. **Search once with a precise query** — broad enough to find relevant pages, specific enough to avoid noise
3. **Assess immediately** — do you have enough to write a comprehensive answer? If yes, stop and write it
4. **Search a second time only if clearly needed** — fill a specific gap, not to find more of the same
5. **Stop at {max_searches} searches** — write your best answer from what you have

**Stop Immediately When**:
- You can answer the user's question comprehensively from current results
- Your last search returned similar information to the previous one

When providing findings:
1. Organize with clear headings (## sections)
2. Cite sources inline as [1], [2], [3]
3. End with ### Sources listing each numbered URL

## Reasoning and Trace Rules
- Include a brief numbered rationale explaining your search decisions.
- If you cannot find relevant sources, report which queries you tried.
"""

SUBAGENT_DELEGATION_INSTRUCTIONS = """# Sub-Agent Research Coordination

Your role is to coordinate research by delegating tasks from your TODO list to specialized research sub-agents.

## Delegation Strategy

**DEFAULT: Start with 1 sub-agent** for most queries:
- "What is quantum computing?" -> 1 sub-agent (general overview)
- "List the top 10 coffee shops in San Francisco" -> 1 sub-agent
- "Summarize the history of the internet" -> 1 sub-agent
- "Research context engineering for AI agents" -> 1 sub-agent (covers all aspects)

**ONLY parallelize when the query EXPLICITLY requires comparison OR has clearly independent aspects:**

**Explicit comparisons** -> 1 sub-agent per element:
- "Compare OpenAI vs Anthropic vs DeepMind AI safety approaches" -> 3 parallel sub-agents
- "Compare Python vs JavaScript for web development" -> 2 parallel sub-agents

**Clearly separated aspects** -> 1 sub-agent per aspect (use sparingly):
- "Research renewable energy adoption in Europe, Asia, and North America" -> 3 parallel sub-agents (geographic separation)
- Only use this pattern when aspects cannot be covered efficiently by a single comprehensive search

## Key Principles
- **Bias towards single sub-agent**: One comprehensive research task is more token-efficient than multiple narrow ones
- **Avoid premature decomposition**: Don't break "research X" into "research X overview", "research X techniques", "research X applications" - just use 1 sub-agent for all of X
- **Parallelize only for clear comparisons**: Use multiple sub-agents when comparing distinct entities or geographically separated data

## Parallel Execution Limits
- Use at most {max_concurrent_research_units} parallel sub-agents per iteration
- Make multiple task() calls in a single response to enable parallel execution
- Each sub-agent returns findings independently

## Research Limits
- Stop after {max_researcher_iterations} delegation rounds if you haven't found adequate sources
- Stop when you have sufficient information to answer comprehensively
- Bias towards focused research over exhaustive exploration"""


# ---------------------------------------------------------------------------
# Config / model init
# ---------------------------------------------------------------------------

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

current_date = datetime.now().strftime("%Y-%m-%d")

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)

research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the subagent. Give one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(
        date=current_date,
        max_searches=MAX_SUBAGENT_SEARCHES,
    ),
    "tools": [tavily_search],
}

# Orchestrator: OpenRouter (DeepSeek V3 free)
'''orchestrator_model = ChatOpenAI(
    model=ORCHESTRATOR_MODEL,
    openai_api_key=_openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    default_headers={
        "HTTP-Referer": "https://github.com/local/research-agent",
        "X-Title": "Research Agent",
    },
)'''

orchestrator_model = ChatAnthropic(
    model=ORCHESTRATOR_MODEL,
    anthropic_api_key=_anthropic_api_key,
    temperature=0.0,
    max_tokens=4096,
)

# Sub-agents: Claude Haiku (Anthropic API directly)
subagent_model = ChatAnthropic(
    model=SUBAGENT_MODEL,
    anthropic_api_key=_anthropic_api_key,
    temperature=0.0,
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Sub-agent runner (Claude Haiku)
# ---------------------------------------------------------------------------

_findings_counter = 0

def _run_subagent(topic: str) -> str:
    """Run the research sub-agent (Claude Haiku) for a single topic."""
    global _findings_counter
    _findings_counter += 1
    my_index = _findings_counter

    log_progress(f"Sub-agent #{my_index} started: {topic}...")

    bound_sub = subagent_model.bind_tools([tavily_search])

    messages = [
        SystemMessage(content=research_sub_agent["system_prompt"]),
        HumanMessage(content=topic),
    ]

    search_count = 0

    for iteration in range(MAX_SUBAGENT_SEARCHES * 2 + 2):
        log_progress(f"  Sub-agent #{my_index} iteration {iteration+1} (searches used: {search_count}/{MAX_SUBAGENT_SEARCHES})...")
        response = invoke_with_retry(bound_sub, messages, provider_name="Anthropic/Haiku")
        messages.append(response)

        if not response.tool_calls:
            findings = response.content or "(no findings)"
            save_findings(my_index, topic, findings)
            return findings

        for tc in response.tool_calls:
            if tc["name"] == "tavily_search":
                search_count += 1
                log_progress(f"  Sub-agent #{my_index} searching [{search_count}/{MAX_SUBAGENT_SEARCHES}]: {tc['args'].get('query','?')[:60]}")
            try:
                result = tavily_search.invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result)[:6000], tool_call_id=tc["id"]))

        # Inject a hard stop if budget is exhausted
        if search_count >= MAX_SUBAGENT_SEARCHES:
            messages.append(HumanMessage(
                content=f"You have used all {MAX_SUBAGENT_SEARCHES} searches. "
                        "Write your final findings now based on what you have. Do not call any more tools."
            ))

    findings = "(sub-agent hit iteration limit)"
    save_findings(my_index, topic, findings)
    return findings


@tool
def task(description: str) -> str:
    """Delegate a research task to a sub-agent.

    Args:
        description: The research task or question for the sub-agent.
    """
    return _run_subagent(description)


# ---------------------------------------------------------------------------
# Orchestrator runner (OpenRouter / DeepSeek V3)
# ---------------------------------------------------------------------------

def _run_agent(question: str) -> dict:
    # Resume check
    existing = load_existing_findings()
    resume_context = ""
    if existing:
        log_progress(f"Found {len(existing)} existing findings files — resuming!")
        pieces = [f"[Previously saved findings #{i}]:\n{existing[i]}" for i in sorted(existing)]
        resume_context = (
            "\n\n## RESUME CONTEXT\n"
            "The following research findings were gathered in a previous run before it was interrupted. "
            "Use these findings directly — do NOT re-research topics that are already covered. "
            "Skip straight to synthesizing a final report from these findings.\n\n"
            + "\n\n---\n\n".join(pieces)
        )

    # Already done check
    final_report_path = OUTPUT_DIR / "final_report.md"
    if final_report_path.exists():
        log_progress("final_report.md already exists! Reading and returning it.")
        return {"messages": [
            SystemMessage(content=""),
            HumanMessage(content=question),
            type("Msg", (), {"content": final_report_path.read_text(encoding="utf-8"), "tool_calls": []})()
        ]}

    with open(OUTPUT_DIR / "progress.md", "a", encoding="utf-8") as f:
        f.write(f"\n# Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Orchestrator: OpenRouter/{ORCHESTRATOR_MODEL}\n")
        f.write(f"Sub-agents:   Anthropic/{SUBAGENT_MODEL}\n")
        f.write(f"Max searches per sub-agent: {MAX_SUBAGENT_SEARCHES}\n")
        f.write(f"Question: {question}\n\n")

    log_progress(f"Orchestrator starting (OpenRouter/{ORCHESTRATOR_MODEL})...")
    log_progress(f"Sub-agents will use Anthropic/{SUBAGENT_MODEL}, max {MAX_SUBAGENT_SEARCHES} searches each")

    orchestrator = orchestrator_model.bind_tools([tavily_search, task, write_file, read_file])

    messages = [
        SystemMessage(content=INSTRUCTIONS + resume_context),
        HumanMessage(content=question),
    ]
    tool_map = {
        "tavily_search": tavily_search,
        "task": task,
        "write_file": write_file,
        "read_file": read_file,
    }

    for iteration in range(20):
        log_progress(f"Orchestrator iteration {iteration+1}...")

        response = invoke_with_retry(orchestrator, messages, provider_name="OpenRouter/DeepSeek")
        messages.append(response)

        if response.content:
            if isinstance(response.content, str):
                content_str = response.content
            else:
                content_str = json.dumps(response.content, indent=2)
            (OUTPUT_DIR / "orchestrator_latest.md").write_text(content_str, encoding="utf-8")

        if not response.tool_calls:
            log_progress("Orchestrator finished (no more tool calls).")
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            log_progress(f"Orchestrator calling: {tool_name}({json.dumps({k: str(v)[:60] for k,v in tc['args'].items()})})")
            fn = tool_map.get(tool_name)
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tool_name}"
            except Exception as e:
                result = f"Tool error: {e}"
                log_progress(f"  Tool error: {e}")
            messages.append(ToolMessage(content=str(result)[:12000], tool_call_id=tc["id"]))

    log_progress("Run complete.")
    return {"messages": messages}


agent = type("Agent", (), {"invoke": staticmethod(lambda inp: _run_agent(inp["messages"][0].content))})()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    question = (
        "CONTEXT:" +
"MALA'IKA is Sety.io's emergency response management platform, being tested with an Uber partnership in Lagos and Abuja. It supports SOS calls/form intake from users (in this case, Uber drivers/riders), Command Center validation by agents, responder dispatch, incident tracking, evidence capture, operational analytics, and Uber-facing reports." +
"Tagline: Technology for Saving Lives"+
"The MVP is focused on the Uber emergency response vertical:" +
"1. An Uber driver/rider triggers an SOS." +
"2. The Command Center receives the request in real time." +
"3. An agent validates the incident by phone or in-app communication." +
"4. The system classifies the incident by category, severity, ride status, and location." +
"5. The agent dispatches the nearest appropriate responder." +
"6. The requester receives status updates." +
"7. The case is resolved, closed, and included in analytics/reports." +
"RESEARCH QUESTION" +
"The main point of your research is on the question of what would be the most optimal in option for handling calls, in terms of price and quality of service in Nigeria. Compare between using Twilio, and Cloudtalk API to set up the emergency lines which command centre agnets will answer when a requester calls, collect transcripts, which we can then use for automatically creating incidents for agents to then manage by parsing the transcript with an LLM. Compare pricing, users reviews and forum discussions about them, especially on how htey perform in Nigeria, etc. return a full breakdown of the process that will be involved, step by step, for each approach, then make the comaprisons. Think through the user flow, and the processes that will be required for this flow to be complete, both on the requester and agent ends."
    )
    print(f"\n{'='*60}")
    print(f"Research question: {question}")
    print(f"Orchestrator: OpenRouter / {ORCHESTRATOR_MODEL}")
    print(f"Sub-agents:   Anthropic / {SUBAGENT_MODEL}")
    print(f"Max searches per sub-agent: {MAX_SUBAGENT_SEARCHES}")
    print(f"{'='*60}\n")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(f"{'='*60}")
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            print(msg.content)

    print(f"\n{'='*60}")
    print("Output files:")
    for p in sorted(OUTPUT_DIR.glob("*.md")):
        print(f"  {p} ({p.stat().st_size} bytes)")