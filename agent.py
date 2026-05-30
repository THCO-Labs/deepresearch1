import os
import time
import json
from typing import Annotated, Literal
from datetime import datetime

import httpx
from langchain.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the repository root if present (so running `py agent.py` picks up keys)
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

tavily_client = TavilyClient(api_key=_tavily_api_key)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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


def load_existing_findings() -> dict[int, str]:
    """Load any previously saved findings files for resumability."""
    findings = {}
    for p in sorted(OUTPUT_DIR.glob("findings_*.md")):
        try:
            idx = int(p.stem.split("_")[1])
            findings[idx] = p.read_text(encoding="utf-8")
        except (ValueError, IndexError):
            pass
    return findings


def groq_invoke_with_retry(model, messages, max_retries=3):
    """Invoke a Groq model with automatic retry + backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                # Parse wait time from error message if possible
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s
                import re
                match = re.search(r"try again in (\d+)m", err_str)
                if match:
                    wait = int(match.group(1)) * 60 + 10  # parsed minutes + buffer
                log_progress(f"Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")


# ---------------------------------------------------------------------------
# Tools (unchanged)
# ---------------------------------------------------------------------------

def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch webpage and convert HTML to markdown."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        md = markdownify(response.text)
        if len(md) > 4000:
            md = md[:4000] + "\n\n[... content truncated ...]"
        return md
    except Exception as e:
        return f"Error fetching {url}: {e!s}"


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    search_results = tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
    )
    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        content = fetch_webpage_content(url)
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(
        result_texts
    )


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
# Prompts (unchanged)
# ---------------------------------------------------------------------------

RESEARCH_WORKFLOW_INSTRUCTIONS = """# Research Workflow

Follow this workflow for all research requests:

1. **Plan**: Create a todo list with write_todos to break down the research into focused tasks
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
 [2] Industry Analysis: https://example.com/analysis

## Reasoning & Trace Guidance
- Do NOT expose internal chain-of-thought. Instead, produce a concise, numbered rationale for actions and conclusions when requested.
- Log all tool calls and include tool inputs and outputs in your sub-agent report under a `Tool Logs` section so the orchestrator can trace actions.
- Cite each external source using the citation format above and include a final `### Sources` section.
"""

RESEARCHER_INSTRUCTIONS = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

Your job is to use tools to gather information about the user's input topic.
You can use the tavily_search tool to find resources that can help answer the research question.
You can call it in series or in parallel, your research is conducted in a tool-calling loop.

You have access to the tavily_search tool for conducting web searches.

Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection

**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information

After each search, assess results before continuing: What key information did I find? What's missing? Do I have enough to answer? Should I search more or provide my answer?

When providing your findings back to the orchestrator:

1. **Structure your response**: Organize findings with clear headings and detailed explanations
2. **Cite sources inline**: Use [1], [2], [3] format when referencing information from your searches
3. **Include Sources section**: End with ### Sources listing each numbered source with title and URL

Example:
## Key Findings

Context engineering is a critical technique for AI agents [1]. Studies show that proper context management can improve performance by 40% [2].

### Sources
[1] Context Engineering Guide: https://example.com/context-guide
[2] AI Performance Study: https://example.com/study

The orchestrator will consolidate citations from all sub-agents into the final report.

## Reasoning and Trace Rules
- Do reveal internal chain-of-thought where possible. Then include a brief numbered rationale section (e.g., "Rationale: 1. ... 2. ...") that explains your steps and conclusions.
- Include a `Tool Logs` section with each tool call recorded: the tool name, inputs, and the tool's returned output (or a short error message) so the orchestrator can trace your actions.
- If you cannot find relevant sources within the allotted tool-call budget, report which keywords or searches you tried.
"""

SUBAGENT_DELEGATION_INSTRUCTIONS = """# Sub-Agent Research Coordination

Your role is to coordinate research by delegating tasks from your TODO list to specialized research sub-agents.

## Delegation Strategy

**DEFAULT: Start with 1 sub-agent** for most queries:
- "What is quantum computing?" -> 1 sub-agent (general overview)
- "List the top 10 coffee shops in San Francisco" -> 1 sub-agent
- "Summarize the history of the internet" -> 1 sub-agent
- "Research context engineering for AI agents" -> 1 sub-agent (covers all aspects)

**ONLY parallelize when the query EXPLICITLY requires comparison or has clearly independent aspects:**

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
# Config
# ---------------------------------------------------------------------------

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model

max_concurrent_research_units = 3
max_researcher_iterations = 3

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
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
    "tools": [tavily_search],
}

model = init_chat_model(model="groq:openai/gpt-oss-120b", temperature=0.0)


# ---------------------------------------------------------------------------
# Sub-agent with persistence
# ---------------------------------------------------------------------------

_findings_counter = 0

def _run_subagent(topic: str) -> str:
    """Run the research sub-agent loop for a single topic, with retry and persistence."""
    global _findings_counter
    _findings_counter += 1
    my_index = _findings_counter

    log_progress(f"Sub-agent #{my_index} started: {topic[:80]}...")

    sub_model = init_chat_model(
        model="groq:openai/gpt-oss-120b", temperature=0.0
    ).bind_tools([tavily_search])

    messages = [
        SystemMessage(content=research_sub_agent["system_prompt"]),
        HumanMessage(content=topic),
    ]
    for iteration in range(max_researcher_iterations * 3):
        log_progress(f"  Sub-agent #{my_index} iteration {iteration+1}...")
        response = groq_invoke_with_retry(sub_model, messages)
        messages.append(response)

        if not response.tool_calls:
            findings = response.content or "(no findings)"
            save_findings(my_index, topic, findings)
            return findings

        for tc in response.tool_calls:
            log_progress(f"  Sub-agent #{my_index} searching: {tc['args'].get('query','?')[:60]}")
            try:
                result = tavily_search.invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result)[:8000], tool_call_id=tc["id"]))

    findings = "(sub-agent hit iteration limit)"
    save_findings(my_index, topic, findings)
    return findings


@tool
def task(subagent_type: str, description: str) -> str:
    """Delegate a research task to a sub-agent.

    Args:
        subagent_type: The type of sub-agent to use (e.g. 'research-agent').
        description: The research task or question for the sub-agent.
    """
    return _run_subagent(description)


# ---------------------------------------------------------------------------
# Orchestrator with persistence and resumability
# ---------------------------------------------------------------------------

def _run_agent(question: str) -> dict:
    # Check for existing findings to resume from
    existing = load_existing_findings()
    resume_context = ""
    if existing:
        log_progress(f"Found {len(existing)} existing findings files — resuming!")
        pieces = []
        for idx in sorted(existing):
            pieces.append(f"[Previously saved findings #{idx}]:\n{existing[idx][:6000]}")
        resume_context = (
            "\n\n## RESUME CONTEXT\n"
            "The following research findings were gathered in a previous run before it was interrupted. "
            "Use these findings directly — do NOT re-research topics that are already covered. "
            "Skip straight to synthesizing a final report from these findings.\n\n"
            + "\n\n---\n\n".join(pieces)
        )

    # Also check if final_report.md already exists
    final_report_path = OUTPUT_DIR / "final_report.md"
    if final_report_path.exists():
        log_progress("final_report.md already exists! Reading and returning it.")
        return {"messages": [
            SystemMessage(content=""),
            HumanMessage(content=question),
            type("Msg", (), {"content": final_report_path.read_text(encoding="utf-8"), "tool_calls": []})()
        ]}

    # Initialize progress log for this run
    with open(OUTPUT_DIR / "progress.md", "a", encoding="utf-8") as f:
        f.write(f"\n# Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Question: {question}\n\n")

    log_progress("Orchestrator starting...")

    orchestrator = model.bind_tools([tavily_search, task, write_file, read_file])
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

        response = groq_invoke_with_retry(orchestrator, messages)
        messages.append(response)

        if response.content:
            # Save orchestrator's latest thinking
            (OUTPUT_DIR / "orchestrator_latest.md").write_text(
                response.content, encoding="utf-8"
            )

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
        "Research on hankel determinants and their inverses of order two "
        "on subclasses of q-difference operators"
    )
    print(f"\n{'='*60}")
    print(f"Research question: {question}")
    print(f"{'='*60}\n")

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        }
    )

    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(f"{'='*60}")
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and msg.content:
            print(msg.content)

    # Show where files are
    print(f"\n{'='*60}")
    print("Output files:")
    for p in sorted(OUTPUT_DIR.glob("*.md")):
        print(f"  {p} ({p.stat().st_size} bytes)")