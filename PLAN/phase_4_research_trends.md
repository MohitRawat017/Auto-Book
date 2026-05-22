# Phase 4 — Research & Trend Agents

## Goal

Add a Research Agent that gathers real-world sources before writing, and a Trend Agent that identifies demand signals for the topic. Research notes should flow into the Planner and Writer, and the final book should include source citations.

**Prerequisite**: Phase 3 is complete — the system produces Markdown + DOCX with images.

**Success Criteria**: Given a topic, the system researches it using a web search API, produces structured research notes with source URLs, feeds them into the planning and writing pipeline, and the final book includes a "Sources" section. The Trend Agent provides optional topic angles to the Planner.

---

## 4.1 — Decision Point: Search API

> **USER DECISION REQUIRED BEFORE IMPLEMENTATION**
>
> The implementer must choose a search API. Options and trade-offs:
>
> | API | Free Tier | Quality | Notes |
> |---|---|---|---|
> | **Tavily** | 1,000 searches/month free | High — built for AI agents, returns clean summaries | Best fit for this project. Recommended. |
> | **SerpAPI** | 100 searches/month free | High — real Google results | More raw data, needs more post-processing |
> | **Brave Search API** | 2,000 queries/month free | Medium-High | Good free tier, privacy-focused |
> | **DuckDuckGo** (`duckduckgo-search`) | Unlimited, no API key | Medium — unofficial, can break | Zero cost but unreliable |
>
> **Recommendation**: Tavily. It returns AI-ready summaries, has a generous free tier, 
> and has a LangChain integration (`langchain-community` includes `TavilySearchResults`).
>
> The plan below assumes Tavily but is written to be adaptable to any provider.
> **Ask the user which API to use before starting implementation.**

---

## 4.2 — New Dependencies

Add to `pyproject.toml` (assuming Tavily):

```toml
dependencies = [
    # ... existing deps ...
    "tavily-python>=0.5.0",               # Tavily search API client
    "langchain-community>=0.3.0",          # For TavilySearchResults tool
]
```

If using pytrends for the Trend Agent:

```toml
    "pytrends>=4.9.0",                    # Google Trends (unofficial)
```

Add to `.env.example`:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

Update `config.py` `Secrets` class:

```python
class Secrets(BaseSettings):
    groq_api_key: str
    kie_api_key: str = ""
    tavily_api_key: str = ""  # NEW — optional for Phase 4
```

Run `uv sync` after updating.

---

## 4.3 — Research Data Models

### File: `auto_book/models/research.py` (NEW)

```python
"""Research-related Pydantic models."""

from datetime import datetime
from pydantic import BaseModel, Field


class ResearchSource(BaseModel):
    """A single source discovered during research."""
    url: str
    title: str = ""
    snippet: str = Field(default="", description="Key excerpt or summary from this source")
    retrieval_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    reliability: str = Field(
        default="unknown",
        description="Reliability assessment: 'high', 'medium', 'low', 'unknown'"
    )


class ResearchNote(BaseModel):
    """
    A structured research note on a specific topic/keyword.
    One ResearchNote may draw from multiple sources.
    """
    topic: str = Field(description="The keyword or topic this note covers")
    summary: str = Field(description="2-4 sentence synthesis of findings on this topic")
    key_facts: list[str] = Field(
        default_factory=list,
        description="Specific facts, statistics, or claims found"
    )
    potential_angles: list[str] = Field(
        default_factory=list,
        description="Interesting angles this research suggests for the book"
    )
    sources: list[ResearchSource] = Field(default_factory=list)
    caution_notes: list[str] = Field(
        default_factory=list,
        description="Warnings about weak claims, contradictory sources, or uncertainty"
    )


class ResearchBundle(BaseModel):
    """
    Complete research output for a book project.
    Contains all notes gathered by the Research Agent.
    """
    topic: str = Field(description="The main book topic that was researched")
    notes: list[ResearchNote] = Field(default_factory=list)
    total_sources: int = 0
    search_queries_used: list[str] = Field(default_factory=list)


class TrendInsight(BaseModel):
    """A single trend signal from the Trend Agent."""
    keyword: str
    trend_direction: str = Field(description="'rising', 'stable', 'declining'")
    related_queries: list[str] = Field(default_factory=list)
    relevance_note: str = Field(
        default="",
        description="Why this trend is relevant to the book topic"
    )


class TrendReport(BaseModel):
    """Complete trend output for a book topic."""
    topic: str
    insights: list[TrendInsight] = Field(default_factory=list)
    suggested_angles: list[str] = Field(
        default_factory=list,
        description="Topic angles suggested by trend data"
    )
```

Update `models/__init__.py` to export the new models.

---

## 4.4 — Research Agent

### File: `auto_book/agents/researcher.py` (NEW)

The Research Agent takes the user brief, generates search queries, calls the search API, and synthesizes the results into structured `ResearchNote` objects.

**Implementation approach:**
1. Use the LLM to generate 5-8 targeted search queries from the user brief.
2. Call the search API for each query.
3. Collect all search results (titles, URLs, snippets).
4. Use the LLM to synthesize raw search results into structured `ResearchNote` objects.
5. Save the full research bundle to disk.

```python
"""
Research Agent — gathers and synthesizes web sources for the book.

Pipeline:
  1. LLM generates search queries from user brief
  2. Search API returns raw results for each query
  3. LLM synthesizes raw results into structured ResearchNotes
  4. Research bundle is saved to disk and passed to Planner/Writer

Fallback: If search API is unavailable or all queries fail, returns an empty
ResearchBundle and the pipeline continues with LLM knowledge only.
"""

import time
import json
from pathlib import Path
from auto_book.agents.llm_client import get_llm
from auto_book.models.research import (
    ResearchBundle, ResearchNote, ResearchSource
)
from auto_book.config import secrets, settings
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from pydantic import BaseModel, Field


# ── Step 1: Query generation ──

class SearchQueries(BaseModel):
    """LLM output: search queries to run."""
    queries: list[str] = Field(description="5-8 targeted search queries")


QUERY_GEN_SYSTEM = """You are a research assistant planning web searches for a book project.

Generate 5-8 specific, targeted search queries that would find useful information 
for writing a book on the given topic. 

RULES:
- Queries should cover different angles: facts, statistics, expert opinions, 
  case studies, counterarguments, and definitions.
- Prefer queries that find authoritative sources (academic, government, reputable media).
- Include at least one query for recent data/statistics.
- Include at least one query for common misconceptions or controversies.
- Keep queries concise and search-engine-friendly.
"""

QUERY_GEN_USER = """Book topic: {brief}
Genre: {genre}

Generate search queries to research this topic."""


def _generate_search_queries(brief: str, genre: str) -> list[str]:
    """Use LLM to generate targeted search queries."""
    logger = get_logger()
    wait_for_rate_limit()

    llm = get_llm("reviewer")  # Cheaper model for query generation
    structured_llm = llm.with_structured_output(SearchQueries)

    try:
        result = structured_llm.invoke([
            {"role": "system", "content": QUERY_GEN_SYSTEM},
            {"role": "user", "content": QUERY_GEN_USER.format(brief=brief, genre=genre)},
        ])
        logger.info(f"Generated {len(result.queries)} search queries")
        return result.queries
    except Exception as e:
        logger.warning(f"Query generation failed: {e}. Using fallback queries.")
        # Fallback: use the brief itself as a query
        return [brief, f"{brief} statistics", f"{brief} expert guide"]


# ── Step 2: Search execution ──

def _execute_search(query: str) -> list[dict]:
    """
    Execute a single search query using the configured search API.
    Returns a list of raw result dicts with: url, title, content/snippet.
    
    NOTE: This implementation assumes Tavily. If using a different provider,
    replace the API call below with the appropriate client.
    """
    logger = get_logger()
    api_key = secrets.tavily_api_key

    if not api_key:
        logger.warning("TAVILY_API_KEY not set — skipping search")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        # Tavily search with content extraction
        response = client.search(
            query=query,
            search_depth="basic",       # "basic" or "advanced" (costs more)
            max_results=5,
            include_answer=False,       # We'll synthesize our own summary
            include_raw_content=False,  # Saves tokens
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", "")[:1000],  # Cap per result
            })

        logger.info(f"Search '{query[:50]}...': {len(results)} results")
        return results

    except Exception as e:
        logger.warning(f"Search failed for '{query[:50]}...': {e}")
        return []


# ── Step 3: Synthesis ──

SYNTHESIS_SYSTEM = """You are a research analyst synthesizing web search results into 
structured research notes for a book project.

RULES:
- Create one ResearchNote per distinct topic or theme found in the results.
- Each note should synthesize information from multiple sources where possible.
- Extract specific facts, statistics, and claims — not vague summaries.
- Note the source URL for each fact so it can be cited later.
- Flag any claims that seem unreliable or contradictory in caution_notes.
- Suggest potential angles that this research opens up for the book.
- Be concise but specific. The writer needs actionable information, not fluff.
"""

SYNTHESIS_USER = """Synthesize the following search results into structured research notes.

Book topic: {topic}

Raw search results:
{raw_results}

Create structured research notes from this data."""


class SynthesisResponse(BaseModel):
    notes: list[ResearchNote] = Field(default_factory=list)


def _synthesize_results(
    topic: str,
    all_results: list[dict],
) -> list[ResearchNote]:
    """Use LLM to synthesize raw search results into structured notes."""
    logger = get_logger()

    if not all_results:
        logger.info("No search results to synthesize")
        return []

    wait_for_rate_limit()

    llm = get_llm("planner")  # Use stronger model for synthesis quality
    structured_llm = llm.with_structured_output(SynthesisResponse)

    # Format raw results for the prompt
    formatted = ""
    for i, r in enumerate(all_results[:20], 1):  # Cap at 20 results
        formatted += f"\n--- Result {i} ---\n"
        formatted += f"Title: {r['title']}\n"
        formatted += f"URL: {r['url']}\n"
        formatted += f"Content: {r['content']}\n"

    try:
        result = structured_llm.invoke([
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": SYNTHESIS_USER.format(
                topic=topic,
                raw_results=formatted,
            )},
        ])
        logger.info(f"Synthesized {len(result.notes)} research notes")
        return result.notes
    except Exception as e:
        logger.warning(f"Research synthesis failed: {e}")
        return []


# ── Main entry point ──

def run_researcher(
    user_brief: str,
    genre: str,
    output_dir: str,
) -> ResearchBundle:
    """
    Run the full research pipeline.

    Args:
        user_brief: The book topic/brief.
        genre: Book genre for context.
        output_dir: Where to save research artifacts.

    Returns:
        A ResearchBundle with structured notes. May be empty if research fails.
    """
    logger = get_logger()
    logger.info("Starting research pipeline")

    # Step 1: Generate queries
    queries = _generate_search_queries(user_brief, genre)

    # Step 2: Execute searches
    all_results = []
    for query in queries:
        results = _execute_search(query)
        all_results.extend(results)
        time.sleep(1)  # Be polite to the search API

    logger.info(f"Total raw results collected: {len(all_results)}")

    # Step 3: Synthesize
    notes = _synthesize_results(user_brief, all_results)

    # Build the bundle
    bundle = ResearchBundle(
        topic=user_brief,
        notes=notes,
        total_sources=len(set(r["url"] for r in all_results if r.get("url"))),
        search_queries_used=queries,
    )

    # Save to disk
    research_dir = Path(output_dir) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    notes_path = research_dir / "notes.json"
    notes_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    # Also save a human-readable Markdown version
    md_path = research_dir / "notes.md"
    md_content = _bundle_to_markdown(bundle)
    md_path.write_text(md_content, encoding="utf-8")

    logger.info(
        f"Research complete: {len(bundle.notes)} notes from "
        f"{bundle.total_sources} sources, saved to {research_dir}"
    )
    return bundle


def _bundle_to_markdown(bundle: ResearchBundle) -> str:
    """Convert a ResearchBundle to human-readable Markdown."""
    lines = [
        f"# Research Notes: {bundle.topic}\n",
        f"*{len(bundle.notes)} notes from {bundle.total_sources} sources*\n",
        f"**Search queries used:**\n",
    ]
    for q in bundle.search_queries_used:
        lines.append(f"- {q}")
    lines.append("")

    for note in bundle.notes:
        lines.append(f"## {note.topic}\n")
        lines.append(f"{note.summary}\n")
        if note.key_facts:
            lines.append("**Key Facts:**")
            for fact in note.key_facts:
                lines.append(f"- {fact}")
        if note.potential_angles:
            lines.append("\n**Potential Angles:**")
            for angle in note.potential_angles:
                lines.append(f"- {angle}")
        if note.caution_notes:
            lines.append("\n**⚠️ Caution:**")
            for caution in note.caution_notes:
                lines.append(f"- {caution}")
        if note.sources:
            lines.append("\n**Sources:**")
            for src in note.sources:
                lines.append(f"- [{src.title}]({src.url}) (reliability: {src.reliability})")
        lines.append("\n---\n")

    return "\n".join(lines)
```

---

## 4.5 — Research Selection for Writer

The Writer Agent (from Phase 2) needs to receive **only the research notes relevant to the current chapter**, not the entire bundle. This avoids wasting context window budget.

### File: `auto_book/utils/research_selector.py` (NEW)

```python
"""
Research selector — picks research notes relevant to a specific chapter.

Uses keyword matching between chapter topics and research note topics.
A more sophisticated approach (embedding similarity) can be added later.
"""

from auto_book.models.research import ResearchBundle, ResearchNote
from auto_book.models.chapter import ChapterPlan
from auto_book.utils.tokens import count_tokens, truncate_to_budget
from auto_book.config import settings


def select_notes_for_chapter(
    chapter_plan: ChapterPlan,
    research_bundle: ResearchBundle | None,
    max_tokens: int | None = None,
) -> str:
    """
    Select and format research notes relevant to a chapter.

    Uses keyword overlap between chapter topics and research note topics.
    Returns formatted text ready to include in the Writer prompt.

    Args:
        chapter_plan: The chapter being written.
        research_bundle: All available research. Can be None.
        max_tokens: Token budget for research context. Defaults to config value.

    Returns:
        Formatted string of relevant research notes, or empty string.
    """
    if not research_bundle or not research_bundle.notes:
        return ""

    budget = max_tokens or settings.context_budget.research_notes

    # Build keyword set from chapter plan
    chapter_keywords = set()
    chapter_keywords.add(chapter_plan.title.lower())
    for topic in chapter_plan.key_topics:
        for word in topic.lower().split():
            if len(word) > 3:  # Skip short words
                chapter_keywords.add(word)

    # Score each research note by keyword overlap
    scored_notes: list[tuple[float, ResearchNote]] = []
    for note in research_bundle.notes:
        note_words = set(note.topic.lower().split())
        note_words.update(
            word.lower() for fact in note.key_facts 
            for word in fact.split() if len(word) > 3
        )
        overlap = len(chapter_keywords & note_words)
        if overlap > 0:
            scored_notes.append((overlap, note))

    # Sort by relevance (highest overlap first)
    scored_notes.sort(key=lambda x: x[0], reverse=True)

    if not scored_notes:
        return ""

    # Format the top notes within token budget
    lines = ["RESEARCH NOTES (use these to support your writing):"]
    for _, note in scored_notes:
        entry = f"\n**{note.topic}**: {note.summary}"
        if note.key_facts:
            entry += "\nFacts: " + "; ".join(note.key_facts[:3])
        if note.sources:
            entry += "\nSource: " + note.sources[0].url

        candidate = "\n".join(lines) + entry
        if count_tokens(candidate) > budget:
            break
        lines.append(entry)

    result = "\n".join(lines)
    return truncate_to_budget(result, budget)
```

### Writer Agent Integration

Update `auto_book/agents/writer.py` to accept and use research notes:

```python
# In run_writer(), add research_notes parameter:

def run_writer(
    chapter_plan: ChapterPlan,
    book_bible: BookBible,
    dynamic_memory: DynamicMemory,
    revision_feedback: list[str] | None = None,
    research_context: str = "",  # NEW — from research_selector
) -> ChapterDraft:
    # ... existing code ...
    
    # Add research to the user prompt if available
    if research_context:
        user_msg += f"\n\n{research_context}"
        user_msg += "\n\nWhen using facts from the research notes, mention the source naturally."
```

### Orchestrator Integration

In the `write_chapter` node, call the research selector before the Writer:

```python
def write_chapter(state: GraphState) -> dict:
    # ... existing code ...
    
    # Select relevant research for this chapter
    from auto_book.utils.research_selector import select_notes_for_chapter
    research_context = select_notes_for_chapter(
        state["chapter_plan"],
        state.get("research_bundle"),  # NEW field in GraphState
    )
    
    draft = run_writer(
        chapter_plan=state["chapter_plan"],
        book_bible=state["book_bible"],
        dynamic_memory=state.get("dynamic_memory", DynamicMemory()),
        revision_feedback=revision_feedback,
        research_context=research_context,  # NEW
    )
```

---

## 4.6 — Sources/Citations in Final Book

### Update Assembler

Add a "Sources" appendix at the end of the book, compiled from all research notes:

```python
# In assemble_markdown(), after all chapters:

def _build_sources_section(research_bundle: ResearchBundle | None) -> str:
    """Build a Sources appendix from research notes."""
    if not research_bundle or not research_bundle.notes:
        return ""

    lines = ["\n## Sources & References\n"]
    seen_urls = set()
    source_num = 1

    for note in research_bundle.notes:
        for source in note.sources:
            if source.url not in seen_urls:
                seen_urls.add(source.url)
                lines.append(
                    f"{source_num}. [{source.title or 'Source'}]({source.url}) "
                    f"— Retrieved {source.retrieval_date}"
                )
                source_num += 1

    if source_num == 1:
        return ""  # No sources to list

    return "\n".join(lines)
```

Add the same to `assemble_docx()` as a final section.

---

## 4.7 — Trend Agent

### File: `auto_book/agents/trend_agent.py` (NEW)

> **DECISION POINT FOR USER**: The Trend Agent uses pytrends (unofficial Google Trends).
> This library scrapes Google Trends and can break without notice.
> 
> **Alternative**: Skip pytrends and use the search API to look for 
> "trending topics in [subject]" — simpler and more reliable.
>
> The plan below implements the **search-based approach** by default, with an 
> optional pytrends path that can be enabled in config.

```python
"""
Trend Agent — identifies demand signals and trending angles for a topic.

Default: Uses the search API to find trending angles (reliable).
Optional: Can use pytrends for Google Trends data (unreliable, may break).
"""

import time
from auto_book.agents.llm_client import get_llm
from auto_book.models.research import TrendReport, TrendInsight
from auto_book.config import settings
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from pydantic import BaseModel, Field


TREND_SYSTEM = """You are a market research analyst specializing in content trends.

Given a book topic and some search results about current trends in this area,
identify:
1. Which subtopics are currently rising in interest
2. What related queries people are searching for
3. What angles would make this book timely and relevant

Be specific and data-driven. Don't guess — base your insights on the provided data.
"""

TREND_USER = """Analyze trends for the book topic: {topic}

Here are current search results about trends in this area:
{trend_data}

Provide a structured trend report with insights and suggested angles.
"""


def run_trend_agent(
    user_brief: str,
    output_dir: str,
) -> TrendReport:
    """
    Run the Trend Agent to identify demand signals.

    Uses the search API to find trend-related content, then synthesizes it.

    Args:
        user_brief: The book topic.
        output_dir: Where to save the trend report.

    Returns:
        A TrendReport with insights. May have empty insights if trending fails.
    """
    logger = get_logger()
    logger.info("Running Trend Agent")

    # Step 1: Search for trend data
    from auto_book.agents.researcher import _execute_search
    trend_queries = [
        f"{user_brief} trending topics 2026",
        f"{user_brief} popular questions",
        f"{user_brief} what people want to know",
    ]

    all_results = []
    for query in trend_queries:
        results = _execute_search(query)
        all_results.extend(results)
        time.sleep(1)

    if not all_results:
        logger.warning("No trend data found — returning empty report")
        return TrendReport(topic=user_brief)

    # Step 2: Synthesize with LLM
    wait_for_rate_limit()
    llm = get_llm("reviewer")
    structured_llm = llm.with_structured_output(TrendReport)

    formatted = ""
    for r in all_results[:10]:
        formatted += f"\n- {r['title']}: {r['content'][:200]}\n"

    try:
        result = structured_llm.invoke([
            {"role": "system", "content": TREND_SYSTEM},
            {"role": "user", "content": TREND_USER.format(
                topic=user_brief,
                trend_data=formatted,
            )},
        ])
        result.topic = user_brief
        logger.info(f"Trend report: {len(result.insights)} insights, {len(result.suggested_angles)} angles")

        # Save to disk
        from pathlib import Path
        trend_path = Path(output_dir) / "research" / "trends.json"
        trend_path.parent.mkdir(parents=True, exist_ok=True)
        trend_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        return result

    except Exception as e:
        logger.warning(f"Trend analysis failed: {e}")
        return TrendReport(topic=user_brief)
```

---

## 4.8 — Orchestrator Updates

### Updated Graph Flow

The graph gains two new nodes at the beginning:

```
collect_input → run_trends → run_research → plan_book → [chapter loop] → assemble → export
```

### New Nodes

```python
def run_trends(state: GraphState) -> dict:
    """Node: Trend Agent gathers demand signals."""
    logger = get_logger()
    from auto_book.agents.trend_agent import run_trend_agent
    
    try:
        trend_report = run_trend_agent(
            state["user_brief"],
            state.get("output_directory", "./output"),
        )
        return {"trend_report": trend_report}
    except Exception as e:
        logger.warning(f"Trend agent failed (non-fatal): {e}")
        return {"trend_report": None}


def run_research(state: GraphState) -> dict:
    """Node: Research Agent gathers and synthesizes sources."""
    logger = get_logger()
    from auto_book.agents.researcher import run_researcher
    
    try:
        bundle = run_researcher(
            state["user_brief"],
            state["genre"],
            state.get("output_directory", "./output"),
        )
        return {"research_bundle": bundle}
    except Exception as e:
        logger.warning(f"Research agent failed (non-fatal): {e}")
        return {"research_bundle": None}
```

### Updated GraphState

Add new fields to `orchestrator/state.py`:

```python
class GraphState(TypedDict, total=False):
    # ... existing fields ...
    
    # Research (Phase 4)
    trend_report: TrendReport | None
    research_bundle: ResearchBundle | None
```

### Updated Planner

Update the Planner Agent to accept trend data and research notes as additional context:

```python
# In planner.py, update the user prompt:

PLANNER_USER_PROMPT = """Create a complete Book Bible for the following book idea:

{brief}

{trend_context}

{research_context}

Return a comprehensive plan...
"""

# Where trend_context and research_context are formatted from the new data:
# trend_context = "TRENDING ANGLES:\n" + "\n".join(trend_report.suggested_angles)
# research_context = "RESEARCH FINDINGS:\n" + summary of top research notes
```

---

## 4.9 — Config Updates

Add research settings to `config.yaml`:

```yaml
# Research Settings
research:
  enabled: true
  search_provider: "tavily"      # "tavily", "serpapi", "brave", "duckduckgo"
  max_queries: 8
  max_results_per_query: 5
  search_depth: "basic"          # "basic" or "advanced" (Tavily-specific)

# Trend Settings
trends:
  enabled: true
  use_pytrends: false            # If true, use pytrends (unreliable)
```

Add corresponding Pydantic models in `config.py`.

---

## 4.10 — Fallback Behavior

> **CRITICAL**: The research and trend pipelines must NEVER cause the book generation to fail.

Implement this fallback chain:

1. **Search API unavailable** → Log warning, return empty `ResearchBundle`. Planner and Writer proceed with LLM knowledge only.
2. **Search returns no results** → Same as above.
3. **Synthesis LLM call fails** → Return raw search snippets as minimal notes (no synthesis).
4. **Trend Agent fails** → Return empty `TrendReport`. Planner ignores trends.
5. **Research notes are empty** → Writer prompt omits the research section entirely. The book is marked as "not research-backed" in metadata.

The orchestrator should always set `research_bundle` and `trend_report` to a valid (possibly empty) object, never `None` — this simplifies downstream null checks.

```python
# In orchestrator, after research node:
if state.get("research_bundle") is None:
    state["research_bundle"] = ResearchBundle(topic=state["user_brief"])
```

---

## 4.11 — Verification Checklist

After completing Phase 4, verify:

- [ ] With a valid `TAVILY_API_KEY`, the system performs web searches and generates research notes.
- [ ] `output/research/notes.json` contains structured research notes with source URLs.
- [ ] `output/research/notes.md` is a human-readable version of the research.
- [ ] `output/research/trends.json` contains trend insights (if trend agent is enabled).
- [ ] The Planner Agent uses research and trend data to inform the Book Bible.
- [ ] The Writer Agent receives chapter-relevant research notes (not all notes).
- [ ] Research notes fit within the configured `context_budget.research_notes` token limit.
- [ ] The final book includes a "Sources & References" section at the end.
- [ ] With `TAVILY_API_KEY` unset, the system still completes with LLM knowledge only.
- [ ] With search API returning errors, the system logs warnings and continues.
- [ ] The book quality is noticeably better with research enabled vs. disabled.

### Quality Checks

- Read the "Sources & References" section — are URLs real and relevant?
- Compare a book generated with research vs. without — does research add value?
- Check that research facts appear naturally in the text, not dumped verbatim.
