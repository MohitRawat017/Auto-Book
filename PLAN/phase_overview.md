# Auto_Book — Phase Overview

## Tech Stack (Decided)

| Component | Choice |
|---|---|
| **Language** | Python 3.12+ |
| **Package Manager** | uv |
| **LLM Provider** | Groq API via LangChain (`langchain-groq`) |
| **LLM Model (Writer/Planner)** | `meta-llama/llama-4-maverick-17b-128e-instruct` (best creative quality on Groq) |
| **LLM Model (Reviewer/Memory)** | `meta-llama/llama-4-scout-17b-16e-instruct` (faster, cheaper for structured eval) |
| **Orchestration** | LangGraph (graph-based state machine) |
| **Structured Outputs** | Pydantic v2 |
| **Image Generation** | kie.ai unified API |
| **Export Formats (MVP)** | Markdown + DOCX (via `python-docx`). PDF deferred. |
| **Config — Secrets** | `.env` file via `pydantic-settings` |
| **Config — Settings** | `config.yaml` via PyYAML |
| **Research (MVP)** | Skipped — LLM parametric knowledge only |
| **Trend Agent (MVP)** | Skipped — added in later phase |
| **Genre** | Genre-agnostic — system asks user at runtime |

---

## Phase Breakdown

### Phase 1 — Foundation & Skeleton
> **Goal**: Project scaffolding, config, data models, and a runnable (but dumb) orchestrator skeleton.

- Project structure and dependency setup
- Configuration system (.env + config.yaml)
- All Pydantic data models (BookBible, ChapterDraft, ReviewDecision, DynamicMemory, RunState, etc.)
- Orchestrator skeleton with LangGraph (nodes defined but using stubs)
- Run state checkpointing (save/resume)
- Structured logging
- CLI entry point

**Deliverable**: You can run `python -m auto_book "Write a book about X"` and it walks through every node printing stub outputs. State is saved to disk after each step.

---

### Phase 2 — Core Writing Loop
> **Goal**: Planner, Writer, and Reviewer agents are live. The system can generate a real book (Markdown only).

- Planner Agent → produces Book Bible
- Writer Agent → writes chapters one at a time using Book Bible + Dynamic Memory
- Reviewer Agent → evaluates chapters, returns pass/revise/fail
- Retry loop with limits (2 revisions, 1 regeneration)
- Dynamic Memory updates after each accepted chapter
- Memory compression when token budget is exceeded
- Context window budget enforcement per agent call
- Output validation (schema + semantic: word count, enum checks, non-empty fields)
- Token usage tracking and rate-limit handling for Groq
- Final Markdown assembly (concatenate chapters + title page + TOC)

**Deliverable**: Given a topic, the system produces a complete ~30-page Markdown book autonomously.

---

### Phase 3 — Export & Image Agent
> **Goal**: DOCX export works. Image Agent generates prompts and optionally calls kie.ai.

- Assembler: Markdown → DOCX conversion via `python-docx`
- Front matter, heading normalization, TOC in DOCX
- Image Agent: identify image opportunities per chapter, generate prompts
- kie.ai integration: call API to generate images from prompts
- Image embedding in Markdown (as links) and DOCX (as embedded images)
- Graceful degradation: if image gen fails, book still exports with placeholders

**Deliverable**: The system outputs a polished DOCX with embedded images alongside the Markdown.

---

### Phase 4 — Research & Trend Agents
> **Goal**: The system can research topics before writing, and optionally gather trend data.

- Research Agent: use a search API (Tavily / SerpAPI / Brave — decided at that time) to gather sources
- Research note structured output (source URL, summary, facts, reliability)
- Research selection: pick relevant notes per chapter using similarity or keyword matching
- Fallback: if research fails, proceed with LLM knowledge and flag the book
- Trend Agent: gather trend signals (pytrends or paid API — decided at that time)
- Trend data fed into Planner for topic angle selection

**Deliverable**: Books are backed by real research with source citations. Trend-informed topic angles.

---

### Phase 5 — Human-in-Loop, PDF, & Polish
> **Goal**: Optional human checkpoints, PDF export, and production hardening.

- Human approval checkpoints in orchestrator (after Book Bible, after each chapter)
- Pause/resume flow with checkpoint persistence
- PDF export (toolchain decided at that time — weasyprint, Pandoc, or browser-based)
- Genre-specific style presets (non-fiction, fiction, how-to)
- Plagiarism / originality light checks
- End-to-end test suite with mock agents
- README, usage docs, and example runs

**Deliverable**: Production-ready autonomous book generator with optional human oversight.

---

## Phase Dependency Graph

```
Phase 1 (Foundation) ──► Phase 2 (Core Loop) ──► Phase 3 (Export + Images)
                                                         │
                                                         ├──► Phase 4 (Research + Trends)
                                                         │
                                                         └──► Phase 5 (Human-in-Loop + PDF + Polish)
```

Phases 4 and 5 can be done in parallel or in either order after Phase 3.
