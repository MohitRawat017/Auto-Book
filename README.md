# Auto_Book

An autonomous book-writing pipeline powered by LLMs. Give it a topic, get back a fully formatted `.docx` book with chapters, images, a cover, and a table of contents.

---

## How It Works

```
User Brief
    │
    ▼
Planner Agent  ──► Google Trends keywords + Cover image (background)
    │ Book Bible (title, chapters, outline)
    ▼
┌─────────────────────────────────────────┐
│  For each chapter:                      │
│  Writer Agent → Reviewer Agent          │
│      └─ (revise if needed)              │
│  Memory Updater                         │
│      └─► Chapter images (background)   │
└─────────────────────────────────────────┘
    │ All chapters accepted
    ▼
Image Queue collect()  ← waits for background threads
    │
    ▼
Assembler  →  book.md  +  book.docx
```

---

## Setup

```bash
git clone <repo>
cd Auto_Book
uv sync
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key
KIE_API_KEY=your_kie_ai_api_key
```

---

## Usage

```bash
# Smoke test (3 chapters, ~500 words each)
python -m auto_book --config configs/smoke.yaml "Your book topic"

# Medium run (6 chapters, ~1000 words each)
python -m auto_book --config configs/medium.yaml "Your book topic"

# Full run (10 chapters, ~1500 words each)
python -m auto_book "Your book topic"

# Custom output directory
python -m auto_book --output ./output/my_book "Your book topic"
```

### Resume after rate limit

```bash
python -m auto_book --resume --output ./output/my_book
```

### Generate/fix images after a run

```bash
python scripts/regenerate_images.py --output ./output/my_book --config configs/full.yaml

# Reassemble only (images already on disk)
python scripts/regenerate_images.py --output ./output/my_book --assemble-only
```

---

## Project Structure

```
Auto_Book/
├── auto_book/
│   ├── __main__.py          # CLI entry point
│   ├── config.py            # Settings + config loading
│   ├── cli.py               # Argument parsing
│   ├── agents/
│   │   ├── planner.py       # Book Bible creation
│   │   ├── writer.py        # Chapter writing
│   │   ├── reviewer.py      # Quality scoring
│   │   ├── memory_updater.py# Continuity tracking
│   │   ├── image_agent.py   # Image prompt expansion + KIE generation
│   │   ├── assembler.py     # Markdown + DOCX export
│   │   └── llm_client.py    # Groq client factory
│   ├── orchestrator/
│   │   ├── graph.py         # LangGraph state machine
│   │   ├── state.py         # Graph state definition
│   │   └── checkpointer.py  # Save/load run checkpoints
│   ├── models/              # Pydantic data models
│   └── utils/
│       ├── llm.py           # Shared LLM response parsing
│       ├── image_queue.py   # Background image generation
│       ├── rate_limiter.py  # Groq rate limit handling
│       ├── trends.py        # Google Trends keyword fetching
│       ├── tokens.py        # Token counting
│       └── logger.py        # JSON structured logging
├── configs/
│   ├── full.yaml            # 10 chapters, 1500 words
│   ├── medium.yaml          # 6 chapters, 1000 words
│   └── smoke.yaml           # 3 chapters, 500 words (testing)
├── resources/
│   └── template.docx        # Word style template
├── scripts/
│   └── regenerate_images.py # Image retry + reassembly tool
├── pyproject.toml
└── README.md
```

---

## Configuration

| Config | Chapters | Words/ch | Tokens | Use case |
|--------|----------|----------|--------|----------|
| `configs/smoke.yaml` | 3 | 500 | ~12k | Testing |
| `configs/medium.yaml` | 6 | 1000 | ~35k | Quality review |
| `configs/full.yaml` | 10 | 1500 | ~57k | Full book |

### Key settings

```yaml
review:
  enabled: true          # false = skip reviewer (saves ~40% tokens)
  max_review_passes: 1   # times reviewer runs per chapter

images:
  enabled: true
  generate_actual: true  # false = placeholder images (saves KIE credits)
  max_images_per_chapter: 4
```

---

## Agents

| Agent | What it does |
|-------|--------------|
| Planner | Creates Book Bible — title, outline, thesis, tone. Fetches Google Trends keywords. |
| Writer | Writes each chapter as Markdown with image anchors |
| Reviewer | Scores drafts (0–10), requests revisions if needed |
| Memory Updater | Summarizes accepted chapters for continuity |
| Image Agent | Expands anchors into prompts, generates via KIE API |
| Assembler | Combines chapters + images into `.md` and `.docx` |

---

## Rate Limits

Groq free tier: **100,000 tokens/day**. Resets at midnight UTC.

If you hit the limit mid-run, swap to a fresh API key and resume:
```bash
python -m auto_book --resume --output ./output/my_book
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | State machine orchestration |
| `langchain-groq` | Groq LLM client |
| `python-docx` | DOCX assembly |
| `httpx` | KIE image API |
| `Pillow` | Image normalization |
| `pytrends` | Google Trends keywords |
| `rich` | Terminal output |
| `pydantic` | Data models |
