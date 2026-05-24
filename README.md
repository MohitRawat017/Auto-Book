# Auto_Book

An autonomous book-writing pipeline powered by LLMs. Give it a topic, get back a fully formatted `.docx` book with chapters, images, a cover, and a table of contents.

---

## How It Works

```
User Brief
    │
    ▼
Planner Agent  ──────────────────────────────► Cover image (background thread)
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

Images generate **in parallel** with chapter writing — by the time the last chapter is done, most images are already ready.

---

## Setup

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
git clone <repo>
cd auto_book
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

### Generate a book

```bash
# Smoke test (3 chapters, ~500 words each, ~12k tokens)
python -m auto_book --config phase3_smoke_config.yaml "Your book topic"

# Medium run (6 chapters, ~1000 words each, ~35k tokens)
python -m auto_book --config config_medium.yaml "Your book topic"

# Full run (10 chapters, ~1500 words each, ~57k tokens)
python -m auto_book --config config.yaml "Your book topic"

# Custom output directory
python -m auto_book --config config.yaml --output ./output/my_book "Your book topic"
```

### Resume after rate limit

```bash
python -m auto_book --resume --output ./output/my_book
```

### Generate/fix images after a run

```bash
# Generate missing images + reassemble book
python regenerate_images.py --output ./output/my_book --config config.yaml

# Just reassemble (images already generated)
python regenerate_images.py --output ./output/my_book --config config.yaml --assemble-only
```

---

## Output

Each run produces:

```
output/my_book/
├── book.docx              # Final formatted book
├── book.md                # Markdown version
├── book_bible.json        # Book plan (title, chapters, thesis)
├── dynamic_memory.json    # Continuity memory across chapters
├── run_state.json         # Checkpoint (resume from here)
├── run.log                # Full run log
├── chapters/              # Individual chapter .md files
│   ├── chapter_01.md
│   └── ...
├── reviews/               # Reviewer decisions per chapter
│   ├── chapter_01_review.json
│   └── ...
└── images/                # Generated images
    ├── COVER.png
    ├── CH1_DIAGRAM.png
    ├── image_manifest.json
    └── image_assets.json
```

---

## Configuration

Three configs are provided:

| Config | Chapters | Words/ch | Est. tokens | Use case |
|--------|----------|----------|-------------|----------|
| `phase3_smoke_config.yaml` | 3 | 500 | ~12k | Testing |
| `config_medium.yaml` | 6 | 1000 | ~35k | Review quality |
| `config.yaml` | 10 | 1500 | ~57k | Full book |

### Key settings

```yaml
# config.yaml

llm:
  writer_model: "llama-3.3-70b-versatile"   # Groq model for writing
  reviewer_model: "openai/gpt-oss-120b"      # Groq model for reviewing
  planner_model: "qwen/qwen3-32b"            # Groq model for planning

book:
  default_chapter_count: 10
  target_words_per_chapter: 1500

review:
  enabled: true          # Set false to skip reviewer entirely (saves ~40% tokens)
  max_review_passes: 1   # How many times reviewer runs per chapter

images:
  enabled: true
  generate_actual: true  # Set false for placeholder images (saves KIE credits)
  max_images_per_chapter: 4
```

### DOCX template

Place a `book_template.docx` in `resources/` to apply custom Word styles. The assembler uses these styles automatically:

| Style name | Used for |
|------------|----------|
| `Title` | Book title on cover page |
| `Subtitle` | Book subtitle |
| `Heading 1` | Chapter headings |
| `Heading 2` | Section headings |
| `Heading 3` | Sub-section headings |
| `Normal` | Body text |
| `TOC Heading` | Table of contents title |
| `toc 1` | TOC chapter entries |
| `toc 2` | TOC section entries |

> **Note:** If the template has duplicate style names (common when copying styles in Word), the assembler falls back to built-in styles with matching colors. Fix by creating a clean template from scratch.

---

## Agents

| Agent | Model role | What it does |
|-------|-----------|--------------|
| Planner | `planner_model` | Creates the Book Bible — title, chapter outline, thesis, tone |
| Writer | `writer_model` | Writes each chapter as Markdown prose with image anchors |
| Reviewer | `reviewer_model` | Scores each draft (0–10), requests revisions if needed |
| Memory Updater | `memory_model` | Summarizes accepted chapters for continuity |
| Image Agent | `reviewer_model` | Expands image anchors into KIE generation prompts |
| Assembler | — | Combines chapters + images into `.md` and `.docx` |

---

## Rate Limits

Groq free tier: **100,000 tokens/day** per account. Resets at midnight UTC (5:30 AM IST).

Approximate token costs per run:

| Config | Tokens | Fits in free tier? |
|--------|--------|-------------------|
| Smoke (3ch) | ~12k | ✅ 8× per day |
| Medium (6ch) | ~35k | ✅ 2× per day |
| Full (10ch) | ~57k | ✅ once per day |

If you hit the limit mid-run, the checkpoint is saved. Swap to a fresh API key (different account) and resume:

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
| `httpx` | KIE image API calls |
| `Pillow` | Image normalization |
| `rich` | Terminal output |
| `pydantic` | Data models + validation |
