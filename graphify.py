"""Visualize the auto_book LangGraph pipeline as Mermaid and ASCII."""

MERMAID = """
flowchart TD
    START([START]) --> collect_input
    collect_input -->|next| plan_book
    collect_input -->|fail| handle_failure
    collect_input -->|rate_limited| END_RL1([END])

    plan_book -->|next| prepare_chapter
    plan_book -->|fail| handle_failure
    plan_book -->|rate_limited| END_RL2([END])

    prepare_chapter -->|write_chapter| write_chapter
    prepare_chapter -->|review_chapter| review_chapter
    prepare_chapter -->|update_memory| update_memory
    prepare_chapter -->|assemble_book| assemble_book
    prepare_chapter -->|fail| handle_failure
    prepare_chapter -->|rate_limited| END_RL3([END])

    write_chapter -->|next| review_chapter
    write_chapter -->|fail| handle_failure
    write_chapter -->|rate_limited| END_RL4([END])

    review_chapter -->|PASS| update_memory
    review_chapter -->|REVISE/FAIL| write_chapter
    review_chapter -->|fail| handle_failure
    review_chapter -->|rate_limited| END_RL5([END])

    update_memory --> check_next

    check_next -->|more chapters| prepare_chapter
    check_next -->|all done| assemble_book

    assemble_book -->|next| export_book
    assemble_book -->|fail| handle_failure
    assemble_book -->|rate_limited| END_RL6([END])

    export_book --> END([END])
    handle_failure --> END2([END])

    style START fill:#4CAF50,color:#fff
    style END fill:#2196F3,color:#fff
    style END2 fill:#f44336,color:#fff
    style handle_failure fill:#f44336,color:#fff
    style END_RL1 fill:#FF9800,color:#fff
    style END_RL2 fill:#FF9800,color:#fff
    style END_RL3 fill:#FF9800,color:#fff
    style END_RL4 fill:#FF9800,color:#fff
    style END_RL5 fill:#FF9800,color:#fff
    style END_RL6 fill:#FF9800,color:#fff
"""

ASCII = """
auto_book Pipeline Graph
========================

  [START]
     │
     ▼
┌─────────────────┐
│  collect_input  │──rate_limited──► [END]
└────────┬────────┘
         │ next
         ▼
┌─────────────────┐
│   plan_book     │──rate_limited──► [END]
└────────┬────────┘
         │ next
         ▼
┌─────────────────┐◄──────────────────────────────────────┐
│ prepare_chapter │──rate_limited──► [END]                 │
└────────┬────────┘                                        │
         │                                                 │
    ┌────┴──────────────────────────────┐                  │
    │ write_chapter  review_chapter     │                  │
    │ update_memory  assemble_book      │                  │
    ▼                                   ▼                  │
┌───────────────┐             ┌──────────────────┐         │
│ write_chapter │◄──REVISE────│ review_chapter   │         │
│               │──rate_lim──►│                  │         │
└───────┬───────┘             └────────┬─────────┘         │
        │ next                         │ PASS               │
        └──────────────────────────────┘                    │
                                       │                    │
                                       ▼                    │
                              ┌─────────────────┐           │
                              │  update_memory  │           │
                              └────────┬────────┘           │
                                       │                    │
                                       ▼                    │
                              ┌─────────────────┐           │
                              │   check_next    │──more────►┘
                              └────────┬────────┘
                                       │ all done
                                       ▼
                              ┌─────────────────┐
                              │  assemble_book  │──rate_limited──► [END]
                              └────────┬────────┘
                                       │ next
                                       ▼
                              ┌─────────────────┐
                              │   export_book   │
                              └────────┬────────┘
                                       │
                                       ▼
                                    [END ✓]

  handle_failure ──────────────────────────────────────────► [END ✗]
  (reached from any node on fail/max-retries)

Nodes
─────
  collect_input   Validate user brief, set defaults
  plan_book       LLM → Book Bible (title, chapters, outline)
  prepare_chapter Extract next ChapterPlan; resume if interrupted
  write_chapter   LLM → ChapterDraft (new or revised)
  review_chapter  LLM → ReviewDecision (PASS / REVISE / FAIL)
  update_memory   Accept draft, update DynamicMemory, save artifacts
  check_next      Decide: more chapters → loop, else → assemble
  assemble_book   Image Agent → ImageAssets
  export_book     Markdown + DOCX export
  handle_failure  Log error, terminate

Retry Logic
───────────
  REVISE  → re-enter write_chapter (max_revisions times)
  FAIL    → re-enter write_chapter (max_regenerations times)
  Soft-accept if score ≥ soft_accept_score after max revisions
"""


def main():
    print(ASCII)
    print("\n" + "=" * 60)
    print("Mermaid diagram (paste into https://mermaid.live):")
    print("=" * 60)
    print(MERMAID)

    # Save files
    with open("graph.mmd", "w") as f:
        f.write(MERMAID.strip())
    print("\nSaved: graph.mmd")


if __name__ == "__main__":
    main()
