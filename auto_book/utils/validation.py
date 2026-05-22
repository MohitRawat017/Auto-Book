"""Semantic validation helpers."""

from auto_book.models.chapter import ChapterDraft
from auto_book.models.review import ReviewDecision, ReviewDecisionEnum


def validate_chapter_draft(
    draft: ChapterDraft,
    min_words: int,
    max_words: int,
) -> list[str]:
    """Validate a chapter draft beyond schema correctness."""

    errors: list[str] = []
    draft.word_count = len(draft.body.split())

    if draft.word_count < min_words:
        errors.append(
            f"Chapter {draft.chapter_number} has {draft.word_count} words; "
            f"minimum is {min_words}."
        )
    if draft.word_count > max_words:
        errors.append(
            f"Chapter {draft.chapter_number} has {draft.word_count} words; "
            f"maximum is {max_words}."
        )
    if not draft.title.strip():
        errors.append(f"Chapter {draft.chapter_number} has an empty title.")
    if not draft.body.strip():
        errors.append(f"Chapter {draft.chapter_number} has an empty body.")

    return errors


def validate_review_decision(review: ReviewDecision) -> list[str]:
    """Validate a review decision beyond schema correctness."""

    errors: list[str] = []

    if review.decision == ReviewDecisionEnum.REVISE and not review.required_fixes:
        errors.append(
            f"Review for chapter {review.chapter_number} requested revision "
            "without required fixes."
        )
    if review.decision == ReviewDecisionEnum.FAIL and not review.problems:
        errors.append(
            f"Review for chapter {review.chapter_number} failed the draft "
            "without listing problems."
        )

    return errors
