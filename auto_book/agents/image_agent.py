"""Image Agent: expands writer anchors into prompts and optional KIE images."""

from __future__ import annotations

import json
import re
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from auto_book.agents.llm_client import get_llm
from auto_book.config import secrets, settings
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.image import ImageAnchor, ImageAsset, ImagePrompt
from auto_book.utils.llm import extract_message_text, load_json_object, strip_code_fence
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import (
    raise_if_rate_limited,
    record_success,
    wait_for_rate_limit,
)
from auto_book.utils.tokens import truncate_to_budget

KIE_CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
IMAGE_ANCHOR_PATTERN = re.compile(r"^\[IMAGE_ANCHOR:\s*([A-Z0-9_]+)\]\s*$")

IMAGE_PROMPT_SYSTEM_PROMPT = """You are an expert AI image art director.

You receive exact image anchors written into a book chapter. Your job is NOT to
choose placement. Your job is to turn the anchor and surrounding prose into a
detailed text-to-image prompt for KIE/Qwen image generation.

Book context:
- Title: {title}
- Genre: {genre}
- Tone: {tone}
- Image direction: {image_direction}

Prompt requirements:
- Be specific and visual, not a short label.
- This is a NON-FICTION book. Generate DIAGRAMS, INFOGRAPHICS, FLOWCHARTS,
  COMPARISON TABLES, STEP-BY-STEP VISUALS, or CONCEPT MAPS — NOT photos,
  portraits, landscapes, or decorative illustrations.
- Describe the visual type explicitly: e.g. "a clean 4-step flowchart showing...",
  "a two-column comparison table contrasting X and Y", "a circular diagram with
  5 labeled segments representing...".
- Describe composition, main objects, layout, labels/text that should appear,
  style, color direction, and what to avoid.
- Use a flat, modern infographic style: white or light background, bold sans-serif
  labels, 2-3 accent colors, clear visual hierarchy, generous whitespace.
- Avoid: photos of people, stock-photo aesthetics, tiny unreadable text, brand
  logos, copyrighted characters, decorative borders, and visual clutter.
- Return ONLY valid JSON, with no Markdown or code fences.

Required JSON shape:
{{
  "anchor_id": "{anchor_id}",
  "chapter_number": {chapter_number},
  "position": "{marker}",
  "prompt": "Detailed image generation prompt...",
  "style": "concise style direction",
  "alt_text": "Concise accessible description"
}}
"""

IMAGE_PROMPT_USER_PROMPT = """Create a detailed image prompt for this exact anchor.

Anchor ID: {anchor_id}
Anchor meaning: {anchor_meaning}
Chapter {chapter_number}: "{chapter_title}"
Chapter summary: {chapter_summary}
Nearest heading: {nearest_heading}

Context before anchor:
{context_before}

Context after anchor:
{context_after}

Return only the JSON image prompt object.
"""


def run_image_agent(
    chapters: list[ChapterDraft],
    book_bible: BookBible,
    output_dir: str,
) -> list[ImageAsset]:
    """Parse writer anchors, create detailed prompts, and optionally generate images."""

    logger = get_logger()
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not settings.images.enabled:
        logger.info("Image Agent disabled by config.")
        _save_json(images_dir / "image_manifest.json", [])
        _save_json(images_dir / "image_plan.json", [])
        _save_json(images_dir / "image_assets.json", [])
        return []

    ordered = sorted(chapters, key=lambda item: item.chapter_number)
    anchors = extract_image_anchors(ordered)
    _save_json(images_dir / "image_manifest.json", [anchor.model_dump() for anchor in anchors])

    if not anchors:
        logger.info("No image anchors found in accepted chapters.")
        _save_json(images_dir / "image_plan.json", [])
        _save_json(images_dir / "image_assets.json", [])
        return []

    if not settings.images.generate_actual:
        assets = [
            ImageAsset(
                anchor_id=anchor.anchor_id,
                chapter_number=anchor.chapter_number,
                position=anchor.marker,
                prompt_used=f"Placeholder for {anchor.anchor_id}",
                alt_text=anchor.anchor_id.replace("_", " ").title(),
                is_placeholder=True,
            )
            for anchor in anchors
        ]
        _save_json(images_dir / "image_plan.json", [])
        _save_json(images_dir / "image_assets.json", [asset.model_dump() for asset in assets])
        logger.info("Image Agent: generate_actual=false, returning %s placeholder(s)", len(assets))
        return assets

    chapter_by_number = {chapter.chapter_number: chapter for chapter in ordered}
    prompts: list[ImagePrompt] = []
    for anchor in anchors:
        chapter = chapter_by_number.get(anchor.chapter_number)
        if chapter is None:
            continue
        prompts.append(plan_image_for_anchor(anchor, chapter, book_bible))

    _save_json(images_dir / "image_plan.json", [prompt.model_dump() for prompt in prompts])
    logger.info("Image prompt expansion complete: %s prompt(s)", len(prompts))

    assets: list[ImageAsset] = []
    for prompt in prompts:
        assets.append(generate_image_via_kie(prompt, output_dir))

    _save_json(images_dir / "image_assets.json", [asset.model_dump() for asset in assets])
    logger.info(
        "Image generation complete: %s generated, %s placeholder(s)",
        sum(1 for asset in assets if not asset.is_placeholder),
        sum(1 for asset in assets if asset.is_placeholder),
    )
    return assets


def extract_image_anchors(chapters: list[ChapterDraft]) -> list[ImageAnchor]:
    """Parse exact standalone image anchors from accepted chapter Markdown."""

    anchors: list[ImageAnchor] = []
    seen: set[str] = set()
    for chapter in sorted(chapters, key=lambda item: item.chapter_number):
        lines = chapter.body.splitlines()
        nearest_heading = ""
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                nearest_heading = stripped.lstrip("#").strip()

            match = IMAGE_ANCHOR_PATTERN.fullmatch(stripped)
            if not match:
                continue

            anchor_id = match.group(1)
            if anchor_id in seen:
                continue
            seen.add(anchor_id)
            anchors.append(
                ImageAnchor(
                    anchor_id=anchor_id,
                    chapter_number=chapter.chapter_number,
                    marker=f"[IMAGE_ANCHOR: {anchor_id}]",
                    nearest_heading=nearest_heading,
                    context_before=_nearby_paragraph(lines, index, direction=-1),
                    context_after=_nearby_paragraph(lines, index, direction=1),
                )
            )

    get_logger().info("Parsed %s image anchor(s)", len(anchors))
    return anchors


def plan_image_for_anchor(
    anchor: ImageAnchor,
    draft: ChapterDraft,
    book_bible: BookBible,
) -> ImagePrompt:
    """Create a detailed image prompt from an exact writer anchor."""

    logger = get_logger()
    chapter_summary = _chapter_summary(book_bible, draft)
    system_msg = IMAGE_PROMPT_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        genre=book_bible.genre,
        tone=book_bible.tone,
        image_direction=book_bible.image_direction
        or "Clean, professional editorial illustrations.",
        anchor_id=anchor.anchor_id,
        chapter_number=anchor.chapter_number,
        marker=anchor.marker,
    )
    user_msg = IMAGE_PROMPT_USER_PROMPT.format(
        anchor_id=anchor.anchor_id,
        anchor_meaning=anchor.anchor_id.replace("_", " ").title(),
        chapter_number=anchor.chapter_number,
        chapter_title=draft.title,
        chapter_summary=chapter_summary,
        nearest_heading=anchor.nearest_heading or draft.title,
        context_before=anchor.context_before or "(No preceding context.)",
        context_after=anchor.context_after or "(No following context.)",
    )

    try:
        wait_for_rate_limit()
        response = get_llm("reviewer").invoke(
            [
                ("system", system_msg),
                ("human", user_msg),
            ]
        )
        record_success()
        payload = load_json_object(strip_code_fence(extract_message_text(response)))
        payload["anchor_id"] = anchor.anchor_id
        payload["chapter_number"] = anchor.chapter_number
        payload["position"] = anchor.marker
        prompt = ImagePrompt.model_validate(payload)
        if len(prompt.prompt.split()) < 35:
            raise ValueError("Image prompt was too short to be useful.")
        logger.info("Image prompt created for %s", anchor.anchor_id)
        return prompt
    except Exception as exc:
        raise_if_rate_limited(exc)
        logger.warning(
            "Image prompt expansion failed for %s: %s. Using fallback prompt.",
            anchor.anchor_id,
            exc,
        )
        return _fallback_image_prompt(anchor, draft, book_bible, chapter_summary)


def generate_image_via_kie(prompt: ImagePrompt, output_dir: str) -> ImageAsset:
    """Generate one image through the KIE market jobs API, or return a placeholder."""

    logger = get_logger()
    if not settings.images.generate_actual:
        return _placeholder_asset(prompt, "Actual image generation disabled by config.")

    if not secrets.kie_api_key:
        return _placeholder_asset(prompt, "KIE_API_KEY is not set.")

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_path = images_dir / _image_filename(prompt)

    try:
        timeout = float(settings.images.timeout_seconds)
        with httpx.Client(timeout=timeout) as client:
            task_id = _create_kie_task(client, prompt)
            image_url = _poll_kie_result(client, task_id)
            response = client.get(image_url)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            _normalize_image_file(file_path)

        logger.info("Image generated for %s: %s", prompt.anchor_id, file_path)
        return ImageAsset(
            anchor_id=prompt.anchor_id,
            chapter_number=prompt.chapter_number,
            position=prompt.position,
            file_path=str(file_path),
            prompt_used=prompt.prompt,
            alt_text=prompt.alt_text,
            is_placeholder=False,
        )
    except Exception as exc:
        logger.warning("Image generation failed for %s: %s", prompt.anchor_id, exc)
        if settings.images.fallback_to_placeholder:
            return _placeholder_asset(prompt, str(exc))
        raise


def generate_cover_image(book_bible: BookBible, output_dir: str) -> ImageAsset:
    """Generate a cover image for the book title page."""

    cover_prompt = (
        f"A professional book cover background for a non-fiction book titled "
        f'"{book_bible.working_title}". '
        f"Genre: {book_bible.genre}. "
        f"Style: modern, clean, editorial. "
        f"Use abstract geometric shapes, subtle gradients, and a dark navy or deep teal "
        f"color palette. No text, no people, no photos. "
        f"The image should work as a full-page background behind white title text. "
        f"Flat design, high contrast, professional publishing aesthetic."
    )
    prompt = ImagePrompt(
        anchor_id="COVER",
        chapter_number=0,
        position="cover",
        prompt=cover_prompt,
        style="modern editorial book cover, abstract geometric",
        alt_text=f"Cover image for {book_bible.working_title}",
    )
    return generate_image_via_kie(prompt, output_dir)


def _create_kie_task(client: httpx.Client, prompt: ImagePrompt) -> str:
    payload: dict[str, Any] = {
        "model": settings.images.default_model,
        "input": {
            "prompt": prompt.prompt,
            "image_size": settings.images.default_size,
            "seed": 0,
            "output_format": settings.images.output_format,
        },
    }
    if settings.images.call_back_url:
        payload["callBackUrl"] = settings.images.call_back_url

    response = client.post(
        KIE_CREATE_TASK_URL,
        headers=_kie_headers(),
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    task_id = ((data.get("data") or {}).get("taskId") or "").strip()
    if not task_id:
        raise ValueError(f"KIE createTask response did not include taskId: {data}")
    return task_id


def _poll_kie_result(client: httpx.Client, task_id: str) -> str:
    logger = get_logger()
    for attempt in range(1, settings.images.max_poll_attempts + 1):
        response = client.get(
            KIE_RECORD_INFO_URL,
            headers=_kie_headers(),
            params={"taskId": task_id},
        )
        response.raise_for_status()
        data = response.json()
        task = data.get("data") or {}
        state = str(task.get("state") or "").lower()

        if state == "success":
            url = _first_result_url(task.get("resultJson"))
            if not url:
                raise ValueError(f"KIE task succeeded without result URL: {task}")
            return url
        if state == "fail":
            raise ValueError(task.get("failMsg") or f"KIE task {task_id} failed.")

        logger.info(
            "KIE task %s state=%s (%s/%s)",
            task_id,
            state or "unknown",
            attempt,
            settings.images.max_poll_attempts,
        )
        time.sleep(settings.images.poll_interval_seconds)

    raise TimeoutError(f"KIE task {task_id} did not complete before polling timeout.")


def _nearby_paragraph(lines: list[str], anchor_index: int, direction: int) -> str:
    indexes = (
        range(anchor_index - 1, -1, -1)
        if direction < 0
        else range(anchor_index + 1, len(lines))
    )
    for index in indexes:
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if IMAGE_ANCHOR_PATTERN.fullmatch(stripped):
            continue
        return truncate_to_budget(stripped, 220)
    return ""


def _chapter_summary(book_bible: BookBible, draft: ChapterDraft) -> str:
    for chapter in book_bible.chapter_outline:
        if chapter.chapter_number == draft.chapter_number:
            return chapter.summary
    return draft.title


def _first_result_url(result_json: object) -> str:
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except JSONDecodeError:
            return ""
    if not isinstance(result_json, dict):
        return ""

    for key in ("resultUrls", "urls", "images"):
        value = result_json.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
    for key in ("url", "imageUrl", "image_url"):
        value = result_json.get(key)
        if isinstance(value, str):
            return value
    return ""


def _normalize_image_file(path: Path) -> None:
    with Image.open(path) as image:
        image.thumbnail((1600, 1600))
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.save(path, format="PNG", optimize=True)


def _fallback_image_prompt(
    anchor: ImageAnchor,
    draft: ChapterDraft,
    book_bible: BookBible,
    chapter_summary: str,
) -> ImagePrompt:
    anchor_meaning = anchor.anchor_id.replace("_", " ").title()
    prompt = (
        f"Create a polished, useful book illustration for anchor {anchor.anchor_id} "
        f"in Chapter {draft.chapter_number}, '{draft.title}', from the book "
        f"'{book_bible.working_title}'. The visual should explain this concept: "
        f"{anchor_meaning}. Use the chapter summary as context: {chapter_summary}. "
        f"Ground the image in the nearby text before the anchor: {anchor.context_before}. "
        f"Also reflect the following text after the anchor: {anchor.context_after}. "
        "Use a clear editorial infographic or concept illustration layout with a "
        "strong focal point, clean spacing, simple readable labels, and no clutter. "
        f"Match this style direction: {book_bible.image_direction or 'modern, professional, accessible'}. "
        "Avoid brand logos, tiny dense text, sensational imagery, and unrelated decorative filler."
    )
    return ImagePrompt(
        anchor_id=anchor.anchor_id,
        chapter_number=anchor.chapter_number,
        position=anchor.marker,
        prompt=prompt,
        style=book_bible.image_direction or "clean editorial infographic",
        alt_text=f"Illustration explaining {anchor_meaning}.",
    )


def _placeholder_asset(prompt: ImagePrompt, reason: str) -> ImageAsset:
    get_logger().info(
        "Using image placeholder for %s: %s",
        prompt.anchor_id or prompt.position,
        reason,
    )
    return ImageAsset(
        anchor_id=prompt.anchor_id,
        chapter_number=prompt.chapter_number,
        position=prompt.position,
        prompt_used=prompt.prompt,
        alt_text=prompt.alt_text,
        is_placeholder=True,
    )


def _image_filename(prompt: ImagePrompt) -> str:
    base = prompt.anchor_id or f"chapter_{prompt.chapter_number:02d}_image"
    safe_name = re.sub(r"[^A-Z0-9_]+", "_", base.upper()).strip("_")
    return f"{safe_name or 'IMAGE'}.png"


def _kie_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {secrets.kie_api_key}",
        "Content-Type": "application/json",
    }


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
