"""Image Agent: plans image placements and optionally generates images via KIE."""

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
from auto_book.models.image import ImageAsset, ImagePrompt
from auto_book.utils.logger import get_logger
from auto_book.utils.rate_limiter import wait_for_rate_limit
from auto_book.utils.tokens import truncate_to_budget

KIE_CREATE_TASK_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_INFO_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"

IMAGE_PLANNER_SYSTEM_PROMPT = """You are an art director for a book project.

Your job is to identify where images would improve reader understanding or
engagement, then create specific image generation prompts.

Book context:
- Title: {title}
- Genre: {genre}
- Image direction: {image_direction}

Rules:
- Suggest only images that genuinely add value.
- Return {max_images} image prompt(s) maximum for this chapter.
- Prefer useful visuals: diagrams, concept illustrations, worksheets, or
  simple infographics for non-fiction; scene or character moments for fiction.
- Make prompts detailed enough for a text-to-image model.
- Include concise accessibility alt text.
- Return ONLY valid JSON, with no Markdown or code fences.

Required JSON shape:
{{
  "images": [
    {{
      "chapter_number": 1,
      "position": "after the introduction",
      "prompt": "A clean editorial illustration...",
      "style": "minimalist editorial illustration",
      "alt_text": "Description for screen readers"
    }}
  ]
}}
"""

IMAGE_PLANNER_USER_PROMPT = """Analyze this chapter and suggest image placements:

Chapter {chapter_number}: "{title}"

{body_excerpt}

Return only the image-plan JSON.
"""


def run_image_agent(
    chapters: list[ChapterDraft],
    book_bible: BookBible,
    output_dir: str,
) -> list[ImageAsset]:
    """Plan and optionally generate images for accepted chapters."""

    logger = get_logger()
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not settings.images.enabled:
        logger.info("Image Agent disabled by config.")
        _save_json(images_dir / "image_plan.json", [])
        _save_json(images_dir / "image_assets.json", [])
        return []

    prompts: list[ImagePrompt] = []
    for chapter in sorted(chapters, key=lambda item: item.chapter_number):
        prompts.extend(plan_images_for_chapter(chapter, book_bible))

    _save_json(images_dir / "image_plan.json", [prompt.model_dump() for prompt in prompts])
    logger.info("Image planning complete: %s prompt(s)", len(prompts))

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


def plan_images_for_chapter(
    draft: ChapterDraft,
    book_bible: BookBible,
) -> list[ImagePrompt]:
    """Use an LLM to create image prompts for one accepted chapter."""

    max_images = max(0, settings.images.max_images_per_chapter)
    if max_images == 0:
        return []

    logger = get_logger()
    system_msg = IMAGE_PLANNER_SYSTEM_PROMPT.format(
        title=book_bible.working_title,
        genre=book_bible.genre,
        image_direction=book_bible.image_direction
        or "Clean, professional editorial illustrations.",
        max_images=max_images,
    )
    user_msg = IMAGE_PLANNER_USER_PROMPT.format(
        chapter_number=draft.chapter_number,
        title=draft.title,
        body_excerpt=truncate_to_budget(draft.body, 1800),
    )

    try:
        wait_for_rate_limit()
        response = get_llm("reviewer").invoke(
            [
                ("system", system_msg),
                ("human", user_msg),
            ]
        )
        payload = _load_json_object(_strip_code_fence(_extract_message_text(response)))
        raw_images = payload.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("Image planner returned a non-list images field.")

        prompts = []
        for raw in raw_images[:max_images]:
            if not isinstance(raw, dict):
                continue
            raw["chapter_number"] = draft.chapter_number
            prompts.append(ImagePrompt.model_validate(raw))

        if prompts:
            logger.info(
                "Image plan for chapter %s: %s prompt(s)",
                draft.chapter_number,
                len(prompts),
            )
            return prompts
    except Exception as exc:
        logger.warning(
            "Image planning failed for chapter %s: %s. Using fallback prompt.",
            draft.chapter_number,
            exc,
        )

    return [_fallback_image_prompt(draft, book_bible)]


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

        logger.info("Image generated for chapter %s: %s", prompt.chapter_number, file_path)
        return ImageAsset(
            chapter_number=prompt.chapter_number,
            position=prompt.position,
            file_path=str(file_path),
            prompt_used=prompt.prompt,
            alt_text=prompt.alt_text,
            is_placeholder=False,
        )
    except Exception as exc:
        logger.warning("Image generation failed for chapter %s: %s", prompt.chapter_number, exc)
        if settings.images.fallback_to_placeholder:
            return _placeholder_asset(prompt, str(exc))
        raise


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


def _fallback_image_prompt(draft: ChapterDraft, book_bible: BookBible) -> ImagePrompt:
    return ImagePrompt(
        chapter_number=draft.chapter_number,
        position="after the chapter introduction",
        prompt=(
            f"Create a clean editorial illustration for Chapter {draft.chapter_number}, "
            f"'{draft.title}', in the book '{book_bible.working_title}'. "
            f"Visualize the chapter's central idea in a useful, non-decorative way. "
            f"Style direction: {book_bible.image_direction or 'professional, clear, modern'}."
        ),
        style=book_bible.image_direction or "clean editorial illustration",
        alt_text=f"Illustration summarizing the main idea of {draft.title}.",
    )


def _placeholder_asset(prompt: ImagePrompt, reason: str) -> ImageAsset:
    get_logger().info(
        "Using image placeholder for chapter %s at %s: %s",
        prompt.chapter_number,
        prompt.position,
        reason,
    )
    return ImageAsset(
        chapter_number=prompt.chapter_number,
        position=prompt.position,
        prompt_used=prompt.prompt,
        alt_text=prompt.alt_text,
        is_placeholder=True,
    )


def _image_filename(prompt: ImagePrompt) -> str:
    safe_position = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.position).strip("_").lower()
    safe_position = safe_position[:32] or "image"
    return f"chapter_{prompt.chapter_number:02d}_{safe_position}.png"


def _kie_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {secrets.kie_api_key}",
        "Content-Type": "application/json",
    }


def _extract_message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _strip_code_fence(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return body


def _load_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Image planner returned JSON, but not an object.")
    return data


def _save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
