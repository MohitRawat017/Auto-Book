"""Background image generation queue — submits KIE jobs in parallel threads."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auto_book.models.book_bible import BookBible
    from auto_book.models.chapter import ChapterDraft
    from auto_book.models.image import ImageAnchor, ImageAsset

from auto_book.utils.logger import get_logger


class ImageGenerationQueue:
    """Thread-pool queue for parallel KIE image generation."""

    def __init__(self, max_workers: int = 3) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="img")
        self._futures: list[Future] = []
        self._cover_future: Future | None = None
        self._lock = threading.Lock()

    def submit_chapter(
        self,
        anchor: ImageAnchor,
        chapter: ChapterDraft,
        book_bible: BookBible,
        output_dir: str,
    ) -> None:
        """Non-blocking. Skip if image file already exists."""
        from auto_book.agents.image_agent import generate_image_via_kie, plan_image_for_anchor
        from auto_book.config import settings

        if not settings.images.generate_actual:
            return

        images_dir = Path(output_dir) / "images"
        from auto_book.agents.image_agent import _image_filename
        from auto_book.models.image import ImagePrompt
        # Build a dummy prompt just to get the filename for the skip check
        candidate = images_dir / f"{anchor.anchor_id}.png"
        if candidate.exists():
            get_logger().debug("Queue: skipping %s (already exists)", anchor.anchor_id)
            return

        def _job():
            try:
                prompt = plan_image_for_anchor(anchor, chapter, book_bible)
                return generate_image_via_kie(prompt, output_dir)
            except Exception as exc:
                get_logger().warning("Queue: image job failed for %s: %s", anchor.anchor_id, exc)
                from auto_book.models.image import ImageAsset
                return ImageAsset(
                    anchor_id=anchor.anchor_id,
                    chapter_number=anchor.chapter_number,
                    position=anchor.marker,
                    prompt_used="",
                    alt_text=anchor.anchor_id.replace("_", " ").title(),
                    is_placeholder=True,
                )

        with self._lock:
            self._futures.append(self._executor.submit(_job))
        get_logger().info("Queue: submitted image job for %s", anchor.anchor_id)

    def submit_cover(self, book_bible: BookBible, output_dir: str) -> None:
        """Non-blocking. Skip if COVER.png already exists."""
        from auto_book.agents.image_agent import generate_cover_image
        from auto_book.config import settings

        if not settings.images.generate_actual:
            return

        cover_path = Path(output_dir) / "images" / "COVER.png"
        if cover_path.exists():
            get_logger().debug("Queue: skipping cover (already exists)")
            return

        def _job():
            try:
                return generate_cover_image(book_bible, output_dir)
            except Exception as exc:
                get_logger().warning("Queue: cover job failed: %s", exc)
                from auto_book.models.image import ImageAsset
                return ImageAsset(
                    anchor_id="COVER", chapter_number=0, position="cover",
                    prompt_used="", alt_text="Cover image", is_placeholder=True,
                )

        with self._lock:
            self._cover_future = self._executor.submit(_job)
        get_logger().info("Queue: submitted cover image job")

    def collect(self, timeout: int = 300) -> tuple[list[ImageAsset], ImageAsset | None]:
        """Block until all submitted jobs finish. Returns (chapter_assets, cover_asset)."""
        from auto_book.models.image import ImageAsset

        with self._lock:
            futures = list(self._futures)
            cover_future = self._cover_future

        assets: list[ImageAsset] = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    assets.append(result)
            except Exception as exc:
                get_logger().warning("Queue: collecting image result failed: %s", exc)

        cover_asset: ImageAsset | None = None
        if cover_future is not None:
            try:
                cover_asset = cover_future.result(timeout=timeout)
            except Exception as exc:
                get_logger().warning("Queue: collecting cover result failed: %s", exc)

        # Also load any already-existing assets that were skipped
        get_logger().info(
            "Queue: collected %s chapter image(s), cover=%s",
            len(assets),
            "yes" if cover_asset and not cover_asset.is_placeholder else "placeholder/none",
        )
        return assets, cover_asset

    def reset(self) -> None:
        """Clear all pending jobs (call at start of each run)."""
        with self._lock:
            self._futures.clear()
            self._cover_future = None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# Module-level singleton
image_queue = ImageGenerationQueue(max_workers=3)
