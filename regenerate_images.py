"""
Standalone script: generate images for a completed run and reassemble the book.

Usage:
    python regenerate_images.py --output ./output/medium_test
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from auto_book.agents.assembler import assemble_docx, assemble_markdown
from auto_book.agents.image_agent import extract_image_anchors, generate_cover_image, generate_image_via_kie, plan_image_for_anchor
from auto_book.config import load_settings, set_settings, secrets
from auto_book.models.book_bible import BookBible
from auto_book.models.chapter import ChapterDraft
from auto_book.models.image import ImageAsset
from auto_book.models.run_state import RunState
from auto_book.utils.logger import get_logger, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images and reassemble book.")
    parser.add_argument("--output", required=True, help="Run output directory")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--assemble-only", action="store_true", help="Skip image generation, just reassemble from saved assets")
    args = parser.parse_args()

    output_dir = args.output
    cfg = load_settings(args.config)
    set_settings(cfg)
    setup_logger(level="INFO", log_to_file=False)
    logger = get_logger()

    # Load run state
    state_path = Path(output_dir) / "run_state.json"
    if not state_path.exists():
        print(f"No run_state.json found in {output_dir}")
        sys.exit(1)

    run_state = RunState.model_validate_json(state_path.read_text(encoding="utf-8"))
    if run_state.book_bible is None:
        print("No Book Bible in run state.")
        sys.exit(1)

    # Load accepted chapter drafts
    chapters_dir = Path(output_dir) / "chapters"
    chapters: list[ChapterDraft] = []
    for md_file in sorted(chapters_dir.glob("chapter_*.md")):
        chapter_number = int(md_file.stem.split("_")[1])
        body = md_file.read_text(encoding="utf-8")
        # Find title from book bible
        outline = next(
            (c for c in run_state.book_bible.chapter_outline if c.chapter_number == chapter_number),
            None,
        )
        title = outline.title if outline else md_file.stem
        chapters.append(ChapterDraft(chapter_number=chapter_number, title=title, body=body))

    if not chapters:
        print("No chapter files found.")
        sys.exit(1)

    logger.info("Loaded %s chapters from %s", len(chapters), chapters_dir)

    # Load existing manifest
    manifest_path = Path(output_dir) / "images" / "image_manifest.json"
    if not manifest_path.exists():
        logger.info("No image manifest found — re-extracting anchors from chapters.")
        anchors = extract_image_anchors(chapters)
    else:
        from auto_book.models.image import ImageAnchor
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        anchors = [ImageAnchor.model_validate(a) for a in raw]

    if not anchors:
        print("No image anchors found. Nothing to generate.")
        sys.exit(0)

    logger.info("Found %s image anchor(s) to generate.", len(anchors))

    assets_path = Path(output_dir) / "images" / "image_assets.json"

    if args.assemble_only:
        # Load already-generated assets and skip to reassembly
        if assets_path.exists():
            raw = json.loads(assets_path.read_text(encoding="utf-8"))
            assets = [ImageAsset.model_validate(a) for a in raw]
            logger.info("Loaded %s saved asset(s), skipping generation.", len(assets))
        else:
            assets = []
        # Merge any PNGs on disk not yet in assets (e.g. from interrupted background queue)
        assets = _merge_disk_images(assets, output_dir, logger)
        cover_path = Path(output_dir) / "images" / "COVER.png"
        cover_asset = ImageAsset(
            anchor_id="COVER", chapter_number=0, position="cover",
            file_path=str(cover_path) if cover_path.exists() else "",
            prompt_used="", alt_text="Cover image",
            is_placeholder=not cover_path.exists(),
        )
        _reassemble(run_state.book_bible, chapters, output_dir, assets, logger, cover_asset)
        return

    # Check KIE key
    if not secrets.kie_api_key:
        print("KIE_API_KEY is not set in .env — cannot generate images.")
        sys.exit(1)

    # Force generate_actual on
    cfg.images.generate_actual = True
    set_settings(cfg)

    # Generate images
    # Generate cover image (skip if already exists)
    cover_path = Path(output_dir) / "images" / "COVER.png"
    if cover_path.exists():
        logger.info("Cover image already exists, skipping.")
        cover_asset = ImageAsset(
            anchor_id="COVER", chapter_number=0, position="cover",
            file_path=str(cover_path), prompt_used="", alt_text="Cover image",
            is_placeholder=False,
        )
    else:
        logger.info("Generating cover image...")
        cover_asset = generate_cover_image(run_state.book_bible, output_dir)
        logger.info("Cover: %s", "generated" if not cover_asset.is_placeholder else "placeholder")

    # Generate images — skip any anchor already generated
    existing_assets: dict[str, ImageAsset] = {}
    if assets_path.exists():
        raw_existing = json.loads(assets_path.read_text(encoding="utf-8"))
        for a in raw_existing:
            asset = ImageAsset.model_validate(a)
            if asset.file_path and Path(asset.file_path).exists():
                existing_assets[asset.anchor_id] = asset

    if existing_assets:
        logger.info("Skipping %s already-generated image(s).", len(existing_assets))

    chapter_by_number = {c.chapter_number: c for c in chapters}
    assets: list[ImageAsset] = []
    for anchor in anchors:
        if anchor.anchor_id in existing_assets:
            logger.info("Skipping %s (already exists).", anchor.anchor_id)
            assets.append(existing_assets[anchor.anchor_id])
            continue
        chapter = chapter_by_number.get(anchor.chapter_number)
        if chapter is None:
            logger.warning("Chapter %s not found for anchor %s", anchor.chapter_number, anchor.anchor_id)
            continue
        logger.info("Expanding prompt for %s...", anchor.anchor_id)
        prompt = plan_image_for_anchor(anchor, chapter, run_state.book_bible)
        logger.info("Generating image for %s...", anchor.anchor_id)
        asset = generate_image_via_kie(prompt, output_dir)
        assets.append(asset)
        logger.info(
            "%s: %s",
            anchor.anchor_id,
            "generated" if not asset.is_placeholder else f"placeholder ({asset.prompt_used})",
        )

    # Save updated assets
    assets_path.write_text(
        json.dumps([a.model_dump() for a in assets], indent=2), encoding="utf-8"
    )
    logger.info("Saved %s asset(s) to %s", len(assets), assets_path)

    # Merge any PNGs on disk not yet in assets
    assets = _merge_disk_images(assets, output_dir, logger)

    _reassemble(run_state.book_bible, chapters, output_dir, assets, logger, cover_asset)


def _merge_disk_images(
    assets: list[ImageAsset], output_dir: str, logger
) -> list[ImageAsset]:
    """Add any PNG files on disk whose anchor_id isn't already in assets."""
    images_dir = Path(output_dir) / "images"
    existing_ids = {a.anchor_id for a in assets}
    added = 0
    for png in sorted(images_dir.glob("*.png")):
        anchor_id = png.stem  # filename without .png = anchor_id
        if anchor_id in existing_ids or anchor_id == "COVER":
            continue
        assets.append(ImageAsset(
            anchor_id=anchor_id,
            chapter_number=0,  # assembler matches by anchor_id, not chapter_number
            position=f"[IMAGE_ANCHOR: {anchor_id}]",
            file_path=str(png),
            prompt_used="",
            alt_text=anchor_id.replace("_", " ").title(),
            is_placeholder=False,
        ))
        existing_ids.add(anchor_id)
        added += 1
    if added:
        logger.info("Merged %s additional image(s) from disk.", added)
    return assets


def _reassemble(book_bible, chapters, output_dir, assets, logger, cover_asset=None):
    logger.info("Reassembling book...")
    assemble_markdown(book_bible, chapters, output_dir, image_assets=assets)
    assemble_docx(book_bible, chapters, output_dir, image_assets=assets, cover_asset=cover_asset)
    logger.info("Done. Book reassembled at %s/book.docx", output_dir)


if __name__ == "__main__":
    main()
