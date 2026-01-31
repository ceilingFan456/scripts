#!/usr/bin/env python3
"""
Recursive media converter:
- .webp (and optionally other image formats) -> .png
- .gif -> .mp4
- deletes originals after successful conversion

Usage:
  python convert_media.py /path/to/root --dry-run
  python convert_media.py /path/to/root
  python convert_media.py /path/to/root --include-jpg --include-bmp --include-tiff
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

# Pillow is used for image conversions
try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow not installed. Install with: pip install pillow")


@dataclass
class Stats:
    images_found: int = 0
    gifs_found: int = 0
    videos_found: int = 0

    images_converted: int = 0
    gifs_converted: int = 0

    images_deleted: int = 0
    gifs_deleted: int = 0

    skipped_existing: int = 0
    failed: int = 0


VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"
}

BASE_IMAGE_EXTS = {".webp"}  # always convert webp -> png
OPTIONAL_IMAGE_EXTS = {
    "jpg": {".jpg", ".jpeg"},
    "bmp": {".bmp"},
    "tiff": {".tif", ".tiff"},
    "png": {".png"},  # usually you don't want to reconvert png
}


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def safe_target_path(src: Path, new_ext: str) -> Path:
    return src.with_suffix(new_ext)


def convert_image_to_png(src: Path, dst: Path) -> None:
    # Convert with Pillow, preserving alpha when present
    with Image.open(src) as im:
        # Ensure compatible mode
        if im.mode in ("P", "LA"):
            im = im.convert("RGBA")
        elif im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA") if "A" in im.getbands() else im.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_gif_to_mp4(src: Path, dst: Path) -> None:
    """
    Uses ffmpeg to convert gif -> mp4.
    - yuv420p improves compatibility
    - scales to even dimensions required by some encoders
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg first.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Root directory to recursively scan")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen, do nothing")
    ap.add_argument("--include-jpg", action="store_true", help="Also convert .jpg/.jpeg -> .png")
    ap.add_argument("--include-bmp", action="store_true", help="Also convert .bmp -> .png")
    ap.add_argument("--include-tiff", action="store_true", help="Also convert .tif/.tiff -> .png")
    ap.add_argument("--skip-if-exists", action="store_true", help="If output exists, skip conversion")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    image_exts = set(BASE_IMAGE_EXTS)
    if args.include_jpg:
        image_exts |= OPTIONAL_IMAGE_EXTS["jpg"]
    if args.include_bmp:
        image_exts |= OPTIONAL_IMAGE_EXTS["bmp"]
    if args.include_tiff:
        image_exts |= OPTIONAL_IMAGE_EXTS["tiff"]

    stats = Stats()

    # Collect first (so we can print counts even in dry-run)
    images: list[Path] = []
    gifs: list[Path] = []
    videos: list[Path] = []

    for p in iter_files(root):
        ext = p.suffix.lower()
        if ext in image_exts:
            images.append(p)
        elif ext == ".gif":
            gifs.append(p)
        elif ext in VIDEO_EXTS:
            videos.append(p)

    stats.images_found = len(images)
    stats.gifs_found = len(gifs)
    stats.videos_found = len(videos)

    print(f"Scanning: {root}")
    print(f"Found images to convert -> png: {stats.images_found}")
    print(f"Found gifs to convert -> mp4:   {stats.gifs_found}")
    print(f"Found other videos (no change): {stats.videos_found}")
    if args.dry_run:
        print("\nDRY RUN: no files will be changed.\n")

    # Convert images
    for src in images:
        try:
            dst = safe_target_path(src, ".png")
            if dst.exists() and args.skip_if_exists:
                stats.skipped_existing += 1
                continue

            print(f"[IMG] {src} -> {dst}")
            if not args.dry_run:
                convert_image_to_png(src, dst)
                stats.images_converted += 1
                # delete original if conversion produced output
                if dst.exists():
                    src.unlink()
                    stats.images_deleted += 1
        except Exception as e:
            stats.failed += 1
            print(f"  !! Failed image: {src}\n     Reason: {e}")

    # Convert gifs
    for src in gifs:
        try:
            dst = safe_target_path(src, ".mp4")
            if dst.exists() and args.skip_if_exists:
                stats.skipped_existing += 1
                continue

            print(f"[GIF] {src} -> {dst}")
            if not args.dry_run:
                convert_gif_to_mp4(src, dst)
                stats.gifs_converted += 1
                if dst.exists():
                    src.unlink()
                    stats.gifs_deleted += 1
        except Exception as e:
            stats.failed += 1
            print(f"  !! Failed gif: {src}\n     Reason: {e}")

    print("\n=== Summary ===")
    print(f"Images found:     {stats.images_found}")
    print(f"Gifs found:       {stats.gifs_found}")
    print(f"Videos found:     {stats.videos_found}")
    print(f"Images converted: {stats.images_converted}")
    print(f"Gifs converted:   {stats.gifs_converted}")
    print(f"Images deleted:   {stats.images_deleted}")
    print(f"Gifs deleted:     {stats.gifs_deleted}")
    print(f"Skipped existing: {stats.skipped_existing}")
    print(f"Failed:           {stats.failed}")

    if stats.gifs_found > 0 and not ffmpeg_available():
        print("\nNote: ffmpeg not found. GIF->MP4 conversions will fail until ffmpeg is installed.")


if __name__ == "__main__":
    main()
