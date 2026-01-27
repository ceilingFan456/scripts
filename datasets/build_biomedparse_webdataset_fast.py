#!/usr/bin/env python3
"""Build BiomedParse WebDataset shards.

Improvements over the previous version:
  1) Recursive discovery: walk DATASET_ROOT and find any "group" that contains
     split image folder + split_mask + split caption folder (or captions beside masks).
  2) Configurable parameters:
       - MAX_SAMPLES_TO_SAVE: number of local inspection samples to save
       - WDS_SUFFIX: suffix for the output folder name
       - MAX_TOTAL_SAMPLES: cap how many samples to write (per split)
       - MAXCOUNT_PER_SHARD: cap samples per tar shard

Expected canonical structure (but discovery is recursive):
  <any_parent>/<split>/              (gt image folder)
  <any_parent>/<split>_mask/         (mask pngs)
  <any_parent>/<split>_simple_caption/ (caption txts matching mask stem)

If caption folder is missing, we will also accept captions "beside" the masks:
  <split>_mask/xxx.png -> <split>_mask/xxx.txt

Each written sample contains:
  - png: concatenated (image | mask) as RGB
  - txt: caption/question text

Usage:
  python build_biomedparse_webdataset_improved.py

Optional environment overrides:
  DATASET_ROOT, WDS_SUFFIX, MAX_TOTAL_SAMPLES, MAXCOUNT_PER_SHARD,
  MAX_SAMPLES_TO_SAVE, RANDOM_SAMPLE_SAVE
"""

import os
import glob
import io
import random
import hashlib
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
import webdataset as wds
from tqdm import tqdm


# ---------------- CONFIG ----------------
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/home/t-qimhuang/disk/datasets/BiomedParseData")
SPLITS = ["train", "test"]

# WebDataset shard settings
MAXCOUNT_PER_SHARD = int(os.environ.get("MAXCOUNT_PER_SHARD", "8000"))

# limit total samples written per split (across all discovered groups)
MAX_TOTAL_SAMPLES = int(os.environ.get("MAX_TOTAL_SAMPLES", "-1"))  # -1 means no limit

# local visual inspection
MAX_SAMPLES_TO_SAVE = int(os.environ.get("MAX_SAMPLES_TO_SAVE", "10"))
RANDOM_SAMPLE_SAVE = os.environ.get("RANDOM_SAMPLE_SAVE", "0") == "1"

# output folder naming
WDS_SUFFIX = os.environ.get("WDS_SUFFIX", "simple_caption")  # becomes wds_<split>_<suffix>

# speed / format knobs
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))  # set yourself
QUEUE_MULT = int(os.environ.get("QUEUE_MULT", "4"))   # inflight futures = NUM_WORKERS*QUEUE_MULT
STORE_MODE = os.environ.get("STORE_MODE", "separate")  # separate|concat
# separate: store raw bytes as {img.png, mask.png, txt}; concat: store {png, txt}


# ---------------- helpers ----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_caption_one_line(path: str) -> str:
    """Read caption, strip, and collapse accidental multi-line output."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if "\n" in txt:
        txt = txt.splitlines()[0].strip()
    return txt


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def concat_image_and_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask = mask.convert("RGB")

    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)

    w, h = image.size
    concat_img = Image.new("RGB", (w * 2, h))
    concat_img.paste(image, (0, 0))
    concat_img.paste(mask, (w, 0))
    return concat_img


def stable_key_from_paths(mask_path: str, image_path: str) -> str:
    """Deterministic key to avoid collisions across multiple groups."""
    h = hashlib.sha1(f"{mask_path}|{image_path}".encode("utf-8")).hexdigest()[:16]
    return f"sample-{h}"


def get_image_path_from_mask(mask_path: str, image_root: str) -> str:
    """Infer image path by removing the last underscore suffix from mask stem.

    mask: patient001_frame01_1_MRI_heart_left+heart+ventricle.png
    img:  patient001_frame01_1_MRI_heart.png

    If no underscore suffix, fall back to same stem.
    """
    basename = os.path.basename(mask_path)
    stem, _ = os.path.splitext(basename)

    try:
        image_stem, _ = stem.rsplit("_", 1)
    except ValueError:
        image_stem = stem

    return os.path.join(image_root, image_stem + ".png")


def mask_path_to_caption_txt(mask_path: str) -> str:
    """Default mapping used in your earlier script.

    train_mask/xxx.png -> train_simple_caption/xxx.txt
    test_mask/xxx.png  -> test_simple_caption/xxx.txt

    If no *_mask segment exists, just replace extension with .txt.
    """
    txt = os.path.splitext(mask_path)[0] + ".txt"
    parts = txt.split(os.sep)
    for i, p in enumerate(parts):
        if p.endswith("_mask"):
            parts[i] = p.replace("_mask", f"_{WDS_SUFFIX}")
            break
    return os.sep.join(parts)


def find_caption_path(mask_path: str, caption_root: Optional[str]) -> Optional[str]:
    """Find caption path for a given mask.

    Priority:
      1) if caption_root exists: <caption_root>/<mask_stem>.txt
      2) fallback: mask beside txt: <mask_root>/<mask_stem>.txt
      3) fallback: your historical mapping (mask_root -> split_<suffix>)

    Returns None if none exist.
    """
    stem = os.path.splitext(os.path.basename(mask_path))[0]

    if caption_root is not None:
        p = os.path.join(caption_root, stem + ".txt")
        if os.path.exists(p):
            return p

    beside = os.path.splitext(mask_path)[0] + ".txt"
    if os.path.exists(beside):
        return beside

    mapped = mask_path_to_caption_txt(mask_path)
    if os.path.exists(mapped):
        return mapped

    return None


@dataclass
class Group:
    dataset_name: str
    split: str
    image_root: str
    mask_root: str
    caption_root: Optional[str]
    base_dir: str


def discover_groups(dataset_root: str, splits: List[str]) -> List[Group]:
    """Recursively find groups that contain image + mask (+ caption optionally)."""
    groups: List[Group] = []

    # treat first-level directories as "dataset_name" if possible; else derive from relative path.
    dataset_root = os.path.abspath(dataset_root)

    for dirpath, dirnames, _filenames in os.walk(dataset_root):
        # Look for <split>_mask directories in this dirpath
        for split in splits:
            mask_dir = os.path.join(dirpath, f"{split}_mask")
            img_dir = os.path.join(dirpath, split)
            if not (os.path.isdir(mask_dir) and os.path.isdir(img_dir)):
                continue

            # Caption folder is optional but preferred
            cap_dir = os.path.join(dirpath, f"{split}_{WDS_SUFFIX}")
            caption_root = cap_dir if os.path.isdir(cap_dir) else None

            # quick sanity: must have at least one png in mask dir and one png in image dir
            has_masks = len(glob.glob(os.path.join(mask_dir, "*.png"))) > 0
            has_imgs = len(glob.glob(os.path.join(img_dir, "*.png"))) > 0
            if not (has_masks and has_imgs):
                continue

            rel = os.path.relpath(dirpath, dataset_root)
            dataset_name = rel.split(os.sep, 1)[0] if rel != "." else os.path.basename(dataset_root)

            groups.append(
                Group(
                    dataset_name=dataset_name,
                    split=split,
                    image_root=img_dir,
                    mask_root=mask_dir,
                    caption_root=caption_root,
                    base_dir=dirpath,
                )
            )

    # de-dup groups by (base_dir, split)
    seen = set()
    uniq: List[Group] = []
    for g in groups:
        k = (os.path.abspath(g.base_dir), g.split)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(g)

    return sorted(uniq, key=lambda x: (x.dataset_name.lower(), x.split, x.base_dir))


def _prepare_sample(mask_path: str, group: Group) -> Tuple[str, Optional[dict], str]:
    """Prepare a single sample.

    Returns (status, sample_or_none, reason).
    status in {"ok","skip","fail"}
    """
    image_path = get_image_path_from_mask(mask_path, group.image_root)
    caption_path = find_caption_path(mask_path, group.caption_root)

    if not os.path.exists(image_path):
        return "skip", None, "missing_image"
    if caption_path is None or not os.path.exists(caption_path):
        return "skip", None, "missing_caption"

    try:
        question = read_caption_one_line(caption_path)
        key = stable_key_from_paths(mask_path, image_path)

        if STORE_MODE == "concat":
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            concat_img = concat_image_and_mask(image, mask)
            sample = {"__key__": key, "png": image_to_png_bytes(concat_img), "txt": question}
        else:
            # fastest path: no decode/re-encode, just pass through bytes
            img_bytes = open(image_path, "rb").read()
            mask_bytes = open(mask_path, "rb").read()
            # choose stable extensions for WDS decoding later
            sample = {"__key__": key, "img.png": img_bytes, "mask.png": mask_bytes, "txt": question}

        return "ok", sample, ""
    except Exception as e:
        return "fail", None, str(e)




def _prepare_sample(mask_path: str, group: Group) -> Tuple[str, Optional[dict], str]:
    """Prepare a single sample (I/O + minimal processing).

    Returns (status, sample_or_none, reason).
      - status: "ok" | "skip" | "fail"
      - reason: short string for logging/metrics
    """
    image_path = get_image_path_from_mask(mask_path, group.image_root)
    caption_path = find_caption_path(mask_path, group.caption_root)

    if not os.path.exists(image_path):
        return "skip", None, "missing_image"
    if caption_path is None or not os.path.exists(caption_path):
        return "skip", None, "missing_caption"

    try:
        question = read_caption_one_line(caption_path)

        key = stable_key_from_paths(mask_path, image_path)

        if STORE_MODE.lower() == "concat":
            # slower: decode + resize + re-encode
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            concat_img = concat_image_and_mask(image, mask)
            sample = {
                "__key__": key,
                "png": image_to_png_bytes(concat_img),
                "txt": question,
            }
        else:
            # fast path: store raw bytes (no decode/re-encode)
            img_bytes = open(image_path, "rb").read()
            mask_bytes = open(mask_path, "rb").read()
            sample = {
                "__key__": key,
                "img.png": img_bytes,
                "mask.png": mask_bytes,
                "txt": question,
            }

        return "ok", sample, ""

    except Exception as e:
        return "fail", None, str(e)


def build_wds_for_group(
    group: Group,
    max_total_samples: int,
) -> Tuple[int, int, int]:
    """Write WebDataset for one group. Returns (written, skipped, failed)."""

    mask_paths = sorted(glob.glob(os.path.join(group.mask_root, "*.png")))
    if not mask_paths:
        print(f"[{group.dataset_name}] {group.split}: no masks found in {group.mask_root}, skip")
        return 0, 0, 0

    # Output dirs under the group base_dir (same behavior as your previous script)
    output_dir = os.path.join(group.base_dir, f"wds_{group.split}_{WDS_SUFFIX}")
    samples_dir = os.path.join(group.base_dir, f"wds_{group.split}_{WDS_SUFFIX}_samples")

    ensure_dir(output_dir)
    ensure_dir(samples_dir)

    safe_name = group.dataset_name.lower().replace(" ", "_")
    output_pattern = os.path.join(output_dir, f"{safe_name}_seg-%06d.tar")

    print(f"\n[{group.dataset_name}] {group.split}")
    print(f"  base_dir   : {group.base_dir}")
    print(f"  image_root : {group.image_root}")
    print(f"  mask_root  : {group.mask_root}")
    print(f"  caption_root: {group.caption_root if group.caption_root else '(fallback beside masks)'}")
    print(f"  masks      : {len(mask_paths)}")
    print(f"  out_wds    : {output_dir}")
    print(f"  out_samples(<= {MAX_SAMPLES_TO_SAVE}): {samples_dir}")

    save_random = RANDOM_SAMPLE_SAVE
    written = 0
    skipped = 0
    failed = 0

    with wds.ShardWriter(output_pattern, maxcount=MAXCOUNT_PER_SHARD) as sink:
        # Parallelize I/O + sample preparation. Writer stays single-threaded.
        max_inflight = max(1, NUM_WORKERS * QUEUE_MULT)
        inflight = set()

        # For random sample saving: store minimal bytes to reconstruct a preview later.
        # Each item: (key, mode, payload, question)
        #   mode="concat": payload=png_bytes
        #   mode="separate": payload=(img_bytes, mask_bytes)
        pending_save_bytes: List[Tuple[str, str, object, str]] = []

        def maybe_record_preview(key: str, sample: dict, question: str) -> None:
            if MAX_SAMPLES_TO_SAVE <= 0:
                return
            if not save_random:
                return
            if len(pending_save_bytes) >= MAX_SAMPLES_TO_SAVE * 5:
                return

            if STORE_MODE.lower() == "concat":
                pending_save_bytes.append((key, "concat", sample["png"], question))
            else:
                pending_save_bytes.append((key, "separate", (sample["img.png"], sample["mask.png"]), question))

        def save_preview_direct(i_written: int, key: str, sample: dict, question: str) -> None:
            """Deterministic first-K saving."""
            if MAX_SAMPLES_TO_SAVE <= 0:
                return
            if save_random:
                return
            if i_written >= MAX_SAMPLES_TO_SAVE:
                return

            if STORE_MODE.lower() == "concat":
                # already a preview image
                with open(os.path.join(samples_dir, f"{key}.png"), "wb") as f:
                    f.write(sample["png"])
            else:
                # build preview only for these few samples
                img = Image.open(io.BytesIO(sample["img.png"]))
                msk = Image.open(io.BytesIO(sample["mask.png"]))
                concat_img = concat_image_and_mask(img, msk)
                concat_img.save(os.path.join(samples_dir, f"{key}.png"))

            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(question + "\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, NUM_WORKERS)) as executor:
            it = iter(mask_paths)

            # Prime the pipeline
            while len(inflight) < max_inflight:
                try:
                    mp = next(it)
                except StopIteration:
                    break
                inflight.add(executor.submit(_prepare_sample, mp, group))

            # Consume as futures complete, and keep pipeline full
            while inflight:
                done, inflight = concurrent.futures.wait(
                    inflight, return_when=concurrent.futures.FIRST_COMPLETED
                )

                for fut in done:
                    if max_total_samples > 0 and written >= max_total_samples:
                        # stop early: cancel remaining futures
                        for r in inflight:
                            r.cancel()
                        inflight = set()
                        break

                    status, sample, reason = fut.result()

                    if status == "skip":
                        skipped += 1
                    elif status == "fail":
                        failed += 1
                        # Keep error light; reason already contains exception str
                        # print(f"[error] {group.dataset_name}/{group.split}: {reason}")
                    else:
                        assert sample is not None
                        key = sample["__key__"]
                        question = sample["txt"]
                        sink.write(sample)
                        maybe_record_preview(key, sample, question)
                        save_preview_direct(written, key, sample, question)
                        written += 1

                    # Refill pipeline
                    while len(inflight) < max_inflight:
                        try:
                            mp = next(it)
                        except StopIteration:
                            break
                        inflight.add(executor.submit(_prepare_sample, mp, group))


    if MAX_SAMPLES_TO_SAVE > 0 and save_random and pending_save_bytes:
        k = min(MAX_SAMPLES_TO_SAVE, len(pending_save_bytes))
        chosen = random.sample(pending_save_bytes, k)
        for key, mode, payload, question in chosen:
            if mode == "concat":
                with open(os.path.join(samples_dir, f"{key}.png"), "wb") as f:
                    f.write(payload)  # type: ignore[arg-type]
            else:
                img_bytes, mask_bytes = payload  # type: ignore[misc]
                img = Image.open(io.BytesIO(img_bytes))
                msk = Image.open(io.BytesIO(mask_bytes))
                concat_img = concat_image_and_mask(img, msk)
                concat_img.save(os.path.join(samples_dir, f"{key}.png"))

            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(question + "\n")
    print(f"[{group.dataset_name}] {group.split} done: wrote={written} skipped={skipped} failed={failed}")
    return written, skipped, failed


def main() -> None:
    groups = discover_groups(DATASET_ROOT, SPLITS)

    if not groups:
        print(f"No valid (image/mask) groups found under: {DATASET_ROOT}")
        print("Expected something like: <dir>/train + <dir>/train_mask (+ optional <dir>/train_<suffix>)")
        return

    # Track totals per split across all groups
    totals: Dict[str, int] = {s: 0 for s in SPLITS}

    for g in groups:
        # Apply per-split cap across all groups
        remaining = MAX_TOTAL_SAMPLES
        if MAX_TOTAL_SAMPLES > 0:
            remaining = max(0, MAX_TOTAL_SAMPLES - totals[g.split])
            if remaining <= 0:
                continue

        w, s, f = build_wds_for_group(g, max_total_samples=remaining)
        totals[g.split] += w

    print("\n=== Summary ===")
    for split in SPLITS:
        print(f"{split}: total_written={totals[split]}")


if __name__ == "__main__":
    main()
