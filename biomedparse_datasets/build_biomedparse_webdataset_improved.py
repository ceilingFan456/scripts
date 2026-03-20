#!/usr/bin/env python3
"""
Recursively build WebDataset shards for BiomedParse-style datasets.

For each dataset folder under --root, we look for:
  <dataset>/<split>/         (images)
  <dataset>/<split>_mask/    (mask pngs)
  <dataset>/<caption_dir>/  (caption txts; same stem as mask png)

Templates (important for --split both):
  --caption_dir "{split}_simple_caption"  -> train_simple_caption / test_simple_caption
  --out_dir     "wds_100_{split}"         -> wds_100_train / wds_100_test
"""

import argparse
import glob
import io
import os
import random
from typing import Dict, List

from PIL import Image
import webdataset as wds
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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
    out = Image.new("RGB", (w * 2, h))
    out.paste(image, (0, 0))
    out.paste(mask, (w, 0))
    return out


def infer_image_path_from_mask(mask_path: str, image_root: str, rule: str) -> str:
    """
    rule:
      - drop_last_underscore: abc_def_x.png -> abc_def.png
      - exact:               abc_def_x.png -> abc_def_x.png
    """
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    if rule == "drop_last_underscore" and "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return os.path.join(image_root, stem + ".png")


def apply_split_template(s: str, split: str) -> str:
    # supports both "{split}" and "%s"
    if "{split}" in s:
        return s.replace("{split}", split)
    if "%s" in s:
        return s % split
    return s


# ----------------------------
# Recursive dataset discovery
# ----------------------------

def is_candidate_dataset_dir(d: str, caption_dir: str, split: str) -> bool:
    img_dir = os.path.join(d, split)
    mask_dir = os.path.join(d, f"{split}_mask")
    cap_dir = os.path.join(d, caption_dir)
    return os.path.isdir(img_dir) and os.path.isdir(mask_dir) and os.path.isdir(cap_dir)


def discover_datasets(root: str, caption_dir: str, split: str) -> List[str]:
    hits = []
    for cur, dirs, _files in os.walk(root):
        if is_candidate_dataset_dir(cur, caption_dir, split):
            hits.append(cur)
            dirs[:] = []  # prune
    return sorted(hits)


# ----------------------------
# WDS writing per dataset
# ----------------------------

def build_one_dataset(
    dataset_dir: str,
    split: str,
    caption_dir: str,
    out_dirname: str,
    *,
    maxcount_per_shard: int,
    max_total_samples: int,
    max_samples_to_save: int,
    random_sample_save: bool,
    image_stem_rule: str,
    skip_missing: bool,
    key_width: int,
) -> Dict[str, int]:
    image_root = os.path.join(dataset_dir, split)
    mask_root = os.path.join(dataset_dir, f"{split}_mask")
    caption_root = os.path.join(dataset_dir, caption_dir)

    out_dir = os.path.join(dataset_dir, out_dirname)
    samples_dir = os.path.join(dataset_dir, out_dirname + "_samples")
    ensure_dir(out_dir)
    ensure_dir(samples_dir)

    mask_paths = sorted(glob.glob(os.path.join(mask_root, "*.png")))
    if not mask_paths:
        return {"wrote": 0, "skipped": 0, "failed": 0}

    # cap upfront, preserving sorted order
    if max_total_samples > 0:
        mask_paths_iter = mask_paths[:max_total_samples]
    else:
        mask_paths_iter = mask_paths

    out_pattern = os.path.join(out_dir, "seg-%06d.tar")

    wrote = skipped = failed = 0
    pending = []  # for random sample saving

    with wds.ShardWriter(out_pattern, maxcount=maxcount_per_shard) as sink:
        pbar = tqdm(
            enumerate(mask_paths_iter),
            total=len(mask_paths_iter),
            desc=f"{os.path.basename(dataset_dir)}/{split}",
            leave=False,
        )

        for idx, mask_path in pbar:
            image_path = infer_image_path_from_mask(mask_path, image_root, image_stem_rule)
            stem = os.path.splitext(os.path.basename(mask_path))[0]
            caption_path = os.path.join(caption_root, stem + ".txt")

            if not os.path.exists(image_path) or not os.path.exists(caption_path):
                if skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing: image={image_path} or caption={caption_path}")

            try:
                image = Image.open(image_path)
                mask = Image.open(mask_path)
                concat_img = concat_image_and_mask(image, mask)

                with open(caption_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if "\n" in txt:
                    txt = txt.splitlines()[0].strip()

                # numeric, deterministic, sorted-order key
                key = str(idx).zfill(key_width)

                sink.write({"__key__": key, "png": image_to_png_bytes(concat_img), "txt": txt})
                wrote += 1

                # sample saving for sanity check: default = first N in sorted order
                if max_samples_to_save > 0:
                    if random_sample_save:
                        if len(pending) < max_samples_to_save * 10:
                            pending.append((key, concat_img.copy(), txt))
                    else:
                        if idx < max_samples_to_save:
                            concat_img.save(os.path.join(samples_dir, f"{key}.png"))
                            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                                f.write(txt + "\n")

            except Exception as e:
                failed += 1
                print(f"[error] {mask_path}: {e}")

    if max_samples_to_save > 0 and random_sample_save and pending:
        chosen = random.sample(pending, k=min(max_samples_to_save, len(pending)))
        for key, img, txt in chosen:
            img.save(os.path.join(samples_dir, f"{key}.png"))
            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(txt + "\n")

    return {"wrote": wrote, "skipped": skipped, "failed": failed}


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing many datasets (searched recursively)")
    ap.add_argument("--split", required=True, choices=["train", "test", "both"])
    ap.add_argument(
        "--caption_dir",
        required=True,
        help='Caption folder name OR template, e.g. "train_simple_caption" or "{split}_simple_caption"',
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help='Output directory name OR template, e.g. "wds_100_train" or "wds_100_{split}"',
    )

    ap.add_argument("--maxcount_per_shard", type=int, default=8000)
    ap.add_argument("--max_total_samples", type=int, default=-1, help="-1 = no limit (per dataset)")
    ap.add_argument("--max_samples_to_save", type=int, default=200, help="How many sanity-check samples to save (sorted order)")
    ap.add_argument("--random_sample_save", action="store_true")

    ap.add_argument("--skip_missing", action="store_true")
    ap.add_argument(
        "--image_stem_rule",
        choices=["drop_last_underscore", "exact"],
        default="drop_last_underscore",
    )

    ap.add_argument("--key_width", type=int, default=9, help="Digits in numeric __key__ (e.g. 000000123)")

    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    total_wrote = total_skipped = total_failed = 0
    total_datasets = 0

    for sp in splits:
        caption_dir = apply_split_template(args.caption_dir, sp)
        out_dir = apply_split_template(args.out_dir, sp)

        ds_list = discover_datasets(args.root, caption_dir, sp)
        if not ds_list:
            print(f"[warn] No datasets found for split={sp} caption_dir={caption_dir} under: {args.root}")
            continue

        print(f"[split={sp}] found {len(ds_list)} dataset(s) | caption_dir={caption_dir} | out_dir={out_dir}")

        for dataset_dir in ds_list:
            total_datasets += 1
            print(f"  -> {dataset_dir}")

            stats = build_one_dataset(
                dataset_dir,
                sp,
                caption_dir,
                out_dir,
                maxcount_per_shard=args.maxcount_per_shard,
                max_total_samples=args.max_total_samples,
                max_samples_to_save=args.max_samples_to_save,
                random_sample_save=args.random_sample_save,
                image_stem_rule=args.image_stem_rule,
                skip_missing=args.skip_missing,
                key_width=args.key_width,
            )
            total_wrote += stats["wrote"]
            total_skipped += stats["skipped"]
            total_failed += stats["failed"]

            print(f"     wrote={stats['wrote']} skipped={stats['skipped']} failed={stats['failed']}")

    print("\n[summary]")
    print(f"  datasets_processed: {total_datasets}")
    print(f"  total_wrote:   {total_wrote}")
    print(f"  total_skipped: {total_skipped}")
    print(f"  total_failed:  {total_failed}")


if __name__ == "__main__":
    main()
