#!/usr/bin/env python3
"""
Build WebDataset shards for ONE dataset folder (e.g., DRIVE/).

Expected structure under --dataset_dir:
  <split>/                (images)
  <split>_mask/           (mask pngs)
  <caption_dir>/          (caption txts; same stem as mask png)

Example:
  python build_wds_single.py \
    --dataset_dir /path/to/DRIVE \
    --split train \
    --caption_dir train_simple_caption \
    --out_dir wds_100_train
"""

import argparse
import glob
import hashlib
import io
import os
import random
from PIL import Image
import webdataset as wds
from tqdm import tqdm
import shutil


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


def stable_key(mask_path: str, image_path: str) -> str:
    h = hashlib.sha1(f"{mask_path}|{image_path}".encode("utf-8")).hexdigest()[:16]
    return f"sample-{h}"


def infer_image_path_from_mask(mask_path: str, image_root: str, rule: str) -> str:
    """
    rule:
      - drop_last_underscore: abc_def_x.png -> abc_def.png
      - exact:               abc_def_x.png -> abc_def_x.png
    """
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    if rule == "drop_last_underscore":
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
    return os.path.join(image_root, stem + ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="Path to dataset folder, e.g. .../DRIVE")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--caption_dir", required=True, help="e.g. train_simple_caption or train_caption")
    ap.add_argument("--out_dir", required=True, help="e.g. wds_100_train")
    ap.add_argument("--maxcount_per_shard", type=int, default=8000)
    ap.add_argument("--max_total_samples", type=int, default=-1, help="-1 = no limit")
    ap.add_argument("--max_samples_to_save", type=int, default=10)
    ap.add_argument("--random_sample_save", action="store_true")
    ap.add_argument(
        "--image_stem_rule",
        choices=["drop_last_underscore", "exact"],
        default="drop_last_underscore",
        help="How to map mask filename -> image filename",
    )
    args = ap.parse_args()

    image_root = os.path.join(args.dataset_dir, args.split)
    mask_root = os.path.join(args.dataset_dir, f"{args.split}_mask")
    caption_root = os.path.join(args.dataset_dir, args.caption_dir)

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Missing image folder: {image_root}")
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f"Missing mask folder: {mask_root}")
    if not os.path.isdir(caption_root):
        raise FileNotFoundError(f"Missing caption folder: {caption_root}")

    out_dir = os.path.join(args.dataset_dir, args.out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    samples_dir = os.path.join(args.dataset_dir, args.out_dir + "_samples")
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)
    ensure_dir(out_dir)
    ensure_dir(samples_dir)

    mask_paths = sorted(glob.glob(os.path.join(mask_root, "*.png")))
    if not mask_paths:
        print(f"No masks found in {mask_root}")
        return

    out_pattern = os.path.join(out_dir, "seg-%06d.tar")

    print(f"[split={args.split}]")
    print(f"  image_root : {image_root}")
    print(f"  mask_root  : {mask_root}")
    print(f"  caption_root: {caption_root}")
    print(f"  out_dir    : {out_dir}")
    print(f"  samples_dir: {samples_dir}")

    pending = []
    written = skipped = failed = 0

    with wds.ShardWriter(out_pattern, maxcount=args.maxcount_per_shard) as sink:
        for mask_path in tqdm(mask_paths, desc=f"Writing {os.path.basename(args.dataset_dir)}/{args.split}"):
            if args.max_total_samples > 0 and written >= args.max_total_samples:
                break

            image_path = infer_image_path_from_mask(mask_path, image_root, args.image_stem_rule)
            stem = os.path.splitext(os.path.basename(mask_path))[0]
            caption_path = os.path.join(caption_root, stem + ".txt")

            if not os.path.exists(image_path) or not os.path.exists(caption_path):
                skipped += 1
                continue

            try:
                image = Image.open(image_path)
                mask = Image.open(mask_path)
                concat_img = concat_image_and_mask(image, mask)

                with open(caption_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if "\n" in txt:
                    txt = txt.splitlines()[0].strip()

                key = stable_key(mask_path, image_path)
                sink.write({"__key__": key, "png": image_to_png_bytes(concat_img), "txt": txt})

                # save inspection samples
                if args.max_samples_to_save > 0:
                    if args.random_sample_save:
                        if len(pending) < args.max_samples_to_save * 5:
                            pending.append((key, concat_img.copy(), txt))
                    else:
                        if written < args.max_samples_to_save:
                            concat_img.save(os.path.join(samples_dir, f"{key}.png"))
                            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                                f.write(txt + "\n")

                written += 1
            except Exception as e:
                failed += 1
                print(f"[error] {mask_path}: {e}")

    if args.max_samples_to_save > 0 and args.random_sample_save and pending:
        chosen = random.sample(pending, k=min(args.max_samples_to_save, len(pending)))
        for key, img, txt in chosen:
            img.save(os.path.join(samples_dir, f"{key}.png"))
            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(txt + "\n")

    print(f"done: wrote={written} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
