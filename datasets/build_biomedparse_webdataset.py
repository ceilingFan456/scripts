#!/usr/bin/env python3
import os
import glob
import io
import random
from typing import Optional, Tuple

from PIL import Image
import webdataset as wds
from tqdm import tqdm


# -------- CONFIG --------
DATASET_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData"   # contains ACDC/, CAMUS/, MSD/, ...
SPLITS = ["train", "test"]

# WebDataset shard settings
MAXCOUNT_PER_SHARD = 8000  # controls how many samples per .tar shard

# local visual inspection
MAX_SAMPLES_TO_SAVE = 10   # save only 10 local samples per (dataset, split)
RANDOM_SAMPLE_SAVE = False # if True, save random 10 among successful; else save first 10


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_image_path_from_mask(mask_path: str, image_root: str) -> str:
    """
    mask: patient001_frame01_1_MRI_heart_left+heart+ventricle.png
    img:  patient001_frame01_1_MRI_heart.png (in image_root)
    """
    basename = os.path.basename(mask_path)
    stem, _ = os.path.splitext(basename)

    try:
        image_stem, _ = stem.rsplit("_", 1)
    except ValueError:
        image_stem = stem

    image_filename = image_stem + ".png"
    return os.path.join(image_root, image_filename)


def mask_path_to_caption_txt(mask_path: str) -> str:
    """
    train_mask/xxx.png -> train_caption/xxx.txt
    test_mask/xxx.png  -> test_caption/xxx.txt
    """
    txt = os.path.splitext(mask_path)[0] + ".txt"
    parts = txt.split(os.sep)
    for i, p in enumerate(parts):
        if p.endswith("_mask"):
            parts[i] = p.replace("_mask", "_caption")
            break
    return os.sep.join(parts)


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


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_wds_for_one_split(
    dataset_name: str,
    dataset_dir: str,
    split: str,
):
    """
    For split=train:
      image_root = dataset_dir/train
      mask_root  = dataset_dir/train_mask
      caption txt beside masks in train_caption
    """

    image_root = os.path.join(dataset_dir, split)
    mask_root = os.path.join(dataset_dir, f"{split}_mask")

    if not os.path.isdir(image_root) or not os.path.isdir(mask_root):
        return

    mask_paths = sorted(glob.glob(os.path.join(mask_root, "*.png")))
    if len(mask_paths) == 0:
        print(f"[{dataset_name}] {split}: no masks found, skip.")
        return

    # Output dirs
    output_dir = os.path.join(dataset_dir, f"wds_{split}")
    samples_dir = os.path.join(dataset_dir, f"wds_{split}_samples")

    ensure_dir(output_dir)
    ensure_dir(samples_dir)

    # shard pattern
    safe_name = dataset_name.lower()
    output_pattern = os.path.join(output_dir, f"{safe_name}_seg-%06d.tar")

    print(f"\n[{dataset_name}] {split}")
    print(f"  image_root: {image_root}")
    print(f"  mask_root : {mask_root}")
    print(f"  masks     : {len(mask_paths)}")
    print(f"  out_wds   : {output_dir}")
    print(f"  out_samples(<= {MAX_SAMPLES_TO_SAVE}): {samples_dir}")

    # Decide which samples to save locally (indices among successful writes)
    # If RANDOM_SAMPLE_SAVE: we will buffer successful keys and later save 10 random.
    # Else: save first 10 successful directly.
    save_random = RANDOM_SAMPLE_SAVE
    pending_save: list[Tuple[str, Image.Image, str]] = []  # (key, concat_img, question)

    count_written = 0
    count_skipped = 0
    count_failed = 0

    with wds.ShardWriter(output_pattern, maxcount=MAXCOUNT_PER_SHARD) as sink:
        for mask_path in tqdm(mask_paths, desc="Writing WebDataset"):
            image_path = get_image_path_from_mask(mask_path, image_root)
            question_path = mask_path_to_caption_txt(mask_path)

            if not os.path.exists(image_path):
                count_skipped += 1
                continue
            if not os.path.exists(question_path):
                count_skipped += 1
                continue

            try:
                image = Image.open(image_path)
                mask = Image.open(mask_path)

                concat_img = concat_image_and_mask(image, mask)
                img_bytes = image_to_png_bytes(concat_img)

                with open(question_path, "r", encoding="utf-8") as f:
                    question = f.read().strip()

                key = f"sample-{count_written:06d}"
                sample = {
                    "__key__": key,
                    "png": img_bytes,
                    "txt": question,
                }
                sink.write(sample)

                # Local sample saving: only keep MAX_SAMPLES_TO_SAVE
                if MAX_SAMPLES_TO_SAVE > 0:
                    if save_random:
                        # buffer for later random pick (cap buffer to avoid RAM blow-up)
                        # Here we just buffer a bit more than needed.
                        if len(pending_save) < MAX_SAMPLES_TO_SAVE * 5:
                            pending_save.append((key, concat_img.copy(), question))
                    else:
                        if count_written < MAX_SAMPLES_TO_SAVE:
                            concat_img.save(os.path.join(samples_dir, f"{key}.png"))
                            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                                f.write(question + "\n")

                count_written += 1

            except Exception as e:
                count_failed += 1
                print(f"[error] {dataset_name}/{split} failed for {mask_path}: {e}")

    # If random saving is enabled, dump random 10 from buffered successful samples
    if MAX_SAMPLES_TO_SAVE > 0 and save_random and len(pending_save) > 0:
        k = min(MAX_SAMPLES_TO_SAVE, len(pending_save))
        chosen = random.sample(pending_save, k)
        for key, concat_img, question in chosen:
            concat_img.save(os.path.join(samples_dir, f"{key}.png"))
            with open(os.path.join(samples_dir, f"{key}.txt"), "w", encoding="utf-8") as f:
                f.write(question + "\n")

    print(f"[{dataset_name}] {split} done: wrote={count_written} skipped={count_skipped} failed={count_failed}")


def main():
    dataset_dirs = sorted(
        os.path.join(DATASET_ROOT, d)
        for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
    )

    if not dataset_dirs:
        print(f"No dataset dirs found under: {DATASET_ROOT}")
        return

    for ds_dir in dataset_dirs:
        dataset_name = os.path.basename(ds_dir)

        # Only process datasets that look like your structure (has at least one *_mask folder)
        has_any_mask = any(os.path.isdir(os.path.join(ds_dir, f"{s}_mask")) for s in SPLITS)
        if not has_any_mask:
            continue

        for split in SPLITS:
            build_wds_for_one_split(dataset_name, ds_dir, split)


if __name__ == "__main__":
    main()
