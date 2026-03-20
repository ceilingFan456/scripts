#!/usr/bin/env python3
import os
import glob
import io
import random

from PIL import Image
import webdataset as wds


# -------- CONFIG --------
# IMAGE_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData/ACDC/train"
# MASK_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData/ACDC/train_mask"
IMAGE_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData/amos22/CT/train"
MASK_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData/amos22/CT/train_mask"

# where to save the webdataset shards
OUTPUT_DIR = "/home/t-qimhuang/disk/datasets/BiomedParseData/amos22/CT/wds_train_simple_caption"
OUTPUT_PATTERN = os.path.join(OUTPUT_DIR, "amos22_seg-%06d.tar")

# where to save visual inspection copies
SAMPLES_DIR = "/home/t-qimhuang/disk/datasets/BiomedParseData/amos22/CT/wds_train_simple_caption_samples"
MAX_SAMPLES = 300  # number of samples you want in this dataset


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_image_path_from_mask(mask_path: str) -> str:
    """
    mask: patient001_frame01_1_MRI_heart_left+heart+ventricle.png
    img:  patient001_frame01_1_MRI_heart.png (in IMAGE_ROOT)
    """
    basename = os.path.basename(mask_path)
    stem, _ = os.path.splitext(basename)

    # split once from right: [base_without_organ, organ_part]
    try:
        image_stem, _ = stem.rsplit("_", 1)
    except ValueError:
        image_stem = stem

    image_filename = image_stem + ".png"
    image_path = os.path.join(IMAGE_ROOT, image_filename)
    return image_path


def concat_image_and_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Horizontally concatenate image and mask."""
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


def build_webdataset():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SAMPLES_DIR)

    # collect all mask paths
    mask_paths = sorted(glob.glob(os.path.join(MASK_ROOT, "*.png")))
    print(f"Found {len(mask_paths)} mask images.")

    if len(mask_paths) == 0:
        print("No mask images found, exiting.")
        return

    # random.shuffle(mask_paths)
    mask_paths = mask_paths[:MAX_SAMPLES]
    print(f"Using {len(mask_paths)} images to build WebDataset.")

    count = 0
    with wds.ShardWriter(OUTPUT_PATTERN, maxcount=MAX_SAMPLES) as sink:
        for mask_path in mask_paths:
            image_path = get_image_path_from_mask(mask_path)
            question_path = os.path.splitext(mask_path)[0] + ".txt"
            question_path = question_path.replace("mask", "caption")

            print(f"Processing mask: {mask_path}")
            print(f"  image: {image_path}")
            print(f"  question: {question_path}")

            if not os.path.exists(image_path):
                print(f"[skip] image not found for mask: {mask_path}")
                continue
            if not os.path.exists(question_path):
                print(f"[skip] question missing: {mask_path}")
                continue

            try:
                # load image & mask
                image = Image.open(image_path)
                mask = Image.open(mask_path)

                # concatenate
                concat_img = concat_image_and_mask(image, mask)
                img_bytes = image_to_png_bytes(concat_img)

                # load question text
                with open(question_path, "r", encoding="utf-8") as f:
                    question = f.read().strip()

                # Write to WDS
                key = f"sample-{count:06d}"
                sample = {
                    "__key__": key,
                    "png": img_bytes,
                    "txt": question,
                }
                sink.write(sample)

                # ---- ALSO SAVE LOCALLY FOR VISUAL CHECK ----
                # save image
                save_img_path = os.path.join(SAMPLES_DIR, f"{key}.png")
                concat_img.save(save_img_path)

                # save txt
                save_txt_path = os.path.join(SAMPLES_DIR, f"{key}.txt")
                with open(save_txt_path, "w", encoding="utf-8") as f:
                    f.write(question + "\n")

                # log
                print(f"[{count}] wrote sample + saved local copy")

                count += 1
                if count >= MAX_SAMPLES:
                    break

            except Exception as e:
                print(f"[error] failed for {mask_path}: {e}")

    print(f"\nDone. Wrote {count} samples.")
    print(f"WebDataset saved under: {OUTPUT_DIR}")
    print(f"Local samples saved under: {SAMPLES_DIR}")


if __name__ == "__main__":
    build_webdataset()

