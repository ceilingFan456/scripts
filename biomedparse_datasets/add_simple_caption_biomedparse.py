#!/usr/bin/env python3
"""
simple_caption.py

Generate simple deterministic captions for BiomedParseData masks.

Caption format (like your screenshot):
    "<Target> in <site> <modality>"

Examples:
    "Neoplastic cells in liver pathology"
    "Enhancing tumor in brain MRI"

It writes:
    .../train_mask/<stem>.png  -> .../train_caption/<stem>.txt
    .../test_mask/<stem>.png   -> .../test_caption/<stem>.txt
"""

import os
import re
import glob
import argparse
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


# ---------------------- Parsing / Formatting ----------------------

SEQ_KEYWORDS = {"T1", "T2", "FLAIR", "ADC", "DWI", "SWI", "PD", "TOF", "MRA"}

def extract_metadata_from_filename(mask_path: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Expected-ish pattern:
      patient101_frame01_0_MRI_heart_left+heart+ventricle.png
    We'll take the last 3 underscore-separated fields as:
      raw_mod = parts[-3], site = parts[-2], target = parts[-1]
    Then:
      - target: replace '+' and '_' with spaces
      - site:   replace '+' and '_' with spaces
      - modality:
          * if raw_mod contains a known MR sequence keyword -> modality="MRI", sequence=raw_mod
          * else modality=raw_mod, sequence=None
    """
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    parts = stem.split("_")

    raw_mod = parts[-3] if len(parts) >= 3 else "imaging"
    site = parts[-2] if len(parts) >= 2 else "region"
    target = parts[-1] if len(parts) >= 1 else "structure"

    def norm(s: str) -> str:
        s = s.replace("+", " ")
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    raw_mod_norm = norm(raw_mod)
    site_norm = norm(site)
    target_norm = norm(target)

    modality = raw_mod_norm
    sequence = None

    # If raw_mod looks like an MRI sequence label, standardize to "MRI"
    upper = raw_mod_norm.upper().replace(" ", "")
    if any(k in upper for k in SEQ_KEYWORDS):
        modality = "MRI"
        sequence = raw_mod_norm

    return target_norm, modality, site_norm, sequence


def mask_to_caption_path(mask_path: str) -> str:
    """
    .../train_mask/xxx.png -> .../train_simple_caption/xxx.txt
    .../test_mask/xxx.png  -> .../test_simple_caption/xxx.txt
    """
    mask_path = os.path.abspath(mask_path)
    parent = os.path.dirname(mask_path)

    if parent.endswith("train_mask"):
        out_root = parent.replace("train_mask", "train_simple_caption")
    elif parent.endswith("test_mask"):
        out_root = parent.replace("test_mask", "test_simple_caption")
    else:
        raise ValueError(f"Unexpected mask directory: {parent}")

    os.makedirs(out_root, exist_ok=True)
    fname = os.path.splitext(os.path.basename(mask_path))[0] + ".txt"
    return os.path.join(out_root, fname)



def to_title_like(s: str) -> str:
    """
    Your screenshot uses capitalized first letter, not Title Case for every word.
    We'll do: first character uppercase, keep rest as-is.
    """
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def build_simple_caption(mask_path: str) -> str:
    target, modality, site, sequence = extract_metadata_from_filename(mask_path)

    # Keep it close to screenshot: "<Target> in <site> <modality>"
    # Example: "Enhancing tumor in brain MRI"
    caption = f"{to_title_like(target)} in {site} {modality}".strip()
    caption = re.sub(r"\s+", " ", caption)
    return caption


# ---------------------- IO / Processing ----------------------

def process_one_mask(mask_path: str, overwrite: bool = False) -> bool:
    """
    Returns True if wrote a caption, False if skipped/existed.
    """
    out_txt = mask_to_caption_path(mask_path)
    if (not overwrite) and os.path.exists(out_txt):
        return False

    caption = build_simple_caption(mask_path)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    return True


def find_mask_dirs(dataset_root: str) -> List[str]:
    target_dirs = []
    for root, dirs, _files in os.walk(dataset_root):
        for d in dirs:
            if d in ["train_mask", "test_mask"]:
                target_dirs.append(os.path.join(root, d))
    target_dirs.sort()
    return target_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="/home/t-qimhuang/disk/datasets/BiomedParseData")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("CAPTION_WORKERS", "64")))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit_per_folder", type=int, default=0, help="0 = no limit (debug only)")
    args = ap.parse_args()

    mask_dirs = find_mask_dirs(args.dataset_root)
    print(f"Found {len(mask_dirs)} mask folders under {args.dataset_root}")

    total_written = 0
    total_seen = 0

    for mask_dir in mask_dirs:
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if args.limit_per_folder and args.limit_per_folder > 0:
            mask_paths = mask_paths[: args.limit_per_folder]

        if not mask_paths:
            continue

        desc = f"{os.path.basename(os.path.dirname(mask_dir))}/{os.path.basename(mask_dir)}"
        total_seen += len(mask_paths)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_mask, p, args.overwrite) for p in mask_paths]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False):
                try:
                    wrote = fut.result()
                    total_written += int(bool(wrote))
                except Exception as e:
                    # keep going; print once per failure
                    print(f"\n[Error] {desc}: {e}")

        print(f">>> {mask_dir}: wrote {total_written} total so far")

    print("\n" + "=" * 40)
    print(f"Done. Seen: {total_seen} masks | Wrote: {total_written} captions")
    print("=" * 40)


if __name__ == "__main__":
    main()


## python simple_caption.py --dataset_root /home/t-qimhuang/disk/datasets/BiomedParseData --workers 128
