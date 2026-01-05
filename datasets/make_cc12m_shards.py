import os
import csv
import tarfile
import urllib.request
from PIL import Image
import io
from tqdm import tqdm

def download_image(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        return img
    except Exception as e:
        return None

def process_tsv_to_shards(
    tsv_path: str,
    output_dir: str,
    shard_size: int = 10000,
    image_folder_name: str = "images",
    caption_ext: str = ".txt",
    img_ext: str = ".jpg",
    max_images: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, image_folder_name)
    os.makedirs(image_dir, exist_ok=True)

    # Read the TSV
    entries = []
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            url, caption = row[0], row[1]
            entries.append((url, caption))
            if max_images and len(entries) >= max_images:
                break

    print(f"Total entries to process: {len(entries)}")

    shard_count = 0
    for shard_start in range(0, len(entries), shard_size):
        shard_end = min(shard_start + shard_size, len(entries))
        shard_entries = entries[shard_start:shard_end]

        shard_filename = os.path.join(
            output_dir,
            f"shard_{shard_count:05d}.tar"
        )
        print(f"Writing shard {shard_count} → {shard_filename} with entries {shard_start} … {shard_end-1}")

        with tarfile.open(shard_filename, "w") as tar:
            for idx, (url, caption) in enumerate(shard_entries):
                img = download_image(url)
                if img is None:
                    continue

                # image file inside tar
                img_name = f"{shard_count:05d}_{idx:05d}{img_ext}"
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                tarinfo = tarfile.TarInfo(name=img_name)
                tarinfo.size = img_bytes.getbuffer().nbytes
                tar.addfile(tarinfo, img_bytes)

                # caption file inside tar
                cap_name = f"{shard_count:05d}_{idx:05d}{caption_ext}"
                cap_bytes = caption.encode('utf-8')
                tarinfo2 = tarfile.TarInfo(name=cap_name)
                tarinfo2.size = len(cap_bytes)
                tar.addfile(tarinfo2, io.BytesIO(cap_bytes))

        shard_count += 1

    print(f"Completed writing {shard_count} shards to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-process CC12M TSV into tar shards")
    parser.add_argument("--tsv_path", type=str, required=True,
                        help="Path to the CC12M .tsv file (image_url \t caption)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory to write shards")
    parser.add_argument("--shard_size", type=int, default=10000,
                        help="Number of samples per shard")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max number of images to process (for subset)")
    args = parser.parse_args()

    process_tsv_to_shards(
        tsv_path=args.tsv_path,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_images=args.max_images
    )

