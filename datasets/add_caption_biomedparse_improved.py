import os
import glob
import asyncio
import time
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from openai import AsyncAzureOpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------- Configuration ----------

AZURE_ENDPOINT = "https://medevalkit.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-5-mini"
API_VERSION = "2024-12-01-preview"

DATASET_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData"

CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "64"))  # Adjust if you hit throttling
MAX_CAPTION_WORDS = int(os.environ.get("MAX_CAPTION_WORDS", "30"))
FORCE_REGEN = os.environ.get("FORCE_REGEN", "0") == "1"  # set to 1 to overwrite all captions

# ---------- Helper Functions ----------

def extract_metadata_from_filename(mask_path: str) -> Tuple[str, str, str, str | None]:
    """
    Heuristic parser for filenames like:
      patient101_frame01_0_MRI_heart_left+heart+ventricle.png
    """
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    parts = stem.split("_")

    target = parts[-1].replace("+", " ") if len(parts) >= 1 else "structure"
    site = parts[-2] if len(parts) >= 2 else "region"
    raw_mod = parts[-3] if len(parts) >= 3 else "imaging"

    modality = raw_mod
    sequence = None
    if any(m in raw_mod.upper() for m in ["T1", "T2", "FLAIR", "ADC"]):
        modality = "MRI"
        sequence = raw_mod

    return target, modality, site, sequence


def mask_to_caption_path(mask_path: str) -> str:
    """
    Converts:
      .../train_mask/x.png -> .../train_caption/x.txt
      .../test_mask/x.png  -> .../test_caption/x.txt
    """
    txt_path = os.path.splitext(mask_path)[0] + ".txt"
    return txt_path.replace("_mask", "_caption")


def count_words(text: str) -> int:
    # Simple word count: split on whitespace
    return len(text.strip().split())


def caption_needs_regen(caption_path: str) -> Tuple[bool, str]:
    """
    Returns (needs_regen, reason).
    Reasons: missing, empty, too_long, force_regen
    """
    if FORCE_REGEN:
        return True, "force_regen"

    if not os.path.exists(caption_path):
        return True, "missing"

    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        # If it can't be read, regenerate
        return True, "unreadable"

    if not content:
        return True, "empty"

    if count_words(content) > MAX_CAPTION_WORDS:
        return True, "too_long"

    return False, ""


def build_messages_for_segmentation_question(mask_path: str) -> List[dict]:
    target, modality, site, sequence = extract_metadata_from_filename(mask_path)
    mod_desc = f"{sequence if sequence else ''} {modality}".strip()

    patterns = [
        f"Write a question asking to identify the {target} in this {mod_desc} scan.",
        f"Write a command to delineate the {target} on this {site} {modality} slice.",
        f"Ask the model to find the pixels corresponding to the {target}.",
        f"Ask where the {target} is located in this {mod_desc} image.",
        f"Formulate a question to identify the {target} pixels within this {site} scan.",
        f"Ask which part of this {mod_desc} corresponds to the {target}.",
        f"Request a mask for the {target} area on this {mod_desc}.",
        f"Direct the model to isolate the {target} from the surrounding {site} tissue.",
        f"Request a segmentation of the {target} within this {site} region.",
    ]
    selected_pattern = random.choice(patterns)

    system_msg = (
        "You are a medical AI specialist. Generate diverse, clear, unambiguous, "
        "and SHORT instructions for anatomical segmentation tasks."
    )

    user_msg = (
        f"Target Structure: {target}\n"
        f"Imaging Context: {mod_desc} of the {site}\n\n"
        f"Task: {selected_pattern}\n\n"
        "Requirements:\n"
        "- Only output the question/instruction text.\n"
        "- Use varied phrasing (don't always start with 'Please').\n"
        "- Professional and direct tone.\n"
        f"- Keep it concise (<= {MAX_CAPTION_WORDS} words).\n"
        f"- If needed, rewrite to be <= {MAX_CAPTION_WORDS} words.\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------- Async Azure Client with Retries ----------

class AsyncLLM:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=AZURE_ENDPOINT,
            api_version=API_VERSION,
            timeout=30.0,
        )

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
    )
    async def get_caption(self, mask_path: str) -> str:
        response = await self.client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=build_messages_for_segmentation_question(mask_path),
        )
        text = (response.choices[0].message.content or "").strip()
        return text

    async def close(self):
        await self.client.close()


# ---------- Dataset Discovery ----------

def discover_datasets(root: str) -> List[Dict[str, str]]:
    """
    Recursively search for directories that contain:
      - train_mask + train
      - test_mask  + test

    Returns list of entries like:
      {"split": "train", "mask_dir": ".../train_mask", "img_dir": ".../train"}
    """
    entries: List[Dict[str, str]] = []

    for current_root, dirs, _files in os.walk(root):
        dirset = set(dirs)
        # We only care when we see *_mask here; then check sibling *_img folder exists
        if "train_mask" in dirset:
            mask_dir = os.path.join(current_root, "train_mask")
            img_dir = os.path.join(current_root, "train")
            if os.path.isdir(img_dir):
                entries.append({"split": "train", "mask_dir": mask_dir, "img_dir": img_dir})
            else:
                print(f"[WARN] Found train_mask but missing train folder: {current_root}")

        if "test_mask" in dirset:
            mask_dir = os.path.join(current_root, "test_mask")
            img_dir = os.path.join(current_root, "test")
            if os.path.isdir(img_dir):
                entries.append({"split": "test", "mask_dir": mask_dir, "img_dir": img_dir})
            else:
                print(f"[WARN] Found test_mask but missing test folder: {current_root}")

    # Stable ordering for nicer logs
    entries.sort(key=lambda x: (x["mask_dir"]))
    return entries


# ---------- Processing Pipeline ----------

async def process_mask(mask_path: str, llm: AsyncLLM, semaphore: asyncio.Semaphore):
    out_txt = mask_to_caption_path(mask_path)

    needs, reason = caption_needs_regen(out_txt)
    if not needs:
        return

    async with semaphore:
        try:
            caption = await llm.get_caption(mask_path)
            caption = caption.strip()

            # Safety: enforce <= MAX_CAPTION_WORDS and handle empty captions
            # Retry up to 3 times if caption is empty or too long
            attempts = 0
            while (not caption or count_words(caption) > MAX_CAPTION_WORDS) and attempts < 3:
                if caption:
                    print(f"[Info] Caption too long ({count_words(caption)} words), retrying... attempt: {attempts + 1}")
                else:
                    print(f"[Info] Caption empty, retrying... attempt: {attempts + 1}")
                caption = (await llm.get_caption(mask_path)).strip()
                attempts += 1
            
            # If still invalid after retries, use fallback caption
            if not caption or count_words(caption) > MAX_CAPTION_WORDS:
                target, modality, site, sequence = extract_metadata_from_filename(mask_path)
                caption = f"Segment {target} in {site} {modality}".strip()
                reason_str = "empty" if not caption else f"too long ({count_words(caption)} words)"
                print(f"[Info] Using fallback caption ({reason_str}): {caption}")

            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(caption + "\n")

        except Exception as e:
            print(f"\n[Error] Failed processing {mask_path} ({reason}): {e}")


async def process_folder(mask_dir: str, llm: AsyncLLM, sem: asyncio.Semaphore):
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        return

    tasks = [process_mask(p, llm, sem) for p in mask_paths]

    desc = f"{os.path.basename(os.path.dirname(mask_dir))}/{os.path.basename(mask_dir)}"
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Folder: {desc}", leave=False):
        await fut


async def main():
    if "AZURE_API_KEY" not in os.environ:
        print("Error: Please set the AZURE_API_KEY environment variable.")
        return

    start_time = time.perf_counter()
    llm = AsyncLLM()
    sem = asyncio.Semaphore(CONCURRENCY)

    # 1) Discover dataset splits recursively (train_mask+train, test_mask+test)
    entries = discover_datasets(DATASET_ROOT)
    print(f"Discovered {len(entries)} (split, mask_dir) entries under: {DATASET_ROOT}")

    # 2) Process folders sequentially (cleaner logs)
    for ent in entries:
        mask_dir = ent["mask_dir"]
        print(f"\n>>> Processing split={ent['split']}  mask_dir={mask_dir}")
        await process_folder(mask_dir, llm, sem)

    await llm.close()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*30}\nAll datasets complete in {elapsed:.2f}s.\n{'='*30}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
