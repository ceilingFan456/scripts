import os
import glob
import asyncio
import time
import random
from typing import List
from tqdm import tqdm
from openai import AsyncAzureOpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------- Configuration ----------

AZURE_ENDPOINT = "https://medevalkit.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-5-mini"
API_VERSION = "2024-12-01-preview"
DATASET_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData"
CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "128")) # Adjusted for safety

# ---------- Logic Functions ----------

def extract_metadata_from_filename(mask_path: str):
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    parts = stem.split('_')
    
    # Mapping for: patient101_frame01_0_MRI_heart_left+heart+ventricle.png
    target = parts[-1].replace("+", " ") if len(parts) >= 1 else "structure"
    site = parts[-2] if len(parts) >= 2 else "region"
    raw_mod = parts[-3] if len(parts) >= 3 else "imaging"
    
    modality = raw_mod
    sequence = None
    if any(m in raw_mod.upper() for m in ['T1', 'T2', 'FLAIR', 'ADC']):
        modality = "MRI"
        sequence = raw_mod
        
    return target, modality, site, sequence

def mask_to_caption_path(mask_path: str) -> str:
    """Converts .../train_mask/x.png to .../train_caption/x.txt"""
    txt_path = os.path.splitext(mask_path)[0] + ".txt"
    return txt_path.replace("_mask", "_caption")

import random

def build_messages_for_segmentation_question(mask_path: str) -> List[dict]:
    # 1. Extract metadata
    target, modality, site, sequence = extract_metadata_from_filename(mask_path)
    mod_desc = f"{sequence if sequence else ''} {modality}".strip()

    # 2. Define 4 distinct "Start Patterns" to ensure variety
    # This prevents the "Please segment..." repetition
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
        "You are a medical AI specialist. You generate diverse, clear, and "
        "unambiguous and short instructions for cardiac and anatomical segmentation tasks."
    )

    user_msg = (
        f"Target Structure: {target}\n"
        f"Imaging Context: {mod_desc} of the {site}\n\n"
        f"Task: {selected_pattern}\n\n"
        "Requirements:\n"
        "- Only output the question/instruction text.\n"
        "- Use varied phrasing (don't always start with 'Please').\n"
        "- Ensure the tone is professional yet direct.\n"
        "- Keep it concise (under 30 words).\n"
        "- Examples of desired variety: 'Where is the {target}?', 'Highlight the {target} boundaries.', 'Show me the {target}.'"
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
            timeout=30.0
        )

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)),
    )
    async def get_caption(self, mask_path: str) -> str:
        response = await self.client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=build_messages_for_segmentation_question(mask_path),
        )
        return response.choices[0].message.content.strip()

    async def close(self):
        await self.client.close()

# ---------- Processing Pipeline ----------

async def process_mask(mask_path: str, llm: AsyncLLM, semaphore: asyncio.Semaphore):
    out_txt = mask_to_caption_path(mask_path)
    if os.path.exists(out_txt):
        return

    async with semaphore:
        try:
            caption = await llm.get_caption(mask_path)
            if caption:
                os.makedirs(os.path.dirname(out_txt), exist_ok=True)
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
        except Exception as e:
            print(f"\n[Error] Failed processing {mask_path}: {e}")

async def process_folder(mask_dir: str, llm: AsyncLLM, sem: asyncio.Semaphore):
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    ## TODO
    ## run it for all the masks in the folder with concurrency control
    # mask_paths = mask_paths[:100]  # Limit to first 100 for testing
    
    if not mask_paths:
        return

    tasks = [process_mask(p, llm, sem) for p in mask_paths]
    
    # Progress bar per folder
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Folder: {os.path.basename(os.path.dirname(mask_dir))}/{os.path.basename(mask_dir)}", leave=False):
        await f

async def main():
    if "AZURE_API_KEY" not in os.environ:
        print("Error: Please set the AZURE_API_KEY environment variable.")
        return

    start_time = time.perf_counter()
    llm = AsyncLLM()
    sem = asyncio.Semaphore(CONCURRENCY)

    # 1. Find all target directories recursively
    target_dirs = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for d in dirs:
            if d in ["train_mask", "test_mask"]:
                target_dirs.append(os.path.join(root, d))

    target_dirs.sort()
    print(f"Starting processing for {len(target_dirs)} folders...")

    # 2. Process folders sequentially (to keep logs/tqdm clean)
    for mask_dir in target_dirs:
        print(f"\n>>> Processing: {mask_dir}")
        await process_folder(mask_dir, llm, sem)

    await llm.close()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*30}\nAll datasets complete in {elapsed:.2f}s.\n{'='*30}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")