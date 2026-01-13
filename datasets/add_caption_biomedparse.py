import os
import glob
import asyncio
import time
from typing import List
from tqdm import tqdm
from openai import AsyncAzureOpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------- Configuration ----------

AZURE_ENDPOINT = "https://medevalkit.openai.azure.com/"
DEPLOYMENT_NAME = "gpt-5-mini"
API_VERSION = "2024-12-01-preview"
DATASET_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData"
CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "128"))

# ---------- Logic Functions ----------

def extract_organ_phrase(mask_filename: str) -> str:
    """Extracts organ name from filename, e.g., 'slice_001_heart.png' -> 'heart'"""
    stem = os.path.splitext(os.path.basename(mask_filename))[0]
    organ_part = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    return organ_part.replace("+", " ")

def mask_to_caption_path(mask_path: str) -> str:
    """Converts .../train_mask/x.png to .../train_caption/x.txt"""
    txt_path = os.path.splitext(mask_path)[0] + ".txt"
    return txt_path.replace("_mask", "_caption")

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
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)),
    )
    async def get_caption(self, organ_phrase: str) -> str:
        response = await self.client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "Be concise. Output ONLY the question."},
                {"role": "user", "content": f"Write one short question to segment the '{organ_phrase}' in this MRI slice."}
            ],
        )
        return response.choices[0].message.content.strip()

    async def close(self):
        await self.client.close()

# ---------- Processing Pipeline ----------

async def process_mask(mask_path: str, llm: AsyncLLM, semaphore: asyncio.Semaphore):
    out_txt = mask_to_caption_path(mask_path)
    if os.path.exists(out_txt):
        return

    organ_phrase = extract_organ_phrase(mask_path)
    
    async with semaphore:
        try:
            caption = await llm.get_caption(organ_phrase)
            if caption:
                os.makedirs(os.path.dirname(out_txt), exist_ok=True)
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
        except Exception as e:
            print(f"\nFailed {mask_path}: {e}")

async def process_folder(mask_dir: str, llm: AsyncLLM, sem: asyncio.Semaphore):
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        return

    tasks = [process_mask(p, llm, sem) for p in mask_paths]
    
    # tqdm progress bar for the current folder
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {os.path.basename(mask_dir)}", leave=False):
        await f

async def main():
    if "AZURE_API_KEY" not in os.environ:
        print("Error: Set AZURE_API_KEY env var.")
        return

    llm = AsyncLLM()
    sem = asyncio.Semaphore(CONCURRENCY)

    # 1. Find all target directories
    target_dirs = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for d in dirs:
            if d in ["train_mask", "test_mask"]:
                target_dirs.append(os.path.join(root, d))

    print(f"Found {len(target_dirs)} mask folders to process.")

    # 2. Process folders one by one, but images inside folder in parallel
    for mask_dir in target_dirs:
        print(f"\nTarget: {mask_dir}")
        await process_folder(mask_dir, llm, sem)

    await llm.close()
    print("\nAll datasets complete.")

if __name__ == "__main__":
    asyncio.run(main())