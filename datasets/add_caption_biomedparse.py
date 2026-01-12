#!/usr/bin/env python3
import os
import glob
import asyncio
from typing import List

from tenacity import (
    retry,
    wait_fixed,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from openai import (
    AzureOpenAI,
    AsyncAzureOpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from tqdm import tqdm


# ---------- Azure OpenAI wrapper (MINIMALLY CHANGED) ----------

def before_retry_fn(retry_state):
    # keep your log style
    print(f"[Retry] Attempt {retry_state.attempt_number} failed with: {retry_state.outcome}")


class azure_openai_llm:
    def __init__(self, model: str = None):
        endpoint = "https://medevalkit.openai.azure.com/"
        self.deployment = "gpt-5-mini"

        subscription_key = os.environ["AZURE_API_KEY"]
        api_version = "2024-12-01-preview"

        # small practical timeout; reusing same client is good
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            timeout=60.0,
        )

        # kept for compatibility AND now used by async path
        self.async_client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
            timeout=60.0,
        )

    # (your sync path preserved)
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    def response(self, messages, **kwargs) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            max_completion_tokens=128,  # reduced (questions are short)
            model=self.deployment,
        )
        return response.choices[0].message.content

    def generate_output(self, messages, **kwargs) -> str:
        try:
            return self.response(messages, **kwargs)
        except Exception as e:
            print(f"LLM failed: {e}")
            return None

    # (new) async path for parallelization
    @retry(
        wait=wait_exponential(min=1, max=30),
        stop=stop_after_attempt(20),
        before=before_retry_fn,
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
    )
    async def aresponse(self, messages, **kwargs) -> str:
        response = await self.async_client.chat.completions.create(
            messages=messages,
            max_completion_tokens=128,  # reduced
            model=self.deployment,
        )
        return response.choices[0].message.content

    async def agenerate_output(self, messages, **kwargs) -> str:
        try:
            return await self.aresponse(messages, **kwargs)
        except Exception as e:
            print(f"LLM failed: {e}")
            return None


# ---------- Core logic (UNCHANGED) ----------

def extract_organ_phrase_from_filename(mask_filename: str) -> str:
    basename = os.path.basename(mask_filename)
    stem, _ = os.path.splitext(basename)
    try:
        _, organ_part = stem.rsplit("_", 1)
    except ValueError:
        organ_part = stem
    return organ_part.replace("+", " ")


def build_messages_for_segmentation_question(organ_phrase: str) -> List[dict]:
    system_msg = (
        "You are a helpful assistant generating segmentation questions for a medical "
        "image segmentation task."
    )

    user_msg = (
        "Given a single cardiac MRI slice, write ONE short and clear and unambiguous question asking "
        f"the model to segment the '{organ_phrase}'.\n\n"
        "Requirements:\n"
        "- Only output the question text.\n"
        "- Do not include any explanation or extra sentences."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def mask_to_caption_txt_path(mask_path: str) -> str:
    txt_path = os.path.splitext(mask_path)[0] + ".txt"
    parts = txt_path.split(os.sep)
    for i, p in enumerate(parts):
        if p.endswith("_mask"):
            parts[i] = p.replace("_mask", "_caption")
            break
    return os.sep.join(parts)


# ---------- Parallel mask processing (NEW, bounded concurrency) ----------

async def generate_questions_for_masks_parallel(mask_root: str, llm: azure_openai_llm, concurrency: int = 16):
    mask_paths = sorted(glob.glob(os.path.join(mask_root, "*.png")))
    if not mask_paths:
        return

    print(f"Found {len(mask_paths)} masks in {mask_root}")

    sem = asyncio.Semaphore(concurrency)

    async def worker(mask_path: str):
        out_txt = mask_to_caption_txt_path(mask_path)

        if os.path.exists(out_txt):
            return

        organ_phrase = extract_organ_phrase_from_filename(mask_path)
        messages = build_messages_for_segmentation_question(organ_phrase)

        async with sem:
            question = await llm.agenerate_output(messages)

        if question is None:
            return

        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(question.strip() + "\n")

    tasks = [asyncio.create_task(worker(p)) for p in mask_paths]

    # progress bar that advances as tasks complete (not in submission order)
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=os.path.basename(mask_root)):
        await fut


# ---------- Dataset-level driver (SEQUENTIAL across datasets/dirs) ----------

def find_mask_dirs(dataset_dir: str):
    res = []
    for split in ["train_mask", "test_mask"]:
        p = os.path.join(dataset_dir, split)
        if os.path.isdir(p):
            res.append(p)

    # handle duplicated nesting: ACDC/ACDC/train_mask
    base = os.path.basename(os.path.normpath(dataset_dir))
    nested = os.path.join(dataset_dir, base)
    if os.path.isdir(nested):
        for split in ["train_mask", "test_mask"]:
            p = os.path.join(nested, split)
            if os.path.isdir(p):
                res.append(p)

    return res


def run_all_datasets(dataset_root: str, concurrency: int = 16):
    llm = azure_openai_llm()

    dataset_dirs = sorted(
        os.path.join(dataset_root, d)
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )

    for ds in dataset_dirs:
        mask_dirs = find_mask_dirs(ds)
        if not mask_dirs:
            continue

        print(f"\n=== Dataset: {os.path.basename(ds)} ===")
        for mask_root in mask_dirs:
            # parallel within each mask_root, but keep dataset traversal sequential
            asyncio.run(generate_questions_for_masks_parallel(mask_root, llm, concurrency=concurrency))


# ---------- Entrypoint ----------

if __name__ == "__main__":
    DATASET_ROOT = "/home/t-qimhuang/disk/datasets/BiomedParseData"  # adjust if needed
    CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "32"))       # tune: 8/16/32
    run_all_datasets(DATASET_ROOT, concurrency=CONCURRENCY)
