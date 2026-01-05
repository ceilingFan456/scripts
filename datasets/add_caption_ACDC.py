#!/usr/bin/env python3
import os
import glob
import asyncio
from typing import List

from tenacity import retry, wait_fixed, stop_after_attempt
from openai import AzureOpenAI, AsyncAzureOpenAI


# ---------- Azure OpenAI wrapper (based on your snippet) ----------

def before_retry_fn(retry_state):
    print(f"[Retry] Attempt {retry_state.attempt_number} failed with: {retry_state.outcome}")


class azure_openai_llm:
    def __init__(self, model: str = None):
        endpoint = "https://medevalkit.openai.azure.com/"
        model_name = "gpt-5-mini"  # not actually used below, but kept for reference
        self.deployment = "gpt-5-mini"

        subscription_key = os.environ["AZURE_API_KEY"]
        api_version = "2024-12-01-preview"

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

        self.async_client = AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    def response(self, messages, **kwargs) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=self.deployment,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    @retry(wait=wait_fixed(10), stop=stop_after_attempt(1000), before=before_retry_fn)
    async def response_async(self, messages, **kwargs) -> str:
        response = await self.async_client.chat.completions.create(
            messages=messages,
            max_completion_tokens=16384,
            model=self.deployment,
        )
        return response.choices[0].message.content

    def generate_output(self, messages, **kwargs) -> str:
        try:
            response = self.response(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.deployment)} response failed: {e}")
        return response

    async def generate_output_async(self, idx, messages, **kwargs):
        try:
            response = await self.response_async(messages, **kwargs)
        except Exception as e:
            response = None
            print(f"get {kwargs.get('model', self.deployment)} response failed: {e}")
        return idx, response


# ---------- Core logic for generating questions ----------

def extract_organ_phrase_from_filename(mask_filename: str) -> str:
    """
    Given a mask filename like:
        patient001_frame01_1_MRI_heart_left+heart+ventricle.png
    return an organ phrase like:
        'left heart ventricle'
    """
    basename = os.path.basename(mask_filename)
    stem, _ = os.path.splitext(basename)

    # Split once from the right: [image_stem, organ_part]
    try:
        image_stem, organ_part = stem.rsplit("_", 1)
    except ValueError:
        # If pattern doesn't match, fall back to using the whole stem
        organ_part = stem

    # Replace '+' with spaces to form phrase
    organ_phrase = organ_part.replace("+", " ")
    return organ_phrase


def build_messages_for_segmentation_question(organ_phrase: str) -> List[dict]:
    """
    Build messages for the chat completion to get ONE short segmentation question.
    """
    system_msg = (
        "You are a helpful assistant generating segmentation questions for a medical "
        "image segmentation task. The output will be used as the natural language "
        "instruction for an AI model that segments specific anatomical structures "
        "on cardiac MRI images."
    )

    user_msg = (
        "Given a single cardiac MRI slice, write ONE short and clear and unambiguous question asking "
        f"the model to segment the '{organ_phrase}'.\n\n"
        "Requirements:\n"
        "- Only output the question text.\n"
        "- Do not include any explanation, notes, or extra sentences.\n"
        "- The question should be imperative or interrogative, e.g., "
        "\"Please segment the left ventricular myocardium.\" or "
        "\"Which pixels correspond to the left heart ventricle?\""
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return messages


def generate_questions_for_masks(mask_root: str):
    """
    For each mask PNG under mask_root, call Azure OpenAI to generate a segmentation
    question and save it as a .txt file next to the mask.

    Example:
        mask: /.../train_mask/patient001_frame01_1_MRI_heart_left+heart+ventricle.png
        txt:  /.../train_mask/patient001_frame01_1_MRI_heart_left+heart+ventricle.txt
    """
    llm = azure_openai_llm()

    caption_root = mask_root.replace("mask", "caption")
    os.makedirs(caption_root, exist_ok=True)

    mask_pattern = os.path.join(mask_root, "*.png")
    mask_paths = sorted(glob.glob(mask_pattern))

    print(f"Found {len(mask_paths)} mask files under {mask_root}")

    for idx, mask_path in enumerate(mask_paths, start=1):
        organ_phrase = extract_organ_phrase_from_filename(mask_path)
        question_txt_path = os.path.splitext(mask_path)[0] + ".txt"

        ## need to swap train_mask to train_caption
        question_txt_path = question_txt_path.replace("mask", "caption")

        # Skip if already exists
        if os.path.exists(question_txt_path):
            print(f"[{idx}/{len(mask_paths)}] Skip existing: {question_txt_path}")
            continue

        print(f"[{idx}/{len(mask_paths)}] Processing: {mask_path}")
        print(f"  -> organ phrase: {organ_phrase}")

        messages = build_messages_for_segmentation_question(organ_phrase)
        question = llm.generate_output(messages)

        if question is None:
            print(f"  !! Failed to generate question for {mask_path}")
            continue

        # Clean up whitespace
        question = question.strip()

        # Save to txt
        with open(question_txt_path, "w", encoding="utf-8") as f:
            f.write(question + "\n")

        print(f"  -> saved question to: {question_txt_path}")


if __name__ == "__main__":
    # Hardcode your ACDC mask folder here.
    # You can also turn this into argparse if you want more flexibility.
    MASK_ROOT = "/home/t-qimhuang/disk/datasets/ACDC/train_mask"

    generate_questions_for_masks(MASK_ROOT)
