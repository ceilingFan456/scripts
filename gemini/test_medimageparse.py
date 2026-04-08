import base64
import json
import os
import re
from io import BytesIO

import requests
from PIL import Image

# ----------------------------
# Config
# ----------------------------
ENDPOINT_URL = "https://e0271228-6805-cghry.eastus2.inference.ml.azure.com/score"
API_KEY = os.getenv("MEDIMAGEPARSE_API_KEY")

IMAGE_PATH = "/home/t-qimhuang/code/grounded_medical_reasoning/datasets/MIMIC-CXR-decoded/p10/p10000764/s57375967/096052b7-d256dc40-453a102b-fa7d01c6-1b22c6b4.png"
PROMPT = "Focal consolidation"

if not API_KEY:
    raise SystemExit("Missing MEDIMAGEPARSE_API_KEY environment variable.")

# ----------------------------
# Encode image as base64
# ----------------------------
def encode_image(path):
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sanitize_filename(text):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("._") or "output"


def extract_image_payload(result):
    if isinstance(result, list) and result:
        result = result[0]

    if not isinstance(result, dict):
        raise SystemExit(f"Unexpected response type: {type(result).__name__}")

    candidates = [result]
    if isinstance(result.get("response"), dict):
        candidates.append(result["response"])
    if isinstance(result.get("result"), dict):
        candidates.append(result["result"])

    for candidate in candidates:
        image_payload = candidate.get("image")
        if isinstance(image_payload, dict) and "data" in image_payload:
            return image_payload, candidate.get("text_features")

        image_features = candidate.get("image_features")
        if isinstance(image_features, str):
            try:
                image_features = json.loads(image_features)
            except json.JSONDecodeError:
                image_features = None
        if isinstance(image_features, dict) and "data" in image_features:
            return image_features, candidate.get("text_features")

    raise SystemExit(f"Could not find an image payload in response:\n{json.dumps(result, indent=2)[:2000]}")


def save_image_payload(image_payload, output_path):
    raw_bytes = base64.b64decode(image_payload["data"])
    shape = image_payload.get("shape")
    dtype = image_payload.get("dtype")

    if not shape:
        raise SystemExit(f"Missing shape in image payload: {json.dumps(image_payload, indent=2)[:1000]}")
    if dtype != "uint8":
        raise SystemExit(f"Unsupported dtype {dtype!r}; expected 'uint8'.")

    if len(shape) == 3 and shape[0] == 1:
        _, height, width = shape
        mode = "L"
    elif len(shape) == 2:
        height, width = shape
        mode = "L"
    elif len(shape) == 3 and shape[2] == 3:
        height, width, _ = shape
        mode = "RGB"
    else:
        raise SystemExit(f"Unsupported image shape: {shape}")

    image = Image.frombytes(mode, (width, height), raw_bytes)
    image.save(output_path)
    return mode, (width, height)


def save_red_overlay(original_image_path, mask_image_path, output_path, alpha=0.35):
    base = Image.open(original_image_path).convert("RGBA")
    mask = Image.open(mask_image_path).convert("L")

    if mask.size != base.size:
        mask = mask.resize(base.size)

    # Colorize the mask as red and use mask intensity to control transparency.
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha_mask = mask.point(lambda px: int(px * alpha))
    overlay.putalpha(alpha_mask)

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    composited.save(output_path)

image_b64 = encode_image(IMAGE_PATH)

# ----------------------------
# Request payload
# ----------------------------
payload = {
    "input_data": {
        "columns": ["image", "text"],
        "index": [0],
        "data": [[image_b64, PROMPT]],
    },
    "params": {},
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# ----------------------------
# Call endpoint
# ----------------------------
response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    json=payload,
    timeout=60
)

try:
    result = response.json()
except ValueError:
    response.raise_for_status()
    raise SystemExit(f"Endpoint returned non-JSON response: {response.text[:1000]}")

if not response.ok:
    raise SystemExit(
        f"Endpoint request failed with status {response.status_code}:\n"
        f"{json.dumps(result, indent=2)}"
    )

image_payload, text_features = extract_image_payload(result)
input_stem = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
prompt_stem = sanitize_filename(PROMPT)
output_path = os.path.join(os.getcwd(), f"{input_stem}_{prompt_stem}_mask.png")
overlay_path = os.path.join(os.getcwd(), f"{input_stem}_{prompt_stem}_overlay_red.png")
mode, size = save_image_payload(image_payload, output_path)
save_red_overlay(IMAGE_PATH, output_path, overlay_path)

summary = {
    "saved_image": output_path,
    "saved_overlay": overlay_path,
    "mode": mode,
    "size": list(size),
    "shape": image_payload.get("shape"),
    "dtype": image_payload.get("dtype"),
    "text_features": text_features,
}
print(json.dumps(summary, indent=2))
