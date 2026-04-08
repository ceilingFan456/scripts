"""Quick smoke test: load Qwen3-VL-8B and print GPU memory usage."""

import torch


def main():
    print("=== GPU Info ===")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  ({props.total_mem / 1024**3:.1f} GB)")

    print("\n=== Loading Qwen3-VL-8B-Instruct ===")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print("Model loaded successfully!")

    print("\n=== GPU Memory Usage ===")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

    print("\n=== Model Device Map ===")
    if hasattr(model, "hf_device_map"):
        unique_devices = set(model.hf_device_map.values())
        print(f"  Devices used: {unique_devices}")
        print(f"  Total layers distributed: {len(model.hf_device_map)}")

    print("\nDone. Qwen3-VL-8B loaded and ready.")


if __name__ == "__main__":
    main()
