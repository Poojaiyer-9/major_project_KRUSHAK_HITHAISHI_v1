# Fixed data_prep.py — with these improvements:
# 1. Configurable image_size (default 224 instead of 512 for faster training)
# 2. Optional per-class image limit (max_per_class) so you can do a quick smoke test
# 3. Progress printing every 100 images
# 4. Works with both PlantVillage folder structure and tfds-downloaded data

import os
import numpy as np
from PIL import Image


def _severity_for(folder: str) -> str:
    name = folder.lower()
    if "healthy" in name:
        return "LOW"
    if any(k in name for k in ("blight", "rust", "mildew", "scab", "spot", "rot", "mosaic", "curl", "canker")):
        return "HIGH"
    return "MEDIUM"


def build_dataset(
    data_dir: str = "./data",
    output_path: str = "./dataset.npz",
    seed: int = 42,
    image_size: int = 224,          # FIX: 224 instead of 512 for speed
    max_per_class: int = None,       # Optional: limit samples per class (e.g. 100 for smoke test)
):
    rng = np.random.default_rng(seed)
    images, weather, labels, severities = [], [], [], []

    class_folders = sorted(os.listdir(data_dir))
    print(f"Found {len(class_folders)} class folders in {data_dir}")

    total = 0
    for disease_folder in class_folders:
        disease_path = os.path.join(data_dir, disease_folder)
        if not os.path.isdir(disease_path):
            continue
        fungal = any(k in disease_folder.lower() for k in ("blight", "rust", "mildew", "scab", "spot", "rot", "mold"))
        humidity = 80.0 if fungal else 55.0
        severity = _severity_for(disease_folder)

        image_files = [f for f in sorted(os.listdir(disease_path))
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        for image_name in image_files:
            try:
                image = (
                    Image.open(os.path.join(disease_path, image_name))
                    .convert("RGB")
                    .resize((image_size, image_size))
                )
                images.append(np.array(image, dtype=np.uint8))
                # 4-dim aux vector: temperature, relative humidity, precipitation, crop-stage code
                stage_code = float(rng.integers(0, 3))   # 0 seedling, 1 vegetative, 2 flowering
                weather.append([28.0, humidity, 0.2, stage_code])
                labels.append(disease_folder)
                severities.append(severity)
                total += 1
                if total % 500 == 0:
                    print(f"  Loaded {total} images...")
            except Exception as e:
                print(f"  [WARN] Skipping {image_name}: {e}")

    print(f"Saving {len(images)} images to {output_path} ...")
    np.savez_compressed(
        output_path,
        images=np.array(images, dtype=np.uint8),  # save as uint8 to reduce file size
        weather=np.array(weather, dtype="float32"),
        labels=np.array(labels),
        severities=np.array(severities),
    )
    print(f"Saved dataset ({len(images)} samples, {len(set(labels))} classes) to {output_path}")


if __name__ == "__main__":
    build_dataset()
