"""
data_prep_final.py — PlantVillage data preparation for Krushak Hithaishi
=========================================================================
Key fix per instruct.md:
  Synthetic weather values are NOISY and overlap heavily between classes.
  The humidity bias (fungal→high, others→low) is intentionally WEAK — a
  large Gaussian noise term (std=15) is added so the weather branch cannot
  simply memorise the label from humidity alone.  This forces the model to
  learn real image features; the weather branch only contributes a marginal
  additional signal, which is the realistic deployment scenario.

Only the 10 tomato classes are kept (as required by instruct.md step 2).
"""

import os
import numpy as np
from PIL import Image

# ── Tomato classes to keep (exactly as PlantVillage names them) ────────────
TOMATO_CLASSES = {
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
}


def _severity_for(folder: str) -> str:
    name = folder.lower()
    if "healthy" in name:
        return "LOW"
    if any(k in name for k in ("blight", "rust", "mildew", "scab", "spot",
                                "rot", "mosaic", "curl", "canker", "mold",
                                "virus", "mites")):
        return "HIGH"
    return "MEDIUM"


def build_dataset(
    data_dir: str = "./data",
    output_path: str = "./dataset.npz",
    seed: int = 42,
    image_size: int = 224,
    max_per_class: int = 300,   # CPU-friendly: ≤300 per class
):
    rng = np.random.default_rng(seed)
    images, weather, labels, severities = [], [], [], []

    all_folders = sorted(os.listdir(data_dir))
    # Filter to only tomato folders
    tomato_folders = [
        f for f in all_folders
        if os.path.isdir(os.path.join(data_dir, f)) and f in TOMATO_CLASSES
    ]
    print(f"[data_prep] Found {len(all_folders)} total class folders")
    print(f"[data_prep] Keeping {len(tomato_folders)} tomato classes")

    total = 0
    for disease_folder in tomato_folders:
        disease_path = os.path.join(data_dir, disease_folder)

        # Weak humidity signal — fungal diseases run slightly higher humidity
        # but heavy noise means no class can be read from weather alone.
        # This is intentional: the model must rely on the image branch for
        # disease identity; weather only provides weak contextual assistance.
        is_fungal = any(k in disease_folder.lower()
                        for k in ("blight", "rust", "mildew", "scab",
                                  "spot", "rot", "mold"))
        base_humidity = 75.0 if is_fungal else 58.0

        severity = _severity_for(disease_folder)

        image_files = [
            f for f in sorted(os.listdir(disease_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        class_count = 0
        for image_name in image_files:
            try:
                img = (
                    Image.open(os.path.join(disease_path, image_name))
                    .convert("RGB")
                    .resize((image_size, image_size))
                )
                images.append(np.array(img, dtype=np.uint8))

                # ── Noisy synthetic weather vector ──────────────────────────
                # temperature:   28 °C ± 5 (std) — disease-agnostic
                # humidity:      base ± 15 (std)  — overlapping distributions
                # precipitation: random 0–5 mm
                # stage_code:    random {0,1,2}   — random crop stage
                temperature   = float(rng.normal(28.0, 5.0))
                humidity      = float(np.clip(rng.normal(base_humidity, 15.0), 20, 100))
                precipitation = float(rng.uniform(0.0, 5.0))
                stage_code    = float(rng.integers(0, 3))

                weather.append([temperature, humidity, precipitation, stage_code])
                labels.append(disease_folder)
                severities.append(severity)
                class_count += 1
                total += 1
                if total % 500 == 0:
                    print(f"  Loaded {total} images so far...")
            except Exception as e:
                print(f"  [WARN] Skipping {image_name}: {e}")

        print(f"  {disease_folder}: {class_count} images")

    print(f"\n[data_prep] Saving {len(images)} images to {output_path} ...")
    np.savez_compressed(
        output_path,
        images=np.array(images, dtype=np.uint8),
        weather=np.array(weather, dtype="float32"),
        labels=np.array(labels),
        severities=np.array(severities),
    )
    classes_found = sorted(set(labels))
    print(f"[data_prep] Done: {len(images)} samples, {len(classes_found)} classes")
    print(f"[data_prep] Classes: {classes_found}")
    return classes_found
