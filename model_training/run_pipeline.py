"""
Krushak Hithaishi — End-to-End Training Pipeline Runner
=======================================================
Run this script using the virtual environment Python:

    .venv\Scripts\python.exe run_pipeline.py

Or activate the venv first:
    .venv\Scripts\activate
    python run_pipeline.py

Steps:
  1. Download PlantVillage dataset via tensorflow_datasets
  2. Run data_prep_fixed  → dataset.npz
  3. Run train_model_fixed → model.keras + disease_labels.txt + severity_labels.txt
  4. Run convert_tflite_fixed → krushak.tflite
  5. Copy outputs to ../backend/models/
  6. Print summary + next steps
"""

import os
import sys
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_MODELS = os.path.join(ROOT, "..", "backend", "models")

# Config — change these to control the pipeline
IMAGE_SIZE = 224            # 224 for fast training, 512 for full quality
MAX_PER_CLASS = None        # Set to e.g. 200 for a quick smoke test, None for full dataset
QUANTIZATION = "float16"    # "int8" or "float16" (float16 is more compatible)


def step(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_tf():
    step("Checking TensorFlow")
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__}")
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
        return True
    except ImportError:
        print("[FAIL] TensorFlow not found. Activate venv:  .venv\\Scripts\\activate")
        return False


def download_plantvillage():
    """Download PlantVillage via tensorflow_datasets and convert to folder structure."""
    step("Step 0 — Download PlantVillage Dataset")
    data_dir = os.path.join(ROOT, "data")

    # Check if data already exists
    if os.path.isdir(data_dir):
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(dirs) >= 10:
            print(f"[OK] ./data/ exists with {len(dirs)} class folders — skipping download")
            return True

    print("[INFO] Downloading PlantVillage via tensorflow_datasets (~1.4 GB) ...")
    print("       This may take 5-15 minutes depending on your internet connection.")
    try:
        import tensorflow_datasets as tfds
        import numpy as np
        from PIL import Image

        # Download and load the dataset
        ds, info = tfds.load(
            "plant_village",
            split="train",
            with_info=True,
            as_supervised=False,
            data_dir=os.path.join(ROOT, "tfds_cache"),
        )
        label_names = info.features["label"].names
        print(f"[OK] Dataset loaded: {info.splits['train'].num_examples} examples, {len(label_names)} classes")

        os.makedirs(data_dir, exist_ok=True)
        counts = {}

        for ex in ds:
            label_id = int(ex["label"].numpy())
            label_name = label_names[label_id]
            cls_dir = os.path.join(data_dir, label_name)
            os.makedirs(cls_dir, exist_ok=True)
            n = counts.get(label_name, 0)

            if MAX_PER_CLASS and n >= MAX_PER_CLASS:
                continue

            img = Image.fromarray(ex["image"].numpy())
            img.save(os.path.join(cls_dir, f"{n:05d}.jpg"))
            counts[label_name] = n + 1

            total = sum(counts.values())
            if total % 1000 == 0:
                print(f"  ... {total} images written so far")

        total = sum(counts.values())
        print(f"[OK] PlantVillage saved: {total} images across {len(counts)} classes")
        return True

    except ImportError:
        print("[FAIL] tensorflow_datasets not installed. Run:")
        print("  .venv\\Scripts\\pip install tensorflow-datasets")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback; traceback.print_exc()
        return False


def run_data_prep():
    step("Step 1 — Data Preparation")
    dataset_path = os.path.join(ROOT, "dataset.npz")
    if os.path.exists(dataset_path):
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"[OK] dataset.npz already exists ({size_mb:.1f} MB) — skipping")
        return True

    sys.path.insert(0, ROOT)
    from data_prep_fixed import build_dataset
    try:
        build_dataset(
            data_dir=os.path.join(ROOT, "data"),
            output_path=dataset_path,
            image_size=IMAGE_SIZE,
            max_per_class=MAX_PER_CLASS,
        )
        return True
    except Exception as e:
        print(f"[FAIL] data_prep failed: {e}")
        import traceback; traceback.print_exc()
        return False


def run_train():
    step("Step 2 — Model Training (20 epochs w/ early stopping)")
    model_path = os.path.join(ROOT, "model.keras")
    labels_path = os.path.join(ROOT, "disease_labels.txt")
    if os.path.exists(model_path) and os.path.exists(labels_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[OK] model.keras ({size_mb:.1f} MB) + disease_labels.txt exist — skipping")
        return True

    sys.path.insert(0, ROOT)
    from train_model_fixed import train
    try:
        train(
            dataset_path=os.path.join(ROOT, "dataset.npz"),
            model_out=model_path,
            labels_out=ROOT,
            image_size=IMAGE_SIZE,
        )
        return True
    except Exception as e:
        print(f"[FAIL] train_model failed: {e}")
        import traceback; traceback.print_exc()
        return False


def run_convert():
    step("Step 3 — TFLite Conversion")
    tflite_path = os.path.join(ROOT, "krushak.tflite")
    if os.path.exists(tflite_path):
        size_kb = os.path.getsize(tflite_path) / 1024
        print(f"[OK] krushak.tflite ({size_kb:.1f} KB) already exists — skipping")
        return True

    sys.path.insert(0, ROOT)
    from convert_tflite_fixed import convert_model
    try:
        convert_model(
            model_path=os.path.join(ROOT, "model.keras"),
            dataset_path=os.path.join(ROOT, "dataset.npz"),
            output_path=tflite_path,
            image_size=IMAGE_SIZE,
            quantization=QUANTIZATION,
        )
        return True
    except Exception as e:
        print(f"[FAIL] convert_tflite failed: {e}")
        import traceback; traceback.print_exc()
        return False


def copy_to_backend():
    step("Step 4 — Copy to backend/models/")
    os.makedirs(BACKEND_MODELS, exist_ok=True)
    files = ["krushak.tflite", "disease_labels.txt", "severity_labels.txt"]
    all_ok = True
    for fname in files:
        src = os.path.join(ROOT, fname)
        dst = os.path.join(BACKEND_MODELS, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size = os.path.getsize(dst) / 1024
            print(f"[OK] {fname} → backend/models/  ({size:.1f} KB)")
        else:
            print(f"[WARN] {fname} missing at {src}")
            all_ok = False
    return all_ok


def print_summary():
    step("Pipeline Complete — Summary")
    files = [
        ("dataset.npz",             ROOT),
        ("model.keras",             ROOT),
        ("krushak.tflite",          ROOT),
        ("disease_labels.txt",      ROOT),
        ("severity_labels.txt",     ROOT),
        ("krushak.tflite",          BACKEND_MODELS),
        ("disease_labels.txt",      BACKEND_MODELS),
        ("severity_labels.txt",     BACKEND_MODELS),
    ]
    for fname, directory in files:
        path = os.path.join(directory, fname)
        label = fname if directory == ROOT else f"{fname} (backend)"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅  {label:40s}  {size:>8.1f} KB")
        else:
            print(f"  ❌  {label:40s}  MISSING")

    print("\n" + "=" * 60)
    print("  NEXT STEPS — Start the Backend")
    print("=" * 60)
    print("""
  Option A: Quick start (Windows PowerShell)
  ------------------------------------------
  cd ..\backend
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  python -m db.seed_shops
  uvicorn main:app --host 0.0.0.0 --port 8000

  Then open: http://localhost:8000/health
  Full API:  http://localhost:8000/docs

  Option B: Or run setup_backend.py (auto-creates venv + seeds DB)
  ---------------------------------------------------------------
  python ..\setup_backend.py
""")


if __name__ == "__main__":
    print("\n🌱  Krushak Hithaishi — Training Pipeline")
    print(f"   Image size  : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   Max/class   : {MAX_PER_CLASS or 'all'}")
    print(f"   Quantization: {QUANTIZATION}")
    print(f"   Working dir : {ROOT}\n")

    if not check_tf():
        sys.exit(1)

    if not download_plantvillage():
        print("\n[ABORT] Cannot proceed without PlantVillage data.")
        print("If you have data already in ./data/, it may be incomplete. Check folder structure.")
        sys.exit(1)

    if not run_data_prep():
        sys.exit(1)

    if not run_train():
        sys.exit(1)

    if not run_convert():
        sys.exit(1)

    copy_to_backend()
    print_summary()
    print("🎉  Pipeline complete!\n")
