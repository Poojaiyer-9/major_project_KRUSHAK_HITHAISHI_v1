"""
run_pipeline_final.py — Krushak Hithaishi End-to-End Training Pipeline
=======================================================================
Follows every step in model_training/instruct.md:

  Step 1 — Check GPU / TF
  Step 2 — Download PlantVillage (tomato classes only, ≤300/class for CPU)
  Step 3 — data_prep_final.py  (noisy, overlapping synthetic weather)
  Step 4 — train_model_final.py (frozen 5 epochs → unfreeze top layers)
  Step 5 — convert_tflite_fixed.py + evaluate_final.py (image-only baseline)
  Step 6 — Copy krushak.tflite + label files → backend/models/
  Step 7 — Print real numbers from evaluate_final.py

Run with the venv Python:
    .venv\\Scripts\\python.exe model_training\\run_pipeline_final.py

Or from the model_training folder:
    python run_pipeline_final.py
"""

import os
import sys
import shutil

ROOT            = os.path.dirname(os.path.abspath(__file__))
BACKEND_MODELS  = os.path.join(ROOT, "..", "backend", "models")
DATA_DIR        = os.path.join(ROOT, "data")
DATASET_PATH    = os.path.join(ROOT, "dataset.npz")
MODEL_PATH      = os.path.join(ROOT, "model.keras")
TFLITE_PATH     = os.path.join(ROOT, "krushak.tflite")

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 224        # 224 for CPU-friendly training
MAX_PER_CLASS   = 300        # per instruct.md: ~300 images/class for CPU
FROZEN_EPOCHS   = 5          # per instruct.md: frozen backbone for first 5 epochs
FINETUNE_EPOCHS = 0          # set to 3 if you want phase-2 fine-tuning (slower)
BATCH_SIZE      = 16
QUANTIZATION    = "float16"  # more compatible than int8 on CPU


def _sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ── Step 1: Check TF + GPU ──────────────────────────────────────────────────
def step1_check_env():
    _sep("Step 1 — Environment Check")
    try:
        import tensorflow as tf
        import numpy as np
        print(f"[OK] TensorFlow {tf.__version__}")
        print(f"[OK] NumPy      {np.__version__}")
        devices = tf.config.list_physical_devices()
        print(f"[OK] Devices    : {devices}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[OK] GPU available: {gpus}")
        else:
            print("[INFO] No GPU detected — using CPU.")
            print(f"       CPU settings: image_size={IMAGE_SIZE}, "
                  f"max_per_class={MAX_PER_CLASS}, "
                  f"frozen_epochs={FROZEN_EPOCHS}")
        return True
    except ImportError as e:
        print(f"[FAIL] {e}  — activate venv:  .venv\\Scripts\\activate")
        return False


# ── Step 2: Download / verify PlantVillage data ─────────────────────────────
def step2_get_data():
    _sep("Step 2 — Get Data (tomato classes only)")
    tomato_classes = {
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

    if os.path.isdir(DATA_DIR):
        existing = {
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
        }
        tomato_found = existing & tomato_classes
        if len(tomato_found) >= 8:
            print(f"[OK] Data already present: {len(tomato_found)} tomato class folders")
            for cls in sorted(tomato_found):
                n = len(os.listdir(os.path.join(DATA_DIR, cls)))
                print(f"     {cls}: {n} images")
            return True

    # Try Kaggle API
    try:
        import kaggle
        print("[INFO] Downloading PlantVillage via Kaggle API (~1.4 GB)...")
        print("       Dataset: vipoooool/new-plant-diseases-dataset")
        kaggle.api.dataset_download_files(
            "vipoooool/new-plant-diseases-dataset",
            path=ROOT,
            unzip=False,
        )
        # Find zip
        import zipfile
        zip_file = next(
            (f for f in os.listdir(ROOT)
             if f.endswith(".zip") and "plant" in f.lower()),
            None
        )
        if zip_file:
            zip_path = os.path.join(ROOT, zip_file)
            print(f"[INFO] Extracting {zip_path} ...")
            extract_to = os.path.join(ROOT, "_extracted")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            _reorganize_kaggle(extract_to)
            os.remove(zip_path)
            return True
    except ImportError:
        print("[INFO] kaggle package not installed.")
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")

    # Manual instructions
    print("""
╔══════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD REQUIRED                                    ║
╠══════════════════════════════════════════════════════════════╣
║  1. Go to:                                                   ║
║     https://www.kaggle.com/datasets/vipoooool/               ║
║     new-plant-diseases-dataset                               ║
║  2. Download the ZIP (≈1.4 GB)                               ║
║  3. Extract so that class folders are at:                    ║
║     model_training/data/Tomato___*/                          ║
║                                                              ║
║  Then re-run this script.                                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    return False


def _reorganize_kaggle(extract_root):
    import shutil as sh
    os.makedirs(DATA_DIR, exist_ok=True)
    for root, dirs, files in os.walk(extract_root):
        jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if jpgs and "valid" not in root.lower():
            cls = os.path.basename(root)
            dst = os.path.join(DATA_DIR, cls)
            os.makedirs(dst, exist_ok=True)
            for i, f in enumerate(jpgs):
                sh.copy2(os.path.join(root, f), os.path.join(dst, f"{i:05d}.jpg"))
            print(f"  {cls}: {len(jpgs)} images")
    sh.rmtree(extract_root, ignore_errors=True)


# ── Step 3: Data preparation ────────────────────────────────────────────────
def step3_data_prep():
    _sep("Step 3 — Data Preparation (noisy weather vectors)")
    if os.path.exists(DATASET_PATH):
        size_mb = os.path.getsize(DATASET_PATH) / 1e6
        print(f"[OK] dataset.npz already exists ({size_mb:.1f} MB) — skipping")
        return True
    sys.path.insert(0, ROOT)
    from data_prep_final import build_dataset
    try:
        build_dataset(
            data_dir=DATA_DIR,
            output_path=DATASET_PATH,
            image_size=IMAGE_SIZE,
            max_per_class=MAX_PER_CLASS,
        )
        return True
    except Exception as e:
        print(f"[FAIL] data_prep_final failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Step 4: Train ───────────────────────────────────────────────────────────
def step4_train():
    _sep(f"Step 4 — Training ({FROZEN_EPOCHS} frozen epochs)")
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"[OK] model.keras ({size_mb:.1f} MB) already exists — skipping")
        return True
    sys.path.insert(0, ROOT)
    from train_model_final import train
    try:
        train(
            dataset_path=DATASET_PATH,
            model_out=MODEL_PATH,
            labels_out=ROOT,
            image_size=IMAGE_SIZE,
            frozen_epochs=FROZEN_EPOCHS,
            finetune_epochs=FINETUNE_EPOCHS,
            batch_size=BATCH_SIZE,
        )
        return True
    except Exception as e:
        print(f"[FAIL] train_model_final failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Step 5: Convert to TFLite + evaluate ────────────────────────────────────
def step5_convert():
    _sep("Step 5 — TFLite Conversion")
    if os.path.exists(TFLITE_PATH):
        size_kb = os.path.getsize(TFLITE_PATH) / 1024
        print(f"[OK] krushak.tflite ({size_kb:.1f} KB) already exists — skipping")
        return True
    sys.path.insert(0, ROOT)
    from convert_tflite_fixed import convert_model
    try:
        convert_model(
            model_path=MODEL_PATH,
            dataset_path=DATASET_PATH,
            output_path=TFLITE_PATH,
            image_size=IMAGE_SIZE,
            quantization=QUANTIZATION,
        )
        return True
    except Exception as e:
        print(f"[FAIL] convert_tflite_fixed failed: {e}")
        import traceback; traceback.print_exc()
        return False


def step5b_evaluate():
    _sep("Step 5b — Evaluate (multimodal vs image-only baseline)")
    sys.path.insert(0, ROOT)
    from evaluate_final import main as eval_main
    try:
        eval_main(
            model_path=TFLITE_PATH,
            dataset_path=DATASET_PATH,
            image_size=IMAGE_SIZE,
        )
        return True
    except Exception as e:
        print(f"[FAIL] evaluate_final failed: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Step 6: Copy to backend/models/ ────────────────────────────────────────
def step6_copy_to_backend():
    _sep("Step 6 — Copy outputs to backend/models/")
    os.makedirs(BACKEND_MODELS, exist_ok=True)
    files = ["krushak.tflite", "disease_labels.txt", "severity_labels.txt"]
    all_ok = True
    for fname in files:
        src = os.path.join(ROOT, fname)
        dst = os.path.join(BACKEND_MODELS, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[OK] {fname} → backend/models/  ({os.path.getsize(dst)/1024:.1f} KB)")
        else:
            print(f"[WARN] {fname} missing — not copied")
            all_ok = False
    return all_ok


def print_summary():
    _sep("Pipeline Summary")
    files = [
        ("dataset.npz",          ROOT),
        ("model.keras",          ROOT),
        ("krushak.tflite",       ROOT),
        ("disease_labels.txt",   ROOT),
        ("severity_labels.txt",  ROOT),
        ("krushak.tflite",       BACKEND_MODELS),
        ("disease_labels.txt",   BACKEND_MODELS),
        ("severity_labels.txt",  BACKEND_MODELS),
    ]
    for fname, directory in files:
        path = os.path.join(directory, fname)
        label = fname if directory == ROOT else f"{fname} (backend)"
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅  {label:45s}  {size:>8.1f} KB")
        else:
            print(f"  ❌  {label:45s}  MISSING")

    print("""
Next: start the backend
──────────────────────────────────────────────────────────────
  cd backend
  ../.venv/Scripts/uvicorn main:app --host 0.0.0.0 --port 8000

  Health check:  http://localhost:8000/health
  API docs:      http://localhost:8000/docs
""")


if __name__ == "__main__":
    print("\n🌾  Krushak Hithaishi — End-to-End Training Pipeline")
    print(f"   Image size     : {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"   Max/class      : {MAX_PER_CLASS}")
    print(f"   Frozen epochs  : {FROZEN_EPOCHS}")
    print(f"   Finetune epochs: {FINETUNE_EPOCHS}")
    print(f"   Quantization   : {QUANTIZATION}")
    print(f"   Working dir    : {ROOT}\n")

    if not step1_check_env():
        sys.exit(1)
    if not step2_get_data():
        print("\n[ABORT] Provide PlantVillage data first (see instructions above).")
        sys.exit(1)
    if not step3_data_prep():
        sys.exit(1)
    if not step4_train():
        sys.exit(1)
    if not step5_convert():
        sys.exit(1)
    step6_copy_to_backend()
    step5b_evaluate()   # evaluate after copying so backend is ready
    print_summary()
    print("🎉  Pipeline complete!\n")
