"""
download_plantvillage.py — Downloads PlantVillage dataset
=========================================================
This script downloads PlantVillage WITHOUT tensorflow-datasets
(which conflicts with TF 2.16's protobuf requirements).

Uses the Kaggle API or a direct HTTP download approach.

Run with the SYSTEM Python (not venv):
    python download_plantvillage.py

Or directly install kaggle in venv and run:
    .venv\Scripts\pip install kaggle
    .venv\Scripts\python download_plantvillage.py
"""

import os
import sys
import zipfile
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def check_data_exists():
    """Return True if data already looks populated."""
    if not os.path.isdir(DATA_DIR):
        return False
    dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    return len(dirs) >= 10


def try_kaggle():
    """Download via Kaggle API (requires ~/.kaggle/kaggle.json)."""
    try:
        import kaggle
        print("[INFO] Downloading via Kaggle API...")
        print("       Dataset: vipoooool/new-plant-diseases-dataset (~1.4 GB)")
        print("       This will take several minutes...\n")

        zip_path = os.path.join(ROOT, "plant_diseases.zip")
        kaggle.api.dataset_download_files(
            "vipoooool/new-plant-diseases-dataset",
            path=ROOT,
            unzip=False,
        )

        # Find the downloaded zip
        for fname in os.listdir(ROOT):
            if fname.endswith(".zip") and "plant" in fname.lower():
                zip_path = os.path.join(ROOT, fname)
                break

        print(f"[OK] Downloaded to {zip_path}")
        print("[INFO] Extracting...")

        # Extract and reorganize
        extract_to = os.path.join(ROOT, "_extracted")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)

        # The Kaggle dataset has train/valid folders — merge into data/
        _reorganize_kaggle_extract(extract_to)
        os.remove(zip_path)
        return True

    except ImportError:
        print("[INFO] kaggle package not found in this environment.")
        return False
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        return False


def _reorganize_kaggle_extract(extract_root):
    """Kaggle dataset has: New Plant Diseases Dataset(Augmented)/train/<class>/*.jpg
    We want: data/<class>/*.jpg
    """
    import shutil

    os.makedirs(DATA_DIR, exist_ok=True)
    # Walk and find all class directories
    for root, dirs, files in os.walk(extract_root):
        jpg_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if jpg_files and "valid" not in root.lower():  # skip validation set
            class_name = os.path.basename(root)
            dst_dir = os.path.join(DATA_DIR, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            for i, fname in enumerate(jpg_files):
                shutil.copy2(
                    os.path.join(root, fname),
                    os.path.join(dst_dir, f"{i:05d}.jpg")
                )
            print(f"  Copied {len(jpg_files):5d} images → data/{class_name}/")

    shutil.rmtree(extract_root, ignore_errors=True)
    total = sum(len(os.listdir(os.path.join(DATA_DIR, d)))
                for d in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, d)))
    n_classes = len(os.listdir(DATA_DIR))
    print(f"\n[OK] PlantVillage ready: {total} images, {n_classes} classes in {DATA_DIR}")


def try_opendatasets():
    """Alternative: opendatasets library can download Kaggle datasets too."""
    try:
        import opendatasets as od
        print("[INFO] Downloading via opendatasets...")
        print("       You'll be prompted for your Kaggle username and API key.")
        od.download("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset", data_dir=ROOT)
        _reorganize_kaggle_extract(os.path.join(ROOT, "new-plant-diseases-dataset"))
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[WARN] opendatasets download failed: {e}")
        return False


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         MANUAL DOWNLOAD REQUIRED                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Go to: https://www.kaggle.com/datasets/vipoooool/        ║
║            new-plant-diseases-dataset                        ║
║                                                              ║
║  2. Download the ZIP (≈1.4 GB)                               ║
║                                                              ║
║  3. Extract and place the class folders in:                  ║
║     model_training/data/<ClassName>/*.jpg                    ║
║                                                              ║
║  Example structure:                                          ║
║     data/Apple___Apple_scab/00001.jpg                        ║
║     data/Apple___Black_rot/00001.jpg                         ║
║     data/Tomato___healthy/00001.jpg                          ║
║     ...38 total class folders                                ║
║                                                              ║
║  Then re-run: .venv\\Scripts\\python run_pipeline.py          ║
║                                                              ║
║  ── OR ── setup Kaggle credentials:                          ║
║  1. Create account at kaggle.com                             ║
║  2. Profile → Settings → API → Create New Token              ║
║  3. Save kaggle.json to C:\\Users\\<you>\\.kaggle\\kaggle.json  ║
║  4. Run: .venv\\Scripts\\pip install kaggle                   ║
║  5. Run: .venv\\Scripts\\python download_plantvillage.py      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    if check_data_exists():
        n = len([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        print(f"[OK] Data already exists: {n} class folders in {DATA_DIR}")
        sys.exit(0)

    print("\n🌱  PlantVillage Dataset Downloader")
    print(f"   Target: {DATA_DIR}\n")

    success = try_kaggle()
    if not success:
        success = try_opendatasets()
    if not success:
        print_manual_instructions()
        sys.exit(1)

    print("\n✅  Dataset ready! Run: .venv\\Scripts\\python run_pipeline.py\n")
