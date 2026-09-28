"""
setup_backend.py — Quick backend setup script
Run from the project root: python setup_backend.py
"""
import os
import sys
import subprocess
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT, "backend")
VENV_DIR = os.path.join(BACKEND_DIR, ".venv")
PY = os.path.join(VENV_DIR, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")
PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "pip")


def run(cmd, cwd=None, check=True):
    print(f"  >> {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd or BACKEND_DIR, shell=isinstance(cmd, str), capture_output=False)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        sys.exit(1)
    return result


print("\n🌱  Krushak Hithaishi — Backend Setup")
print(f"   Backend dir: {BACKEND_DIR}\n")

# 1. Create venv
if not os.path.exists(PY):
    print("[1/4] Creating virtual environment...")
    run([sys.executable, "-m", "venv", VENV_DIR])
else:
    print("[1/4] Virtual environment already exists")

# 2. Install requirements (without heavy ML deps)
print("\n[2/4] Installing requirements...")
# Use lighter requirements for backend (no TF training deps needed)
backend_reqs = os.path.join(BACKEND_DIR, "requirements.txt")
run([PIP, "install", "-r", backend_reqs, "-q"])

# 3. Create static/voice dir
voice_dir = os.path.join(BACKEND_DIR, "static", "voice")
os.makedirs(voice_dir, exist_ok=True)
print(f"\n[3/4] Created {voice_dir}")

# 4. Seed the database
print("\n[4/4] Seeding shop database...")
env = os.environ.copy()
env["PYTHONPATH"] = BACKEND_DIR
seed_result = subprocess.run([PY, "-m", "db.seed_shops"], cwd=BACKEND_DIR, env=env)
if seed_result.returncode != 0:
    print("[WARN] seed_shops failed — shops endpoint may not return results")

print("\n" + "=" * 50)
print("  Backend ready! Start with:")
print("=" * 50)
print(f"""
  cd backend
  .venv\\Scripts\\activate       (Windows)
  # or: source .venv/bin/activate  (Linux/Mac)
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  Test endpoints:
    GET  http://localhost:8000/health
    GET  http://localhost:8000/docs       (Swagger UI)
    POST http://localhost:8000/detect     (with image + form data)
    POST http://localhost:8000/shops/nearby
""")
