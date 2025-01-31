import os
import shutil
import subprocess
import sys

# Define project structure
FOLDERS = [
    "notebooks",
    "config",
    "tests",
    ".github/workflows"
]

FILES = {
    ".gitignore": """
.venv/
*.pyc
__pycache__/
config/service_account.json
""",
    
    "config/settings.yaml": """
bigquery_project: "your-gcp-project-id"
dataset_id: "your-dataset-id"
table_id: "your-table-id"
service_account_path: "config/service_account.json"
""",

    ".github/workflows/ci.yml": """
name: CI Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          python -m venv venv
          source venv/bin/activate
          pip install -r requirements.txt

      - name: Run tests
        run: |
          source venv/bin/activate
          pytest --maxfail=1 --disable-warnings -q
"""
}

# Step 1: Delete old files and folders
def clean_old_setup():
    print("🔄 Cleaning up old setup...")
    folders_to_remove = [".venv", ".github", "config", "notebooks", "tests"]
    
    for folder in folders_to_remove:
        if os.path.exists(folder):
            print(f"Removing {folder}...")
            shutil.rmtree(folder)
    
    print("✅ Old setup removed.")

# Step 2: Create new virtual environment
def setup_virtualenv():
    print("🐍 Creating a new virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
    print("✅ Virtual environment created.")

# Step 3: Install dependencies
def install_dependencies():
    print("📦 Installing dependencies...")
    subprocess.run([os.path.join(".venv", "bin", "pip"), "install", "--upgrade", "pip"], check=True)
    dependencies = ["pytest", "google-cloud-bigquery", "pandas", "seaborn", "matplotlib"]
    subprocess.run([os.path.join(".venv", "bin", "pip"), "install"] + dependencies, check=True)
    print("✅ Dependencies installed.")

# Step 4: Create folder structure
def create_folders():
    print("📂 Creating project folder structure...")
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created: {folder}")
    print("✅ Folders created.")

# Step 5: Create files
def create_files():
    print("📝 Creating configuration files...")
    for filename, content in FILES.items():
        with open(filename, "w") as f:
            f.write(content.strip())
        print(f"📝 Created: {filename}")
    print("✅ Files created.")

# Run all steps
if __name__ == "__main__":
    try:
        clean_old_setup()
        create_folders()
        setup_virtualenv()
        install_dependencies()
        create_files()
        
        print("\n🚀 Setup complete! Next steps:")
        print("1️⃣ Activate the environment: `source .venv/bin/activate`")
        print("2️⃣ Add your service account JSON file to `config/service_account.json`")
        print("3️⃣ Start working in `notebooks/`")
    
    except Exception as e:
        print(f"❌ Error occurred: {e}")
