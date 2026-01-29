import os
import urllib.request
import zipfile

# Config
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "../data/glove")
ZIP_FILE = "glove.6B.zip"

def download_glove():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    zip_path = os.path.join(TARGET_DIR, ZIP_FILE)
    
    # 1. Download
    if not os.path.exists(zip_path):
        print(f"⬇️ Downloading GloVe embeddings (This may take a minute)...")
        print(f"   Source: {GLOVE_URL}")
        try:
            urllib.request.urlretrieve(GLOVE_URL, zip_path)
            print("   ✅ Download complete.")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return
    else:
        print("   ✅ Loop: Zip file already exists.")

    # 2. Extract
    print("📦 Extracting glove.6B.100d.txt...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We only need the 100d version for now
            zip_ref.extract("glove.6B.100d.txt", TARGET_DIR)
        print(f"✨ Ready! File located at: {os.path.join(TARGET_DIR, 'glove.6B.100d.txt')}")
    except zipfile.BadZipFile:
        print("❌ Error: Zip file is corrupted. Please delete it and try again.")

if __name__ == "__main__":
    download_glove()
