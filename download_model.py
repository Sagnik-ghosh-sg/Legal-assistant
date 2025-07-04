# download_model.py

import os
import requests

MODEL_DIR = "models/gemma"
MODEL_FILENAME = "gemma-2-2b-it.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL = f"https://huggingface.co/TheBloke/gemma-2b-it-GGUF/resolve/main/{MODEL_FILENAME}"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists. Skipping download.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("⬇️ Downloading GGUF model from Hugging Face (~1.6 GB)...")

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("✅ Model download complete.")

if __name__ == "__main__":
    download_model()
