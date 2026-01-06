# This script downloads a GGUF model from Hugging Face Hub to a persistent
# volume using Modal. It is intended to be run locally to set up the model
# before deploying the llama.cpp server.
#
# --- USAGE ---
# modal run llama-download-modal.py

import modal
from pathlib import Path
import subprocess

# --- Variables ---
#REPO_ID = "mradermacher/Qwen3-VL-8B-NSFW-Caption-V4.5-GGUF"
#FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.Q8_0.gguf"
#MPROJ_FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf"
REPO_ID = "SicariusSicariiStuff/Llama-3.3-8B-Instruct-128K_Abliterated_GGUF"
FILENAME = "Llama-3.3_8B_Abliterated-Q8_0.gguf"
MPROJ_FILENAME = None

# --- Configuration ---
# Define the base Docker image from ghcr.io, a persistent volume for models,
# and a directory path for storing models inside the container.
model_volume = modal.Volume.from_name("llama-models-store", create_if_missing=True)
MODEL_DIR = "/models"
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("llama-cpp-downloader")

@app.function(
    image=download_image, volumes={MODEL_DIR: model_volume}, timeout=60 * 5
)
def download_model(repo_id: str, filename: str, mproj_filename: str = None):
    from huggingface_hub import hf_hub_download
    
    """
    Downloads the specified GGUF model and (if provided) a multimodal projector
    file from Hugging Face Hub to the persistent volume. This function is
    idempotent; it will only download files that don't already exist.
    """
    # Download the main model file
    model_path = Path(MODEL_DIR) / filename
    if not model_path.exists():
        print(f"Downloading model: {repo_id}/{filename}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODEL_DIR
        )
        model_volume.commit()
        print("Model download complete.")
    
    # Download the multimodal projector file (for vision models like InternVL)
    if mproj_filename:
        mproj_path = Path(MODEL_DIR) / mproj_filename
        if not mproj_path.exists():
            print(f"Downloading mmproj: {repo_id}/{mproj_filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=mproj_filename,
                local_dir=MODEL_DIR
            )
            model_volume.commit()
            print("Mmproj download complete.")

@app.local_entrypoint()
def main():
    """
    This local entrypoint provides instructions on how to deploy and run the server.
    """
    download_model.remote(REPO_ID, FILENAME, MPROJ_FILENAME)
