# llama_cpp_server.py
#
# This script creates a Modal deployment for the llama.cpp server, allowing you
# to run various GGUF models on Modal's GPU cloud.
#
# --- USAGE ---
#
# 1. First, ensure you have downloaded the desired GGUF model to the persistent
#    volume using the `llama-download-modal.py` script.
#
# 2. Deploy this script:
#    modal deploy llama-serve-modal.py
#
# 3. To stop the deployment, use:
#    modal app stop llama-cpp-server
#
# After running, Modal will provide a public URL. You can use this URL with any
# OpenAI-compatible client library or tools like `curl` to interact with the model.
# The server exposes standard endpoints like `/v1/chat/completions`.

import modal
from pathlib import Path
import subprocess

# --- Variables ---
#FILENAME = "Qwen2.5-VL-7B-NSFW-Caption-V4.Q8_0.gguf"
#MPROJ_FILENAME = "Qwen2.5-VL-7B-NSFW-Caption-V4.mmproj-Q8_0.gguf"
FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.Q8_0.gguf"
MPROJ_FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf"
ALIAS = "InternVL3_5-1B"
N_GPU_LAYERS = "99"
CTX_SIZE = "12288" # 4096*3
BATCH = "2048"
UBATCH = "2048"
PARALLEL = "3"
API_KEY = "ByU8dGHlt8chOIKT"
#TAG = "12.9.1-devel-ubuntu22.04"
TAG = "13.0.1-devel-ubuntu24.04"
GPU = "L4" # T4, L4, A10 Available GPUs: https://modal.com/pricing, https://modal.com/docs/guide/gpu#specifying-gpu-type

# --- Configuration ---
# Define the base Docker image from ghcr.io, a persistent volume for models,
# and a directory path for storing models inside the container.
model_volume = modal.Volume.from_name("llama-models-store", create_if_missing=True)
MODEL_DIR = "/models"
llama_image = (
    #modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.11")
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev", "ccache")
    .run_commands("git clone --depth 1 --branch b6992 https://github.com/ggml-org/llama.cpp", force_build=False)
    .run_commands(
        "nvidia-smi",
        gpu=GPU
    )
    .run_commands( # https://developer.nvidia.com/cuda-gpus
        'cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100"',
        gpu=GPU 
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-server",
        gpu=GPU
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
    .env(
        {
            "LLAMA_ARG_MODEL": f"{MODEL_DIR}/{FILENAME}",
            "LLAMA_ARG_MMPROJ": f"{MODEL_DIR}/{MPROJ_FILENAME}",
            "LLAMA_ARG_ALIAS": ALIAS,
            "LLAMA_ARG_CTX_SIZE": CTX_SIZE,
            "LLAMA_ARG_BATCH": BATCH,
            "LLAMA_ARG_UBATCH": UBATCH,
            "LLAMA_ARG_N_GPU_LAYERS": N_GPU_LAYERS,
            "LLAMA_ARG_N_PARALLEL": PARALLEL,
            "LLAMA_API_KEY": API_KEY
        }
    )
    
)
app = modal.App("llama-cpp-server")

@app.function(
    image=llama_image,
    gpu=GPU,  
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 5,  # 5 minutes max input runtime
    scaledown_window=300,  # Timeout after 5 minutes of inactivity.
    min_containers=1,  # Keep at least one container running for fast startup
)
@modal.concurrent(max_inputs=3)
@modal.web_server(port=8080, startup_timeout=180)
def serve():
    import subprocess
    cmd = [ "/llama.cpp/llama-server --port 8080 --host 0.0.0.0" ]
    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)
    print("Serving llama.cpp API on port 8080")
