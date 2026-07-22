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
#    modal app stop -y llama-cpp-server
#
# After running, Modal will provide a public URL. You can use this URL with any
# OpenAI-compatible client library or tools like `curl` to interact with the model.
# The server exposes standard endpoints like `/v1/chat/completions`.

import subprocess

import modal

# --- Variables ---
FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.Q8_0.gguf"
MPROJ_FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf"
# CHAT_TEMPLATE_FILE = "chat_template-instruct.jinja"
ALIAS = "InternVL3_5-1B"
N_GPU_LAYERS = "99"
CTX_SIZE = "24576"  # 8192*3
BATCH = "2048"
UBATCH = "512"
PARALLEL = "3"
# Pre-built llama.cpp image tag from ghcr.io/ggml-org/llama.cpp.
# The project publishes server images tagged as `server-cudaNN`, e.g.
# `server-cuda13` for CUDA 13 builds. Bump this when upgrading llama.cpp.
TAG_IMAGE = "server-cuda13-b10088"
GPU = "L4"  # T4, L4, A10 Available GPUs: https://modal.com/pricing, https://modal.com/docs/guide/gpu#specifying-gpu-type

# --- Configuration ---
# Use the pre-built llama.cpp server-cuda image from ghcr.io to avoid the
# lengthy CUDA + cmake build inside Modal. Pre-built images are published by
# the llama.cpp project at ghcr.io/ggml-org/llama.cpp:server-cudaNN.
# See: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
model_volume = modal.Volume.from_name("llama-models-store", create_if_missing=True)
MODEL_DIR = "/models"
llama_image = (
    modal.Image.from_registry(
        f"ghcr.io/ggml-org/llama.cpp:{TAG_IMAGE}",
        add_python="3.12",
        force_build=False
    )
    .entrypoint([])  # remove the base container entrypoint so Modal can run our server
    .env(
        {
            "LLAMA_ARG_MODEL": f"{MODEL_DIR}/{FILENAME}",
            "LLAMA_ARG_MMPROJ": f"{MODEL_DIR}/{MPROJ_FILENAME}",
            # "LLAMA_ARG_CHAT_TEMPLATE_FILE": f"{MODEL_DIR}/{CHAT_TEMPLATE_FILE}",
            "LLAMA_ARG_ALIAS": ALIAS,
            "LLAMA_ARG_CTX_SIZE": CTX_SIZE,
            "LLAMA_ARG_BATCH": BATCH,
            "LLAMA_ARG_UBATCH": UBATCH,
            "LLAMA_ARG_N_GPU_LAYERS": N_GPU_LAYERS,
            "LLAMA_ARG_N_PARALLEL": PARALLEL,
        }
    )
)
app = modal.App("llama-cpp-server")


@app.function(
    image=llama_image,
    gpu=GPU,
    cpu=8.0,  # default is 0.125
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 5,  # 5 minutes max input runtime
    scaledown_window=300,  # Timeout after 5 minutes of inactivity.
    min_containers=0,  # Keep at least one container running for fast startup
    secrets=[
        modal.Secret.from_name("LLAMA_API_KEY")
    ],  # load LLAMA_API_KEY from secrets
)
@modal.concurrent(max_inputs=3)
@modal.web_server(port=8080, startup_timeout=180)
def serve():

    cmd = [
        # "/app/llama-server --port 8080 --host 0.0.0.0 --fit on --jinja --chat-template-kwargs '{\"enable_thinking\": false}'"
        "/app/llama-server --port 8080 --host 0.0.0.0 --fit on --jinja"
    ]
    print(cmd)
    subprocess.Popen(" ".join(cmd), shell=True)
    print("Serving llama.cpp API on port 8080")
