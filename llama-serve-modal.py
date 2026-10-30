# llama-serve-modal.py
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
#
# --- TUNING NOTES (A10 / 24 GB) ---
#
# This config is sized for image tagging, not chat: one frame per request, a
# short prompt, and a small JSON reply. Frames are capped at MAX_PIXELS=640000
# (800x800), which is only ~350-625 visual tokens, so prefill is a small slice
# of each request and generation dominates. That means the goal is many
# concurrent slots on one GPU, not a GPU with more FLOPs.
#
# Decoding is memory-bandwidth bound: every decode step streams the ~8.7 GB of
# Q8_0 weights once, no matter how many sequences are being decoded. Running 6
# slots instead of 3 therefore roughly doubles aggregate tokens/s on the *same*
# A10, because the weight read is amortised over twice as many tokens.
#
# VRAM budget with the settings below: ~8.7 GB weights + ~1.4 GB mmproj +
# ~1.8 GB KV (q8_0) + ~1.5-2 GB compute buffers ~= 14 GB of 22.8 GB usable.

import subprocess

import modal

# --- Variables ---
FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.Q8_0.gguf"
MPROJ_FILENAME = "Qwen3-VL-8B-NSFW-Caption-V4.5.mmproj-Q8_0.gguf"
# CHAT_TEMPLATE_FILE = "chat_template-instruct.jinja"
# Must match API_MODEL_NAME used by the client. llama.cpp echoes this in /v1/models.
ALIAS = "InternVL3_5-1B"
N_GPU_LAYERS = "99"

PARALLEL = 6  # concurrent requests == number of server slots
# One frame (~350-625 image tokens) + prompt (~150) + JSON reply (<=400) is
# ~1.2k tokens. 4096 leaves 3x headroom and still fits multi-frame requests.
PER_SLOT_CTX = 4096
CTX_SIZE = PER_SLOT_CTX * PARALLEL  # shared KV pool, 24576 tokens
MAX_INPUTS = PARALLEL  # must track PARALLEL, otherwise Modal throttles below the slot count

BATCH = "2048"  # logical batch: tokens per decode step across all slots
UBATCH = "512"  # physical batch: tokens per GEMM launch / vision encode chunk

# Biggest prefill lever for a dynamic-resolution vision model: caps visual
# tokens per image (Qwen3-VL can otherwise emit 2-4k tokens per frame).
# In practice this never binds here: frames are capped at MAX_PIXELS=640000
# (800x800) and Qwen3-VL packs 32x32 px per token, so a frame is ~350-625
# tokens. Kept as a guard in case MAX_PIXELS is ever raised.
IMAGE_MAX_TOKENS = "1024"
# NOTE: llama.cpp warns at load time that Qwen-VL "requires at minimum 1024
# image tokens ... on grounding tasks". That is about --image-min-tokens (the
# floor), and it is only a concern for bounding-box/pointing accuracy. Setting
# it to 1024 would upsample every 625-token frame to 1024 tokens, i.e. ~60%
# more prefill for interpolated pixels, with no new information for captioning.
# If tagging quality drops or positions look wrong, uncomment the line below.
# IMAGE_MIN_TOKENS = "1024"
# Vision encoder chunk size. Default is 1024 tokens, which can split a single
# image across several ViT passes; 2048 encodes each frame in one pass.
MTMD_BATCH_MAX_TOKENS = "2048"

CPU = 8.0  # base64 + JPEG decode happen on CPU, so this matters at high concurrency
MEMORY = 12288  # MiB; needed because the model is read into RAM (load-mode none)
GPU = "A10"  # T4, L4, A10. Pricing/specs: https://modal.com/pricing
# Pre-built llama.cpp image tag from ghcr.io/ggml-org/llama.cpp.
# The project publishes server images tagged as `server-cudaNN`, e.g.
# `server-cuda13` for CUDA 13 builds. Bump this when upgrading llama.cpp.
TAG_IMAGE = "server-cuda13-b12004"

# --- Configuration ---
# Use the pre-built llama.cpp server-cuda image from ghcr.io to avoid the
# lengthy CUDA + cmake build inside Modal. Pre-built images are published by
# the llama.cpp project at ghcr.io/ggml-org/llama.cpp:server-cudaNN.
# See: https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp
model_volume = modal.Volume.from_name("llama-models-store", create_if_missing=True)
MODEL_DIR = "/models"
llama_image = (
    modal.Image.from_registry(
        f"ghcr.io/ggml-org/llama.cpp:{TAG_IMAGE}", add_python="3.12"
    )
    .entrypoint([])  # remove the base container entrypoint so Modal can run our server
    .env(
        {
            "LLAMA_ARG_MODEL": f"{MODEL_DIR}/{FILENAME}",
            "LLAMA_ARG_MMPROJ": f"{MODEL_DIR}/{MPROJ_FILENAME}",
            # "LLAMA_ARG_CHAT_TEMPLATE_FILE": f"{MODEL_DIR}/{CHAT_TEMPLATE_FILE}",
            "LLAMA_ARG_ALIAS": ALIAS,
            "LLAMA_ARG_CTX_SIZE": str(CTX_SIZE),
            "LLAMA_ARG_BATCH": BATCH,
            "LLAMA_ARG_UBATCH": UBATCH,
            "LLAMA_ARG_N_GPU_LAYERS": N_GPU_LAYERS,
            "LLAMA_ARG_N_PARALLEL": str(PARALLEL),
            # Continuous batching is on by default; set it explicitly so a
            # changed default can't silently serialise every request.
            "LLAMA_ARG_CONT_BATCHING": "1",
            # Flash attention must be on for quantised KV cache.
            "LLAMA_ARG_FLASH_ATTN": "on",
            # q8_0 KV halves the cache (~1.8 GB instead of ~3.6 GB for 24576
            # tokens), which is what makes 6 slots fit comfortably on 24 GB.
            "LLAMA_ARG_CACHE_TYPE_K": "q8_0",
            "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
            # One shared KV pool instead of a fixed ctx/parallel slice per slot:
            # short requests don't strand a slot's reserved context. Note the
            # default is only "on" when --parallel is left on auto.
            "LLAMA_ARG_KV_UNIFIED": "1",
            # Reuse of a shifted prompt prefix is NOT supported together with an
            # mmproj (llama.cpp disables it with a warning), so don't set
            # LLAMA_ARG_CACHE_REUSE here. Prefix caching for identical prompts
            # (LLAMA_ARG_CACHE_PROMPT) is separate, on by default, and still works.
            # Vision: fewer tokens per frame, larger encode chunks.
            "LLAMA_ARG_IMAGE_MAX_TOKENS": IMAGE_MAX_TOKENS,
            "LLAMA_ARG_MTMD_BATCH_MAX_TOKENS": MTMD_BATCH_MAX_TOKENS,
            # Tagging needs JSON, not chain-of-thought. Thinking tokens are pure
            # decode cost and Qwen3 templates default to thinking on.
            "LLAMA_ARG_REASONING": "off",
            # Prometheus metrics on /metrics (prompt vs decode throughput), used
            # to decide whether a faster GPU is actually worth paying for.
            "LLAMA_ARG_ENDPOINT_METRICS": "1",
            # The volume is a network mount; mmap over it page-faults badly on
            # cold start. Read the weights into RAM once instead (see MEMORY).
            "LLAMA_ARG_LOAD_MODE": "none",
            "LLAMA_ARG_THREADS": str(int(CPU)),
            "LLAMA_ARG_THREADS_HTTP": str(int(CPU)),
        }
    )
)
app = modal.App("llama-cpp-server")


@app.function(
    image=llama_image,
    gpu=GPU,
    cpu=CPU,  # default is 0.125
    memory=MEMORY,
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 5,  # 5 minutes max input runtime
    scaledown_window=300,  # Timeout after 5 minutes of inactivity.
    min_containers=0,  # Keep at least one container running for fast startup
    secrets=[
        modal.Secret.from_name("LLAMA_API_KEY")
    ],  # load LLAMA_API_KEY from secrets; llama-server reads it natively
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=8080, startup_timeout=600)  # loading ~10 GB from the volume is slow
def serve():
    # Passed as a list (no shell) so the JSON in --chat-template-kwargs, if it's
    # ever re-enabled, survives quoting intact.
    cmd = [
        "/app/llama-server",
        "--port",
        "8080",
        "--host",
        "0.0.0.0",
        "--fit",
        "on",
        "--jinja",
        # Shared KV pool plus a per-request ceiling, so one runaway generation
        # can't starve the other slots.
        "--kv-unified",
        "--kv-unified-per-slot",
        str(PER_SLOT_CTX),
        # Leave any reasoning tags in the content rather than splitting them out.
        "--reasoning-format",
        "none",
    ]
    print(cmd)
    subprocess.Popen(cmd)
    print("Serving llama.cpp API on port 8080")
