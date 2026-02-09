# AWS DLC PyTorch Training (includes PyTorch + CUDA + build tooling)
# Good for compiling flash-attn during build.
FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.23

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_ROOT=/models

# System deps:
# - ffmpeg: to write .m4b/.m4a
# - libsndfile1: soundfile backend
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps recommended by Qwen repo:
# - qwen-tts
# - flash-attn (FlashAttention 2) to reduce GPU memory usage
# - huggingface_hub[cli] for huggingface-cli download
RUN pip install -U pip \
 && pip install -U qwen-tts soundfile requests boto3 "huggingface_hub[cli]" \
 && MAX_JOBS=4 pip install -U flash-attn --no-build-isolation

# Where we bake model weights
RUN mkdir -p ${MODEL_ROOT}

# ---- Bake weights into the image (repo-recommended commands) ----
# For preset voices (CustomVoice 0.6B), bake tokenizer + model.
RUN huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
      --local-dir ${MODEL_ROOT}/Qwen3-TTS-Tokenizer-12Hz \
 && huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
      --local-dir ${MODEL_ROOT}/Qwen3-TTS-12Hz-0.6B-CustomVoice

# App
WORKDIR /app
COPY generate_tts.py /app/generate_tts.py

ENV QWEN_MODEL_PATH=${MODEL_ROOT}/Qwen3-TTS-12Hz-0.6B-CustomVoice

CMD ["python", "/app/generate_tts.py"]
