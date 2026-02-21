# MiniCPM-V 4.5 Web Demo — build and run endlessly with GPU
# Build:  docker build -t minicpm-v45-demo .
# Run:    docker run --gpus all -p 8081:8081 -v $(pwd)/models:/app/models -d --restart unless-stopped minicpm-v45-demo

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Use requirements-docker.txt only (do not use requirements.txt — may have uv freeze / networkx 3.6.1).
COPY requirements-docker.txt .
RUN python3 -m pip install --no-cache-dir -r requirements-docker.txt

# App code
COPY web_demos/ web_demos/
COPY scripts/ scripts/

# Model dir is mounted at /app/models/MiniCPM-V-4_5
ENV MODEL_PATH=/app/models/MiniCPM-V-4_5
EXPOSE 8081

# Run endlessly (Gradio server keeps running)
CMD ["python", "web_demos/web_demo_4_5.py", \
     "--model-path", "/app/models/MiniCPM-V-4_5", \
     "--device", "cuda", \
     "--dtype", "fp16"]
