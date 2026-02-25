FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip

WORKDIR /app

# Install Python dependencies
COPY requirementsapp.txt .
RUN pip install --no-cache-dir -r requirementsapp.txt

# Copy source
COPY src/ ./src/

EXPOSE 8081

CMD ["python", "src/main.py", \
     "--model-path", "./models/MiniCPM-V-4_5", \
     "--device", "cuda", \
     "--dtype", "bf16", \
     "--port", "8081", \
     "--host", "0.0.0.0"]