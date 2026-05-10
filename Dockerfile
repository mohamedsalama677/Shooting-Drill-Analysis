FROM python:3.11-slim

# System dependencies:
#   libgl1, libglib2.0-0, libgomp1, ffmpeg — OpenCV / MediaPipe / video I/O
#   git — required for `pip install git+https://...` (used for CLIP, a YOLO-World dep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make pip resilient on slow / unstable connections
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5 \
    PIP_NO_CACHE_DIR=1

# Install CPU-only PyTorch FIRST from the dedicated wheel index.
# Without this, ultralytics pulls the full CUDA torch (~3 GB of NVIDIA libs).
# CPU torch is ~200 MB and is all we need for this drill-analysis project.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 \
    torchvision==0.19.1

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-install the CLIP fork that YOLO-World needs for its text encoder.
# Done at build time so it doesn't try to auto-install at first run.
RUN pip install git+https://github.com/ultralytics/CLIP.git

COPY . .

# Volumes: mount input videos and receive output files from the host
VOLUME ["/app/data/input", "/app/data/output", "/app/models"]

ENTRYPOINT ["python", "main.py"]
