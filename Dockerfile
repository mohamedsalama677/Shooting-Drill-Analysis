FROM python:3.11-slim

# System dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Volumes: mount input videos and receive output files from the host
VOLUME ["/app/data/input", "/app/data/output", "/app/models"]

ENTRYPOINT ["python", "main.py"]
