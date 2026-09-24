FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set CUDA environment variables for bitsandbytes
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Don't write .pyc files. The repo is bind-mounted from the host, and the
# container runs as root — pycache files end up root-owned and break the
# GitHub Actions runner's workspace cleanup on the next deploy.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and system dependencies for new OCR improvements
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libpq-dev \
    gcc \
    g++ \
    libenchant-2-2 \
    git \
    # SSH tools for host machine access (LoRA conversion, llama-server control)
    openssh-client \
    sshpass \
    # VNC dependencies for interactive browser sessions
    xvfb \
    x11vnc \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Reinstall bitsandbytes from source with CUDA support
# This ensures proper CUDA 12.x compilation
RUN pip uninstall -y bitsandbytes && \
    pip install bitsandbytes --no-cache-dir


# Install Playwright browsers with system dependencies
# Firefox for Reddit scraping, Chromium for Patreon publishing
RUN playwright install-deps firefox chromium && \
    playwright install firefox chromium

# Copy application code
COPY . .

# Create data directory for videos
RUN mkdir -p /data/videos

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
