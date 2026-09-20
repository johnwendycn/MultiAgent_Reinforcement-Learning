# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DISPLAY=:1

# Install system dependencies, C++ build toolchain, OpenGL, SDL2, Boost, and Xvfb
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        curl \
        wget \
        cmake \
        build-essential \
        libgl1-mesa-dev \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-ttf-dev \
        libsdl2-gfx-dev \
        libboost-all-dev \
        libdirectfb-dev \
        libst-dev \
        mesa-utils \
        xvfb \
        x11vnc \
        ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Alias python to python3.11
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /content/semantic_football_marl

# Copy requirements and install pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source tree and configuration files
COPY . .

# Launch Xvfb virtual framebuffer on container start if rendering
CMD ["bash", "-c", "Xvfb :1 -screen 0 1024x768x24 > /dev/null 2>&1 & python scripts/train.py"]
