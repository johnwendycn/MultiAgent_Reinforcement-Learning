#!/usr/bin/env bash
# ==============================================================================
# Google Colab Setup Script for Google Research Football (GRF)
# Bypasses Boost / CMake compilation errors on Python 3.11 using prebuilt binary
# ==============================================================================
set -e

echo "[1/4] Installing system dependencies (Python 3.10, SDL2, Boost, OpenGL, Xvfb)..."
apt-get update -qq
apt-get install -y -qq git cmake build-essential libgl1-mesa-dev \
    libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
    libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc wget

echo "Installing clean Python 3.10 runtime into /usr/local..."
wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -u -p /usr/local
/usr/local/bin/pip install -q ipykernel ipython "setuptools<66" wheel

echo "[2/4] Fetching official Google Research prebuilt game engine binary..."
rm -rf /tmp/football
git clone -q -b v2.9 https://github.com/google-research/football.git /tmp/football
mkdir -p /tmp/football/third_party/gfootball_engine/lib
wget -q https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so \
    -O /tmp/football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so

echo "[3/4] Installing gfootball with GFOOTBALL_USE_PREBUILT_SO=1..."
cd /tmp/football
GFOOTBALL_USE_PREBUILT_SO=1 python3 -m pip install -q .
cd - > /dev/null

echo "[4/4] Installing pinned project dependencies from requirements.txt..."
python3 -m pip install --no-cache-dir -r requirements.txt

echo "=============================================================================="
echo "[SUCCESS] Google Research Football and all pinned packages installed cleanly!"
echo "=============================================================================="
