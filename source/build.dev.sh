#!/bin/sh
set -e
cd "$(dirname "$0")"

# ==== Linux x86_64 (Debug) ====
printf '=== Building DEBUG libbinding.a for Linux x86_64 ===\n\n'
docker build -f Dockerfile.linux -t zeus-linux-builder-debug .
docker run --rm -e DEBUG=1 -e CCACHE_DIR=/cache \
    -v "$(pwd)/../lib/linux:/output" \
    -v zeus-build-cache-linux-x64:/cache \
    -v zeus-build-linux-x64:/build/build \
    zeus-linux-builder-debug \
    sh -c "make libbinding.a && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Linux x86_64 ===\n'

# ==== Linux ARM64 (Debug) ====
printf '\n=== Building DEBUG libbinding.a for Linux ARM64 ===\n\n'
docker build -f Dockerfile.linux-arm64 -t zeus-linux-arm64-builder-debug .
docker run --rm -e DEBUG=1 -e CCACHE_DIR=/cache \
    -v "$(pwd)/../lib/linux-arm64:/output" \
    -v zeus-build-cache-linux-arm64:/cache \
    -v zeus-build-linux-arm64:/build/build \
    zeus-linux-arm64-builder-debug \
    sh -c "make libbinding.a CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Linux ARM64 ===\n'

# ==== Windows x86_64 (Debug) ====
printf '\n=== Building DEBUG libbinding.a for Windows x86_64 ===\n\n'
docker build -f Dockerfile.windows -t zeus-windows-builder-debug .
docker run --rm -e DEBUG=1 -e CCACHE_DIR=/cache \
    -v "$(pwd)/../lib/windows:/output" \
    -v zeus-build-cache-windows-x64:/cache \
    -v zeus-build-windows-x64:/build/build \
    zeus-windows-builder-debug \
    sh -c "make libbinding.a CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ AR=x86_64-w64-mingw32-ar && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Windows x86_64 ===\n'
