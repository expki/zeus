#!/bin/sh
set -e
cd "$(dirname "$0")"

# ==== Linux x86_64 (Debug) ====
printf '=== Building DEBUG libbinding.a for Linux x86_64 ===\n\n'
docker build -f Dockerfile.linux -t zeus-linux-builder .
docker run --rm -e DEBUG=1 \
    -v "$(pwd)/../lib/linux:/output" \
    zeus-linux-builder \
    sh -c "make clean && make libbinding.a && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Linux x86_64 ===\n'

# ==== Linux ARM64 (Debug) ====
printf '\n=== Building DEBUG libbinding.a for Linux ARM64 ===\n\n'
docker build -f Dockerfile.linux-arm64 -t zeus-linux-arm64-builder .
docker run --rm -e DEBUG=1 \
    -v "$(pwd)/../lib/linux-arm64:/output" \
    zeus-linux-arm64-builder \
    sh -c "make clean && make libbinding.a CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Linux ARM64 ===\n'

# ==== Windows x86_64 (Debug) ====
printf '\n=== Building DEBUG libbinding.a for Windows x86_64 ===\n\n'
docker build -f Dockerfile.windows -t zeus-windows-builder .
docker run --rm -e DEBUG=1 \
    -v "$(pwd)/../lib/windows:/output" \
    zeus-windows-builder \
    sh -c "make clean && make libbinding.a CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ AR=x86_64-w64-mingw32-ar && cp *.a /output/"
printf '\n=== Built DEBUG libbinding.a for Windows x86_64 ===\n'
