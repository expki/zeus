#!/bin/sh
set -e
cd "$(dirname "$0")/../" # we need the root source because of the replace github.com/expki/zeus => ../

# ==== Linux ====
printf '=== Building zeus example for Linux x86_64 ===\n\n'
docker build -f ./example/Dockerfile.linux -t zeus-linux-example .
docker run --rm \
    -v "$(pwd)/example/build:/output" \
    zeus-linux-example \
    sh -c "cd /build/example && CGO_ENABLED=1 GOOS=linux GOARCH=amd64 GOAMD64=v3 go build -o /output/example ."
printf '\n=== Built zeus example for Linux x86_64 ===\n'

# ==== Windows ====
printf  '\n=== Building zeus example for Windows x86_64 ===\n\n'
docker build -f ./example/Dockerfile.windows -t zeus-windows-example .
docker run --rm \
    -v "$(pwd)/example/build:/output" \
    zeus-windows-example \
    sh -c "cd /build/example && CGO_ENABLED=1 GOOS=windows GOARCH=amd64 GOAMD64=v3 CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ go build -o /output/example.exe ."
printf  '\n=== Built zeus example for Windows x86_64 ===\n'
