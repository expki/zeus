# Contributing to Zeus

Thank you for your interest in contributing to Zeus!
This document covers how to build the project from source and submit changes.

## Project Structure

```
zeus/
├── *.go                 # Go API (public interface)
├── lib/
│   ├── linux/          # Pre-built static libraries for Linux x86_64
│   └── windows/        # Pre-built static libraries for Windows x86_64
├── source/
│   ├── binding.cpp/h   # C API wrapper for llama.cpp
│   ├── llama.cpp/      # Upstream llama.cpp (git submodule)
│   ├── Makefile        # Build system for native libraries
│   ├── build.sh        # Docker-based build script
│   ├── Dockerfile.linux    # Linux build environment
│   └── Dockerfile.windows  # Windows cross-compile environment
└── example/
    ├── main.go         # Example application
    └── build.sh        # Docker-based example build
```

### Architecture Layers

```
Go API (*.go)           # Public Go interface with functional options
    ↓
CGO bindings            # #cgo directives in model.go
    ↓
source/binding.cpp/h    # C API wrapper for llama.cpp
    ↓
source/llama.cpp/       # Upstream llama.cpp (git submodule)
```

## Prerequisites

### For Using Zeus (no build required)

- Go 1.25+
- x86_64 Linux or Windows
  - Linux: libvulkan1 or libvulkan-dev

Pre-built static libraries are included in `lib/`. Most users can simply `go get github.com/expki/zeus`.

### For Building from Source

- Linux / Windows WSL x86_64
- Docker

The build uses Docker containers with all necessary toolchains pre-configured:
- **Linux builds**: `gcc:11-bookworm` with CMake and Vulkan SDK
- **Windows cross-compilation**: `llvm-mingw` with CMake and Vulkan headers

## Building from Source

### Clone with Submodules

```bash
git clone --recurse-submodules https://github.com/expki/zeus
cd zeus
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Build Libraries

Run the build script from the source directory:

```bash
cd source
./build.sh
```

This builds static libraries for both Linux and Windows using Docker:

1. Builds `zeus-linux-builder` image from `Dockerfile.linux`
2. Compiles libraries and copies to `lib/linux/`
3. Builds `zeus-windows-builder` image from `Dockerfile.windows`
4. Cross-compiles libraries and copies to `lib/windows/`

### Build Example Executables

```bash
cd example
./build.sh
```

This creates `build/example` (Linux) and `build/example.exe` (Windows).

## Running the Example

```bash
# With pre-built libraries (most common)
go run ./example -m /path/to/model.gguf -p "Hello!"

# With local build
cd example
go run . -m /path/to/model.gguf -p "Hello!"
```

### Example Options

```
-m string    Path to GGUF model file (uses embedded model if not specified)
-t string    Chat template: chatml, gemma, llama2, llama3, vicuna, zephyr (default "chatml")
-c int       Context size (0 = use model's native context)
-n int       Max tokens to generate (0 = till context is full)
-s string    System prompt (default "You are a helpful assistant.")
-p string    Prompt/query to send to the model
-r int       Seed for reproducible output (-1 = random)
-timeout int Generation timeout in seconds (default 60)
-v           Show verbose logs
```

## Code Style

### Go Code

- Use functional options pattern for configuration
- Document exported functions and types
- I don't care about conventions, simply write good code

### C++ Code

- Follow existing code style in `binding.cpp`
- Keep the C API minimal, this is a Golang repository

## Running Tests

```bash
go test ./...
```

## Submitting Changes

1. Fork the repository
2. Create a bugfix (`git checkout -b bugfix/my-bug`) or feature (`git checkout -b feature/my-feature`) branch
3. Make your changes
4. Run tests and formatting
5. Commit with minimal messages
6. Push to your fork
7. Open a Pull Request

### What to Include in PRs

- Whatever your heart desires, if it moves the library forwards I will approve it

## Updating llama.cpp

The `source/llama.cpp` directory is a git submodule. To update:

```bash
cd source/llama.cpp
git fetch origin
git checkout <version-tag>
cd ..
git add llama.cpp
git commit -m "Update llama.cpp to <version>"
```

After updating, rebuild the libraries:

```bash
cd source
./build.sh
```

Test thoroughly - I don't like bugs and llama.cpp interface will change and may require updates to `binding.cpp`.

## Questions/Issues?

Open an issue on GitHub and lets start discussing.
