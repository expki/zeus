# Zeus

[![CI](https://github.com/expki/zeus/actions/workflows/ci.yml/badge.svg)](https://github.com/expki/zeus/actions/workflows/ci.yml)

Go bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp). Run LLMs locally with zero setup.

## What is Zeus?

Zeus brings the power of llama.cpp to Go applications. [llama.cpp](https://github.com/ggerganov/llama.cpp) is a high-performance C++ library for running Large Language Models, known for its efficiency, broad model support, and ability to run on consumer hardware.

Zeus wraps llama.cpp with a clean Go API, handling all the complexity of CGO bindings, memory management, and cross-platform builds. The result is a library that lets you run any GGUF model with just a few lines of Go code.

### Key Features

- **Zero Setup** - Pre-built static libraries included. No compilation, no cmake, no toolchains.
- **Universal Model Support** - Works with any GGUF model: Llama, Mistral, Qwen, Phi, Gemma, and hundreds more.
- **Portable** - x86_64 builds for Linux and Windows that work on CPUs from 2013 onwards.
- **GPU Acceleration** - Vulkan support for GPU inference, with automatic CPU fallback.
- **Sensible Defaults** - Works out of the box. Configure only what you need.
- **Memory Efficient** - KV cache quantization to run larger contexts on limited RAM.
- **Developer Idiocracy** - This library expects no assumed knowlege from developers and it cannot be used incorrectly.

## Quick Start

### Add Library

```bash
go get github.com/expki/zeus
```

### Use Library

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/expki/zeus"
)

func main() {
    model, err := zeus.LoadModel("model.gguf")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()

    chat := model.NewChat()
    chat.AddMessage(zeus.RoleSystem, "You are a helpful assistant.")

    for token, err := range chat.GenerateSequence(context.Background(), "Hello!") {
        if err != nil {
            log.Fatal(err)
        }
        fmt.Print(token.Text)
    }
}
```

### Build App

```bash
CGO_ENABLED=1 go build -o myapp .
```


## Requirements

- Go 1.25+
- x86_64 Linux or Windows
  - Linux: libvulkan1
- Any GGUF model file

## Documentation

- [API Reference](DOC.md) - Complete documentation of all interfaces, methods, and options
- [Contributing](CONTRIBUTING.md) - Building from source and contributing guidelines

## Acknowledgments

This project was inspired by [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp), which pioneered Go bindings for llama.cpp. Zeus builds on that foundation with a focus on simplicity, portability, and pre-built binaries.

## License

Unlicensed
