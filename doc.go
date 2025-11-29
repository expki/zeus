// Package zeus provides Go bindings for llama.cpp, enabling local LLM inference
// with pre-built static libraries for Linux and Windows x86_64.
//
// Zeus is designed for simplicity - load any GGUF model and start generating
// text with sensible defaults. No compilation required, no external dependencies.
//
// # Quick Start
//
// The simplest way to use Zeus is with the Chat API:
//
//	model, err := zeus.LoadModel("model.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer model.Close()
//
//	chat := model.NewChat()
//	chat.AddMessage(zeus.RoleSystem, "You are a helpful assistant.")
//
//	for tok, err := range chat.GenerateSequence(ctx, "Hello!") {
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    fmt.Print(tok.Text)
//	}
//
// # Core Abstractions
//
// Zeus provides three main abstractions for different use cases:
//
//   - [Model]: Load and manage GGUF models. Provides tokenization, embeddings,
//     and model information. Thread-safe for concurrent use.
//
//   - [Session]: Token-level generation with state tracking. Use when you need
//     precise control over the prompt format or are working with non-chat models.
//     Supports checkpoint/backtrack for branching conversations.
//
//   - [Chat]: Message-level conversations with automatic template handling.
//     Manages conversation history and applies the model's chat template.
//     Best for chatbot-style applications.
//
// # Configuration
//
// Zeus uses Go's functional options pattern for clean, readable configuration:
//
//	model, err := zeus.LoadModel("model.gguf",
//	    zeus.WithContextSize(4096),
//	    zeus.WithKVCacheType(zeus.KVCacheQ8_0),
//	)
//
//	for tok, err := range session.GenerateSequence(ctx, prompt,
//	    zeus.WithMaxTokens(512),
//	    zeus.WithTemperature(0.7),
//	) {
//	    // ...
//	}
//
// Most options have sensible defaults. You only need to configure what you
// want to change.
//
// # Streaming Generation
//
// Zeus provides two streaming interfaces:
//
//   - [iter.Seq2] via GenerateSequence: Returns tokens one at a time using
//     Go 1.23+ range-over-func. Best for token-by-token processing.
//
//   - [io.ReadCloser] via Generate: Returns generated text as a byte stream.
//     Best for piping to HTTP responses or other io.Writers.
//
// # GPU Acceleration
//
// Zeus includes pre-built Vulkan support for GPU acceleration. Enable it
// with [WithGPULayers]:
//
//	model, err := zeus.LoadModel("model.gguf",
//	    zeus.WithGPULayers(zeus.GPULayersAll),
//	)
//
// GPU acceleration is optional - Zeus falls back to CPU if Vulkan is unavailable.
//
// # Thread Safety
//
// Model is safe for concurrent use from multiple goroutines. The KV cache is
// protected by a mutex, so generation operations are serialized. Multiple
// Sessions or Chats can exist simultaneously, but only one can generate at
// a time.
//
// [Close] is safe to call multiple times and will wait for any ongoing
// generation to complete.
//
// # Error Handling
//
// Zeus provides sentinel errors for common conditions that can be checked
// with [errors.Is]:
//
//   - [ErrModelClosed]: Operation attempted on a closed model
//   - [ErrEmbeddingsDisabled]: Embeddings requested but model loaded without [WithEmbeddings]
//   - [ErrPromptTooLong]: Prompt exceeds context size
//   - [ErrDecodeFailed]: Decode operation failed during generation
//
// Typed errors provide additional context and can be checked with [errors.As]:
//
//   - [ModelLoadError]: Details about model loading failures
//   - [GenerationError]: Details about generation failures
//   - [TokenizeError]: Details about tokenization failures
//
// For complete API documentation, see DOC.md in the repository.
package zeus
