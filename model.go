package zeus

/*
#cgo CXXFLAGS: -I${SRCDIR}/source/llama.cpp/include -I${SRCDIR}/source/llama.cpp/ggml/include -I${SRCDIR}/source/llama.cpp/common -I${SRCDIR}/source
#cgo linux LDFLAGS: -L${SRCDIR}/lib/linux/ -lbinding -lcommon -lllama -lggml -lggml-vulkan -lggml-cpu -lggml-base -lvulkan -lm -lstdc++ -static-libgcc -static-libstdc++
#cgo windows LDFLAGS: -L${SRCDIR}/lib/windows/ -lbinding -lcommon -lllama -lggml -lggml-vulkan -lggml-cpu -lggml-base -lvulkan-1 -lstdc++ -lwinpthread -static
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"runtime"
	"sync"
	"unsafe"
)

// Model represents a loaded LLM model.
type Model interface {
	// NewSession creates a new empty session for text generation.
	NewSession() *session

	// NewChat creates a new empty chat for conversation.
	// Uses the model's embedded template by default.
	NewChat(opts ...ChatOption) Chat

	// Embeddings extracts embeddings for the given text.
	// The model must be loaded with WithEmbeddings() option.
	Embeddings(ctx context.Context, text string) ([]float32, error)

	// EmbeddingsBatch extracts embeddings for multiple texts in a single call.
	// The model must be loaded with WithEmbeddings() option.
	EmbeddingsBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Tokenize converts text to token IDs.
	Tokenize(text string, addSpecial bool) ([]int, error)

	// TokenizeCount returns number of token IDs the text represents.
	TokenizeCount(text string, addSpecial bool) (int, error)

	// Detokenize converts token IDs back to text.
	Detokenize(tokens []int) (string, error)

	// DetokenizeLength returns string length the token IDs represents.
	DetokenizeLength(tokens []int) (int, error)

	// BOS returns the beginning-of-sequence token ID.
	BOS() int

	// EOS returns the end-of-sequence token ID.
	EOS() int

	// TokenToText converts a single token ID to its text representation.
	TokenToText(token int) string

	// IsSpecialToken returns true if the token is a special/control token.
	IsSpecialToken(token int) bool

	// IsEOG returns true if the token is an end-of-generation token.
	IsEOG(token int) bool

	// SpecialTokens returns all special token IDs.
	SpecialTokens() SpecialTokens

	// VocabSize returns the vocabulary size.
	VocabSize() int

	// ContextSize returns the effective context window size.
	ContextSize() int

	// TrainContextSize returns the model's original training context size.
	TrainContextSize() int

	// EmbeddingSize returns the embedding dimension.
	EmbeddingSize() int

	// Info returns model metadata and architecture details.
	Info() ModelInfo

	// ChatTemplate returns the model's embedded chat template string.
	// Returns empty string if no template is embedded.
	ChatTemplate() string

	// ApplyChatTemplate formats messages using a chat template.
	// Uses model's embedded template by default.
	ApplyChatTemplate(messages []ChatMessage, opts ...ChatTemplateOption) (string, error)

	// Close releases model resources.
	Close() error
}

// model is a loaded LLM model.
// it is safe for concurrent use from multiple goroutines.
type model struct {
	ptr     unsafe.Pointer
	config  ModelConfig
	kvMutex sync.Mutex // protects generation (exclusive access to KV cache)

	closed     bool
	closedLock sync.RWMutex
}

// LoadModel loads a model from a GGUF file.
func LoadModel(path string, opts ...ModelOption) (Model, error) {
	cfg := DefaultModelConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Create C config
	cConfig := C.binding_model_config_default()
	cConfig.context_size = C.int32_t(cfg.ContextSize)
	cConfig.batch_size = C.int32_t(cfg.BatchSize)
	cConfig.seed = C.int32_t(cfg.Seed)
	cConfig.gpu_layers = C.int32_t(cfg.GPULayers)
	cConfig.main_gpu = C.int32_t(cfg.MainGPU)
	cConfig.kv_cache_type = C.int32_t(cfg.KVCacheType)
	cConfig.rope_freq_base = C.float(cfg.RopeFreqBase)
	cConfig.rope_freq_scale = C.float(cfg.RopeFreqScale)
	cConfig.use_mmap = C.bool(cfg.UseMMap)
	cConfig.use_mlock = C.bool(cfg.UseMlock)
	cConfig.use_numa = C.bool(cfg.UseNUMA)
	cConfig.embeddings = C.bool(cfg.Embeddings)

	// Handle LoRA path
	var cLoraPath *C.char
	if cfg.LoraAdapter != "" {
		cLoraPath = C.CString(cfg.LoraAdapter)
		defer C.free(unsafe.Pointer(cLoraPath))
		cConfig.lora_path = cLoraPath
	}

	// Handle tensor split
	var tensorSplitPtr *C.float
	if len(cfg.TensorSplit) > 0 {
		tensorSplitPtr = (*C.float)(C.malloc(C.size_t(len(cfg.TensorSplit)) * C.size_t(unsafe.Sizeof(C.float(0)))))
		defer C.free(unsafe.Pointer(tensorSplitPtr))
		for i, v := range cfg.TensorSplit {
			*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(tensorSplitPtr)) + uintptr(i)*unsafe.Sizeof(C.float(0)))) = C.float(v)
		}
		cConfig.tensor_split = tensorSplitPtr
		cConfig.tensor_split_count = C.int32_t(len(cfg.TensorSplit))
	}

	// Load model
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.binding_load_model(cPath, &cConfig)
	if ptr == nil {
		return nil, &ModelLoadError{Path: path, Reason: "failed to load model"}
	}

	m := &model{
		ptr:    ptr,
		config: cfg,
	}

	// Register finalizer as safety net if developer forgets defer model.Close()
	runtime.SetFinalizer(m, (*model).Close)

	// Spawn background warmup if enabled
	if cfg.Warmup {
		go m.warmup()
	}

	return m, nil
}

// Close releases model resources.
// It is safe to call Close multiple times.
func (m *model) Close() error {
	if m == nil {
		return nil
	}

	// If already closed, return early
	m.closedLock.RLock()
	if m.closed {
		m.closedLock.RUnlock()
		return nil
	}
	m.closedLock.RUnlock()

	// Wait for any ongoing generation to complete
	m.kvMutex.Lock()
	defer m.kvMutex.Unlock()
	m.closedLock.Lock()
	defer m.closedLock.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true

	runtime.SetFinalizer(m, nil)

	if m.ptr != nil {
		C.binding_free_model(m.ptr)
		m.ptr = nil
	}

	return nil
}

func (m *model) isClosed() bool {
	if m == nil {
		return true
	}
	m.closedLock.RLock()
	defer m.closedLock.RUnlock()
	return m.closed
}

// warmup runs a minimal decode to initialize GPU kernels.
func (m *model) warmup() {
	if m == nil || m.isClosed() {
		return
	}

	m.kvMutex.Lock()
	defer m.kvMutex.Unlock()

	C.binding_warmup(m.ptr)
}
