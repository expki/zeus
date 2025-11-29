package zeus

/*
#include "source/binding.h"
*/
import "C"

// ContextSize returns the effective context window size.
func (m *model) ContextSize() int {
	if m == nil {
		return -1
	}
	if m.isClosed() {
		return -1
	}
	return int(C.binding_get_context_size(m.ptr))
}

// TrainContextSize returns the model's original training context size.
func (m *model) TrainContextSize() int {
	if m == nil {
		return -1
	}
	if m.isClosed() {
		return -1
	}
	return int(C.binding_get_train_context_size(m.ptr))
}

// EmbeddingSize returns the embedding dimension.
func (m *model) EmbeddingSize() int {
	if m == nil {
		return -1
	}
	if m.isClosed() {
		return -1
	}
	return int(C.binding_get_embedding_size(m.ptr))
}

// ModelInfo provides model metadata and architecture details.
type ModelInfo struct {
	Description  string // Model description (e.g., "LLaMA v2 7B Q4_K_M")
	Architecture string // Architecture name from metadata
	QuantType    string // Quantization type (e.g., "Q4_K_M")
	Parameters   uint64 // Total parameter count
	Size         uint64 // Model size in bytes
	Layers       int    // Number of layers
	Heads        int    // Number of attention heads
	HeadsKV      int    // Number of KV heads
	VocabSize    int    // Vocabulary size
}

// Info returns model metadata and architecture details.
func (m *model) Info() ModelInfo {
	if m == nil || m.isClosed() {
		return ModelInfo{}
	}

	info := C.binding_get_model_info(m.ptr)
	return ModelInfo{
		Description:  C.GoString(&info.description[0]),
		Architecture: C.GoString(&info.architecture[0]),
		QuantType:    C.GoString(&info.quant_type[0]),
		Parameters:   uint64(info.parameters),
		Size:         uint64(info.size),
		Layers:       int(info.layers),
		Heads:        int(info.heads),
		HeadsKV:      int(info.heads_kv),
		VocabSize:    int(info.vocab_size),
	}
}
