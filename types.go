package zeus

// KVCacheType represents the data type for KV cache storage.
// Lower precision types use less memory but may reduce quality slightly.
type KVCacheType int

const (
	KVCacheF32  KVCacheType = iota // Full precision (32-bit float)
	KVCacheF16                     // Half precision (16-bit float) - default
	KVCacheQ8_0                    // 8-bit quantized
	KVCacheQ4_0                    // 4-bit quantized - lowest memory
)

// String returns the llama.cpp string representation.
func (k KVCacheType) String() string {
	switch k {
	case KVCacheF32:
		return "f32"
	case KVCacheF16:
		return "f16"
	case KVCacheQ8_0:
		return "q8_0"
	case KVCacheQ4_0:
		return "q4_0"
	default:
		return "f16"
	}
}

// MirostatMode controls the Mirostat adaptive sampling algorithm.
type MirostatMode int

const (
	MirostatDisabled MirostatMode = iota // Standard sampling (top-k, top-p, temperature)
	Mirostat1                            // Mirostat v1 algorithm
	Mirostat2                            // Mirostat v2 algorithm
)

// StopReason indicates why text generation stopped.
type StopReason int

const (
	StopReasonEOS          StopReason = iota // End of sequence token encountered
	StopReasonMaxTokens                      // Reached maximum token limit
	StopReasonStopSequence                   // Matched a stop sequence
	StopReasonCancelled                      // Context was cancelled
	StopReasonError                          // An error occurred
)

func (s StopReason) String() string {
	switch s {
	case StopReasonEOS:
		return "eos"
	case StopReasonMaxTokens:
		return "max_tokens"
	case StopReasonStopSequence:
		return "stop_sequence"
	case StopReasonCancelled:
		return "cancelled"
	case StopReasonError:
		return "error"
	default:
		return "unknown"
	}
}

// Token represents a single generated token.
type Token struct {
	Text string // The token text (may not be valid UTF-8 for partial tokens)
	ID   int    // Token ID from the vocabulary
}

// TokenProb represents a token with its probability.
type TokenProb struct {
	Token int     // Token ID
	Text  string  // Token text
	Prob  float32 // Probability (0-1)
	Logit float32 // Raw logit value
}

// TokenWithLogprobs extends Token with probability information.
type TokenWithLogprobs struct {
	Token             // Embedded - ID, Text
	Prob  float32     // Probability of selected token
	Logit float32     // Logit of selected token
	TopK  []TokenProb // Top-K alternatives (if requested)
}

// GPULayersAll is a constant to offload all model layers to GPU.
// llama.cpp will offload as many layers as fit in available VRAM.
const GPULayersAll = 999
