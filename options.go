package zeus

/*
#include "source/binding.h"
*/
import "C"

import "time"

func init() {
	SetVerbose(false)
}

// SetVerbose enables or disables verbose logging from llama.cpp.
func SetVerbose(verbose bool) {
	C.binding_set_verbose(C.bool(verbose))
}

// ModelConfig holds configuration for model loading.
type ModelConfig struct {
	ContextSize   int         // Context window size (0 = model's native context)
	Seed          int         // Random seed for model initialization
	BatchSize     int         // Batch size for prompt processing
	GPULayers     int         // Number of layers to offload to GPU (GPULayersAll for all)
	MainGPU       int         // Primary GPU device index for multi-GPU systems
	TensorSplit   []float32   // Distribution of layers across GPUs (e.g., [0.5, 0.5])
	KVCacheType   KVCacheType // Data type for KV cache storage
	RopeFreqBase  float32     // RoPE frequency base (0 = from model)
	RopeFreqScale float32     // RoPE frequency scale (0 = from model)
	LoraAdapter   string      // Path to LoRA adapter file
	UseMMap       bool        // Use memory mapping for model loading
	UseMlock      bool        // Lock model in memory (prevent swapping)
	UseNUMA       bool        // Enable NUMA optimizations
	Embeddings    bool        // Enable embedding extraction mode
}

// DefaultModelConfig returns a ModelConfig with sensible defaults.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		ContextSize:   0, // Use model's native context size
		Seed:          0,
		BatchSize:     512,
		GPULayers:     GPULayersAll,
		MainGPU:       0,
		KVCacheType:   KVCacheF16,
		RopeFreqBase:  0, // From model
		RopeFreqScale: 0, // From model
		UseMMap:       true,
		UseMlock:      false,
		UseNUMA:       false,
		Embeddings:    false,
	}
}

// ModelOption configures model loading.
type ModelOption func(*ModelConfig)

// WithContextSize sets the context window size.
// Use 0 to use the model's native context size.
func WithContextSize(n int) ModelOption {
	return func(c *ModelConfig) { c.ContextSize = n }
}

// WithSeed sets the random seed for model initialization.
func WithSeed(seed int) ModelOption {
	return func(c *ModelConfig) { c.Seed = seed }
}

// WithBatchSize sets the batch size for prompt processing.
func WithBatchSize(n int) ModelOption {
	return func(c *ModelConfig) { c.BatchSize = n }
}

// WithGPULayers sets the number of layers to offload to GPU.
// Use GPULayersAll to offload all layers, 0 for CPU only.
func WithGPULayers(n int) ModelOption {
	return func(c *ModelConfig) { c.GPULayers = n }
}

// WithMainGPU sets the primary GPU device index for multi-GPU systems.
func WithMainGPU(gpu int) ModelOption {
	return func(c *ModelConfig) { c.MainGPU = gpu }
}

// WithTensorSplit sets the distribution of layers across multiple GPUs.
// []float32{0.5, 0.5} splits evenly between two GPUs.
func WithTensorSplit(split []float32) ModelOption {
	return func(c *ModelConfig) { c.TensorSplit = split }
}

// WithKVCacheType sets the data type for KV cache storage.
func WithKVCacheType(t KVCacheType) ModelOption {
	return func(c *ModelConfig) { c.KVCacheType = t }
}

// WithRopeFreqBase sets the RoPE frequency base.
// Use 0 to use the model's default value.
func WithRopeFreqBase(base float32) ModelOption {
	return func(c *ModelConfig) { c.RopeFreqBase = base }
}

// WithRopeFreqScale sets the RoPE frequency scale.
// Use 0 to use the model's default value.
func WithRopeFreqScale(scale float32) ModelOption {
	return func(c *ModelConfig) { c.RopeFreqScale = scale }
}

// WithLoRA loads a LoRA adapter from the specified path.
func WithLoRA(path string) ModelOption {
	return func(c *ModelConfig) { c.LoraAdapter = path }
}

// WithMMap enables or disables memory mapping for model loading.
// Disabling forces loading the entire model into memory, might improve preformance.
func WithMMap(enable bool) ModelOption {
	return func(c *ModelConfig) { c.UseMMap = enable }
}

// WithMlock enables or disables memory locking.
// When enabled, the model is locked in RAM to prevent swapping.
func WithMlock(enable bool) ModelOption {
	return func(c *ModelConfig) { c.UseMlock = enable }
}

// WithNUMA enables or disables NUMA optimizations.
func WithNUMA(enable bool) ModelOption {
	return func(c *ModelConfig) { c.UseNUMA = enable }
}

// WithEmbeddings enables embedding extraction mode.
// Required to use the Embeddings() method.
func WithEmbeddings() ModelOption {
	return func(c *ModelConfig) { c.Embeddings = true }
}

// GenerateConfig holds configuration for text generation.
type GenerateConfig struct {
	MaxTokens        int          // Maximum tokens to generate (0 = unlimited)
	Temperature      float32      // Sampling temperature (higher = more random)
	TopK             int          // Top-K sampling (0 = disabled)
	TopP             float32      // Nucleus sampling probability
	MinP             float32      // Minimum probability threshold
	RepeatPenalty    float32      // Repetition penalty (1.0 = no penalty)
	RepeatLastN      int          // Number of tokens to consider for repetition penalty
	FrequencyPenalty float32      // Frequency-based penalty
	PresencePenalty  float32      // Presence-based penalty
	Mirostat         MirostatMode // Mirostat sampling mode
	MirostatTau      float32      // Mirostat target entropy
	MirostatEta      float32      // Mirostat learning rate
	StopSequences    []string     // Sequences that stop generation
	IgnoreEOS        bool         // Continue past end-of-sequence token
	Grammar          string       // GBNF grammar to constrain output
	Seed             int          // Random seed for sampling (-1 = random)
	Threads          int          // Number of threads for generation (0 = autodetect)
	ReasoningEnabled bool         // Enable model thinking/reasoning (default: true)
}

// DefaultGenerateConfig returns a GenerateConfig with sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:        0,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.95,
		MinP:             0.05,
		RepeatPenalty:    1.1,
		RepeatLastN:      64,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		Mirostat:         MirostatDisabled,
		MirostatTau:      5.0,
		MirostatEta:      0.1,
		IgnoreEOS:        false,
		Seed:             -1,
		Threads:          -1,
		ReasoningEnabled: true,
	}
}

// GenerateOption configures text generation.
type GenerateOption func(*GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
// Use 0 for unlimited (generates until EOS or context full).
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature.
// Higher values (e.g., 1.0) make output more random.
// Lower values (e.g., 0.2) make output more deterministic.
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK sets the top-K sampling value.
// Only the K most likely tokens are considered. Use 0 to disable.
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets the nucleus sampling probability.
// Tokens are sampled from the smallest set whose cumulative probability exceeds P.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithMinP sets the minimum probability threshold.
// Tokens with probability below this are excluded.
func WithMinP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.MinP = p }
}

// WithRepeatPenalty sets the repetition penalty.
// Values > 1.0 discourage repetition, 1.0 = no penalty.
func WithRepeatPenalty(penalty float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = penalty }
}

// WithRepeatLastN sets how many recent tokens to consider for repetition penalty.
func WithRepeatLastN(n int) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatLastN = n }
}

// WithFrequencyPenalty sets the frequency-based penalty.
// Penalizes tokens based on their frequency in the generated text.
func WithFrequencyPenalty(penalty float32) GenerateOption {
	return func(c *GenerateConfig) { c.FrequencyPenalty = penalty }
}

// WithPresencePenalty sets the presence-based penalty.
// Penalizes tokens that have already appeared in the generated text.
func WithPresencePenalty(penalty float32) GenerateOption {
	return func(c *GenerateConfig) { c.PresencePenalty = penalty }
}

// WithMirostat enables Mirostat adaptive sampling.
func WithMirostat(mode MirostatMode, tau, eta float32) GenerateOption {
	return func(c *GenerateConfig) {
		c.Mirostat = mode
		c.MirostatTau = tau
		c.MirostatEta = eta
	}
}

// WithStopSequences sets sequences that will stop generation when encountered.
func WithStopSequences(seqs ...string) GenerateOption {
	return func(c *GenerateConfig) { c.StopSequences = seqs }
}

// WithIgnoreEOS enables generation past the end-of-sequence token.
func WithIgnoreEOS() GenerateOption {
	return func(c *GenerateConfig) { c.IgnoreEOS = true }
}

// WithGrammar constrains output to match a GBNF grammar.
func WithGrammar(grammar string) GenerateOption {
	return func(c *GenerateConfig) { c.Grammar = grammar }
}

// WithGenerateSeed sets the random seed for sampling.
// Use -1 for random seed.
func WithGenerateSeed(seed int) GenerateOption {
	return func(c *GenerateConfig) { c.Seed = seed }
}

// WithThreads sets the number of threads for generation.
// Default value is -1, which autodetect the amount of cores
func WithThreads(n int) GenerateOption {
	return func(c *GenerateConfig) { c.Threads = n }
}

// WithReasoningEnabled enables or disables model thinking/reasoning.
// When disabled, thinking tags are closed immediately to prevent reasoning output.
// This is equivalent to llama.cpp server's --reasoning-budget 0.
func WithReasoningEnabled(enabled bool) GenerateOption {
	return func(c *GenerateConfig) { c.ReasoningEnabled = enabled }
}

// ChatTemplateConfig holds options for ApplyChatTemplate.
type ChatTemplateConfig struct {
	Template             string  // Empty = use model's embedded template
	AddAssistant         bool    // Add assistant turn prefix (default: true)
	AutoCompactThreshold float32 // Ratio after which automatic compact occurs (Chat only)
}

// ChatTemplateOption is a functional option for ApplyChatTemplate.
type ChatTemplateOption func(*ChatTemplateConfig)

// DefaultChatTemplateConfig returns the default chat template configuration.
func DefaultChatTemplateConfig() ChatTemplateConfig {
	return ChatTemplateConfig{
		Template:             "",
		AddAssistant:         true,
		AutoCompactThreshold: 0.8,
	}
}

// WithChatTemplate specifies a built-in template name (e.g., "chatml", "llama3").
func WithChatTemplate(name string) ChatTemplateOption {
	return func(c *ChatTemplateConfig) { c.Template = name }
}

// WithAddAssistant controls whether to append the assistant turn prefix.
func WithAddAssistant(add bool) ChatTemplateOption {
	return func(c *ChatTemplateConfig) { c.AddAssistant = add }
}

// WithAutoCompactThreshold sets the context usage ratio at which Chat automatically
// compacts the conversation. When context usage exceeds this threshold (e.g., 0.8 = 80%),
// older messages are summarized to free space. Use 0 to disable auto-compaction.
func WithAutoCompactThreshold(threshold float32) ChatTemplateOption {
	return func(c *ChatTemplateConfig) { c.AutoCompactThreshold = threshold }
}

// WithTools registers tools for the chat to use with GenerateWithTools.
func WithTools(tools ...Tool) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.Tools = tools }
}

// WithMaxIterations sets the maximum number of agentic loop iterations.
// Each iteration may contain multiple tool calls. Default is 10.
func WithMaxIterations(n int) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.MaxIterations = n }
}

// WithMaxToolCalls sets the maximum total tool calls across all iterations.
// Default is 25.
func WithMaxToolCalls(n int) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.MaxToolCalls = n }
}

// WithToolTimeout sets the per-tool execution timeout. Default is 30 seconds.
func WithToolTimeout(d time.Duration) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.ToolTimeout = d }
}

// WithChatFormat sets the tool call format for parsing model output.
// Different models use different formats. Normally this is auto-detected by llama.cpp
// when applying tool templates, so you typically don't need to set this explicitly.
func WithChatFormat(format ChatFormat) ChatOption {
	return func(c *ChatConfig) { c.ChatFormat = format }
}

// WithToolChoice sets how the model should use tools.
//   - ToolChoiceAuto (default): Model decides when to use tools
//   - ToolChoiceNone: Never use tools
//   - ToolChoiceRequired: Must use a tool
func WithToolChoice(choice ToolChoice) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.ToolChoice = choice }
}

// WithParallelToolCalls controls whether the model can make multiple tool calls
// in a single response. Default is true.
func WithParallelToolCalls(parallel bool) ChatOption {
	return func(c *ChatConfig) { c.AgentConfig.ParallelToolCalls = parallel }
}
