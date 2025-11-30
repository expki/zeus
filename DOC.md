# Zeus API Reference

Complete API documentation for the Zeus Go library.

## Package Overview

```go
import "github.com/expki/zeus"
```

Zeus provides Go bindings for llama.cpp with four core abstractions:
- **Model** - Load and manage GGUF models
- **Session** - Token-level generation with state tracking
- **Chat** - Message-level conversations with template support
- **Tools/Agent** - Function calling and agentic tool execution

---

## Functions

### LoadModel

```go
func LoadModel(path string, opts ...ModelOption) (Model, error)
```

Loads a model from a GGUF file. Returns a Model interface for inference operations.

### SetVerbose

```go
func SetVerbose(verbose bool)
```

Enables or disables verbose logging from llama.cpp.

### Default Configuration Functions

```go
func DefaultModelConfig() ModelConfig
func DefaultGenerateConfig() GenerateConfig
func DefaultChatConfig() ChatConfig
func DefaultChatTemplateConfig() ChatTemplateConfig
func DefaultAgentConfig() AgentConfig
```

Return configuration structs with sensible default values.

---

## Model Interface

The primary interface for interacting with loaded LLM models. Thread-safe for concurrent use.

### Session & Chat Creation

| Method | Returns | Description |
|--------|---------|-------------|
| `NewSession()` | `*Session` | Create a new token-level generation session |
| `NewChat(opts ...ChatOption)` | `Chat` | Create a new message-level chat conversation |

### Text Generation Info

| Method | Returns | Description |
|--------|---------|-------------|
| `ContextSize()` | `int` | Effective context window size |
| `TrainContextSize()` | `int` | Model's original training context size |

### Tokenization

| Method | Returns | Description |
|--------|---------|-------------|
| `Tokenize(text string, addSpecial bool)` | `[]int, error` | Convert text to token IDs |
| `TokenizeCount(text string, addSpecial bool)` | `int, error` | Count tokens without allocating slice |
| `Detokenize(tokens []int)` | `string, error` | Convert token IDs back to text |
| `DetokenizeLength(tokens []int)` | `int, error` | Get output length without allocating |
| `TokenToText(token int)` | `string` | Convert single token ID to text |

### Special Tokens

| Method | Returns | Description |
|--------|---------|-------------|
| `BOS()` | `int` | Beginning-of-sequence token ID |
| `EOS()` | `int` | End-of-sequence token ID |
| `IsSpecialToken(token int)` | `bool` | Check if token is a special/control token |
| `IsEOG(token int)` | `bool` | Check if token is end-of-generation |
| `SpecialTokens()` | `SpecialTokens` | Get all special token IDs |
| `VocabSize()` | `int` | Vocabulary size |

### Embeddings

| Method | Returns | Description |
|--------|---------|-------------|
| `Embeddings(ctx context.Context, text string)` | `[]float32, error` | Extract embeddings for text (requires `WithEmbeddings()`) |
| `EmbeddingsBatch(ctx context.Context, texts []string)` | `[][]float32, error` | Batch embedding extraction |
| `EmbeddingSize()` | `int` | Embedding vector dimension |

### Model Information

| Method | Returns | Description |
|--------|---------|-------------|
| `Info()` | `ModelInfo` | Model metadata and architecture details |
| `ChatTemplate()` | `string` | Model's embedded chat template string |
| `ApplyChatTemplate(messages []ChatMessage, opts ...ChatTemplateOption)` | `string, error` | Format messages using chat template |

### Lifecycle

| Method | Returns | Description |
|--------|---------|-------------|
| `Close()` | `error` | Release model resources (safe to call multiple times) |

---

## Session Interface

Represents a conversation state that tracks token history. Use for lower-level control over generation.

### Generation

| Method | Returns | Description |
|--------|---------|-------------|
| `Generate(ctx context.Context, prompt string, opts ...GenerateOption)` | `io.ReadCloser` | Stream generated text as bytes |
| `GenerateSequence(ctx context.Context, prompt string, opts ...GenerateOption)` | `iter.Seq2[Token, error]` | Stream tokens via iterator |
| `GenerateSequenceWithLogprobs(ctx context.Context, prompt string, topK int, opts ...GenerateOption)` | `iter.Seq2[TokenWithLogprobs, error]` | Stream tokens with probability info |

### State Management

| Method | Returns | Description |
|--------|---------|-------------|
| `Checkpoint()` | `Session` | Create snapshot of current session state |
| `Backtrack()` | `Session, bool` | Return to state before last Generate call |
| `Tokens()` | `[]int` | Copy of token history |
| `Text()` | `string, error` | Full session text (detokenized) |
| `TokenCount()` | `int` | Number of tokens in session |
| `ContextUsed()` | `float64` | Percentage of context window used (0.0-1.0) |
| `Model()` | `Model` | Parent model reference |

---

## Chat Interface

Represents a conversation that tracks message history. Automatically applies chat templates.

### Generation

| Method | Returns | Description |
|--------|---------|-------------|
| `Generate(ctx context.Context, userMessage string, opts ...GenerateOption)` | `io.ReadCloser` | Send message, stream response as bytes |
| `GenerateSequence(ctx context.Context, userMessage string, opts ...GenerateOption)` | `iter.Seq2[Token, error]` | Send message, stream tokens via iterator |
| `GenerateWithTools(ctx context.Context, userMessage string, opts ...GenerateOption)` | `iter.Seq2[AgentEvent, error]` | Agentic loop with automatic tool execution |

### Message Management

| Method | Returns | Description |
|--------|---------|-------------|
| `AddMessage(role Role, content string)` | - | Add message to history without generating |
| `Messages()` | `[]ChatMessage` | Copy of message history |
| `MessageCount()` | `int` | Number of messages in conversation |
| `Compact(ctx context.Context)` | `error` | Summarize older messages to free context space |

### Tool Management

| Method | Returns | Description |
|--------|---------|-------------|
| `Tools()` | `[]Tool` | Registered tools for this chat |

### State Management

| Method | Returns | Description |
|--------|---------|-------------|
| `Checkpoint()` | `Chat` | Create snapshot of current chat state |
| `Backtrack()` | `Chat, bool` | Return to state before last Generate call |
| `Model()` | `Model` | Parent model reference |

---

## Tool Interface

Implement this interface to create custom tools for agentic function calling.

```go
type Tool interface {
    // Name returns the unique identifier for this tool.
    Name() string

    // Description returns a human-readable description of what this tool does.
    // This is provided to the model to help it decide when to use the tool.
    Description() string

    // Parameters returns the list of parameters this tool accepts.
    Parameters() []ToolParameter

    // Execute runs the tool with the given arguments and returns the result.
    // The result should be a string that can be fed back to the model.
    // Return an error if the tool execution fails.
    Execute(ctx context.Context, args map[string]any) (string, error)
}
```

### Example Tool Implementation

```go
type WeatherTool struct{}

func (t *WeatherTool) Name() string { return "get_weather" }

func (t *WeatherTool) Description() string {
    return "Get the current weather for a location"
}

func (t *WeatherTool) Parameters() []zeus.ToolParameter {
    return []zeus.ToolParameter{
        {Name: "location", Type: "string", Description: "City name", Required: true},
        {Name: "units", Type: "string", Description: "Temperature units", Enum: []string{"celsius", "fahrenheit"}},
    }
}

func (t *WeatherTool) Execute(ctx context.Context, args map[string]any) (string, error) {
    location := args["location"].(string)
    // ... fetch weather data
    return fmt.Sprintf("Weather in %s: 22°C, sunny", location), nil
}
```

---

## Model Options

Options for `LoadModel()`. All have sensible defaults.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithContextSize(n)` | `int` | `0` (model native) | Context window size |
| `WithSeed(seed)` | `int` | `0` | Random seed for model initialization |
| `WithBatchSize(n)` | `int` | `512` | Batch size for prompt processing |
| `WithGPULayers(n)` | `int` | `GPULayersAll` | Layers to offload to GPU (0 = CPU only) |
| `WithMainGPU(gpu)` | `int` | `0` | Primary GPU device index |
| `WithTensorSplit(split)` | `[]float32` | `nil` | Distribution of layers across GPUs |
| `WithKVCacheType(t)` | `KVCacheType` | `KVCacheF16` | KV cache precision |
| `WithRopeFreqBase(base)` | `float32` | `0` (from model) | RoPE frequency base |
| `WithRopeFreqScale(scale)` | `float32` | `0` (from model) | RoPE frequency scale |
| `WithLoRA(path)` | `string` | `""` | Path to LoRA adapter file |
| `WithMMap(enable)` | `bool` | `true` | Memory-map model file |
| `WithMlock(enable)` | `bool` | `false` | Lock model in RAM (prevent swapping) |
| `WithNUMA(enable)` | `bool` | `false` | Enable NUMA optimizations |
| `WithEmbeddings()` | - | `false` | Enable embedding extraction mode |

---

## Generate Options

Options for `Generate()` and `GenerateSequence()` methods.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithMaxTokens(n)` | `int` | `0` (unlimited) | Maximum tokens to generate |
| `WithTemperature(t)` | `float32` | `0.8` | Sampling temperature (higher = more random) |
| `WithTopK(k)` | `int` | `40` | Top-K sampling (0 = disabled) |
| `WithTopP(p)` | `float32` | `0.95` | Nucleus sampling probability |
| `WithMinP(p)` | `float32` | `0.05` | Minimum probability threshold |
| `WithRepeatPenalty(p)` | `float32` | `1.1` | Repetition penalty (1.0 = none) |
| `WithRepeatLastN(n)` | `int` | `64` | Tokens to consider for repeat penalty |
| `WithFrequencyPenalty(p)` | `float32` | `0.0` | Frequency-based penalty |
| `WithPresencePenalty(p)` | `float32` | `0.0` | Presence-based penalty |
| `WithMirostat(mode, tau, eta)` | `MirostatMode, float32, float32` | `Disabled, 5.0, 0.1` | Mirostat adaptive sampling |
| `WithStopSequences(seqs...)` | `...string` | `nil` | Stop generation on these sequences |
| `WithIgnoreEOS()` | - | `false` | Continue past end-of-sequence token |
| `WithGrammar(grammar)` | `string` | `""` | GBNF grammar constraint |
| `WithGenerateSeed(seed)` | `int` | `-1` (random) | Random seed for sampling |
| `WithThreads(n)` | `int` | `-1` (auto) | CPU threads for generation |
| `WithReasoningEnabled(enabled)` | `bool` | `true` | Enable model thinking/reasoning output |

---

## Chat Options

Options for `NewChat()`.

### Template Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithChatTemplate(name)` | `string` | `""` (from model) | Built-in template name (chatml, llama3, etc.) |
| `WithAddAssistant(add)` | `bool` | `true` | Append assistant turn prefix |
| `WithAutoCompactThreshold(threshold)` | `float32` | `0.8` | Context usage ratio for auto-compaction (0 = disabled) |

### Tool/Agent Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `WithTools(tools...)` | `...Tool` | `nil` | Register tools for GenerateWithTools |
| `WithMaxIterations(n)` | `int` | `10` | Maximum agentic loop iterations |
| `WithMaxToolCalls(n)` | `int` | `25` | Maximum total tool calls |
| `WithToolTimeout(d)` | `time.Duration` | `30s` | Per-tool execution timeout |
| `WithToolChoice(choice)` | `ToolChoice` | `ToolChoiceAuto` | How the model should use tools |
| `WithParallelToolCalls(parallel)` | `bool` | `true` | Allow multiple tool calls per response |
| `WithChatFormat(format)` | `ChatFormat` | auto-detected | Tool call format for parsing |

---

## Types

### Token Types

```go
// Token represents a single generated token.
type Token struct {
    Text string // Token text (may not be valid UTF-8 for partial tokens)
    ID   int    // Token ID from vocabulary
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
    TopK  []TokenProb // Top-K alternative tokens (if requested)
}
```

### Chat Types

```go
// Role represents the role of a message sender.
type Role string

const (
    RoleSystem    Role = "system"
    RoleUser      Role = "user"
    RoleAssistant Role = "assistant"
    RoleTool      Role = "tool"      // Tool result messages
)

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
    Role    Role
    Content string
}
```

### Tool Types

```go
// ToolParameter describes a single parameter for a tool.
type ToolParameter struct {
    Name        string   // Parameter name
    Type        string   // Type: "string", "number", "boolean", "array", "object"
    Description string   // Human-readable description
    Required    bool     // Whether this parameter is required
    Enum        []string // Optional: allowed values for this parameter
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
    ID        string         // Unique identifier for this call
    Name      string         // Name of the tool to invoke
    Arguments map[string]any // Parsed arguments from the model
}

// ToolResult represents the outcome of executing a tool.
type ToolResult struct {
    CallID  string // Corresponds to ToolCall.ID
    Content string // Result content to feed back to the model
    IsError bool   // Whether this result represents an error
}

// ToolChoice controls how the model should use tools.
type ToolChoice int

const (
    ToolChoiceAuto     ToolChoice = iota // Model decides when to use tools
    ToolChoiceNone                       // Never use tools
    ToolChoiceRequired                   // Must use a tool
)
```

### Agent Event Types

```go
// AgentEventType indicates the type of event during agentic loop execution.
type AgentEventType int

const (
    AgentEventToken         AgentEventType = iota // A token was generated
    AgentEventToolCallStart                       // A tool call is starting
    AgentEventToolCallEnd                         // A tool call completed
    AgentEventError                               // An error occurred
    AgentEventDone                                // The agentic loop completed
)

// AgentEvent represents an event during agentic loop execution.
type AgentEvent struct {
    Type     AgentEventType // Type of event
    Token    *Token         // For AgentEventToken: the generated token
    ToolCall *ToolCall      // For AgentEventToolCallStart/End: the tool call
    Result   *ToolResult    // For AgentEventToolCallEnd: the result
    Error    error          // For AgentEventError: the error that occurred
}
```

### Chat Format

```go
// ChatFormat represents the tool call format for parsing model output.
type ChatFormat int32
```

### Model Metadata

```go
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

// SpecialTokens contains all special token IDs for the model.
type SpecialTokens struct {
    BOS int // Beginning of sequence (-1 if not available)
    EOS int // End of sequence
    EOT int // End of turn
    PAD int // Padding
    SEP int // Separator
    NL  int // Newline
}
```

### KVCacheType

Controls memory usage vs quality tradeoff for the KV cache.

```go
type KVCacheType int

const (
    KVCacheF32  KVCacheType = iota // Full precision (32-bit float)
    KVCacheF16                     // Half precision (16-bit float) - default
    KVCacheQ8_0                    // 8-bit quantized
    KVCacheQ4_0                    // 4-bit quantized
)
```

| Type | Memory/Token | Quality | Use Case |
|------|--------------|---------|----------|
| `KVCacheF32` | 4 bytes | Full precision | Maximum quality |
| `KVCacheF16` | 2 bytes | Excellent | Default, good balance |
| `KVCacheQ8_0` | ~1 byte | Very good | Longer contexts |
| `KVCacheQ4_0` | ~0.5 bytes | Good | Maximum context length |

### MirostatMode

```go
type MirostatMode int

const (
    MirostatDisabled MirostatMode = iota // Standard sampling (top-k, top-p, temperature)
    Mirostat1                            // Mirostat v1 algorithm
    Mirostat2                            // Mirostat v2 algorithm
)
```

### StopReason

```go
type StopReason int

const (
    StopReasonEOS          StopReason = iota // End of sequence token
    StopReasonMaxTokens                      // Token limit reached
    StopReasonStopSequence                   // Stop sequence matched
    StopReasonCancelled                      // Context cancelled
    StopReasonError                          // Error occurred
)
```

### Constants

```go
// GPULayersAll offloads all model layers to GPU.
// llama.cpp will offload as many layers as fit in available VRAM.
const GPULayersAll = 999
```

---

## Error Handling

### Sentinel Errors

Use with `errors.Is()`:

| Error | Description |
|-------|-------------|
| `ErrModelClosed` | Operation attempted on a closed model |
| `ErrEmbeddingsDisabled` | Embeddings requested but model loaded without `WithEmbeddings()` |
| `ErrPromptTooLong` | Prompt exceeds context size |
| `ErrDecodeFailed` | Decode operation failed |
| `ErrSessionIsNil` | Session is nil |
| `ErrModelIsNil` | Model is nil |
| `ErrChatIsNil` | Chat is nil |
| `ErrNoToolsRegistered` | GenerateWithTools called without registered tools |
| `ErrMaxIterationsExceeded` | Maximum agent loop iterations reached |
| `ErrMaxToolCallsExceeded` | Maximum total tool calls reached |
| `ErrTemplateApply` | Failed to apply chat template with tools |
| `ErrToolTemplateUnsupported` | Model does not support native tool templates |

### Typed Errors

Use with `errors.As()` for additional context:

| Error Type | Fields | Description |
|------------|--------|-------------|
| `ModelLoadError` | `Path`, `Reason` | Model loading failure |
| `GenerationError` | `Stage`, `Message` | Generation failure (stage: tokenize, decode, sample) |
| `TokenizeError` | `Text`, `Message` | Tokenization failure |
| `EmbeddingError` | `Message` | Embedding extraction failure |
| `ChatTemplateError` | `Message` | Chat template failure |
| `ToolExecutionError` | `ToolName`, `CallID`, `Err` | Tool execution failure |

Example:

```go
model, err := zeus.LoadModel("model.gguf")
if err != nil {
    var loadErr *zeus.ModelLoadError
    if errors.As(err, &loadErr) {
        fmt.Printf("Failed to load %s: %s\n", loadErr.Path, loadErr.Reason)
    }
}
```

---

## Concurrency

### Thread Safety

- **Model**: Safe for concurrent use from multiple goroutines
- **KV Cache**: Protected by mutex; generation operations are serialized
- **Sessions/Chats**: Multiple can exist simultaneously, but only one generates at a time
- **Close()**: Safe to call multiple times; waits for ongoing generation

### Shared KV Cache

All Sessions and Chats from the same Model share one KV cache. When switching between unrelated sessions, the cache is recomputed from the common prefix. Use `Checkpoint()` to save state before branching.

---

## Examples

### Basic Chat

```go
model, _ := zeus.LoadModel("model.gguf")
defer model.Close()

chat := model.NewChat()
chat.AddMessage(zeus.RoleSystem, "You are a helpful assistant.")

reader := chat.Generate(ctx, "Hello!")
response, _ := io.ReadAll(reader)
fmt.Println(string(response))
```

### Streaming Tokens

```go
for tok, err := range chat.GenerateSequence(ctx, "Tell me a story") {
    if err != nil {
        log.Fatal(err)
    }
    fmt.Print(tok.Text)
}
```

### Agentic Tool Execution

```go
chat := model.NewChat(
    zeus.WithTools(&WeatherTool{}, &CalculatorTool{}),
    zeus.WithMaxIterations(5),
)
chat.AddMessage(zeus.RoleSystem, "You are a helpful assistant with access to tools.")

for event, err := range chat.GenerateWithTools(ctx, "What's the weather in Paris?") {
    if err != nil {
        log.Fatal(err)
    }
    switch event.Type {
    case zeus.AgentEventToken:
        fmt.Print(event.Token.Text)
    case zeus.AgentEventToolCallStart:
        fmt.Printf("\n[Calling %s...]\n", event.ToolCall.Name)
    case zeus.AgentEventToolCallEnd:
        fmt.Printf("[Result: %s]\n", event.Result.Content)
    case zeus.AgentEventDone:
        fmt.Println("\n[Done]")
    }
}
```

### Branching Conversation

```go
session := model.NewSession()

// Generate initial response
for tok, _ := range session.GenerateSequence(ctx, "Hello") {
    fmt.Print(tok.Text)
}

// Save state before branching
checkpoint := session.Checkpoint()

// Try one continuation
for tok, _ := range session.GenerateSequence(ctx, " How are you?") {
    fmt.Print(tok.Text)
}

// Restore and try different continuation
session, _ = checkpoint.Backtrack()
for tok, _ := range session.GenerateSequence(ctx, " What's your name?") {
    fmt.Print(tok.Text)
}
```

### Auto-Compacting Long Conversations

```go
// Chat will automatically summarize older messages when context reaches 80%
chat := model.NewChat(
    zeus.WithAutoCompactThreshold(0.8),
)
```

### Embeddings

```go
model, _ := zeus.LoadModel("model.gguf", zeus.WithEmbeddings())
defer model.Close()

embedding, _ := model.Embeddings(ctx, "Hello world")
fmt.Printf("Embedding dimension: %d\n", len(embedding))

// Batch embeddings
embeddings, _ := model.EmbeddingsBatch(ctx, []string{"Hello", "World"})
```
