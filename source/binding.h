#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// ============================================================================
// Result Codes
// ============================================================================

typedef enum {
    BINDING_OK = 0,
    BINDING_DONE = 1,                   // Generation complete (not an error)
    BINDING_ERR_INVALID_ARGS = -1,
    BINDING_ERR_MODEL_LOAD = -2,
    BINDING_ERR_CONTEXT_CREATE = -3,
    BINDING_ERR_DECODE = -4,
    BINDING_ERR_PROMPT_TOO_LONG = -5,
    BINDING_ERR_EMBEDDINGS_DISABLED = -6,
    BINDING_ERR_TOKENIZE = -7,
} binding_result;

// Token returned by step-based generation
typedef struct {
    const char *text;   // Token text (valid until next call or free)
    int32_t id;         // Token ID
} binding_token;

// Token probability for logprobs
typedef struct {
    int32_t token_id;
    float prob;         // Probability (0-1)
    float logit;        // Raw logit value
} binding_token_prob;

// Token with logprobs information
typedef struct {
    const char *text;   // Token text (valid until next call or free)
    int32_t id;         // Token ID
    float prob;         // Probability of selected token
    float logit;        // Logit of selected token
    binding_token_prob *top_k;  // Array of top-k alternatives (caller must free)
    int32_t top_k_count;        // Number of top-k alternatives
} binding_token_with_logprobs;

// Cache preparation result
typedef struct {
    int32_t common_prefix;      // Number of tokens matching cached state
    int32_t tokens_to_process;  // Number of new tokens to process
    binding_result status;      // BINDING_OK or error code
} binding_cache_prepare_result;

// ============================================================================
// Configuration Structures
// ============================================================================

// Model loading configuration
typedef struct {
    int32_t context_size;       // 0 = from model
    int32_t batch_size;         // default 512
    int32_t seed;               // random seed
    int32_t gpu_layers;         // 0 = CPU only, 999 = all
    int32_t main_gpu;           // main GPU index
    int32_t kv_cache_type;      // 0=f32, 1=f16, 2=q8_0, 3=q4_0
    float rope_freq_base;       // 0 = from model
    float rope_freq_scale;      // 0 = from model
    bool use_mmap;
    bool use_mlock;
    bool use_numa;
    bool embeddings;
    const char *lora_path;      // NULL for none
    const float *tensor_split;  // NULL for default
    int32_t tensor_split_count;
} binding_model_config;

// Text generation configuration
typedef struct {
    int32_t max_tokens;         // 0 = unlimited
    int32_t seed;               // -1 = random
    int32_t threads;
    int32_t top_k;
    float top_p;
    float min_p;
    float temperature;
    float repeat_penalty;
    int32_t repeat_last_n;
    float frequency_penalty;
    float presence_penalty;
    int32_t mirostat;           // 0=disabled, 1=v1, 2=v2
    float mirostat_tau;
    float mirostat_eta;
    bool ignore_eos;
    const char *grammar;        // NULL for none
    const char **stop_sequences;
    int32_t stop_count;
    bool reasoning_enabled;     // Enable model thinking/reasoning
} binding_generate_config;

// ============================================================================
// Logging
// ============================================================================

// Enable or disable verbose logging
void binding_set_verbose(bool verbose);

// ============================================================================
// Default Configurations
// ============================================================================

binding_model_config binding_model_config_default(void);
binding_generate_config binding_generate_config_default(void);

// ============================================================================
// Model Operations
// ============================================================================

// Load a model from file
// Returns NULL on failure
void *binding_load_model(const char *path, const binding_model_config *config);

// Free model resources
void binding_free_model(void *model);

// Get model properties
int32_t binding_get_context_size(void *model);
int32_t binding_get_train_context_size(void *model);
int32_t binding_get_embedding_size(void *model);

// ============================================================================
// KV Cache Management
// ============================================================================

// Prepare KV cache for a token sequence.
// Compares provided tokens with current cache state:
// - If tokens share a prefix with cache, reuses that portion
// - If tokens diverge, truncates cache at divergence point
// - Returns info about common prefix and tokens needing processing
binding_cache_prepare_result binding_prepare_cache(
    void *model,
    const int32_t *tokens,
    int32_t token_count
);

// Get number of tokens currently in KV cache
int32_t binding_get_cache_token_count(void *model);

// ============================================================================
// Text Generation
// ============================================================================

// Start a new generation session from pre-tokenized input.
// Caller should call binding_prepare_cache first to set up cache state.
// tokens are the NEW tokens to process (after the cached prefix).
// Returns an opaque handle, or NULL on failure
void *binding_generate_start(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    const binding_generate_config *config
);

// Generate the next token
// Returns BINDING_OK with token filled, BINDING_DONE when complete, or error
binding_result binding_generate_next(void *generation, binding_token *out_token);

// Generate the next token with logprobs
// top_k: number of top alternatives to return (0 = just selected token)
// Returns BINDING_OK with token filled, BINDING_DONE when complete, or error
// Caller must free out_token->top_k if top_k_count > 0
binding_result binding_generate_next_with_logprobs(
    void *generation,
    binding_token_with_logprobs *out_token,
    int32_t top_k
);

// Free generation resources
void binding_generate_free(void *generation);

// ============================================================================
// Embeddings
// ============================================================================

// Get embeddings for text
// out_embeddings must be pre-allocated
// out_size receives the embedding dimension
binding_result binding_get_embeddings(
    void *model,
    const char *text,
    float *out_embeddings,
    int32_t *out_size
);

// ============================================================================
// Tokenization
// ============================================================================

// Get the number of tokens for text without tokenizing
binding_result binding_tokenize_count(
    void *model,
    const char *text,
    bool add_special,
    int32_t *out_count
);

// Tokenize text into pre-allocated buffer (use binding_tokenize_count first)
binding_result binding_tokenize(
    void *model,
    const char *text,
    bool add_special,
    int32_t *out_tokens
);

// Get the length of detokenized text without detokenizing
binding_result binding_detokenize_length(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    int32_t *out_length
);

// Detokenize tokens into pre-allocated buffer (use binding_detokenize_length first)
binding_result binding_detokenize(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    char *out_text
);

// ============================================================================
// Token Utilities
// ============================================================================

// Special tokens structure
typedef struct {
    int32_t bos;    // Beginning of sequence
    int32_t eos;    // End of sequence
    int32_t eot;    // End of turn
    int32_t pad;    // Padding
    int32_t sep;    // Separator
    int32_t nl;     // Newline
} binding_special_tokens;

// Convert a single token ID to its text representation
// Returns number of bytes written, or negative on error
int32_t binding_token_to_text(void *model, int32_t token, char *buf, int32_t buf_size);

// Check if token is a special/control token
bool binding_is_special_token(void *model, int32_t token);

// Check if token is an end-of-generation token
bool binding_is_eog_token(void *model, int32_t token);

// Get all special token IDs
binding_special_tokens binding_get_special_tokens(void *model);

// Get vocabulary size
int32_t binding_vocab_size(void *model);

// ============================================================================
// Model Information
// ============================================================================

// Model information structure
typedef struct {
    char description[256];
    char architecture[64];
    char quant_type[32];
    uint64_t parameters;
    uint64_t size;
    int32_t layers;
    int32_t heads;
    int32_t heads_kv;
    int32_t vocab_size;
} binding_model_info;

// Get model information
binding_model_info binding_get_model_info(void *model);

// ============================================================================
// Batch Embeddings
// ============================================================================

// Get embeddings for multiple texts
// out_embeddings must be pre-allocated with text_count * embedding_size floats
// Returns BINDING_OK on success
binding_result binding_get_embeddings_batch(
    void *model,
    const char **texts,
    int32_t text_count,
    float *out_embeddings,
    int32_t *out_size
);

// ============================================================================
// Chat Templates
// ============================================================================

// Tool call in an assistant message
typedef struct {
    const char *name;       // Tool function name
    const char *arguments;  // JSON string of arguments
    const char *id;         // Tool call ID
} binding_chat_tool_call;

// Chat message for template formatting
typedef struct {
    const char *role;
    const char *content;
    // For assistant messages: tool calls made
    const binding_chat_tool_call *tool_calls;
    int32_t tool_call_count;
    // For tool result messages: which tool call this is a response to
    const char *tool_name;
    const char *tool_call_id;
} binding_chat_message;

// Get the model's chat template (returns NULL if not embedded)
const char *binding_get_chat_template(void *model);

// Get the required buffer size for applying a chat template
// Returns required buffer size, or negative on error
int32_t binding_apply_chat_template_length(
    void *model,
    const char *tmpl,
    const binding_chat_message *messages,
    int32_t message_count,
    bool add_assistant
);

// Apply chat template to format messages into pre-allocated buffer
// Use binding_apply_chat_template_length first to get required size
// Returns number of bytes written, or negative on error
int32_t binding_apply_chat_template(
    void *model,
    const char *tmpl,
    const binding_chat_message *messages,
    int32_t message_count,
    bool add_assistant,
    char *buf,
    int32_t buf_size
);

// ============================================================================
// Tool Call Parsing
// ============================================================================

// Chat format is passed as int32_t for forward compatibility with llama.cpp.
// Values correspond to common_chat_format enum in llama.cpp.
// Use binding_chat_format_name() to get a human-readable name.

// Parsed tool call structure
typedef struct {
    const char* name;       // Tool function name
    const char* arguments;  // JSON string of arguments
    const char* id;         // Tool call ID (format-specific)
} binding_tool_call;

// Parse result structure
typedef struct {
    const char* content;            // Non-tool-call text content
    const char* reasoning_content;  // Reasoning/thinking content (if any)
    binding_tool_call* tool_calls;  // Array of tool calls
    int32_t tool_call_count;        // Number of tool calls
    bool success;                   // Whether parsing succeeded
} binding_parse_result;

// Parse tool calls from model output
// response: the raw model output
// format: the chat format to use for parsing (int32_t for forward compatibility)
// is_partial: true if response may be incomplete (streaming)
// Returns a parse result that must be freed with binding_free_parse_result
binding_parse_result* binding_parse_tool_calls(
    const char* response,
    int32_t format,
    bool is_partial
);

// Free parse result
void binding_free_parse_result(binding_parse_result* result);

// Get format name for debugging
const char* binding_chat_format_name(int32_t format);

// ============================================================================
// Tool Template Application
// ============================================================================

// Tool definition structure (matches common_chat_tool)
typedef struct {
    const char* name;        // Tool function name
    const char* description; // Human-readable description
    const char* parameters;  // JSON schema string for parameters
} binding_chat_tool_def;

// Tool choice enum
typedef enum {
    BINDING_TOOL_CHOICE_AUTO = 0,     // Model decides when to use tools
    BINDING_TOOL_CHOICE_NONE,         // Never use tools
    BINDING_TOOL_CHOICE_REQUIRED,     // Must use a tool
} binding_tool_choice;

// Template application result (matches common_chat_params)
typedef struct {
    const char* prompt;              // Formatted prompt with tools
    const char* grammar;             // GBNF grammar for output (may be empty)
    int32_t format;                  // Detected format (forward compatible with llama.cpp)
    bool grammar_lazy;               // Apply grammar only after trigger
    const char** grammar_triggers;   // Trigger patterns (null-terminated array)
    int32_t trigger_count;           // Number of triggers
    const char** additional_stops;   // Extra stop sequences (null-terminated array)
    int32_t stop_count;              // Number of stops
} binding_chat_params;

// Apply chat template with tools
// model: the model handle (contains embedded chat template)
// messages: array of chat messages
// message_count: number of messages
// tools: array of tool definitions
// tool_count: number of tools
// tool_choice: how tools should be used
// parallel_tool_calls: allow multiple tool calls
// add_generation_prompt: add assistant turn prefix
// Returns params that must be freed with binding_free_chat_params
binding_chat_params* binding_apply_chat_template_with_tools(
    void* model,
    const binding_chat_message* messages,
    int32_t message_count,
    const binding_chat_tool_def* tools,
    int32_t tool_count,
    binding_tool_choice tool_choice,
    bool parallel_tool_calls,
    bool add_generation_prompt
);

// Free chat params
void binding_free_chat_params(binding_chat_params* params);

#ifdef __cplusplus
}
#endif

#endif // BINDING_H
