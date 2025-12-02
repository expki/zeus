// binding.cpp - Go bindings for llama.cpp

#include "llama.h"
#include "binding.h"
#include "chat.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ============================================================================
// Internal Structures
// ============================================================================

struct llama_binding_state {
    llama_model *model;
    llama_context *ctx;
    const llama_vocab *vocab;
    bool embeddings_enabled;
    int n_embd;

    // KV cache tracking for prefix caching
    std::vector<llama_token> cached_tokens;
    int cached_n_past;
};

// Generation state for step-based API
struct llama_generation_state {
    llama_binding_state *model_state;
    llama_sampler *sampler;
    binding_generate_config config;

    std::vector<llama_token> prompt_tokens;
    std::vector<std::string> stop_sequences;
    std::string generated_text;
    std::string current_token_text;  // Buffer for returning token text

    int n_past;
    int n_consumed;
    int n_remain;
    bool finished;

    // Thinking suppression: tokens to inject when closing thinking block
    std::vector<llama_token> pending_inject_tokens;
    size_t inject_index;
};

// ============================================================================
// Logging
// ============================================================================

static bool g_verbose = false;

static void llama_log_callback_null(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void binding_set_verbose(bool verbose) {
    g_verbose = verbose;
    if (!verbose) {
        llama_log_set(llama_log_callback_null, nullptr);
    } else {
        llama_log_set(nullptr, nullptr);
    }
}

// ============================================================================
// Default Configurations
// ============================================================================

binding_model_config binding_model_config_default(void) {
    binding_model_config config;
    config.context_size = 0;
    config.batch_size = 512;
    config.seed = 0;
    config.gpu_layers = 999;
    config.main_gpu = 0;
    config.kv_cache_type = 1; // f16
    config.rope_freq_base = 0;
    config.rope_freq_scale = 0;
    config.use_mmap = true;
    config.use_mlock = false;
    config.use_numa = false;
    config.embeddings = false;
    config.lora_path = nullptr;
    config.tensor_split = nullptr;
    config.tensor_split_count = 0;
    return config;
}

binding_generate_config binding_generate_config_default(void) {
    binding_generate_config config;
    config.max_tokens = 0;
    config.seed = -1;
    config.threads = 4;
    config.top_k = 40;
    config.top_p = 0.95f;
    config.min_p = 0.05f;
    config.temperature = 0.8f;
    config.repeat_penalty = 1.1f;
    config.repeat_last_n = 64;
    config.frequency_penalty = 0.0f;
    config.presence_penalty = 0.0f;
    config.mirostat = 0;
    config.mirostat_tau = 5.0f;
    config.mirostat_eta = 0.1f;
    config.ignore_eos = false;
    config.grammar = nullptr;
    config.stop_sequences = nullptr;
    config.stop_count = 0;
    config.reasoning_enabled = true;
    return config;
}

// ============================================================================
// Helper Functions
// ============================================================================

static ggml_type parse_kv_cache_type(int32_t type) {
    switch (type) {
        case 0: return GGML_TYPE_F32;
        case 1: return GGML_TYPE_F16;
        case 2: return GGML_TYPE_Q8_0;
        case 3: return GGML_TYPE_Q4_0;
        default: return GGML_TYPE_F16;
    }
}

static llama_sampler *create_sampler(const llama_vocab *vocab, const binding_generate_config *config, int32_t n_vocab) {
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;

    llama_sampler *smpl = llama_sampler_chain_init(sparams);

    // Add penalties
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        config->repeat_last_n,
        config->repeat_penalty,
        config->frequency_penalty,
        config->presence_penalty
    ));

    // Add sampling based on mirostat setting
    if (config->mirostat == 1) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config->temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
            n_vocab,
            config->seed,
            config->mirostat_tau,
            config->mirostat_eta,
            100
        ));
    } else if (config->mirostat == 2) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config->temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
            config->seed,
            config->mirostat_tau,
            config->mirostat_eta
        ));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(config->top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_typical(1.0f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config->top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(config->min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(config->temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(config->seed));
    }

    // Add grammar if specified
    if (config->grammar != nullptr && config->grammar[0] != '\0') {
        llama_sampler *grammar_smpl = llama_sampler_init_grammar(vocab, config->grammar, "root");
        if (grammar_smpl != nullptr) {
            llama_sampler_chain_add(smpl, grammar_smpl);
        }
    }

    return smpl;
}

static std::vector<llama_token> tokenize_text(const llama_vocab *vocab, const std::string &text, bool add_special) {
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), nullptr, 0, add_special, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    std::vector<llama_token> tokens(n_tokens);
    int32_t actual = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_special, true);

    if (actual < 0) {
        tokens.clear();
        return tokens;
    }

    tokens.resize(actual);
    return tokens;
}

static std::string token_to_piece(const llama_vocab *vocab, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
    if (n < 0) {
        return "";
    }
    return std::string(buf, n);
}

// Check if text ends with a thinking start tag and return the corresponding closing tag
// Returns empty string if no thinking tag found
static std::string get_thinking_close_tag(const std::string &text) {
    // Check for various thinking start tags (in order of specificity)
    static const struct { const char *start; const char *end; } thinking_tags[] = {
        {"<think>\n",          "</think>"},
        {"<think>",            "</think>"},
        {"<|START_THINKING|>", "<|END_THINKING|>"},
        {"<|inner_prefix|>",   "<|inner_suffix|>"},
    };

    for (const auto &tag : thinking_tags) {
        size_t start_len = strlen(tag.start);
        if (text.length() >= start_len &&
            text.compare(text.length() - start_len, start_len, tag.start) == 0) {
            return tag.end;
        }
    }
    return "";
}

// ============================================================================
// Model Operations
// ============================================================================

void *binding_load_model(const char *path, const binding_model_config *config) {
    if (path == nullptr || config == nullptr) {
        return nullptr;
    }

    if (g_verbose) {
        fprintf(stderr, "binding: loading model from '%s'\n", path);
    }

    llama_backend_init();

    if (config->use_numa) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }

    // Model parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config->gpu_layers;
    model_params.use_mmap = config->use_mmap;
    model_params.use_mlock = config->use_mlock;
    model_params.main_gpu = config->main_gpu;

    // Tensor split (static storage - shared across models)
    static float tensor_split_values[128] = {0};
    if (config->tensor_split != nullptr && config->tensor_split_count > 0) {
        for (int i = 0; i < config->tensor_split_count && i < 128; ++i) {
            tensor_split_values[i] = config->tensor_split[i];
        }
        model_params.tensor_split = tensor_split_values;
    }

    // Load model
    llama_model *model = llama_model_load_from_file(path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "binding: failed to load model from '%s'\n", path);
        return nullptr;
    }

    // Context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config->context_size > 0 ? config->context_size : 0;
    ctx_params.n_batch = config->batch_size > 0 ? config->batch_size : 512;
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.embeddings = config->embeddings;

    // KV cache type
    ggml_type kv_type = parse_kv_cache_type(config->kv_cache_type);
    ctx_params.type_k = kv_type;
    ctx_params.type_v = kv_type;

    if (config->rope_freq_base > 0.0f) {
        ctx_params.rope_freq_base = config->rope_freq_base;
    }
    if (config->rope_freq_scale > 0.0f) {
        ctx_params.rope_freq_scale = config->rope_freq_scale;
    }

    // Create context
    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "binding: failed to create context\n");
        llama_model_free(model);
        return nullptr;
    }

    // Load LoRA adapter if specified
    if (config->lora_path != nullptr && config->lora_path[0] != '\0') {
        llama_adapter_lora *adapter = llama_adapter_lora_init(model, config->lora_path);
        if (adapter == nullptr) {
            fprintf(stderr, "binding: failed to load LoRA adapter from '%s'\n", config->lora_path);
        } else {
            int32_t ret = llama_set_adapter_lora(ctx, adapter, 1.0f);
            if (ret != 0) {
                fprintf(stderr, "binding: failed to set LoRA adapter\n");
                llama_adapter_lora_free(adapter);
            }
        }
    }

    // Create binding state
    llama_binding_state *state = new llama_binding_state;
    state->model = model;
    state->ctx = ctx;
    state->vocab = llama_model_get_vocab(model);
    state->embeddings_enabled = config->embeddings;
    state->n_embd = llama_model_n_embd(model);
    state->cached_n_past = 0;

    if (g_verbose) {
        fprintf(stderr, "binding: model loaded (n_ctx=%d, n_embd=%d)\n",
                llama_n_ctx(ctx), state->n_embd);
    }

    return state;
}

void binding_free_model(void *model) {
    if (model == nullptr) return;

    llama_binding_state *state = (llama_binding_state *)model;

    if (state->ctx) {
        llama_free(state->ctx);
    }
    if (state->model) {
        llama_model_free(state->model);
    }

    delete state;
    llama_backend_free();
}

int32_t binding_get_context_size(void *model) {
    if (model == nullptr) return 0;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_n_ctx(state->ctx);
}

int32_t binding_get_train_context_size(void *model) {
    if (model == nullptr) return 0;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_model_n_ctx_train(state->model);
}

int32_t binding_get_embedding_size(void *model) {
    if (model == nullptr) return 0;
    llama_binding_state *state = (llama_binding_state *)model;
    return state->n_embd;
}

// ============================================================================
// KV Cache Management
// ============================================================================

// Find common prefix length between cached tokens and new tokens
static int find_common_prefix(
    const std::vector<llama_token>& cached,
    int cached_count,
    const int32_t* new_tokens,
    int new_count
) {
    int max_check = std::min({cached_count, new_count, (int)cached.size()});
    for (int i = 0; i < max_check; i++) {
        if (cached[i] != (llama_token)new_tokens[i]) {
            return i;
        }
    }
    return max_check;
}

binding_cache_prepare_result binding_prepare_cache(
    void *model,
    const int32_t *tokens,
    int32_t token_count
) {
    binding_cache_prepare_result result = {0, token_count, BINDING_OK};

    if (model == nullptr || (tokens == nullptr && token_count > 0)) {
        result.status = BINDING_ERR_INVALID_ARGS;
        return result;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    llama_memory_t mem = llama_get_memory(state->ctx);

    // Handle empty token case
    if (token_count == 0) {
        llama_memory_clear(mem, true);
        state->cached_tokens.clear();
        state->cached_n_past = 0;
        return result;
    }

    // Find common prefix between cached and new tokens
    int common = find_common_prefix(
        state->cached_tokens,
        state->cached_n_past,
        tokens,
        token_count
    );

    result.common_prefix = common;
    result.tokens_to_process = token_count - common;

    if (common == 0) {
        // No overlap - clear everything
        llama_memory_clear(mem, true);
        state->cached_tokens.clear();
        state->cached_n_past = 0;
    } else if (common < state->cached_n_past) {
        // Partial overlap - remove divergent portion
        llama_memory_seq_rm(mem, 0, common, -1);
        state->cached_tokens.resize(common);
        state->cached_n_past = common;
    }
    // If common == cached_n_past, cache is already in correct state

    if (g_verbose) {
        fprintf(stderr, "binding: prepare_cache - common=%d, to_process=%d, cached_n_past=%d\n",
                common, result.tokens_to_process, state->cached_n_past);
    }

    return result;
}

int32_t binding_get_cache_token_count(void *model) {
    if (model == nullptr) return 0;
    llama_binding_state *state = (llama_binding_state *)model;
    return state->cached_n_past;
}

// ============================================================================
// Text Generation
// ============================================================================

void *binding_generate_start(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    const binding_generate_config *config
) {
    if (model == nullptr || config == nullptr) {
        return nullptr;
    }
    if (tokens == nullptr && token_count > 0) {
        return nullptr;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    llama_context *ctx = state->ctx;
    const llama_vocab *vocab = state->vocab;

    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Set threads
    llama_set_n_threads(ctx, config->threads, config->threads);

    // Check total tokens fit in context
    if (state->cached_n_past + token_count > n_ctx - 4) {
        return nullptr;  // Would exceed context
    }

    // Create generation state
    llama_generation_state *gen = new llama_generation_state;
    gen->model_state = state;
    gen->config = *config;

    // Copy tokens to process (these are NEW tokens after cache)
    gen->prompt_tokens.reserve(token_count);
    for (int32_t i = 0; i < token_count; i++) {
        gen->prompt_tokens.push_back((llama_token)tokens[i]);
    }

    // Start from cached position
    gen->n_past = state->cached_n_past;
    gen->n_consumed = 0;
    gen->n_remain = config->max_tokens > 0 ? config->max_tokens : 99999999;
    gen->finished = false;
    gen->inject_index = 0;

    // Copy stop sequences
    if (config->stop_sequences != nullptr && config->stop_count > 0) {
        for (int i = 0; i < config->stop_count; i++) {
            gen->stop_sequences.push_back(config->stop_sequences[i]);
        }
    }

    // Create sampler
    gen->sampler = create_sampler(vocab, config, n_vocab);

    // Note: KV cache is NOT cleared - caller should have called binding_prepare_cache first

    return gen;
}

binding_result binding_generate_next(void *generation, binding_token *out_token) {
    if (generation == nullptr || out_token == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_generation_state *gen = (llama_generation_state *)generation;

    if (gen->finished) {
        return BINDING_DONE;
    }

    llama_binding_state *state = gen->model_state;
    llama_context *ctx = state->ctx;
    const llama_vocab *vocab = state->vocab;

    // First, check if we have pending tokens to inject (for thinking suppression)
    if (gen->inject_index < gen->pending_inject_tokens.size()) {
        llama_token id = gen->pending_inject_tokens[gen->inject_index++];

        // Decode the injected token
        std::vector<llama_token> inject_tok = {id};
        llama_batch batch = llama_batch_get_one(inject_tok.data(), 1);
        if (llama_decode(ctx, batch)) {
            gen->finished = true;
            return BINDING_ERR_DECODE;
        }
        gen->n_past++;

        // Track in cache
        state->cached_tokens.push_back(id);
        state->cached_n_past = gen->n_past;

        // Get token text
        gen->current_token_text = token_to_piece(vocab, id);
        gen->generated_text += gen->current_token_text;

        out_token->text = gen->current_token_text.c_str();
        out_token->id = id;

        // Clear pending tokens when done
        if (gen->inject_index >= gen->pending_inject_tokens.size()) {
            gen->pending_inject_tokens.clear();
            gen->inject_index = 0;
        }

        return BINDING_OK;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);

    // Process any remaining prompt tokens first
    while ((int)gen->prompt_tokens.size() > gen->n_consumed) {
        std::vector<llama_token> batch_tokens;

        while ((int)gen->prompt_tokens.size() > gen->n_consumed && (int)batch_tokens.size() < 512) {
            batch_tokens.push_back(gen->prompt_tokens[gen->n_consumed]);
            gen->n_consumed++;
        }

        if (!batch_tokens.empty()) {
            // Handle context overflow
            if (gen->n_past + (int)batch_tokens.size() > n_ctx) {
                const int n_left = gen->n_past;
                gen->n_past = std::max(1, n_ctx / 4);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, gen->n_past, n_left);
                llama_memory_seq_add(llama_get_memory(ctx), 0, gen->n_past, n_left, -(n_left - gen->n_past));
                // Update cache tracking after overflow handling
                state->cached_tokens.resize(gen->n_past);
                state->cached_n_past = gen->n_past;
            }

            llama_batch batch = llama_batch_get_one(batch_tokens.data(), (int)batch_tokens.size());
            if (llama_decode(ctx, batch)) {
                gen->finished = true;
                return BINDING_ERR_DECODE;
            }
            gen->n_past += (int)batch_tokens.size();

            // Track processed prompt tokens in cache
            for (const auto& tok : batch_tokens) {
                state->cached_tokens.push_back(tok);
            }
            state->cached_n_past = gen->n_past;
        }
    }

    // Check if we've exhausted token limit
    if (gen->n_remain <= 0) {
        gen->finished = true;
        return BINDING_DONE;
    }

    // Sample next token
    llama_token id = llama_sampler_sample(gen->sampler, ctx, -1);

    // Check for EOS
    bool is_eog = llama_vocab_is_eog(vocab, id);
    if (!gen->config.ignore_eos && is_eog) {
        gen->finished = true;
        return BINDING_DONE;
    }

    // Decode to batch
    std::vector<llama_token> new_token = {id};
    llama_batch batch = llama_batch_get_one(new_token.data(), 1);

    // Handle context overflow for the new token
    if (gen->n_past + 1 > n_ctx) {
        const int n_left = gen->n_past;
        gen->n_past = std::max(1, n_ctx / 4);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, gen->n_past, n_left);
        llama_memory_seq_add(llama_get_memory(ctx), 0, gen->n_past, n_left, -(n_left - gen->n_past));
        // Update cache tracking after overflow handling
        state->cached_tokens.resize(gen->n_past);
        state->cached_n_past = gen->n_past;
    }

    if (llama_decode(ctx, batch)) {
        gen->finished = true;
        return BINDING_ERR_DECODE;
    }
    gen->n_past++;
    gen->n_remain--;

    // Track generated token in cache
    state->cached_tokens.push_back(id);
    state->cached_n_past = gen->n_past;

    // Get token string
    gen->current_token_text = token_to_piece(vocab, id);
    gen->generated_text += gen->current_token_text;

    // Fill output token
    out_token->text = gen->current_token_text.c_str();
    out_token->id = id;

    // Check for thinking tag and queue closing tokens if reasoning is disabled
    if (!gen->config.reasoning_enabled) {
        std::string close_tag = get_thinking_close_tag(gen->generated_text);
        if (!close_tag.empty()) {
            // Tokenize the closing tag and queue for injection
            gen->pending_inject_tokens = tokenize_text(vocab, close_tag, false);
            gen->inject_index = 0;
        }
    }

    // Check for stop sequences
    for (const auto &stop : gen->stop_sequences) {
        if (gen->generated_text.length() >= stop.length() &&
            gen->generated_text.compare(gen->generated_text.length() - stop.length(), stop.length(), stop) == 0) {
            gen->finished = true;
            // Still return the token that triggered the stop
            return BINDING_OK;
        }
    }

    return BINDING_OK;
}

binding_result binding_generate_next_with_logprobs(
    void *generation,
    binding_token_with_logprobs *out_token,
    int32_t top_k
) {
    if (generation == nullptr || out_token == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    // Initialize output
    out_token->text = nullptr;
    out_token->id = 0;
    out_token->prob = 0.0f;
    out_token->logit = 0.0f;
    out_token->top_k = nullptr;
    out_token->top_k_count = 0;

    llama_generation_state *gen = (llama_generation_state *)generation;

    if (gen->finished) {
        return BINDING_DONE;
    }

    llama_binding_state *state = gen->model_state;
    llama_context *ctx = state->ctx;
    const llama_vocab *vocab = state->vocab;
    const int32_t n_ctx = llama_n_ctx(ctx);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // First, check if we have pending tokens to inject (for thinking suppression)
    if (gen->inject_index < gen->pending_inject_tokens.size()) {
        llama_token id = gen->pending_inject_tokens[gen->inject_index++];

        // Decode the injected token
        std::vector<llama_token> inject_tok = {id};
        llama_batch batch = llama_batch_get_one(inject_tok.data(), 1);
        if (llama_decode(ctx, batch)) {
            gen->finished = true;
            return BINDING_ERR_DECODE;
        }
        gen->n_past++;

        state->cached_tokens.push_back(id);
        state->cached_n_past = gen->n_past;

        gen->current_token_text = token_to_piece(vocab, id);
        gen->generated_text += gen->current_token_text;

        out_token->text = gen->current_token_text.c_str();
        out_token->id = id;
        out_token->prob = 1.0f;  // Injected tokens have probability 1
        out_token->logit = 0.0f;

        if (gen->inject_index >= gen->pending_inject_tokens.size()) {
            gen->pending_inject_tokens.clear();
            gen->inject_index = 0;
        }

        return BINDING_OK;
    }

    // Process any remaining prompt tokens first
    while ((int)gen->prompt_tokens.size() > gen->n_consumed) {
        std::vector<llama_token> batch_tokens;

        while ((int)gen->prompt_tokens.size() > gen->n_consumed && (int)batch_tokens.size() < 512) {
            batch_tokens.push_back(gen->prompt_tokens[gen->n_consumed]);
            gen->n_consumed++;
        }

        if (!batch_tokens.empty()) {
            // Handle context overflow
            if (gen->n_past + (int)batch_tokens.size() > n_ctx) {
                const int n_left = gen->n_past;
                gen->n_past = std::max(1, n_ctx / 4);
                llama_memory_seq_rm(llama_get_memory(ctx), 0, gen->n_past, n_left);
                llama_memory_seq_add(llama_get_memory(ctx), 0, gen->n_past, n_left, -(n_left - gen->n_past));
                state->cached_tokens.resize(gen->n_past);
                state->cached_n_past = gen->n_past;
            }

            llama_batch batch = llama_batch_get_one(batch_tokens.data(), (int)batch_tokens.size());
            if (llama_decode(ctx, batch)) {
                gen->finished = true;
                return BINDING_ERR_DECODE;
            }
            gen->n_past += (int)batch_tokens.size();

            for (const auto& tok : batch_tokens) {
                state->cached_tokens.push_back(tok);
            }
            state->cached_n_past = gen->n_past;
        }
    }

    // Check if we've exhausted token limit
    if (gen->n_remain <= 0) {
        gen->finished = true;
        return BINDING_DONE;
    }

    // Get logits before sampling
    float *logits = llama_get_logits_ith(ctx, -1);
    if (logits == nullptr) {
        gen->finished = true;
        return BINDING_ERR_DECODE;
    }

    // Sample next token
    llama_token id = llama_sampler_sample(gen->sampler, ctx, -1);

    // Get selected token's logit
    float selected_logit = logits[id];

    // Compute softmax to get probabilities
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < n_vocab; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }

    // Compute selected token probability
    float selected_prob = expf(selected_logit - max_logit) / sum_exp;

    // Check for EOS
    bool is_eog = llama_vocab_is_eog(vocab, id);
    if (!gen->config.ignore_eos && is_eog) {
        gen->finished = true;
        return BINDING_DONE;
    }

    // Decode to batch
    std::vector<llama_token> new_token = {id};
    llama_batch batch = llama_batch_get_one(new_token.data(), 1);

    // Handle context overflow for the new token
    if (gen->n_past + 1 > n_ctx) {
        const int n_left = gen->n_past;
        gen->n_past = std::max(1, n_ctx / 4);
        llama_memory_seq_rm(llama_get_memory(ctx), 0, gen->n_past, n_left);
        llama_memory_seq_add(llama_get_memory(ctx), 0, gen->n_past, n_left, -(n_left - gen->n_past));
        state->cached_tokens.resize(gen->n_past);
        state->cached_n_past = gen->n_past;
    }

    if (llama_decode(ctx, batch)) {
        gen->finished = true;
        return BINDING_ERR_DECODE;
    }
    gen->n_past++;
    gen->n_remain--;

    state->cached_tokens.push_back(id);
    state->cached_n_past = gen->n_past;

    // Get token string
    gen->current_token_text = token_to_piece(vocab, id);
    gen->generated_text += gen->current_token_text;

    // Fill output token
    out_token->text = gen->current_token_text.c_str();
    out_token->id = id;
    out_token->prob = selected_prob;
    out_token->logit = selected_logit;

    // Compute top-k if requested
    if (top_k > 0) {
        // Build vector of (token_id, logit) pairs
        std::vector<std::pair<int32_t, float>> token_logits(n_vocab);
        for (int32_t i = 0; i < n_vocab; i++) {
            token_logits[i] = {i, logits[i]};
        }

        // Partial sort to get top-k
        int32_t k = std::min(top_k, n_vocab);
        std::partial_sort(token_logits.begin(), token_logits.begin() + k, token_logits.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Allocate and fill top-k array
        out_token->top_k = (binding_token_prob *)malloc(k * sizeof(binding_token_prob));
        if (out_token->top_k != nullptr) {
            out_token->top_k_count = k;
            for (int32_t i = 0; i < k; i++) {
                int32_t tok_id = token_logits[i].first;
                float tok_logit = token_logits[i].second;
                float tok_prob = expf(tok_logit - max_logit) / sum_exp;

                out_token->top_k[i].token_id = tok_id;
                out_token->top_k[i].logit = tok_logit;
                out_token->top_k[i].prob = tok_prob;
            }
        }
    }

    // Check for thinking tag and queue closing tokens if reasoning is disabled
    if (!gen->config.reasoning_enabled) {
        std::string close_tag = get_thinking_close_tag(gen->generated_text);
        if (!close_tag.empty()) {
            gen->pending_inject_tokens = tokenize_text(vocab, close_tag, false);
            gen->inject_index = 0;
        }
    }

    // Check for stop sequences
    for (const auto &stop : gen->stop_sequences) {
        if (gen->generated_text.length() >= stop.length() &&
            gen->generated_text.compare(gen->generated_text.length() - stop.length(), stop.length(), stop) == 0) {
            gen->finished = true;
            return BINDING_OK;
        }
    }

    return BINDING_OK;
}

void binding_generate_free(void *generation) {
    if (generation == nullptr) return;

    llama_generation_state *gen = (llama_generation_state *)generation;

    if (gen->sampler != nullptr) {
        llama_sampler_free(gen->sampler);
    }

    delete gen;
}

// ============================================================================
// Embeddings
// ============================================================================

binding_result binding_get_embeddings(
    void *model,
    const char *text,
    float *out_embeddings,
    int32_t *out_size
) {
    if (model == nullptr || text == nullptr || out_embeddings == nullptr || out_size == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;

    if (!state->embeddings_enabled) {
        return BINDING_ERR_EMBEDDINGS_DISABLED;
    }

    llama_context *ctx = state->ctx;
    const llama_vocab *vocab = state->vocab;

    // Tokenize
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = tokenize_text(vocab, text, add_bos);

    if (tokens.empty()) {
        return BINDING_ERR_TOKENIZE;
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Decode
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

    if (llama_decode(ctx, batch)) {
        return BINDING_ERR_DECODE;
    }

    // Get embeddings
    const float *embeddings = llama_get_embeddings(ctx);
    if (embeddings == nullptr) {
        return BINDING_ERR_EMBEDDINGS_DISABLED;
    }

    *out_size = state->n_embd;
    for (int i = 0; i < state->n_embd; i++) {
        out_embeddings[i] = embeddings[i];
    }

    return BINDING_OK;
}

// ============================================================================
// Tokenization
// ============================================================================

binding_result binding_tokenize_count(
    void *model,
    const char *text,
    bool add_special,
    int32_t *out_count
) {
    if (model == nullptr || text == nullptr || out_count == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    const llama_vocab *vocab = state->vocab;

    // Use llama_tokenize with nullptr to get count
    int32_t n_tokens = llama_tokenize(vocab, text, (int32_t)strlen(text), nullptr, 0, add_special, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    *out_count = n_tokens;
    return BINDING_OK;
}

binding_result binding_tokenize(
    void *model,
    const char *text,
    bool add_special,
    int32_t *out_tokens
) {
    if (model == nullptr || text == nullptr || out_tokens == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    const llama_vocab *vocab = state->vocab;

    // Get count first to know the capacity
    int32_t n_tokens = llama_tokenize(vocab, text, (int32_t)strlen(text), nullptr, 0, add_special, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    // Tokenize into buffer
    int32_t result = llama_tokenize(vocab, text, (int32_t)strlen(text), out_tokens, n_tokens, add_special, true);
    if (result < 0) {
        return BINDING_ERR_TOKENIZE;
    }

    return BINDING_OK;
}

binding_result binding_detokenize_length(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    int32_t *out_length
) {
    if (model == nullptr || tokens == nullptr || out_length == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    int32_t total_len = 0;

    for (int32_t i = 0; i < token_count; i++) {
        std::string piece = token_to_piece(state->vocab, tokens[i]);
        total_len += (int32_t)piece.length();
    }

    *out_length = total_len;
    return BINDING_OK;
}

binding_result binding_detokenize(
    void *model,
    const int32_t *tokens,
    int32_t token_count,
    char *out_text
) {
    if (model == nullptr || tokens == nullptr || out_text == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    char *ptr = out_text;

    for (int32_t i = 0; i < token_count; i++) {
        std::string piece = token_to_piece(state->vocab, tokens[i]);
        memcpy(ptr, piece.c_str(), piece.length());
        ptr += piece.length();
    }
    *ptr = '\0';

    return BINDING_OK;
}

// ============================================================================
// Chat Templates
// ============================================================================

const char *binding_get_chat_template(void *model) {
    if (model == nullptr) return nullptr;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_model_chat_template(state->model, nullptr);
}

// Helper to get template string from model or use provided
static const char *get_template_str(void *model, const char *tmpl) {
    if (tmpl != nullptr) return tmpl;
    if (model == nullptr) return nullptr;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_model_chat_template(state->model, nullptr);
}

int32_t binding_apply_chat_template_length(
    void *model,
    const char *tmpl,
    const binding_chat_message *messages,
    int32_t message_count,
    bool add_assistant
) {
    if (message_count <= 0 || messages == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    // Convert to llama_chat_message array
    std::vector<llama_chat_message> chat(message_count);
    for (int i = 0; i < message_count; i++) {
        chat[i].role = messages[i].role;
        chat[i].content = messages[i].content;
    }

    const char *template_str = get_template_str(model, tmpl);

    return llama_chat_apply_template(
        template_str,
        chat.data(),
        chat.size(),
        add_assistant,
        nullptr,
        0
    );
}

int32_t binding_apply_chat_template(
    void *model,
    const char *tmpl,
    const binding_chat_message *messages,
    int32_t message_count,
    bool add_assistant,
    char *buf,
    int32_t buf_size
) {
    if (message_count <= 0 || messages == nullptr || buf == nullptr || buf_size <= 0) {
        return BINDING_ERR_INVALID_ARGS;
    }

    // Convert to llama_chat_message array
    std::vector<llama_chat_message> chat(message_count);
    for (int i = 0; i < message_count; i++) {
        chat[i].role = messages[i].role;
        chat[i].content = messages[i].content;
    }

    const char *template_str = get_template_str(model, tmpl);

    return llama_chat_apply_template(
        template_str,
        chat.data(),
        chat.size(),
        add_assistant,
        buf,
        buf_size
    );
}

// ============================================================================
// Token Utilities
// ============================================================================

int32_t binding_token_to_text(void *model, int32_t token, char *buf, int32_t buf_size) {
    if (model == nullptr || buf == nullptr || buf_size <= 0) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;
    int32_t n = llama_token_to_piece(state->vocab, token, buf, buf_size, 0, false);
    return n;
}

bool binding_is_special_token(void *model, int32_t token) {
    if (model == nullptr) return false;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_vocab_is_control(state->vocab, token);
}

bool binding_is_eog_token(void *model, int32_t token) {
    if (model == nullptr) return false;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_vocab_is_eog(state->vocab, token);
}

binding_special_tokens binding_get_special_tokens(void *model) {
    binding_special_tokens tokens = {-1, -1, -1, -1, -1, -1};
    if (model == nullptr) return tokens;

    llama_binding_state *state = (llama_binding_state *)model;
    const llama_vocab *vocab = state->vocab;

    tokens.bos = llama_vocab_bos(vocab);
    tokens.eos = llama_vocab_eos(vocab);
    tokens.eot = llama_vocab_eot(vocab);
    tokens.pad = llama_vocab_pad(vocab);
    tokens.sep = llama_vocab_sep(vocab);
    tokens.nl = llama_vocab_nl(vocab);

    return tokens;
}

int32_t binding_vocab_size(void *model) {
    if (model == nullptr) return 0;
    llama_binding_state *state = (llama_binding_state *)model;
    return llama_vocab_n_tokens(state->vocab);
}

// ============================================================================
// Model Information
// ============================================================================

binding_model_info binding_get_model_info(void *model) {
    binding_model_info info = {};
    if (model == nullptr) return info;

    llama_binding_state *state = (llama_binding_state *)model;
    const llama_model *lmodel = state->model;
    const llama_vocab *vocab = state->vocab;

    // Get description
    llama_model_desc(lmodel, info.description, sizeof(info.description));

    // Get architecture from metadata
    llama_model_meta_val_str(lmodel, "general.architecture", info.architecture, sizeof(info.architecture));

    // Get quantization type from metadata
    llama_model_meta_val_str(lmodel, "general.file_type", info.quant_type, sizeof(info.quant_type));
    // If file_type not available, try to get from description
    if (info.quant_type[0] == '\0') {
        // Description often contains quant type, extract it
        const char *desc = info.description;
        const char *quant_prefixes[] = {"Q4_", "Q5_", "Q8_", "Q2_", "Q3_", "Q6_", "F16", "F32", "IQ"};
        for (const char *prefix : quant_prefixes) {
            const char *found = strstr(desc, prefix);
            if (found) {
                // Copy until space or end
                int i = 0;
                while (found[i] && found[i] != ' ' && i < 31) {
                    info.quant_type[i] = found[i];
                    i++;
                }
                info.quant_type[i] = '\0';
                break;
            }
        }
    }

    // Get numeric info
    info.parameters = llama_model_n_params(lmodel);
    info.size = llama_model_size(lmodel);
    info.layers = llama_model_n_layer(lmodel);
    info.heads = llama_model_n_head(lmodel);
    info.heads_kv = llama_model_n_head_kv(lmodel);
    info.vocab_size = llama_vocab_n_tokens(vocab);

    return info;
}

// ============================================================================
// Batch Embeddings
// ============================================================================

binding_result binding_get_embeddings_batch(
    void *model,
    const char **texts,
    int32_t text_count,
    float *out_embeddings,
    int32_t *out_size
) {
    if (model == nullptr || texts == nullptr || text_count <= 0 ||
        out_embeddings == nullptr || out_size == nullptr) {
        return BINDING_ERR_INVALID_ARGS;
    }

    llama_binding_state *state = (llama_binding_state *)model;

    if (!state->embeddings_enabled) {
        return BINDING_ERR_EMBEDDINGS_DISABLED;
    }

    llama_context *ctx = state->ctx;
    const llama_vocab *vocab = state->vocab;
    const int32_t n_embd = state->n_embd;

    *out_size = n_embd;

    // Process each text sequentially
    for (int32_t t = 0; t < text_count; t++) {
        const char *text = texts[t];
        if (text == nullptr) {
            // Fill with zeros for null text
            for (int i = 0; i < n_embd; i++) {
                out_embeddings[t * n_embd + i] = 0.0f;
            }
            continue;
        }

        // Tokenize
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        std::vector<llama_token> tokens = tokenize_text(vocab, text, add_bos);

        if (tokens.empty()) {
            // Fill with zeros for empty tokenization
            for (int i = 0; i < n_embd; i++) {
                out_embeddings[t * n_embd + i] = 0.0f;
            }
            continue;
        }

        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Decode
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

        if (llama_decode(ctx, batch)) {
            return BINDING_ERR_DECODE;
        }

        // Get embeddings
        const float *embeddings = llama_get_embeddings(ctx);
        if (embeddings == nullptr) {
            return BINDING_ERR_EMBEDDINGS_DISABLED;
        }

        // Copy embeddings
        for (int i = 0; i < n_embd; i++) {
            out_embeddings[t * n_embd + i] = embeddings[i];
        }
    }

    return BINDING_OK;
}

// ============================================================================
// Tool Call Parsing
// ============================================================================

binding_parse_result* binding_parse_tool_calls(
    const char* response,
    int32_t format,
    bool is_partial
) {
    auto* result = new binding_parse_result();
    result->content = nullptr;
    result->reasoning_content = nullptr;
    result->tool_calls = nullptr;
    result->tool_call_count = 0;
    result->success = false;

    if (response == nullptr) {
        return result;
    }

    try {
        // Set up syntax for parsing
        common_chat_syntax syntax;
        syntax.format = static_cast<common_chat_format>(format);
        syntax.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        syntax.reasoning_in_content = false;
        syntax.thinking_forced_open = false;
        syntax.parse_tool_calls = true;

        // Parse the response
        common_chat_msg parsed = common_chat_parse(response, is_partial, syntax);

        // Copy content
        result->content = strdup(parsed.content.c_str());

        // Copy reasoning content if present
        if (!parsed.reasoning_content.empty()) {
            result->reasoning_content = strdup(parsed.reasoning_content.c_str());
        }

        // Copy tool calls
        result->tool_call_count = static_cast<int32_t>(parsed.tool_calls.size());
        if (result->tool_call_count > 0) {
            result->tool_calls = new binding_tool_call[result->tool_call_count];
            for (int32_t i = 0; i < result->tool_call_count; i++) {
                result->tool_calls[i].name = strdup(parsed.tool_calls[i].name.c_str());
                result->tool_calls[i].arguments = strdup(parsed.tool_calls[i].arguments.c_str());
                result->tool_calls[i].id = strdup(parsed.tool_calls[i].id.c_str());
            }
        }

        result->success = true;
    } catch (const std::exception& e) {
        if (g_verbose) {
            fprintf(stderr, "binding: parse_tool_calls error: %s\n", e.what());
        }
        // Return empty result on error
        result->content = strdup(response);  // Keep original content on error
    } catch (...) {
        result->content = strdup(response);  // Keep original content on error
    }

    return result;
}

void binding_free_parse_result(binding_parse_result* result) {
    if (result == nullptr) {
        return;
    }

    free((void*)result->content);
    free((void*)result->reasoning_content);

    if (result->tool_calls != nullptr) {
        for (int32_t i = 0; i < result->tool_call_count; i++) {
            free((void*)result->tool_calls[i].name);
            free((void*)result->tool_calls[i].arguments);
            free((void*)result->tool_calls[i].id);
        }
        delete[] result->tool_calls;
    }

    delete result;
}

const char* binding_chat_format_name(int32_t format) {
    return common_chat_format_name(static_cast<common_chat_format>(format));
}

// ============================================================================
// Tool Template Application
// ============================================================================

binding_chat_params* binding_apply_chat_template_with_tools(
    void* model,
    const binding_chat_message* messages,
    int32_t message_count,
    const binding_chat_tool_def* tools,
    int32_t tool_count,
    binding_tool_choice tool_choice,
    bool parallel_tool_calls,
    bool add_generation_prompt
) {
    auto* result = new binding_chat_params();
    result->prompt = nullptr;
    result->grammar = nullptr;
    result->format = 0;  // CONTENT_ONLY
    result->grammar_lazy = false;
    result->grammar_triggers = nullptr;
    result->trigger_count = 0;
    result->additional_stops = nullptr;
    result->stop_count = 0;

    llama_binding_state *state = (llama_binding_state *)model;
    if (state == nullptr || state->model == nullptr) {
        result->prompt = strdup("");
        result->grammar = strdup("");
        return result;
    }

    try {
        // Get model's chat templates (returns unique_ptr)
        auto tmpls = common_chat_templates_init(state->model, "");
        if (!tmpls) {
            result->prompt = strdup("");
            result->grammar = strdup("");
            return result;
        }

        // Build inputs
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja = true;
        inputs.parallel_tool_calls = parallel_tool_calls;

        // Convert messages
        for (int32_t i = 0; i < message_count; i++) {
            common_chat_msg msg;
            msg.role = messages[i].role ? messages[i].role : "";
            msg.content = messages[i].content ? messages[i].content : "";

            // Handle tool calls in assistant messages
            if (messages[i].tool_calls != nullptr && messages[i].tool_call_count > 0) {
                for (int32_t j = 0; j < messages[i].tool_call_count; j++) {
                    common_chat_tool_call tc;
                    tc.name = messages[i].tool_calls[j].name ? messages[i].tool_calls[j].name : "";
                    tc.arguments = messages[i].tool_calls[j].arguments ? messages[i].tool_calls[j].arguments : "";
                    tc.id = messages[i].tool_calls[j].id ? messages[i].tool_calls[j].id : "";
                    msg.tool_calls.push_back(tc);
                }
            }

            // Handle tool result messages
            if (messages[i].tool_name != nullptr) {
                msg.tool_name = messages[i].tool_name;
            }
            if (messages[i].tool_call_id != nullptr) {
                msg.tool_call_id = messages[i].tool_call_id;
            }

            inputs.messages.push_back(msg);
        }

        // Convert tools
        for (int32_t i = 0; i < tool_count; i++) {
            common_chat_tool tool;
            tool.name = tools[i].name ? tools[i].name : "";
            tool.description = tools[i].description ? tools[i].description : "";
            tool.parameters = tools[i].parameters ? tools[i].parameters : "{}";
            inputs.tools.push_back(tool);
        }

        // Set tool choice
        switch (tool_choice) {
            case BINDING_TOOL_CHOICE_NONE:
                inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
                break;
            case BINDING_TOOL_CHOICE_REQUIRED:
                inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                break;
            default:
                inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        }

        // Apply template
        auto params = common_chat_templates_apply(tmpls.get(), inputs);

        // Copy results
        result->prompt = strdup(params.prompt.c_str());
        result->grammar = strdup(params.grammar.c_str());
        result->format = static_cast<int32_t>(params.format);
        result->grammar_lazy = params.grammar_lazy;

        // Copy triggers
        result->trigger_count = static_cast<int32_t>(params.grammar_triggers.size());
        if (result->trigger_count > 0) {
            result->grammar_triggers = new const char*[result->trigger_count + 1];
            for (size_t i = 0; i < params.grammar_triggers.size(); i++) {
                result->grammar_triggers[i] = strdup(params.grammar_triggers[i].value.c_str());
            }
            result->grammar_triggers[result->trigger_count] = nullptr;
        }

        // Copy stops
        result->stop_count = static_cast<int32_t>(params.additional_stops.size());
        if (result->stop_count > 0) {
            result->additional_stops = new const char*[result->stop_count + 1];
            for (size_t i = 0; i < params.additional_stops.size(); i++) {
                result->additional_stops[i] = strdup(params.additional_stops[i].c_str());
            }
            result->additional_stops[result->stop_count] = nullptr;
        }

        // Note: tmpls is a unique_ptr, automatically cleaned up
    } catch (const std::exception& e) {
        if (g_verbose) {
            fprintf(stderr, "binding: apply_chat_template_with_tools error: %s\n", e.what());
        }
        if (result->prompt == nullptr) result->prompt = strdup("");
        if (result->grammar == nullptr) result->grammar = strdup("");
    } catch (...) {
        if (g_verbose) {
            fprintf(stderr, "binding: apply_chat_template_with_tools unknown error\n");
        }
        if (result->prompt == nullptr) result->prompt = strdup("");
        if (result->grammar == nullptr) result->grammar = strdup("");
    }

    return result;
}

void binding_free_chat_params(binding_chat_params* params) {
    if (params == nullptr) {
        return;
    }

    free((void*)params->prompt);
    free((void*)params->grammar);

    if (params->grammar_triggers != nullptr) {
        for (int32_t i = 0; i < params->trigger_count; i++) {
            free((void*)params->grammar_triggers[i]);
        }
        delete[] params->grammar_triggers;
    }

    if (params->additional_stops != nullptr) {
        for (int32_t i = 0; i < params->stop_count; i++) {
            free((void*)params->additional_stops[i]);
        }
        delete[] params->additional_stops;
    }

    delete params;
}
