package zeus

/*
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"io"
	"iter"
	"slices"
	"unsafe"
)

// Session represents a conversation state that tracks token history.
// Generate/GenerateSequence mutate the session, appending new tokens.
// Use Checkpoint() before generation to save state for branching.
// Note: All sessions share one KV cache. Switching between unrelated sessions recomputes all tokens.
type Session interface {
	// Generate processes the prompt, returns new tokens and moves the session forward.
	Generate(ctx context.Context, prompt string, opts ...GenerateOption) io.ReadCloser

	// GenerateSequence processes the prompt, returns new tokens as yield iterator and moves the session forward.
	GenerateSequence(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq2[Token, error]

	// GenerateSequenceWithLogprobs is like GenerateSequence but returns tokens with probability information.
	// topK specifies how many top alternatives to include (0 = just the selected token's prob/logit).
	GenerateSequenceWithLogprobs(ctx context.Context, prompt string, topK int, opts ...GenerateOption) iter.Seq2[TokenWithLogprobs, error]

	// Checkpoint creates a snapshot of the current session state.
	// Both the original and checkpoint can be used independently for branching.
	// Note: All sessions share one KV cache; switching between divergent sessions recomputes tokens from the common prefix.
	Checkpoint() Session

	// Backtrack returns to the state before the last Generate call.
	// Returns ok=false if this is the initial session (no parent).
	Backtrack() (Session, bool)

	// Tokens returns a copy of the token history for this session.
	Tokens() []int

	// Text returns the full text of this session (detokenized).
	Text() (string, error)

	// TokenCount returns the number of tokens in this session.
	TokenCount() int

	// ContextUsed returns the percentage of context used in this session.
	ContextUsed() float64

	// Model returns the parent model.
	Model() Model
}

// NewSession creates a new empty session for text generation.
func (m *model) NewSession() *session {
	if m == nil {
		return nil
	}
	return &session{
		model:  m,
		tokens: nil,
		parent: nil,
	}
}

type session struct {
	model  *model
	tokens []int
	parent *session
}

// Generate processes the prompt, returns new tokens and moves the session forward.
func (s *session) Generate(ctx context.Context, prompt string, opts ...GenerateOption) io.ReadCloser {
	pr, pw := io.Pipe()
	if s == nil {
		pw.CloseWithError(ErrSessionIsNil)
		return pr
	}

	go func() {
		defer pw.Close()

		for tok, err := range s.GenerateSequence(ctx, prompt, opts...) {
			if err != nil {
				pw.CloseWithError(err)
				return
			}
			if _, err := pw.Write([]byte(tok.Text)); err != nil {
				return
			}
		}
	}()

	return pr
}

// GenerateSequence processes the prompt, returns new tokens as yield iterator and moves the session forward.
func (s *session) GenerateSequence(ctx context.Context, prompt string, opts ...GenerateOption) iter.Seq2[Token, error] {
	return func(yield func(Token, error) bool) {
		if s == nil {
			yield(Token{}, ErrSessionIsNil)
			return
		}
		if s.model.isClosed() {
			yield(Token{}, ErrModelClosed)
			return
		}

		// Tokenize the prompt
		promptTokens, err := s.model.Tokenize(prompt, len(s.tokens) == 0)
		if err != nil {
			yield(Token{}, err)
			return
		}

		// Combine existing tokens + prompt tokens
		allTokens := slices.Concat(s.tokens, promptTokens)
		if len(allTokens) <= 0 {
			return
		}

		s.model.kvMutex.Lock()
		defer s.model.kvMutex.Unlock()

		// Prepare KV cache
		cTokens := make([]C.int32_t, len(allTokens))
		for i, t := range allTokens {
			cTokens[i] = C.int32_t(t)
		}
		var tokensPtr *C.int32_t = &cTokens[0]

		result := C.binding_prepare_cache(s.model.ptr, tokensPtr, C.int32_t(len(allTokens)))
		if result.status != C.BINDING_OK {
			yield(Token{}, &GenerationError{Stage: "prepare_cache", Message: "failed to prepare cache"})
			return
		}

		tokensToProcess := allTokens[result.common_prefix:]

		cfg := DefaultGenerateConfig()
		for _, opt := range opts {
			opt(&cfg)
		}

		// Prepare C config
		cConfig := (*C.binding_generate_config)(C.malloc(C.sizeof_binding_generate_config))
		defer C.free(unsafe.Pointer(cConfig))
		*cConfig = C.binding_generate_config_default()
		cConfig.max_tokens = C.int32_t(cfg.MaxTokens)
		cConfig.seed = C.int32_t(cfg.Seed)
		cConfig.threads = C.int32_t(cfg.Threads)
		cConfig.top_k = C.int32_t(cfg.TopK)
		cConfig.top_p = C.float(cfg.TopP)
		cConfig.min_p = C.float(cfg.MinP)
		cConfig.temperature = C.float(cfg.Temperature)
		cConfig.repeat_penalty = C.float(cfg.RepeatPenalty)
		cConfig.repeat_last_n = C.int32_t(cfg.RepeatLastN)
		cConfig.frequency_penalty = C.float(cfg.FrequencyPenalty)
		cConfig.presence_penalty = C.float(cfg.PresencePenalty)
		cConfig.mirostat = C.int32_t(cfg.Mirostat)
		cConfig.mirostat_tau = C.float(cfg.MirostatTau)
		cConfig.mirostat_eta = C.float(cfg.MirostatEta)
		cConfig.ignore_eos = C.bool(cfg.IgnoreEOS)
		cConfig.reasoning_enabled = C.bool(cfg.ReasoningEnabled)

		// Handle grammar
		var cGrammar *C.char
		if cfg.Grammar != "" {
			cGrammar = C.CString(cfg.Grammar)
			defer C.free(unsafe.Pointer(cGrammar))
			cConfig.grammar = cGrammar
		}

		// Handle lazy grammar
		cConfig.grammar_lazy = C.bool(cfg.grammarLazy)
		if cfg.grammarLazy && len(cfg.grammarTriggers) > 0 {
			// Allocate C trigger array
			cTriggersSize := C.size_t(len(cfg.grammarTriggers)) * C.size_t(unsafe.Sizeof(C.binding_grammar_trigger{}))
			cTriggers := (*C.binding_grammar_trigger)(C.malloc(cTriggersSize))
			defer C.free(unsafe.Pointer(cTriggers))

			triggersSlice := unsafe.Slice(cTriggers, len(cfg.grammarTriggers))
			var cTriggerStrings []*C.char

			for i, trigger := range cfg.grammarTriggers {
				triggersSlice[i]._type = C.binding_trigger_type(trigger.Type)
				triggersSlice[i].token = C.int32_t(trigger.Token)
				if trigger.Value != "" {
					cStr := C.CString(trigger.Value)
					cTriggerStrings = append(cTriggerStrings, cStr)
					triggersSlice[i].value = cStr
				} else {
					triggersSlice[i].value = nil
				}
			}
			// Defer freeing trigger strings
			defer func() {
				for _, s := range cTriggerStrings {
					C.free(unsafe.Pointer(s))
				}
			}()

			cConfig.grammar_triggers = cTriggers
			cConfig.grammar_trigger_count = C.int32_t(len(cfg.grammarTriggers))
		}

		// Handle stop sequences
		var cStopSequences **C.char
		if len(cfg.StopSequences) > 0 {
			cStopSequences = (**C.char)(C.malloc(C.size_t(len(cfg.StopSequences)) * C.size_t(unsafe.Sizeof((*C.char)(nil)))))
			defer C.free(unsafe.Pointer(cStopSequences))

			stopPtrs := make([]*C.char, len(cfg.StopSequences))
			for i, seq := range cfg.StopSequences {
				stopPtrs[i] = C.CString(seq)
				defer C.free(unsafe.Pointer(stopPtrs[i]))
				*(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cStopSequences)) + uintptr(i)*unsafe.Sizeof((*C.char)(nil)))) = stopPtrs[i]
			}
			cConfig.stop_sequences = cStopSequences
			cConfig.stop_count = C.int32_t(len(cfg.StopSequences))
		}

		// Convert tokens to process to C array
		var newTokensPtr *C.int32_t
		if len(tokensToProcess) > 0 {
			cNewTokens := make([]C.int32_t, len(tokensToProcess))
			for i, t := range tokensToProcess {
				cNewTokens[i] = C.int32_t(t)
			}
			newTokensPtr = &cNewTokens[0]
		}

		gen := C.binding_generate_start(s.model.ptr, newTokensPtr, C.int32_t(len(tokensToProcess)), cConfig)
		if gen == nil {
			yield(Token{}, ErrPromptTooLong)
			return
		}
		defer C.binding_generate_free(gen)

		// Save current state as parent for Backtrack support
		s.parent = &session{
			model:  s.model,
			tokens: s.tokens,
			parent: s.parent,
		}
		s.tokens = allTokens

		var cToken C.binding_token
		for {
			select {
			case <-ctx.Done():
				yield(Token{}, ctx.Err())
				return
			default:
			}

			genResult := C.binding_generate_next(gen, &cToken)

			switch genResult {
			case C.BINDING_OK:
				tok := Token{
					Text: C.GoString(cToken.text),
					ID:   int(cToken.id),
				}
				s.tokens = append(s.tokens, tok.ID) // Track generated token
				if !yield(tok, nil) {
					return
				}
			case C.BINDING_DONE:
				return
			case C.BINDING_ERR_DECODE:
				yield(Token{}, ErrDecodeFailed)
				return
			default:
				yield(Token{}, &GenerationError{Stage: "generate", Message: "generation failed"})
				return
			}
		}
	}
}

// GenerateSequenceWithLogprobs is like GenerateSequence but returns tokens with probability information.
// topK specifies how many top alternatives to include (0 = just the selected token's prob/logit).
func (s *session) GenerateSequenceWithLogprobs(ctx context.Context, prompt string, topK int, opts ...GenerateOption) iter.Seq2[TokenWithLogprobs, error] {
	return func(yield func(TokenWithLogprobs, error) bool) {
		if s == nil {
			yield(TokenWithLogprobs{}, ErrSessionIsNil)
			return
		}
		if s.model == nil {
			yield(TokenWithLogprobs{}, ErrModelIsNil)
			return
		}
		if s.model.isClosed() {
			yield(TokenWithLogprobs{}, ErrModelClosed)
			return
		}

		// Tokenize the prompt
		promptTokens, err := s.model.Tokenize(prompt, len(s.tokens) == 0)
		if err != nil {
			yield(TokenWithLogprobs{}, err)
			return
		}

		// Combine existing tokens + prompt tokens
		allTokens := slices.Concat(s.tokens, promptTokens)
		if len(allTokens) <= 0 {
			return
		}

		s.model.kvMutex.Lock()
		defer s.model.kvMutex.Unlock()

		// Prepare KV cache
		cTokens := make([]C.int32_t, len(allTokens))
		for i, t := range allTokens {
			cTokens[i] = C.int32_t(t)
		}
		var tokensPtr *C.int32_t = &cTokens[0]

		result := C.binding_prepare_cache(s.model.ptr, tokensPtr, C.int32_t(len(allTokens)))
		if result.status != C.BINDING_OK {
			yield(TokenWithLogprobs{}, &GenerationError{Stage: "prepare_cache", Message: "failed to prepare cache"})
			return
		}

		tokensToProcess := allTokens[result.common_prefix:]

		cfg := DefaultGenerateConfig()
		for _, opt := range opts {
			opt(&cfg)
		}

		// Prepare C config
		cConfig := (*C.binding_generate_config)(C.malloc(C.sizeof_binding_generate_config))
		defer C.free(unsafe.Pointer(cConfig))
		*cConfig = C.binding_generate_config_default()
		cConfig.max_tokens = C.int32_t(cfg.MaxTokens)
		cConfig.seed = C.int32_t(cfg.Seed)
		cConfig.threads = C.int32_t(cfg.Threads)
		cConfig.top_k = C.int32_t(cfg.TopK)
		cConfig.top_p = C.float(cfg.TopP)
		cConfig.min_p = C.float(cfg.MinP)
		cConfig.temperature = C.float(cfg.Temperature)
		cConfig.repeat_penalty = C.float(cfg.RepeatPenalty)
		cConfig.repeat_last_n = C.int32_t(cfg.RepeatLastN)
		cConfig.frequency_penalty = C.float(cfg.FrequencyPenalty)
		cConfig.presence_penalty = C.float(cfg.PresencePenalty)
		cConfig.mirostat = C.int32_t(cfg.Mirostat)
		cConfig.mirostat_tau = C.float(cfg.MirostatTau)
		cConfig.mirostat_eta = C.float(cfg.MirostatEta)
		cConfig.ignore_eos = C.bool(cfg.IgnoreEOS)
		cConfig.reasoning_enabled = C.bool(cfg.ReasoningEnabled)

		// Handle grammar
		var cGrammar *C.char
		if cfg.Grammar != "" {
			cGrammar = C.CString(cfg.Grammar)
			defer C.free(unsafe.Pointer(cGrammar))
			cConfig.grammar = cGrammar
		}

		// Handle lazy grammar
		cConfig.grammar_lazy = C.bool(cfg.grammarLazy)
		if cfg.grammarLazy && len(cfg.grammarTriggers) > 0 {
			// Allocate C trigger array
			cTriggersSize := C.size_t(len(cfg.grammarTriggers)) * C.size_t(unsafe.Sizeof(C.binding_grammar_trigger{}))
			cTriggers := (*C.binding_grammar_trigger)(C.malloc(cTriggersSize))
			defer C.free(unsafe.Pointer(cTriggers))

			triggersSlice := unsafe.Slice(cTriggers, len(cfg.grammarTriggers))
			var cTriggerStrings []*C.char

			for i, trigger := range cfg.grammarTriggers {
				triggersSlice[i]._type = C.binding_trigger_type(trigger.Type)
				triggersSlice[i].token = C.int32_t(trigger.Token)
				if trigger.Value != "" {
					cStr := C.CString(trigger.Value)
					cTriggerStrings = append(cTriggerStrings, cStr)
					triggersSlice[i].value = cStr
				} else {
					triggersSlice[i].value = nil
				}
			}
			// Defer freeing trigger strings
			defer func() {
				for _, s := range cTriggerStrings {
					C.free(unsafe.Pointer(s))
				}
			}()

			cConfig.grammar_triggers = cTriggers
			cConfig.grammar_trigger_count = C.int32_t(len(cfg.grammarTriggers))
		}

		// Handle stop sequences
		var cStopSequences **C.char
		if len(cfg.StopSequences) > 0 {
			cStopSequences = (**C.char)(C.malloc(C.size_t(len(cfg.StopSequences)) * C.size_t(unsafe.Sizeof((*C.char)(nil)))))
			defer C.free(unsafe.Pointer(cStopSequences))

			stopPtrs := make([]*C.char, len(cfg.StopSequences))
			for i, seq := range cfg.StopSequences {
				stopPtrs[i] = C.CString(seq)
				defer C.free(unsafe.Pointer(stopPtrs[i]))
				*(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cStopSequences)) + uintptr(i)*unsafe.Sizeof((*C.char)(nil)))) = stopPtrs[i]
			}
			cConfig.stop_sequences = cStopSequences
			cConfig.stop_count = C.int32_t(len(cfg.StopSequences))
		}

		// Convert tokens to process to C array
		var newTokensPtr *C.int32_t
		if len(tokensToProcess) > 0 {
			cNewTokens := make([]C.int32_t, len(tokensToProcess))
			for i, t := range tokensToProcess {
				cNewTokens[i] = C.int32_t(t)
			}
			newTokensPtr = &cNewTokens[0]
		}

		gen := C.binding_generate_start(s.model.ptr, newTokensPtr, C.int32_t(len(tokensToProcess)), cConfig)
		if gen == nil {
			yield(TokenWithLogprobs{}, ErrPromptTooLong)
			return
		}
		defer C.binding_generate_free(gen)

		// Save current state as parent for Backtrack support
		s.parent = &session{
			model:  s.model,
			tokens: s.tokens,
			parent: s.parent,
		}
		s.tokens = allTokens

		var cToken C.binding_token_with_logprobs
		for {
			select {
			case <-ctx.Done():
				yield(TokenWithLogprobs{}, ctx.Err())
				return
			default:
			}

			genResult := C.binding_generate_next_with_logprobs(gen, &cToken, C.int32_t(topK))

			switch genResult {
			case C.BINDING_OK:
				tok := TokenWithLogprobs{
					Token: Token{
						Text: C.GoString(cToken.text),
						ID:   int(cToken.id),
					},
					Prob:  float32(cToken.prob),
					Logit: float32(cToken.logit),
				}

				// Copy top-k alternatives if present
				if cToken.top_k_count > 0 && cToken.top_k != nil {
					tok.TopK = make([]TokenProb, int(cToken.top_k_count))
					for i := 0; i < int(cToken.top_k_count); i++ {
						cProb := *(*C.binding_token_prob)(unsafe.Pointer(uintptr(unsafe.Pointer(cToken.top_k)) + uintptr(i)*unsafe.Sizeof(C.binding_token_prob{})))
						tok.TopK[i] = TokenProb{
							Token: int(cProb.token_id),
							Text:  s.model.TokenToText(int(cProb.token_id)),
							Prob:  float32(cProb.prob),
							Logit: float32(cProb.logit),
						}
					}
					// Free the C-allocated top_k array
					C.free(unsafe.Pointer(cToken.top_k))
				}

				s.tokens = append(s.tokens, tok.ID) // Track generated token
				if !yield(tok, nil) {
					return
				}
			case C.BINDING_DONE:
				return
			case C.BINDING_ERR_DECODE:
				yield(TokenWithLogprobs{}, ErrDecodeFailed)
				return
			default:
				yield(TokenWithLogprobs{}, &GenerationError{Stage: "generate", Message: "generation failed"})
				return
			}
		}
	}
}

// Checkpoint creates a snapshot of the current session state.
// Both the original and checkpoint can be used independently for branching.
// Note: All sessions share one KV cache; switching between divergent sessions recomputes tokens from the common prefix.
func (s *session) Checkpoint() Session {
	if s == nil {
		return nil
	}
	return &session{
		model:  s.model,
		tokens: s.tokens,
		parent: s.parent,
	}
}

// Backtrack returns to the state before the last Generate call.
// Returns ok=false if this is the initial session (no parent).
func (s *session) Backtrack() (Session, bool) {
	if s == nil {
		return nil, false
	}
	if s.parent == nil {
		return s, false
	}
	return s.parent, true
}

// Tokens returns a copy of the token history for this session.
func (s *session) Tokens() []int {
	if s == nil {
		return nil
	}
	return slices.Clone(s.tokens)
}

// Text returns the full text of this session (detokenized).
func (s *session) Text() (string, error) {
	if s == nil {
		return "", ErrSessionIsNil
	}
	if len(s.tokens) <= 0 {
		return "", nil
	}
	return s.model.Detokenize(s.tokens)
}

// TokenCount returns the number of tokens in this session.
func (s *session) TokenCount() int {
	if s == nil {
		return -1
	}
	return len(s.tokens)
}

// ContextUsed returns the percentage of context used in this session.
func (s *session) ContextUsed() float64 {
	if s == nil {
		return -1
	}
	return float64(len(s.tokens)) / float64(s.model.ContextSize())
}

// Model returns the parent model.
func (s *session) Model() Model {
	if s == nil {
		return nil
	}
	return s.model
}
