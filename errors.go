package zeus

import (
	"errors"
	"fmt"
)

// Sentinel errors for use with errors.Is()
var (
	ErrModelClosed        = errors.New("zeus: model is closed")
	ErrEmbeddingsDisabled = errors.New("zeus: model loaded without embeddings support")
	ErrPromptTooLong      = errors.New("zeus: prompt exceeds context size")
	ErrDecodeFailed       = errors.New("zeus: decode operation failed")
	ErrSessionIsNil       = errors.New("zeus: session is nil and not defined")
	ErrModelIsNil         = errors.New("zeus: model is nil and not defined")
	ErrChatIsNil          = errors.New("zeus: chat is nil and not defined")
)

// ModelLoadError provides details about model loading failures.
type ModelLoadError struct {
	Path   string
	Reason string
}

func (e *ModelLoadError) Error() string {
	return fmt.Sprintf("zeus: failed to load model %q: %s", e.Path, e.Reason)
}

// GenerationError provides details about text generation failures.
type GenerationError struct {
	Stage   string // "tokenize", "decode", "sample"
	Message string
}

func (e *GenerationError) Error() string {
	return fmt.Sprintf("zeus: generation failed during %s: %s", e.Stage, e.Message)
}

// TokenizeError provides details about tokenization failures.
type TokenizeError struct {
	Text    string
	Message string
}

func (e *TokenizeError) Error() string {
	if len(e.Text) > 50 {
		return fmt.Sprintf("zeus: tokenization failed for %q...: %s", e.Text[:50], e.Message)
	}
	return fmt.Sprintf("zeus: tokenization failed for %q: %s", e.Text, e.Message)
}

// EmbeddingError provides details about embedding extraction failures.
type EmbeddingError struct {
	Message string
}

func (e *EmbeddingError) Error() string {
	return fmt.Sprintf("zeus: embedding extraction failed: %s", e.Message)
}

// ChatTemplateError provides details about chat template failures.
type ChatTemplateError struct {
	Message string
}

func (e *ChatTemplateError) Error() string {
	return fmt.Sprintf("zeus: chat template error: %s", e.Message)
}
