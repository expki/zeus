package zeus

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestSentinelErrors_Is(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{"ErrModelClosed", ErrModelClosed},
		{"ErrEmbeddingsDisabled", ErrEmbeddingsDisabled},
		{"ErrPromptTooLong", ErrPromptTooLong},
		{"ErrDecodeFailed", ErrDecodeFailed},
		{"ErrSessionIsNil", ErrSessionIsNil},
		{"ErrModelIsNil", ErrModelIsNil},
		{"ErrChatIsNil", ErrChatIsNil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// errors.Is should work
			if !errors.Is(tt.err, tt.err) {
				t.Errorf("errors.Is(%v, %v) = false, want true", tt.err, tt.err)
			}

			// Wrapped errors should also work
			wrapped := fmt.Errorf("wrapped: %w", tt.err)
			if !errors.Is(wrapped, tt.err) {
				t.Errorf("errors.Is(wrapped, %v) = false, want true", tt.err)
			}
		})
	}
}

func TestSentinelErrors_NotEqual(t *testing.T) {
	if errors.Is(ErrModelClosed, ErrModelIsNil) {
		t.Error("errors.Is(ErrModelClosed, ErrModelIsNil) should be false")
	}
}

func TestModelLoadError_Format(t *testing.T) {
	err := &ModelLoadError{
		Path:   "/path/to/model.gguf",
		Reason: "file not found",
	}

	msg := err.Error()

	if !strings.Contains(msg, "/path/to/model.gguf") {
		t.Errorf("error should contain path: %s", msg)
	}
	if !strings.Contains(msg, "file not found") {
		t.Errorf("error should contain reason: %s", msg)
	}
	if !strings.Contains(msg, "zeus:") {
		t.Errorf("error should have zeus prefix: %s", msg)
	}
}

func TestGenerationError_Format(t *testing.T) {
	err := &GenerationError{
		Stage:   "decode",
		Message: "decode failed",
	}

	msg := err.Error()

	if !strings.Contains(msg, "decode") {
		t.Errorf("error should contain stage: %s", msg)
	}
	if !strings.Contains(msg, "decode failed") {
		t.Errorf("error should contain message: %s", msg)
	}
	if !strings.Contains(msg, "zeus:") {
		t.Errorf("error should have zeus prefix: %s", msg)
	}
}

func TestTokenizeError_Format(t *testing.T) {
	err := &TokenizeError{
		Text:    "short text",
		Message: "tokenization failed",
	}

	msg := err.Error()

	if !strings.Contains(msg, "short text") {
		t.Errorf("error should contain text: %s", msg)
	}
	if !strings.Contains(msg, "tokenization failed") {
		t.Errorf("error should contain message: %s", msg)
	}
}

func TestTokenizeError_Format_Long(t *testing.T) {
	// Text longer than 50 characters should be truncated
	longText := strings.Repeat("a", 100)
	err := &TokenizeError{
		Text:    longText,
		Message: "failed",
	}

	msg := err.Error()

	// Should contain first 50 chars + "..."
	if !strings.Contains(msg, strings.Repeat("a", 50)) {
		t.Errorf("error should contain truncated text: %s", msg)
	}
	if !strings.Contains(msg, "...") {
		t.Errorf("error should contain ellipsis for long text: %s", msg)
	}
	// Should NOT contain the full 100 chars
	if strings.Contains(msg, strings.Repeat("a", 100)) {
		t.Errorf("error should truncate long text: %s", msg)
	}
}

func TestEmbeddingError_Format(t *testing.T) {
	err := &EmbeddingError{
		Message: "embedding failed",
	}

	msg := err.Error()

	if !strings.Contains(msg, "embedding") {
		t.Errorf("error should contain 'embedding': %s", msg)
	}
	if !strings.Contains(msg, "embedding failed") {
		t.Errorf("error should contain message: %s", msg)
	}
}

func TestChatTemplateError_Format(t *testing.T) {
	err := &ChatTemplateError{
		Message: "template error",
	}

	msg := err.Error()

	if !strings.Contains(msg, "template") {
		t.Errorf("error should contain 'template': %s", msg)
	}
	if !strings.Contains(msg, "template error") {
		t.Errorf("error should contain message: %s", msg)
	}
}
