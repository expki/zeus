package zeus

import (
	"errors"
	"testing"
)

func TestTokenize_Basic(t *testing.T) {
	tokens, err := testModel.Tokenize("Hello world", false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}
	if len(tokens) == 0 {
		t.Error("Tokenize() returned empty slice")
	}
}

func TestTokenize_WithSpecial(t *testing.T) {
	tokensWithSpecial, err := testModel.Tokenize("Hello", true)
	if err != nil {
		t.Fatalf("Tokenize(addSpecial=true) error = %v", err)
	}

	tokensNoSpecial, err := testModel.Tokenize("Hello", false)
	if err != nil {
		t.Fatalf("Tokenize(addSpecial=false) error = %v", err)
	}

	// Log the difference (behavior is model-specific)
	t.Logf("Tokenize with special: %d tokens, without: %d tokens", len(tokensWithSpecial), len(tokensNoSpecial))
	t.Logf("BOS token = %d, first token with special = %d", testModel.BOS(), tokensWithSpecial[0])
}

func TestTokenize_Empty(t *testing.T) {
	tokens, err := testModel.Tokenize("", false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}
	// Empty string should return empty tokens (or minimal)
	if len(tokens) > 0 {
		t.Logf("Empty string tokenized to %d tokens (model-specific)", len(tokens))
	}
}

func TestTokenizeCount(t *testing.T) {
	text := "The quick brown fox"

	count, err := testModel.TokenizeCount(text, false)
	if err != nil {
		t.Fatalf("TokenizeCount() error = %v", err)
	}

	tokens, err := testModel.Tokenize(text, false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}

	if count != len(tokens) {
		t.Errorf("TokenizeCount() = %d, len(Tokenize()) = %d, want equal", count, len(tokens))
	}
}

func TestDetokenize_Basic(t *testing.T) {
	tokens, err := testModel.Tokenize("Hello world", false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}

	text, err := testModel.Detokenize(tokens)
	if err != nil {
		t.Fatalf("Detokenize() error = %v", err)
	}

	if text == "" {
		t.Error("Detokenize() returned empty string")
	}
}

func TestTokenize_Roundtrip(t *testing.T) {
	original := "The quick brown fox"

	tokens, err := testModel.Tokenize(original, false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}

	result, err := testModel.Detokenize(tokens)
	if err != nil {
		t.Fatalf("Detokenize() error = %v", err)
	}

	if result != original {
		t.Errorf("Roundtrip failed: got %q, want %q", result, original)
	}
}

func TestDetokenize_Empty(t *testing.T) {
	text, err := testModel.Detokenize([]int{})
	if err != nil {
		t.Fatalf("Detokenize() error = %v", err)
	}
	if text != "" {
		t.Errorf("Detokenize([]) = %q, want empty", text)
	}
}

func TestDetokenizeLength(t *testing.T) {
	tokens, err := testModel.Tokenize("Hello world", false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}

	length, err := testModel.DetokenizeLength(tokens)
	if err != nil {
		t.Fatalf("DetokenizeLength() error = %v", err)
	}

	text, err := testModel.Detokenize(tokens)
	if err != nil {
		t.Fatalf("Detokenize() error = %v", err)
	}

	if length != len(text) {
		t.Errorf("DetokenizeLength() = %d, len(Detokenize()) = %d, want equal", length, len(text))
	}
}

func TestTokenToText(t *testing.T) {
	// Get EOS token and convert to text
	eos := testModel.EOS()
	text := testModel.TokenToText(eos)

	// EOS token should have some text representation
	t.Logf("EOS token %d -> %q", eos, text)
}

func TestIsSpecialToken(t *testing.T) {
	bos := testModel.BOS()
	eos := testModel.EOS()

	if !testModel.IsSpecialToken(bos) {
		t.Errorf("IsSpecialToken(BOS=%d) = false, want true", bos)
	}
	if !testModel.IsSpecialToken(eos) {
		t.Errorf("IsSpecialToken(EOS=%d) = false, want true", eos)
	}

	// Tokenize a regular word and check it's not special
	tokens, err := testModel.Tokenize("hello", false)
	if err != nil {
		t.Fatalf("Tokenize() error = %v", err)
	}
	if len(tokens) > 0 {
		// First token of "hello" should not be special
		if testModel.IsSpecialToken(tokens[0]) {
			t.Logf("Token %d for 'hello' is marked as special (may be model-specific)", tokens[0])
		}
	}
}

func TestIsEOG(t *testing.T) {
	eos := testModel.EOS()

	if !testModel.IsEOG(eos) {
		t.Errorf("IsEOG(EOS=%d) = false, want true", eos)
	}
}

func TestTokenize_ClosedModel(t *testing.T) {
	// Load a fresh model and close it
	m, err := LoadModel(testModelPath, WithContextSize(512))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	m.Close()

	_, err = m.Tokenize("test", false)
	if !errors.Is(err, ErrModelClosed) {
		t.Errorf("Tokenize on closed model error = %v, want ErrModelClosed", err)
	}
}

func TestTokenize_NilModel(t *testing.T) {
	var m *model = nil

	_, err := m.Tokenize("test", false)
	if !errors.Is(err, ErrModelIsNil) {
		t.Errorf("Tokenize on nil model error = %v, want ErrModelIsNil", err)
	}
}
