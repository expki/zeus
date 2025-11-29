package zeus

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestNewSession(t *testing.T) {
	session := testModel.NewSession()
	if session == nil {
		t.Fatal("NewSession() returned nil")
	}
}

func TestSession_GenerateSequence_Basic(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var tokens []Token
	for tok, err := range session.GenerateSequence(ctx, "Count: 1, 2,", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
		tokens = append(tokens, tok)
	}

	if len(tokens) == 0 {
		t.Error("GenerateSequence() should generate at least one token")
	}
	if len(tokens) > 10 {
		t.Errorf("GenerateSequence() generated %d tokens, want <= 10", len(tokens))
	}
}

func TestSession_GenerateSequence_Cancelled(t *testing.T) {
	session := testModel.NewSession()

	// Create a context that we'll cancel immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	var sawError bool
	for _, err := range session.GenerateSequence(ctx, "Hello", WithMaxTokens(5)) {
		if err != nil {
			sawError = true
			if !errors.Is(err, context.Canceled) {
				t.Errorf("expected context.Canceled, got %v", err)
			}
			break
		}
	}

	if !sawError {
		t.Log("Generation completed before cancellation could take effect")
	}
}

func TestSession_Tokens(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Initially empty
	if tokens := session.Tokens(); len(tokens) != 0 {
		t.Errorf("fresh session Tokens() = %v, want empty", tokens)
	}

	// Generate some tokens
	for _, err := range session.GenerateSequence(ctx, "Hi", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Should have tokens now
	tokens := session.Tokens()
	if len(tokens) == 0 {
		t.Error("Tokens() should not be empty after generation")
	}

	// Verify it's a copy
	tokens[0] = -999
	if session.Tokens()[0] == -999 {
		t.Error("Tokens() should return a copy, not the original slice")
	}
}

func TestSession_Text(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Initially empty
	text, err := session.Text()
	if err != nil {
		t.Fatalf("Text() error = %v", err)
	}
	if text != "" {
		t.Errorf("fresh session Text() = %q, want empty", text)
	}

	// Generate some tokens
	for _, err := range session.GenerateSequence(ctx, "Hi", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Should have text now
	text, err = session.Text()
	if err != nil {
		t.Fatalf("Text() error = %v", err)
	}
	if text == "" {
		t.Error("Text() should not be empty after generation")
	}
}

func TestSession_TokenCount(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if session.TokenCount() != 0 {
		t.Errorf("fresh session TokenCount() = %d, want 0", session.TokenCount())
	}

	// Generate some tokens
	for _, err := range session.GenerateSequence(ctx, "Hi", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	count := session.TokenCount()
	tokens := session.Tokens()
	if count != len(tokens) {
		t.Errorf("TokenCount() = %d, len(Tokens()) = %d, want equal", count, len(tokens))
	}
}

func TestSession_ContextUsed(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Initially zero
	if cu := session.ContextUsed(); cu != 0 {
		t.Errorf("fresh session ContextUsed() = %f, want 0", cu)
	}

	// Generate some tokens
	for _, err := range session.GenerateSequence(ctx, "Hi", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	cu := session.ContextUsed()
	if cu <= 0 || cu > 1 {
		t.Errorf("ContextUsed() = %f, want in range (0, 1]", cu)
	}
}

func TestSession_Model(t *testing.T) {
	session := testModel.NewSession()
	m := session.Model()

	if m == nil {
		t.Fatal("Model() returned nil")
	}
	if m != testModel {
		t.Error("Model() should return parent model")
	}
}

func TestSession_Checkpoint(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Generate some tokens
	for _, err := range session.GenerateSequence(ctx, "One", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Create checkpoint
	checkpoint := session.Checkpoint()
	checkpointCount := checkpoint.TokenCount()

	// Generate more tokens on original session
	for _, err := range session.GenerateSequence(ctx, " Two", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Checkpoint should be unchanged
	if checkpoint.TokenCount() != checkpointCount {
		t.Errorf("checkpoint TokenCount changed from %d to %d", checkpointCount, checkpoint.TokenCount())
	}

	// Original session should have more tokens
	if session.TokenCount() <= checkpointCount {
		t.Errorf("session should have more tokens than checkpoint")
	}
}

func TestSession_Backtrack(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Generate first batch
	for _, err := range session.GenerateSequence(ctx, "First", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}
	firstCount := session.TokenCount()

	// Generate second batch
	for _, err := range session.GenerateSequence(ctx, " Second", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Backtrack
	previous, ok := session.Backtrack()
	if !ok {
		t.Fatal("Backtrack() returned ok=false")
	}

	// Should be back to first count
	if previous.TokenCount() != firstCount {
		t.Errorf("after Backtrack() TokenCount() = %d, want %d", previous.TokenCount(), firstCount)
	}
}

func TestSession_Backtrack_Initial(t *testing.T) {
	session := testModel.NewSession()

	// Backtrack on fresh session should return ok=false
	_, ok := session.Backtrack()
	if ok {
		t.Error("Backtrack() on fresh session should return ok=false")
	}
}

func TestSession_Generate_Reader(t *testing.T) {
	session := testModel.NewSession()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	reader := session.Generate(ctx, "Hello", WithMaxTokens(5))
	defer reader.Close()

	buf := make([]byte, 1024)
	n, err := reader.Read(buf)
	if err != nil && n == 0 {
		t.Fatalf("Read() error = %v, n = %d", err, n)
	}

	if n == 0 {
		t.Error("Read() returned 0 bytes")
	}
}

func TestSession_NilReceiver(t *testing.T) {
	var s *session = nil

	if s.Tokens() != nil {
		t.Error("nil session Tokens() should return nil")
	}
	if s.TokenCount() != -1 {
		t.Errorf("nil session TokenCount() = %d, want -1", s.TokenCount())
	}
	if s.ContextUsed() != -1 {
		t.Errorf("nil session ContextUsed() = %f, want -1", s.ContextUsed())
	}
	if s.Model() != nil {
		t.Error("nil session Model() should return nil")
	}
	if cp := s.Checkpoint(); cp != nil {
		t.Error("nil session Checkpoint() should return nil")
	}
	if _, ok := s.Backtrack(); ok {
		t.Error("nil session Backtrack() should return ok=false")
	}
}
