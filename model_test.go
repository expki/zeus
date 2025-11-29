package zeus

import (
	"errors"
	"testing"
)

func TestLoadModel_Basic(t *testing.T) {
	// Use the shared testModel loaded in TestMain
	if testModel == nil {
		t.Fatal("testModel should not be nil")
	}
}

func TestLoadModel_WithOptions(t *testing.T) {
	m, err := LoadModel(testModelPath,
		WithContextSize(512),
		WithKVCacheType(KVCacheQ8_0),
		WithSeed(42),
	)
	if err != nil {
		t.Errorf("model should not error = %v", err)
	}
	if m == nil {
		t.Fatal("model should not be nil")
	}
	defer m.Close()

	// Verify context size was applied
	if cs := m.ContextSize(); cs != 512 {
		t.Errorf("ContextSize() = %d, want 512", cs)
	}
}

func TestLoadModel_InvalidPath(t *testing.T) {
	_, err := LoadModel("/nonexistent/path/to/model.gguf")
	if err == nil {
		t.Fatal("expected error for invalid path")
	}

	var loadErr *ModelLoadError
	if !errors.As(err, &loadErr) {
		t.Errorf("expected *ModelLoadError, got %T", err)
	}
}

func TestModel_Close(t *testing.T) {
	m, err := LoadModel(testModelPath,
		WithContextSize(100),
		WithKVCacheType(KVCacheQ8_0),
		WithSeed(42),
	)
	if err != nil {
		t.Errorf("model should not error = %v", err)
	}

	// Close should not error
	if err := m.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Operations on closed model should return appropriate values
	if cs := m.ContextSize(); cs != -1 {
		t.Errorf("ContextSize() on closed model = %d, want -1", cs)
	}

	// Tokenize should return ErrModelClosed
	_, err = m.Tokenize("test", false)
	if !errors.Is(err, ErrModelClosed) {
		t.Errorf("Tokenize on closed model error = %v, want ErrModelClosed", err)
	}
}

func TestModel_DoubleClose(t *testing.T) {
	m, err := LoadModel(testModelPath,
		WithContextSize(100),
		WithKVCacheType(KVCacheQ8_0),
		WithSeed(42),
	)
	if err != nil {
		t.Errorf("model should not error = %v", err)
	}

	// First close
	if err := m.Close(); err != nil {
		t.Errorf("first Close() error = %v", err)
	}

	// Second close should not panic and should return nil
	if err := m.Close(); err != nil {
		t.Errorf("second Close() error = %v", err)
	}
}

func TestModel_Info(t *testing.T) {
	info := testModel.Info()

	if info.Architecture == "" {
		t.Error("Info().Architecture should not be empty")
	}
	if info.VocabSize <= 0 {
		t.Errorf("Info().VocabSize = %d, want > 0", info.VocabSize)
	}
	if info.Layers <= 0 {
		t.Errorf("Info().Layers = %d, want > 0", info.Layers)
	}
}

func TestModel_ContextSize(t *testing.T) {
	cs := testModel.ContextSize()
	if cs <= 0 {
		t.Errorf("ContextSize() = %d, want > 0", cs)
	}
}

func TestModel_TrainContextSize(t *testing.T) {
	tcs := testModel.TrainContextSize()
	if tcs <= 0 {
		t.Errorf("TrainContextSize() = %d, want > 0", tcs)
	}
}

func TestModel_EmbeddingSize(t *testing.T) {
	es := testModel.EmbeddingSize()
	if es <= 0 {
		t.Errorf("EmbeddingSize() = %d, want > 0", es)
	}
}

func TestModel_VocabSize(t *testing.T) {
	vs := testModel.VocabSize()
	info := testModel.Info()

	if vs <= 0 {
		t.Errorf("VocabSize() = %d, want > 0", vs)
	}
	if vs != info.VocabSize {
		t.Errorf("VocabSize() = %d, want %d (from Info())", vs, info.VocabSize)
	}
}

func TestModel_ChatTemplate(t *testing.T) {
	tmpl := testModel.ChatTemplate()
	// SmolLM2 should have a chat template
	if tmpl == "" {
		t.Error("ChatTemplate() should not be empty for SmolLM2")
	}
}

func TestModel_BOS_EOS(t *testing.T) {
	bos := testModel.BOS()
	eos := testModel.EOS()

	if bos < 0 {
		t.Errorf("BOS() = %d, want >= 0", bos)
	}
	if eos < 0 {
		t.Errorf("EOS() = %d, want >= 0", eos)
	}
}

func TestModel_SpecialTokens(t *testing.T) {
	st := testModel.SpecialTokens()

	// EOS should be valid
	if st.EOS < 0 {
		t.Errorf("SpecialTokens().EOS = %d, want >= 0", st.EOS)
	}

	// EOS from SpecialTokens should match EOS()
	if st.EOS != testModel.EOS() {
		t.Errorf("SpecialTokens().EOS = %d, EOS() = %d, want equal", st.EOS, testModel.EOS())
	}
}
