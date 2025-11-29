package zeus

import "testing"

func TestDefaultModelConfig(t *testing.T) {
	cfg := DefaultModelConfig()

	if cfg.ContextSize != 0 {
		t.Errorf("ContextSize = %d, want 0 (model native)", cfg.ContextSize)
	}
	if cfg.BatchSize != 512 {
		t.Errorf("BatchSize = %d, want 512", cfg.BatchSize)
	}
	if cfg.GPULayers != GPULayersAll {
		t.Errorf("GPULayers = %d, want %d (GPULayersAll)", cfg.GPULayers, GPULayersAll)
	}
	if cfg.KVCacheType != KVCacheF16 {
		t.Errorf("KVCacheType = %v, want KVCacheF16", cfg.KVCacheType)
	}
	if !cfg.UseMMap {
		t.Error("UseMMap = false, want true")
	}
	if cfg.Embeddings {
		t.Error("Embeddings = true, want false")
	}
}

func TestDefaultGenerateConfig(t *testing.T) {
	cfg := DefaultGenerateConfig()

	if cfg.MaxTokens != 0 {
		t.Errorf("MaxTokens = %d, want 0 (unlimited)", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.8 {
		t.Errorf("Temperature = %f, want 0.8", cfg.Temperature)
	}
	if cfg.TopK != 40 {
		t.Errorf("TopK = %d, want 40", cfg.TopK)
	}
	if cfg.TopP != 0.95 {
		t.Errorf("TopP = %f, want 0.95", cfg.TopP)
	}
	if cfg.RepeatPenalty != 1.1 {
		t.Errorf("RepeatPenalty = %f, want 1.1", cfg.RepeatPenalty)
	}
	if cfg.Mirostat != MirostatDisabled {
		t.Errorf("Mirostat = %v, want MirostatDisabled", cfg.Mirostat)
	}
	if cfg.Seed != -1 {
		t.Errorf("Seed = %d, want -1 (random)", cfg.Seed)
	}
}

func TestDefaultChatConfig(t *testing.T) {
	cfg := DefaultChatConfig()

	if cfg.Template != "" {
		t.Errorf("Template = %q, want empty", cfg.Template)
	}
	if !cfg.AddAssistant {
		t.Error("AddAssistant = false, want true")
	}
}

func TestDefaultChatTemplateConfig(t *testing.T) {
	cfg := DefaultChatTemplateConfig()

	if cfg.Template != "" {
		t.Errorf("Template = %q, want empty", cfg.Template)
	}
	if !cfg.AddAssistant {
		t.Error("AddAssistant = false, want true")
	}
}

func TestModelOption_WithContextSize(t *testing.T) {
	cfg := DefaultModelConfig()
	WithContextSize(2048)(&cfg)

	if cfg.ContextSize != 2048 {
		t.Errorf("ContextSize = %d, want 2048", cfg.ContextSize)
	}
}

func TestModelOption_WithKVCacheType(t *testing.T) {
	cfg := DefaultModelConfig()
	WithKVCacheType(KVCacheQ4_0)(&cfg)

	if cfg.KVCacheType != KVCacheQ4_0 {
		t.Errorf("KVCacheType = %v, want KVCacheQ4_0", cfg.KVCacheType)
	}
}

func TestModelOption_WithEmbeddings(t *testing.T) {
	cfg := DefaultModelConfig()
	WithEmbeddings()(&cfg)

	if !cfg.Embeddings {
		t.Error("Embeddings = false, want true")
	}
}

func TestModelOption_WithSeed(t *testing.T) {
	cfg := DefaultModelConfig()
	WithSeed(42)(&cfg)

	if cfg.Seed != 42 {
		t.Errorf("Seed = %d, want 42", cfg.Seed)
	}
}

func TestModelOption_WithGPULayers(t *testing.T) {
	cfg := DefaultModelConfig()
	WithGPULayers(10)(&cfg)

	if cfg.GPULayers != 10 {
		t.Errorf("GPULayers = %d, want 10", cfg.GPULayers)
	}
}

func TestGenerateOption_WithMaxTokens(t *testing.T) {
	cfg := DefaultGenerateConfig()
	WithMaxTokens(100)(&cfg)

	if cfg.MaxTokens != 100 {
		t.Errorf("MaxTokens = %d, want 100", cfg.MaxTokens)
	}
}

func TestGenerateOption_WithTemperature(t *testing.T) {
	cfg := DefaultGenerateConfig()
	WithTemperature(0.5)(&cfg)

	if cfg.Temperature != 0.5 {
		t.Errorf("Temperature = %f, want 0.5", cfg.Temperature)
	}
}

func TestGenerateOption_WithStopSequences(t *testing.T) {
	cfg := DefaultGenerateConfig()
	WithStopSequences("stop1", "stop2")(&cfg)

	if len(cfg.StopSequences) != 2 {
		t.Fatalf("len(StopSequences) = %d, want 2", len(cfg.StopSequences))
	}
	if cfg.StopSequences[0] != "stop1" {
		t.Errorf("StopSequences[0] = %q, want 'stop1'", cfg.StopSequences[0])
	}
	if cfg.StopSequences[1] != "stop2" {
		t.Errorf("StopSequences[1] = %q, want 'stop2'", cfg.StopSequences[1])
	}
}

func TestChatTemplateOption_WithAddAssistant(t *testing.T) {
	cfg := DefaultChatTemplateConfig()
	WithAddAssistant(false)(&cfg)

	if cfg.AddAssistant {
		t.Error("AddAssistant = true, want false")
	}
}

func TestChatTemplateOption_WithChatTemplate(t *testing.T) {
	cfg := DefaultChatTemplateConfig()
	WithChatTemplate("chatml")(&cfg)

	if cfg.Template != "chatml" {
		t.Errorf("Template = %q, want 'chatml'", cfg.Template)
	}
}
