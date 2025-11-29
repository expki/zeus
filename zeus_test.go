package zeus

import (
	"log"
)

// Shared model for tests - loaded once in TestMain
const testModelPath = "example/models/SmolLM2-135M-Instruct-Q2_K.gguf"

var testModel = func() Model {
	model, err := LoadModel(testModelPath, WithContextSize(100), WithSeed(100))
	if err != nil {
		log.Fatalf("unable to open model %s: %v\n", testModelPath, err)
	}
	return model
}()
