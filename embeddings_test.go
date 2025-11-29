package zeus

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestEmbeddings_Disabled(t *testing.T) {
	// testModel was loaded without WithEmbeddings()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := testModel.Embeddings(ctx, "Hello world")
	if !errors.Is(err, ErrEmbeddingsDisabled) {
		t.Errorf("Embeddings() without WithEmbeddings() error = %v, want ErrEmbeddingsDisabled", err)
	}
}
