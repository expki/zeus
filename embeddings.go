package zeus

/*
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"unsafe"
)

// Embeddings extracts embeddings for the given text.
// The model must be loaded with WithEmbeddings() option.
func (m *model) Embeddings(ctx context.Context, text string) ([]float32, error) {
	if m == nil {
		return nil, ErrModelIsNil
	}
	if m.isClosed() {
		return nil, ErrModelClosed
	}

	m.kvMutex.Lock()
	if !m.config.Embeddings {
		m.kvMutex.Unlock()
		return nil, ErrEmbeddingsDisabled
	}

	var response []float32
	var err error
	wait := make(chan struct{}, 1)
	go func() {
		defer m.kvMutex.Unlock()
		defer close(wait)
		cText := C.CString(text)
		defer C.free(unsafe.Pointer(cText))

		// Get embedding size
		embSize := m.EmbeddingSize()
		if embSize <= 0 {
			err = &EmbeddingError{Message: "invalid embedding size"}
			return
		}
		if ctx.Err() != nil {
			err = ctx.Err()
			return
		}

		// Allocate buffer
		embeddings := make([]float32, embSize)
		var outSize C.int32_t

		result := C.binding_get_embeddings(
			m.ptr,
			cText,
			(*C.float)(unsafe.Pointer(&embeddings[0])),
			&outSize,
		)

		switch result {
		case C.BINDING_OK:
			response = embeddings[:outSize]
		case C.BINDING_ERR_EMBEDDINGS_DISABLED:
			err = ErrEmbeddingsDisabled
		case C.BINDING_ERR_DECODE:
			err = &EmbeddingError{Message: "decode failed"}
		case C.BINDING_ERR_TOKENIZE:
			err = &EmbeddingError{Message: "tokenization failed"}
		default:
			err = &EmbeddingError{Message: "embedding extraction failed"}
		}
	}()

	// Wait for request
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-wait:
		return response, err
	}
}

// EmbeddingsBatch extracts embeddings for multiple texts in a single call.
// The model must be loaded with WithEmbeddings() option.
func (m *model) EmbeddingsBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if m == nil {
		return nil, ErrModelIsNil
	}
	if m.isClosed() {
		return nil, ErrModelClosed
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	m.kvMutex.Lock()
	if !m.config.Embeddings {
		m.kvMutex.Unlock()
		return nil, ErrEmbeddingsDisabled
	}

	var response [][]float32
	var err error
	wait := make(chan struct{}, 1)
	go func() {
		defer m.kvMutex.Unlock()
		defer close(wait)

		// Get embedding size
		embSize := m.EmbeddingSize()
		if embSize <= 0 {
			err = &EmbeddingError{Message: "invalid embedding size"}
			return
		}
		if ctx.Err() != nil {
			err = ctx.Err()
			return
		}

		// Convert texts to C strings
		cTexts := make([]*C.char, len(texts))
		for i, text := range texts {
			cTexts[i] = C.CString(text)
		}
		defer func() {
			for _, cText := range cTexts {
				C.free(unsafe.Pointer(cText))
			}
		}()

		// Allocate flat buffer for all embeddings
		totalSize := len(texts) * embSize
		embeddings := make([]float32, totalSize)
		var outSize C.int32_t

		result := C.binding_get_embeddings_batch(
			m.ptr,
			(**C.char)(unsafe.Pointer(&cTexts[0])),
			C.int32_t(len(texts)),
			(*C.float)(unsafe.Pointer(&embeddings[0])),
			&outSize,
		)

		switch result {
		case C.BINDING_OK:
			// Reshape flat array to [][]float32
			embDim := int(outSize)
			response = make([][]float32, len(texts))
			for i := range texts {
				start := i * embDim
				end := start + embDim
				response[i] = make([]float32, embDim)
				copy(response[i], embeddings[start:end])
			}
		case C.BINDING_ERR_EMBEDDINGS_DISABLED:
			err = ErrEmbeddingsDisabled
		case C.BINDING_ERR_DECODE:
			err = &EmbeddingError{Message: "decode failed"}
		default:
			err = &EmbeddingError{Message: "batch embedding extraction failed"}
		}
	}()

	// Wait for request
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-wait:
		return response, err
	}
}
