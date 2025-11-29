package zeus

/*
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// Tokenize converts text to token IDs.
func (m *model) Tokenize(text string, addSpecial bool) ([]int, error) {
	if m == nil {
		return nil, ErrModelIsNil
	}
	if m.isClosed() {
		return nil, ErrModelClosed
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Get the token count first
	count, err := m.TokenizeCount(text, addSpecial)
	if err != nil {
		return nil, err
	}
	if count <= 0 {
		return []int{}, nil
	}

	// Allocate buffer and tokenize
	tokens := make([]C.int32_t, count)
	result := C.binding_tokenize(
		m.ptr,
		cText,
		C.bool(addSpecial),
		&tokens[0],
	)
	if result != C.BINDING_OK {
		return nil, &TokenizeError{Text: text, Message: "tokenization failed"}
	}

	// Convert to Go slice
	goTokens := make([]int, count)
	for i := range count {
		goTokens[i] = int(tokens[i])
	}

	return goTokens, nil
}

// TokenizeCount returns number of token IDs the text represents.
func (m *model) TokenizeCount(text string, addSpecial bool) (int, error) {
	if m == nil {
		return 0, ErrModelIsNil
	}
	if m.isClosed() {
		return 0, ErrModelClosed
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var count C.int32_t
	result := C.binding_tokenize_count(
		m.ptr,
		cText,
		C.bool(addSpecial),
		&count,
	)
	if result != C.BINDING_OK {
		return 0, &TokenizeError{Text: text, Message: "failed to count tokens"}
	}

	return int(count), nil
}

// Detokenize converts token IDs to text.
func (m *model) Detokenize(tokens []int) (string, error) {
	if m == nil {
		return "", ErrModelIsNil
	}
	if m.isClosed() {
		return "", ErrModelClosed
	}

	// Get the required buffer length first
	length, err := m.DetokenizeLength(tokens)
	if err != nil {
		return "", err
	}
	if length <= 0 {
		return "", nil
	}

	// Convert to C array
	cTokens := make([]C.int32_t, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int32_t(t)
	}

	// Allocate buffer and detokenize (add 1 for null terminator)
	textBuf := make([]byte, length+1)
	result := C.binding_detokenize(
		m.ptr,
		&cTokens[0],
		C.int32_t(len(tokens)),
		(*C.char)(unsafe.Pointer(&textBuf[0])),
	)
	if result != C.BINDING_OK {
		return "", &TokenizeError{Text: "", Message: "detokenization failed"}
	}

	return string(textBuf[:length]), nil
}

// DetokenizeLength returns string length the token IDs represents.
func (m *model) DetokenizeLength(tokens []int) (int, error) {
	if m == nil {
		return 0, ErrModelIsNil
	}
	if m.isClosed() {
		return 0, ErrModelClosed
	}
	if len(tokens) <= 0 {
		return 0, nil
	}

	// Convert to C array
	cTokens := make([]C.int32_t, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.int32_t(t)
	}

	var length C.int32_t
	result := C.binding_detokenize_length(
		m.ptr,
		&cTokens[0],
		C.int32_t(len(tokens)),
		&length,
	)
	if result != C.BINDING_OK {
		return 0, &TokenizeError{Text: "", Message: "failed to get detokenize length"}
	}

	return int(length), nil
}

// BOS returns the beginning-of-sequence token ID.
func (m *model) BOS() int {
	return m.SpecialTokens().BOS
}

// EOS returns the end-of-sequence token ID.
func (m *model) EOS() int {
	return m.SpecialTokens().EOS
}

// TokenToText converts a single token ID to its text representation.
func (m *model) TokenToText(token int) string {
	if m == nil || m.isClosed() {
		return ""
	}
	buf := make([]byte, 256)
	n := C.binding_token_to_text(m.ptr, C.int32_t(token), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}

// IsSpecialToken returns true if the token is a special/control token.
func (m *model) IsSpecialToken(token int) bool {
	if m == nil || m.isClosed() {
		return false
	}
	return bool(C.binding_is_special_token(m.ptr, C.int32_t(token)))
}

// IsEOG returns true if the token is an end-of-generation token.
func (m *model) IsEOG(token int) bool {
	if m == nil || m.isClosed() {
		return false
	}
	return bool(C.binding_is_eog_token(m.ptr, C.int32_t(token)))
}

// SpecialTokens contains all special token IDs for the model.
type SpecialTokens struct {
	BOS int // Beginning of sequence (-1 if not available)
	EOS int // End of sequence
	EOT int // End of turn
	PAD int // Padding
	SEP int // Separator
	NL  int // Newline
}

// SpecialTokens returns all special token IDs.
func (m *model) SpecialTokens() SpecialTokens {
	if m == nil || m.isClosed() {
		return SpecialTokens{-1, -1, -1, -1, -1, -1}
	}
	tokens := C.binding_get_special_tokens(m.ptr)
	return SpecialTokens{
		BOS: int(tokens.bos),
		EOS: int(tokens.eos),
		EOT: int(tokens.eot),
		PAD: int(tokens.pad),
		SEP: int(tokens.sep),
		NL:  int(tokens.nl),
	}
}

// VocabSize returns the vocabulary size.
func (m *model) VocabSize() int {
	if m == nil || m.isClosed() {
		return 0
	}
	return int(C.binding_vocab_size(m.ptr))
}
