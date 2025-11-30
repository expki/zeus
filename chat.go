package zeus

/*
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// Role represents the role of a message sender in a conversation.
type Role string

// Standard chat roles.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool" // Tool result messages
)

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    Role
	Content string
}

// ChatTemplate returns the model's embedded chat template string.
// Returns empty string if no template is embedded in the model.
func (m *model) ChatTemplate() string {
	if m == nil || m.isClosed() {
		return ""
	}
	result := C.binding_get_chat_template(m.ptr)
	if result == nil {
		return ""
	}
	return C.GoString(result)
}

// ApplyChatTemplate formats messages using a chat template.
// Uses model's embedded template by default.
func (m *model) ApplyChatTemplate(messages []ChatMessage, opts ...ChatTemplateOption) (string, error) {
	if m == nil {
		return "", ErrModelIsNil
	}
	if m.isClosed() {
		return "", ErrModelClosed
	}
	if len(messages) == 0 {
		return "", nil
	}

	// Apply options
	cfg := DefaultChatTemplateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Convert to C array
	cMessages := make([]C.binding_chat_message, len(messages))
	cStrings := make([]*C.char, len(messages)*2)
	for i, msg := range messages {
		cStrings[i*2] = C.CString(string(msg.Role))
		cStrings[i*2+1] = C.CString(msg.Content)
		cMessages[i].role = cStrings[i*2]
		cMessages[i].content = cStrings[i*2+1]
	}
	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	var cTemplate *C.char
	if cfg.Template != "" {
		cTemplate = C.CString(cfg.Template)
		defer C.free(unsafe.Pointer(cTemplate))
	}

	// Get required buffer size
	size := C.binding_apply_chat_template_length(
		m.ptr, cTemplate, &cMessages[0], C.int32_t(len(messages)),
		C.bool(cfg.AddAssistant),
	)
	if size < 0 {
		return "", &ChatTemplateError{Message: "failed to get template length"}
	}
	if size == 0 {
		return "", nil
	}

	// Allocate and fill buffer
	buf := make([]byte, size+1)
	result := C.binding_apply_chat_template(
		m.ptr, cTemplate, &cMessages[0], C.int32_t(len(messages)),
		C.bool(cfg.AddAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(size+1),
	)
	if result < 0 {
		return "", &ChatTemplateError{Message: "failed to apply template"}
	}

	return string(buf[:result]), nil
}
