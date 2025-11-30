package zeus

/*
#include "source/binding.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"
)

// Tool defines a callable function that the model can invoke.
// Implement this interface to create custom tools.
type Tool interface {
	// Name returns the unique identifier for this tool.
	Name() string

	// Description returns a human-readable description of what this tool does.
	// This is provided to the model to help it decide when to use the tool.
	Description() string

	// Parameters returns the list of parameters this tool accepts.
	Parameters() []ToolParameter

	// Execute runs the tool with the given arguments and returns the result.
	// The result should be a string that can be fed back to the model.
	// Return an error if the tool execution fails.
	Execute(ctx context.Context, args map[string]any) (string, error)
}

// ToolParameter describes a single parameter for a tool.
type ToolParameter struct {
	Name        string   // Parameter name
	Type        string   // Type: "string", "number", "boolean", "array", "object"
	Description string   // Human-readable description
	Required    bool     // Whether this parameter is required
	Enum        []string // Optional: allowed values for this parameter
}

// ToolCall represents a tool invocation requested by the model.
type ToolCall struct {
	ID        string         // Unique identifier for this call
	Name      string         // Name of the tool to invoke
	Arguments map[string]any // Parsed arguments from the model
}

// ToolResult represents the outcome of executing a tool.
type ToolResult struct {
	CallID  string // Corresponds to ToolCall.ID
	Content string // Result content to feed back to the model
	IsError bool   // Whether this result represents an error
}

// AgentEventType indicates the type of event during agentic loop execution.
type AgentEventType int

const (
	AgentEventToken         AgentEventType = iota // A token was generated
	AgentEventToolCallStart                       // A tool call is starting
	AgentEventToolCallEnd                         // A tool call completed
	AgentEventError                               // An error occurred
	AgentEventDone                                // The agentic loop completed
)

// String returns the string representation of the event type.
func (t AgentEventType) String() string {
	switch t {
	case AgentEventToken:
		return "token"
	case AgentEventToolCallStart:
		return "tool_call_start"
	case AgentEventToolCallEnd:
		return "tool_call_end"
	case AgentEventError:
		return "error"
	case AgentEventDone:
		return "done"
	default:
		return "unknown"
	}
}

// AgentEvent represents an event during agentic loop execution.
type AgentEvent struct {
	Type     AgentEventType // Type of event
	Token    *Token         // For AgentEventToken: the generated token
	ToolCall *ToolCall      // For AgentEventToolCallStart/End: the tool call
	Result   *ToolResult    // For AgentEventToolCallEnd: the result
	Error    error          // For AgentEventError: the error that occurred
}

// ChatFormat represents the tool call format for parsing model output.
// Different models use different formats for function/tool calling.
// This is forward-compatible: new formats added to llama.cpp work automatically
// even if not listed here. Use String() to get a readable name.
type ChatFormat int32

// Known chat formats. This list may be incomplete - llama.cpp may support
// additional formats that will work automatically.
const (
	ChatFormatContentOnly             ChatFormat = 0  // No tool calls, content only
	ChatFormatGeneric                 ChatFormat = 1  // Generic format with JSON
	ChatFormatMistralNemo             ChatFormat = 2  // Mistral Nemo format
	ChatFormatMagistral               ChatFormat = 3  // Magistral format
	ChatFormatLlama3X                 ChatFormat = 4  // Llama 3.x format
	ChatFormatLlama3XWithBuiltinTools ChatFormat = 5  // Llama 3.x with builtin tools
	ChatFormatDeepSeekR1              ChatFormat = 6  // DeepSeek R1 format
	ChatFormatFireFunctionV2          ChatFormat = 7  // FireFunction v2 format
	ChatFormatFunctionaryV32          ChatFormat = 8  // Functionary v3.2 format
	ChatFormatFunctionaryV31Llama31   ChatFormat = 9  // Functionary v3.1 Llama 3.1 format
	ChatFormatDeepSeekV31             ChatFormat = 10 // DeepSeek V3.1 format
	ChatFormatHermes2Pro              ChatFormat = 11 // Hermes 2 Pro format (Qwen 2.5, Hermes 2/3)
	ChatFormatCommandR7B              ChatFormat = 12 // Command R7B format
	ChatFormatGranite                 ChatFormat = 13 // Granite format
	ChatFormatGPTOSS                  ChatFormat = 14 // GPT-OSS format
	ChatFormatSeedOSS                 ChatFormat = 15 // Seed-OSS format
	ChatFormatNemotronV2              ChatFormat = 16 // Nemotron V2 format
	ChatFormatApertus                 ChatFormat = 17 // Apertus format
	ChatFormatLFM2WithJSONTools       ChatFormat = 18 // LFM2 with JSON tools format
	ChatFormatGLM45                   ChatFormat = 19 // GLM 4.5 format
	ChatFormatMiniMaxM2               ChatFormat = 20 // MiniMax-M2 format
	ChatFormatKimiK2                  ChatFormat = 21 // Kimi K2 format
	ChatFormatQwen3CoderXML           ChatFormat = 22 // Qwen3 Coder format
	ChatFormatApriel15                ChatFormat = 23 // Apriel 1.5 format
	ChatFormatXiaomiMiMo              ChatFormat = 24 // Xiaomi MiMo format
)

// String returns the name of the chat format from llama.cpp.
func (f ChatFormat) String() string {
	cName := C.binding_chat_format_name(C.int32_t(f))
	if cName == nil {
		return fmt.Sprintf("unknown-%d", f)
	}
	return C.GoString(cName)
}

// ParseResult contains the parsed tool calls from model output.
type ParseResult struct {
	Content          string     // Non-tool-call text content
	ReasoningContent string     // Reasoning/thinking content (if any)
	ToolCalls        []ToolCall // Parsed tool calls
}

// parseToolCallsNative uses llama.cpp's native parser to extract tool calls.
func parseToolCallsNative(response string, format ChatFormat) ParseResult {
	if response == "" {
		return ParseResult{Content: response}
	}

	cResponse := C.CString(response)
	defer C.free(unsafe.Pointer(cResponse))

	result := C.binding_parse_tool_calls(cResponse, C.int32_t(format), C.bool(false))
	if result == nil {
		return ParseResult{Content: response}
	}
	defer C.binding_free_parse_result(result)

	parsed := ParseResult{}

	if result.content != nil {
		parsed.Content = C.GoString(result.content)
	}

	if result.reasoning_content != nil {
		parsed.ReasoningContent = C.GoString(result.reasoning_content)
	}

	if result.tool_call_count <= 0 || result.tool_calls == nil {
		return parsed
	}

	// Convert C array to Go slice
	count := int(result.tool_call_count)
	cToolCalls := unsafe.Slice(result.tool_calls, count)

	for i := 0; i < count; i++ {
		tc := cToolCalls[i]
		toolCall := ToolCall{
			Name: C.GoString(tc.name),
		}
		if tc.id != nil {
			toolCall.ID = C.GoString(tc.id)
		}
		if tc.arguments != nil {
			argsJSON := C.GoString(tc.arguments)
			// Parse JSON arguments
			if argsJSON != "" {
				var args map[string]any
				if err := json.Unmarshal([]byte(argsJSON), &args); err == nil {
					toolCall.Arguments = args
				} else {
					// If JSON parsing fails, store as raw string
					toolCall.Arguments = map[string]any{"_raw": argsJSON}
				}
			}
		}
		if toolCall.Arguments == nil {
			toolCall.Arguments = make(map[string]any)
		}
		parsed.ToolCalls = append(parsed.ToolCalls, toolCall)
	}

	return parsed
}

// ToolChoice controls how the model should use tools during generation.
type ToolChoice int

const (
	ToolChoiceAuto     ToolChoice = iota // Model decides when to use tools
	ToolChoiceNone                       // Never use tools
	ToolChoiceRequired                   // Must use a tool
)

// ChatParams contains the result of applying a chat template with tools.
// This includes the formatted prompt, grammar constraints, and other metadata.
type ChatParams struct {
	Prompt          string     // Formatted prompt with tools embedded
	Grammar         string     // GBNF grammar for constraining output (may be empty)
	Format          ChatFormat // Detected chat format
	GrammarLazy     bool       // Apply grammar only after trigger patterns
	GrammarTriggers []string   // Patterns that activate grammar
	AdditionalStops []string   // Extra stop sequences
}

// applyChatTemplateWithTools applies the model's chat template with tool definitions.
// This uses llama.cpp's native template system to format messages with tools.
func (m *model) applyChatTemplateWithTools(
	messages []ChatMessage,
	tools []Tool,
	toolChoice ToolChoice,
	parallelToolCalls bool,
) (*ChatParams, error) {
	if m == nil || m.isClosed() {
		return nil, ErrModelClosed
	}

	// Convert messages to C structures
	cMessages := make([]C.binding_chat_message, len(messages))
	cStrings := make([]*C.char, 0, len(messages)*2)

	for i, msg := range messages {
		roleStr := C.CString(string(msg.Role))
		contentStr := C.CString(msg.Content)
		cStrings = append(cStrings, roleStr, contentStr)
		cMessages[i] = C.binding_chat_message{
			role:    roleStr,
			content: contentStr,
		}
	}
	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	// Convert tools to C structures
	cTools := make([]C.binding_chat_tool_def, len(tools))
	toolStrings := make([]*C.char, 0, len(tools)*3)

	for i, tool := range tools {
		nameStr := C.CString(tool.Name())
		descStr := C.CString(tool.Description())
		paramsJSON := toolParametersToJSON(tool.Parameters())
		paramsStr := C.CString(paramsJSON)
		toolStrings = append(toolStrings, nameStr, descStr, paramsStr)
		cTools[i] = C.binding_chat_tool_def{
			name:        nameStr,
			description: descStr,
			parameters:  paramsStr,
		}
	}
	defer func() {
		for _, s := range toolStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	// Call the C function
	var messagesPtr *C.binding_chat_message
	if len(cMessages) > 0 {
		messagesPtr = &cMessages[0]
	}

	var toolsPtr *C.binding_chat_tool_def
	if len(cTools) > 0 {
		toolsPtr = &cTools[0]
	}

	result := C.binding_apply_chat_template_with_tools(
		m.ptr,
		messagesPtr,
		C.int32_t(len(messages)),
		toolsPtr,
		C.int32_t(len(tools)),
		C.binding_tool_choice(toolChoice),
		C.bool(parallelToolCalls),
		C.bool(true), // add_generation_prompt
	)

	if result == nil {
		return nil, ErrTemplateApply
	}
	defer C.binding_free_chat_params(result)

	// Convert result to Go struct
	params := &ChatParams{
		Format:      ChatFormat(result.format),
		GrammarLazy: bool(result.grammar_lazy),
	}

	if result.prompt != nil {
		params.Prompt = C.GoString(result.prompt)
	}
	if result.grammar != nil {
		params.Grammar = C.GoString(result.grammar)
	}

	// Copy triggers
	if result.trigger_count > 0 && result.grammar_triggers != nil {
		count := int(result.trigger_count)
		cTriggers := unsafe.Slice(result.grammar_triggers, count)
		params.GrammarTriggers = make([]string, count)
		for i := 0; i < count; i++ {
			if cTriggers[i] != nil {
				params.GrammarTriggers[i] = C.GoString(cTriggers[i])
			}
		}
	}

	// Copy additional stops
	if result.stop_count > 0 && result.additional_stops != nil {
		count := int(result.stop_count)
		cStops := unsafe.Slice(result.additional_stops, count)
		params.AdditionalStops = make([]string, count)
		for i := 0; i < count; i++ {
			if cStops[i] != nil {
				params.AdditionalStops[i] = C.GoString(cStops[i])
			}
		}
	}

	return params, nil
}

// toolParametersToJSON converts tool parameters to a JSON schema string.
func toolParametersToJSON(params []ToolParameter) string {
	if len(params) == 0 {
		return "{}"
	}

	schema := map[string]any{
		"type":       "object",
		"properties": make(map[string]any),
		"required":   []string{},
	}

	properties := schema["properties"].(map[string]any)
	required := []string{}

	for _, p := range params {
		prop := map[string]any{
			"type":        p.Type,
			"description": p.Description,
		}
		if len(p.Enum) > 0 {
			prop["enum"] = p.Enum
		}
		properties[p.Name] = prop

		if p.Required {
			required = append(required, p.Name)
		}
	}

	if len(required) > 0 {
		schema["required"] = required
	}

	data, _ := json.Marshal(schema)
	return string(data)
}
