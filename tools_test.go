package zeus

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// MockTool implements Tool interface for testing
type MockTool struct {
	name        string
	description string
	params      []ToolParameter
	handler     func(ctx context.Context, args map[string]any) (string, error)
}

func (m MockTool) Name() string                { return m.name }
func (m MockTool) Description() string         { return m.description }
func (m MockTool) Parameters() []ToolParameter { return m.params }
func (m MockTool) Execute(ctx context.Context, args map[string]any) (string, error) {
	if m.handler != nil {
		return m.handler(ctx, args)
	}
	return "mock result", nil
}

func TestAgentEventType_String(t *testing.T) {
	tests := []struct {
		input    AgentEventType
		expected string
	}{
		{AgentEventToken, "token"},
		{AgentEventToolCallStart, "tool_call_start"},
		{AgentEventToolCallEnd, "tool_call_end"},
		{AgentEventError, "error"},
		{AgentEventDone, "done"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.input.String(); got != tt.expected {
				t.Errorf("AgentEventType(%d).String() = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestAgentEventType_String_Invalid(t *testing.T) {
	invalid := AgentEventType(99)
	if got := invalid.String(); got != "unknown" {
		t.Errorf("AgentEventType(99).String() = %q, want 'unknown'", got)
	}
}

func TestToolExecutionError_Format(t *testing.T) {
	innerErr := errors.New("connection timeout")
	err := &ToolExecutionError{
		ToolName: "get_weather",
		CallID:   "call_123",
		Err:      innerErr,
	}

	msg := err.Error()

	if !strings.Contains(msg, "get_weather") {
		t.Errorf("error should contain tool name: %s", msg)
	}
	if !strings.Contains(msg, "call_123") {
		t.Errorf("error should contain call ID: %s", msg)
	}
	if !strings.Contains(msg, "connection timeout") {
		t.Errorf("error should contain inner error: %s", msg)
	}
	if !strings.Contains(msg, "zeus:") {
		t.Errorf("error should have zeus prefix: %s", msg)
	}
}

func TestToolExecutionError_Unwrap(t *testing.T) {
	innerErr := errors.New("inner error")
	err := &ToolExecutionError{
		ToolName: "test",
		CallID:   "123",
		Err:      innerErr,
	}

	if !errors.Is(err, innerErr) {
		t.Error("errors.Is should find inner error")
	}

	unwrapped := err.Unwrap()
	if unwrapped != innerErr {
		t.Errorf("Unwrap() = %v, want %v", unwrapped, innerErr)
	}
}

func TestToolSentinelErrors(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{"ErrNoToolsRegistered", ErrNoToolsRegistered},
		{"ErrMaxIterationsExceeded", ErrMaxIterationsExceeded},
		{"ErrMaxToolCallsExceeded", ErrMaxToolCallsExceeded},
		{"ErrTemplateApply", ErrTemplateApply},
		{"ErrToolTemplateUnsupported", ErrToolTemplateUnsupported},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !errors.Is(tt.err, tt.err) {
				t.Errorf("errors.Is(%v, %v) = false, want true", tt.err, tt.err)
			}

			// Should have zeus prefix
			if !strings.Contains(tt.err.Error(), "zeus:") {
				t.Errorf("error should have zeus prefix: %s", tt.err.Error())
			}
		})
	}
}

func TestDefaultAgentConfig(t *testing.T) {
	cfg := DefaultAgentConfig()

	if cfg.MaxIterations != 10 {
		t.Errorf("MaxIterations = %d, want 10", cfg.MaxIterations)
	}
	if cfg.MaxToolCalls != 25 {
		t.Errorf("MaxToolCalls = %d, want 25", cfg.MaxToolCalls)
	}
	if cfg.ToolTimeout != 30*1e9 { // 30 seconds in nanoseconds
		t.Errorf("ToolTimeout = %v, want 30s", cfg.ToolTimeout)
	}
}

func TestToolParameter(t *testing.T) {
	param := ToolParameter{
		Name:        "location",
		Type:        "string",
		Description: "City name",
		Required:    true,
		Enum:        []string{"Tokyo", "Paris", "London"},
	}

	if param.Name != "location" {
		t.Errorf("Name = %q, want 'location'", param.Name)
	}
	if param.Type != "string" {
		t.Errorf("Type = %q, want 'string'", param.Type)
	}
	if !param.Required {
		t.Error("Required should be true")
	}
	if len(param.Enum) != 3 {
		t.Errorf("len(Enum) = %d, want 3", len(param.Enum))
	}
}

func TestToolCall(t *testing.T) {
	call := ToolCall{
		ID:        "call_123",
		Name:      "get_weather",
		Arguments: map[string]any{"location": "Tokyo", "unit": "celsius"},
	}

	if call.ID != "call_123" {
		t.Errorf("ID = %q, want 'call_123'", call.ID)
	}
	if call.Name != "get_weather" {
		t.Errorf("Name = %q, want 'get_weather'", call.Name)
	}
	if call.Arguments["location"] != "Tokyo" {
		t.Errorf("Arguments[location] = %v, want 'Tokyo'", call.Arguments["location"])
	}
}

func TestToolResult(t *testing.T) {
	result := ToolResult{
		CallID:  "call_123",
		Content: `{"temp": 22}`,
		IsError: false,
	}

	if result.CallID != "call_123" {
		t.Errorf("CallID = %q, want 'call_123'", result.CallID)
	}
	if result.IsError {
		t.Error("IsError should be false")
	}

	errorResult := ToolResult{
		CallID:  "call_456",
		Content: "connection failed",
		IsError: true,
	}

	if !errorResult.IsError {
		t.Error("IsError should be true for error result")
	}
}

func TestAgentEvent(t *testing.T) {
	token := &Token{Text: "Hello", ID: 1}
	event := AgentEvent{
		Type:  AgentEventToken,
		Token: token,
	}

	if event.Type != AgentEventToken {
		t.Errorf("Type = %v, want AgentEventToken", event.Type)
	}
	if event.Token != token {
		t.Error("Token should be set")
	}
}

func TestFormatToolResult(t *testing.T) {
	call := ToolCall{Name: "get_weather"}

	// Success result
	result := ToolResult{Content: `{"temp": 22}`, IsError: false}
	formatted := formatToolResult(call, result)
	if !strings.Contains(formatted, "get_weather") {
		t.Errorf("formatted should contain tool name: %s", formatted)
	}
	if !strings.Contains(formatted, `{"temp": 22}`) {
		t.Errorf("formatted should contain result: %s", formatted)
	}
	if strings.Contains(formatted, "error") {
		t.Errorf("formatted should not contain 'error' for success: %s", formatted)
	}

	// Error result
	errorResult := ToolResult{Content: "connection failed", IsError: true}
	formattedError := formatToolResult(call, errorResult)
	if !strings.Contains(formattedError, "error") {
		t.Errorf("formatted should contain 'error' for error result: %s", formattedError)
	}
}

func TestChatFormatString(t *testing.T) {
	tests := []struct {
		format   ChatFormat
		expected string
	}{
		{ChatFormatContentOnly, "Content-only"},
		{ChatFormatGeneric, "Generic"},
		{ChatFormatHermes2Pro, "Hermes 2 Pro"},
		{ChatFormatLlama3X, "Llama 3.x"},
		{ChatFormatMistralNemo, "Mistral Nemo"},
		{ChatFormatDeepSeekR1, "DeepSeek R1"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			got := tt.format.String()
			if got != tt.expected {
				t.Errorf("ChatFormat(%d).String() = %q, want %q", tt.format, got, tt.expected)
			}
		})
	}
}

func TestParseToolCallsNative_Hermes2Pro(t *testing.T) {
	// Hermes 2 Pro format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
	response := `I'll check the weather for you.
<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}</tool_call>`

	result := parseToolCallsNative(response, ChatFormatHermes2Pro)

	if !strings.Contains(result.Content, "I'll check the weather") {
		t.Errorf("content should contain text before tool call: %s", result.Content)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.ToolCalls))
	}
	if result.ToolCalls[0].Name != "get_weather" {
		t.Errorf("tool call name = %q, want 'get_weather'", result.ToolCalls[0].Name)
	}
	if result.ToolCalls[0].Arguments["location"] != "Tokyo" {
		t.Errorf("tool call arguments[location] = %v, want 'Tokyo'", result.ToolCalls[0].Arguments["location"])
	}
}

func TestParseToolCallsNative_NoToolCalls(t *testing.T) {
	response := "This is a regular response without any tool calls."
	result := parseToolCallsNative(response, ChatFormatHermes2Pro)

	if result.Content != response {
		t.Errorf("content = %q, want %q", result.Content, response)
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(result.ToolCalls))
	}
}

func TestParseToolCallsNative_Empty(t *testing.T) {
	result := parseToolCallsNative("", ChatFormatHermes2Pro)

	if result.Content != "" {
		t.Errorf("content should be empty, got %q", result.Content)
	}
	if len(result.ToolCalls) != 0 {
		t.Errorf("expected 0 tool calls, got %d", len(result.ToolCalls))
	}
}

func TestParseToolCallsNative_Generic(t *testing.T) {
	// Generic format uses JSON objects
	response := `{"tool_call": {"name": "get_weather", "arguments": {"location": "Paris"}}}`

	result := parseToolCallsNative(response, ChatFormatGeneric)

	// For generic format, tool calls may be parsed or content returned
	// depending on how the native parser handles it
	if result.Content == "" && len(result.ToolCalls) == 0 {
		t.Error("expected either content or tool calls")
	}
}

func TestToolParametersToJSON(t *testing.T) {
	params := []ToolParameter{
		{Name: "location", Type: "string", Description: "City name", Required: true},
		{Name: "unit", Type: "string", Description: "Temperature unit", Enum: []string{"celsius", "fahrenheit"}},
	}

	result := toolParametersToJSON(params)

	// Should be valid JSON
	if !strings.Contains(result, `"type":"object"`) {
		t.Errorf("result should contain object type: %s", result)
	}
	if !strings.Contains(result, `"location"`) {
		t.Errorf("result should contain location property: %s", result)
	}
	if !strings.Contains(result, `"required"`) {
		t.Errorf("result should contain required array: %s", result)
	}
}

func TestToolParametersToJSON_Empty(t *testing.T) {
	result := toolParametersToJSON(nil)
	if result != "{}" {
		t.Errorf("empty params should return {}, got %s", result)
	}
}

func TestToolChoiceConstants(t *testing.T) {
	// Ensure constants match expected values
	if ToolChoiceAuto != 0 {
		t.Errorf("ToolChoiceAuto should be 0, got %d", ToolChoiceAuto)
	}
	if ToolChoiceNone != 1 {
		t.Errorf("ToolChoiceNone should be 1, got %d", ToolChoiceNone)
	}
	if ToolChoiceRequired != 2 {
		t.Errorf("ToolChoiceRequired should be 2, got %d", ToolChoiceRequired)
	}
}

func TestDefaultAgentConfig_NewFields(t *testing.T) {
	cfg := DefaultAgentConfig()

	if cfg.ToolChoice != ToolChoiceAuto {
		t.Errorf("ToolChoice should default to Auto, got %d", cfg.ToolChoice)
	}
	if !cfg.ParallelToolCalls {
		t.Error("ParallelToolCalls should default to true")
	}
}
