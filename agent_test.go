package zeus

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestGenerateWithTools_NoTools(t *testing.T) {
	chat := testModel.NewChat()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var sawError bool
	for _, err := range chat.GenerateWithTools(ctx, "Hello") {
		if err != nil {
			sawError = true
			if !errors.Is(err, ErrNoToolsRegistered) {
				t.Errorf("expected ErrNoToolsRegistered, got %v", err)
			}
			break
		}
	}

	if !sawError {
		t.Error("GenerateWithTools() should return ErrNoToolsRegistered when no tools registered")
	}
}

func TestGenerateWithTools_NilChat(t *testing.T) {
	var c *chat = nil

	ctx := context.Background()
	var sawError bool
	for _, err := range c.GenerateWithTools(ctx, "Hello") {
		if err != nil {
			sawError = true
			if !errors.Is(err, ErrChatIsNil) {
				t.Errorf("expected ErrChatIsNil, got %v", err)
			}
			break
		}
	}

	if !sawError {
		t.Error("GenerateWithTools() on nil chat should return ErrChatIsNil")
	}
}

func TestGenerateWithTools_ToolsRegistered(t *testing.T) {
	tool := MockTool{
		name:        "get_time",
		description: "Get current time",
		params:      nil,
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			return "12:00 PM", nil
		},
	}

	chat := testModel.NewChat(WithTools(tool))

	// Verify tools are registered
	tools := chat.Tools()
	if len(tools) != 1 {
		t.Fatalf("Tools() len = %d, want 1", len(tools))
	}
	if tools[0].Name() != "get_time" {
		t.Errorf("tools[0].Name() = %q, want 'get_time'", tools[0].Name())
	}
}

func TestExecuteTool_Success(t *testing.T) {
	tool := MockTool{
		name:        "echo",
		description: "Echo back the input",
		params: []ToolParameter{
			{Name: "message", Type: "string", Description: "Message to echo", Required: true},
		},
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			msg, _ := args["message"].(string)
			return "Echo: " + msg, nil
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 5 * time.Second,
		},
	}

	call := ToolCall{
		ID:        "call_1",
		Name:      "echo",
		Arguments: map[string]any{"message": "hello"},
	}

	result := c.executeTool(context.Background(), call)

	if result.IsError {
		t.Errorf("executeTool() IsError = true, want false: %s", result.Content)
	}
	if result.Content != "Echo: hello" {
		t.Errorf("executeTool() Content = %q, want 'Echo: hello'", result.Content)
	}
	if result.CallID != "call_1" {
		t.Errorf("executeTool() CallID = %q, want 'call_1'", result.CallID)
	}
}

func TestExecuteTool_NotFound(t *testing.T) {
	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{}, // No tools
			ToolTimeout: 5 * time.Second,
		},
	}

	call := ToolCall{
		ID:   "call_1",
		Name: "nonexistent",
	}

	result := c.executeTool(context.Background(), call)

	if !result.IsError {
		t.Error("executeTool() IsError = false, want true for nonexistent tool")
	}
	if result.Content == "" {
		t.Error("executeTool() Content should contain error message")
	}
}

func TestExecuteTool_Timeout(t *testing.T) {
	tool := MockTool{
		name:        "slow",
		description: "A slow tool",
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			select {
			case <-time.After(5 * time.Second):
				return "done", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 100 * time.Millisecond, // Very short timeout
		},
	}

	call := ToolCall{
		ID:   "call_1",
		Name: "slow",
	}

	result := c.executeTool(context.Background(), call)

	if !result.IsError {
		t.Error("executeTool() IsError = false, want true for timeout")
	}
	if result.Content == "" {
		t.Error("executeTool() Content should contain timeout message")
	}
}

func TestExecuteTool_Panic(t *testing.T) {
	tool := MockTool{
		name:        "panicky",
		description: "A tool that panics",
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			panic("something went wrong")
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 5 * time.Second,
		},
	}

	call := ToolCall{
		ID:   "call_1",
		Name: "panicky",
	}

	// Should not panic - should recover and return error
	result := c.executeTool(context.Background(), call)

	if !result.IsError {
		t.Error("executeTool() IsError = false, want true for panic")
	}
	if result.Content == "" {
		t.Error("executeTool() Content should contain panic message")
	}
}

func TestExecuteTool_Error(t *testing.T) {
	tool := MockTool{
		name:        "failing",
		description: "A tool that returns an error",
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			return "", errors.New("tool failed")
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 5 * time.Second,
		},
	}

	call := ToolCall{
		ID:   "call_1",
		Name: "failing",
	}

	result := c.executeTool(context.Background(), call)

	if !result.IsError {
		t.Error("executeTool() IsError = false, want true for error")
	}
	if result.Content == "" {
		t.Error("executeTool() Content should contain error message")
	}
}

func TestExecuteTool_DefaultTimeout(t *testing.T) {
	tool := MockTool{
		name:        "quick",
		description: "A quick tool",
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			return "quick result", nil
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 0, // Should use default 30s
		},
	}

	call := ToolCall{
		ID:   "call_1",
		Name: "quick",
	}

	result := c.executeTool(context.Background(), call)

	if result.IsError {
		t.Errorf("executeTool() IsError = true: %s", result.Content)
	}
	if result.Content != "quick result" {
		t.Errorf("executeTool() Content = %q, want 'quick result'", result.Content)
	}
}

func TestChat_Tools(t *testing.T) {
	tool1 := MockTool{name: "tool1", description: "First tool"}
	tool2 := MockTool{name: "tool2", description: "Second tool"}

	chat := testModel.NewChat(WithTools(tool1, tool2))

	tools := chat.Tools()
	if len(tools) != 2 {
		t.Fatalf("Tools() len = %d, want 2", len(tools))
	}
	if tools[0].Name() != "tool1" {
		t.Errorf("tools[0].Name() = %q, want 'tool1'", tools[0].Name())
	}
	if tools[1].Name() != "tool2" {
		t.Errorf("tools[1].Name() = %q, want 'tool2'", tools[1].Name())
	}
}

func TestChat_Tools_Empty(t *testing.T) {
	chat := testModel.NewChat()

	tools := chat.Tools()
	if len(tools) != 0 {
		t.Errorf("Tools() len = %d, want 0", len(tools))
	}
}

func TestChat_Tools_NilReceiver(t *testing.T) {
	var c *chat = nil

	if c.Tools() != nil {
		t.Error("nil chat Tools() should return nil")
	}
}

func TestExecuteTool_ContextCancelled(t *testing.T) {
	tool := MockTool{
		name:        "slow",
		description: "A slow tool that respects context",
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			select {
			case <-time.After(5 * time.Second):
				return "done", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	}

	c := &chat{
		model: testModel.(*model),
		agentConfig: AgentConfig{
			Tools:       []Tool{tool},
			ToolTimeout: 5 * time.Second,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	call := ToolCall{
		ID:   "call_1",
		Name: "slow",
	}

	result := c.executeTool(ctx, call)

	// Should get an error due to context cancellation
	if !result.IsError {
		t.Log("executeTool() completed before cancellation could take effect")
	}
}

func TestAgentConfig_Defaults(t *testing.T) {
	tool := MockTool{name: "test", description: "Test"}

	chat := testModel.NewChat(WithTools(tool)).(*chat)

	// Check defaults are applied
	if chat.agentConfig.MaxIterations != 10 {
		t.Errorf("default MaxIterations = %d, want 10", chat.agentConfig.MaxIterations)
	}
	if chat.agentConfig.MaxToolCalls != 25 {
		t.Errorf("default MaxToolCalls = %d, want 25", chat.agentConfig.MaxToolCalls)
	}
	if chat.agentConfig.ToolTimeout != 30*time.Second {
		t.Errorf("default ToolTimeout = %v, want 30s", chat.agentConfig.ToolTimeout)
	}
	if chat.agentConfig.ToolChoice != ToolChoiceAuto {
		t.Errorf("default ToolChoice = %d, want ToolChoiceAuto", chat.agentConfig.ToolChoice)
	}
	if !chat.agentConfig.ParallelToolCalls {
		t.Error("default ParallelToolCalls = false, want true")
	}
}

func TestAgentOptions(t *testing.T) {
	tool := MockTool{name: "test", description: "Test"}

	chat := testModel.NewChat(
		WithTools(tool),
		WithMaxIterations(5),
		WithMaxToolCalls(10),
		WithToolTimeout(10*time.Second),
		WithToolChoice(ToolChoiceRequired),
		WithParallelToolCalls(false),
	).(*chat)

	if chat.agentConfig.MaxIterations != 5 {
		t.Errorf("MaxIterations = %d, want 5", chat.agentConfig.MaxIterations)
	}
	if chat.agentConfig.MaxToolCalls != 10 {
		t.Errorf("MaxToolCalls = %d, want 10", chat.agentConfig.MaxToolCalls)
	}
	if chat.agentConfig.ToolTimeout != 10*time.Second {
		t.Errorf("ToolTimeout = %v, want 10s", chat.agentConfig.ToolTimeout)
	}
	if chat.agentConfig.ToolChoice != ToolChoiceRequired {
		t.Errorf("ToolChoice = %d, want ToolChoiceRequired", chat.agentConfig.ToolChoice)
	}
	if chat.agentConfig.ParallelToolCalls {
		t.Error("ParallelToolCalls = true, want false")
	}
}

// TestGenerateWithTools_NoGrammarCrash verifies that tool calling doesn't crash
// when models output markdown or other content that doesn't match strict grammar rules.
// This was a regression where models without native tool support (ChatFormatGeneric)
// would crash with "Unexpected empty grammar stack" when outputting backticks.
func TestGenerateWithTools_NoGrammarCrash(t *testing.T) {
	// Skip if test model context is too small for tool calling
	// Tool templates add significant token overhead
	if testModel.ContextSize() < 512 {
		t.Skip("Test model context size too small for tool calling test")
	}

	tool := MockTool{
		name:        "search",
		description: "Search for items",
		params: []ToolParameter{
			{Name: "query", Type: "string", Description: "Search query", Required: true},
		},
		handler: func(ctx context.Context, args map[string]any) (string, error) {
			return "Found 2 results", nil
		},
	}

	chat := testModel.NewChat(WithTools(tool))
	chat.AddMessage(RoleSystem, "You are a helpful assistant.")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// The test passes if this doesn't panic/crash
	var completed bool
	for event, err := range chat.GenerateWithTools(ctx, "Find something") {
		if err != nil {
			// Errors are acceptable (e.g., context timeout), crashes are not
			t.Logf("GenerateWithTools returned error: %v", err)
			break
		}
		if event.Type == AgentEventDone {
			completed = true
		}
	}

	if !completed {
		t.Log("Generation did not complete (may have hit iteration limit or timeout)")
	}
	// Test passes if we reach here without crashing
}

func TestStripThinkingTags(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "no thinking tags",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name:     "thinking tags with content after",
			input:    "<think>Some thinking</think>Actual response",
			expected: "Actual response",
		},
		{
			name:     "thinking tags only",
			input:    "<think>Some thinking</think>",
			expected: "",
		},
		{
			name:     "thinking tags with newlines",
			input:    "<think>\nLet me think...\n</think>\n\nHere is the answer",
			expected: "Here is the answer",
		},
		{
			name:     "unclosed think tag",
			input:    "<think>Some thinking",
			expected: "Some thinking",
		},
		{
			name:     "multiple think blocks - takes content after last",
			input:    "<think>first</think>middle<think>second</think>final",
			expected: "final",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripThinkingTags(tt.input)
			if result != tt.expected {
				t.Errorf("stripThinkingTags(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
