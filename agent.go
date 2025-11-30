package zeus

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"
	"time"
)

// AgentConfig holds configuration for agentic tool execution.
type AgentConfig struct {
	Tools             []Tool        // Registered tools
	MaxIterations     int           // Maximum agentic loop iterations (default: 10)
	MaxToolCalls      int           // Maximum total tool calls (default: 25)
	ToolTimeout       time.Duration // Per-tool execution timeout (default: 30s)
	ToolChoice        ToolChoice    // How the model should use tools (default: Auto)
	ParallelToolCalls bool          // Allow multiple tool calls in one response (default: true)
}

// DefaultAgentConfig returns an AgentConfig with sensible defaults.
func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		Tools:             nil,
		MaxIterations:     10,
		MaxToolCalls:      25,
		ToolTimeout:       30 * time.Second,
		ToolChoice:        ToolChoiceAuto,
		ParallelToolCalls: true,
	}
}

// GenerateWithTools executes an agentic loop, auto-executing tools until the model produces a final response without tool calls.
func (c *chat) GenerateWithTools(ctx context.Context, userMessage string, opts ...GenerateOption) iter.Seq2[AgentEvent, error] {
	return func(yield func(AgentEvent, error) bool) {
		if c == nil {
			yield(AgentEvent{Type: AgentEventError, Error: ErrChatIsNil}, ErrChatIsNil)
			return
		}
		if c.model == nil {
			yield(AgentEvent{Type: AgentEventError, Error: ErrModelIsNil}, ErrModelIsNil)
			return
		}
		if c.model.isClosed() {
			yield(AgentEvent{Type: AgentEventError, Error: ErrModelClosed}, ErrModelClosed)
			return
		}
		if len(c.agentConfig.Tools) == 0 {
			yield(AgentEvent{Type: AgentEventError, Error: ErrNoToolsRegistered}, ErrNoToolsRegistered)
			return
		}
		if c.session == nil {
			yield(AgentEvent{Type: AgentEventError, Error: ErrSessionIsNil}, ErrSessionIsNil)
			return
		}

		// Autocompact
		if c.config.AutoCompactThreshold > 0 && c.config.AutoCompactThreshold < 1 && (float32(c.session.TokenCount())/float32(c.model.ContextSize())) > c.config.AutoCompactThreshold {
			err := c.Compact(ctx)
			if err != nil {
				yield(AgentEvent{Type: AgentEventError, Error: err}, err)
				return
			}
		}

		// Save state for potential backtrack
		c.parent = &chat{
			model:         c.model,
			messages:      c.messages,
			parent:        c.parent,
			config:        c.config,
			agentConfig:   c.agentConfig,
			chatFormat:    c.chatFormat,
			session:       c.session.Checkpoint(),
			lastFormatted: c.lastFormatted,
		}

		// Add user message
		c.messages = append(slices.Clone(c.messages), ChatMessage{Role: RoleUser, Content: userMessage})

		totalToolCalls := 0
		maxIterations := c.agentConfig.MaxIterations
		if maxIterations <= 0 {
			maxIterations = 10
		}
		maxToolCalls := c.agentConfig.MaxToolCalls
		if maxToolCalls <= 0 {
			maxToolCalls = 25
		}

		for iteration := 0; iteration < maxIterations; iteration++ {
			// Check context cancellation
			if ctx.Err() != nil {
				yield(AgentEvent{Type: AgentEventError, Error: ctx.Err()}, ctx.Err())
				return
			}

			// Generate response with tool instructions
			response, err := c.generateWithToolPrompt(ctx, opts...)
			if err != nil {
				yield(AgentEvent{Type: AgentEventError, Error: err}, err)
				return
			}

			// Parse for tool calls using native parser
			parsed := parseToolCallsNative(response, c.chatFormat)
			content := parsed.Content
			toolCalls := parsed.ToolCalls

			// No tool calls = final response
			if len(toolCalls) == 0 {
				// Yield final response content
				if content != "" {
					if !yield(AgentEvent{Type: AgentEventToken, Token: &Token{Text: content}}, nil) {
						return
					}
				}
				yield(AgentEvent{Type: AgentEventDone}, nil)
				return
			}

			// Check tool call limit
			if totalToolCalls+len(toolCalls) > maxToolCalls {
				err := ErrMaxToolCallsExceeded
				yield(AgentEvent{Type: AgentEventError, Error: err}, err)
				return
			}

			// Update the assistant message with cleaned content (without tool call markup)
			// Must be done before adding tool results since assistant message is currently last
			if content != "" {
				for i := len(c.messages) - 1; i >= 0; i-- {
					if c.messages[i].Role == RoleAssistant {
						c.messages[i].Content = content
						break
					}
				}
			}

			// Execute tool calls
			for _, call := range toolCalls {
				// Signal tool call start
				if !yield(AgentEvent{Type: AgentEventToolCallStart, ToolCall: &call}, nil) {
					return
				}

				// Find and execute tool
				result := c.executeTool(ctx, call)
				totalToolCalls++

				// Signal tool call end
				if !yield(AgentEvent{Type: AgentEventToolCallEnd, ToolCall: &call, Result: &result}, nil) {
					return
				}

				// Add tool result to conversation
				c.messages = append(slices.Clone(c.messages), ChatMessage{
					Role:    RoleTool,
					Content: formatToolResult(call, result),
				})
			}

		}

		// Max iterations reached
		err := ErrMaxIterationsExceeded
		yield(AgentEvent{Type: AgentEventError, Error: err}, err)
	}
}

// generateWithToolPrompt generates a response with tool instructions injected.
func (c *chat) generateWithToolPrompt(ctx context.Context, opts ...GenerateOption) (string, error) {
	chatParams, err := c.model.applyChatTemplateWithTools(
		c.messages,
		c.agentConfig.Tools,
		c.agentConfig.ToolChoice,
		c.agentConfig.ParallelToolCalls,
	)
	if err != nil {
		return "", err
	}
	if chatParams == nil || chatParams.Prompt == "" {
		return "", ErrToolTemplateUnsupported
	}

	formatted := chatParams.Prompt
	c.chatFormat = chatParams.Format

	// Add grammar and stop sequences to generation options if provided
	if chatParams.Grammar != "" && !chatParams.GrammarLazy {
		opts = append(opts, WithGrammar(chatParams.Grammar))
	}
	if len(chatParams.AdditionalStops) > 0 {
		opts = append(opts, WithStopSequences(chatParams.AdditionalStops...))
	}

	// Compute delta
	var delta string
	if strings.HasPrefix(formatted, c.lastFormatted) {
		delta = formatted[len(c.lastFormatted):]
	} else {
		c.session = c.model.NewSession()
		delta = formatted
	}

	// Generate
	var response strings.Builder
	for tok, err := range c.session.GenerateSequence(ctx, delta, opts...) {
		if err != nil {
			return "", err
		}
		response.WriteString(tok.Text)
	}

	result := response.String()

	// Add assistant response to messages
	c.messages = append(slices.Clone(c.messages), ChatMessage{Role: RoleAssistant, Content: result})

	// Update lastFormatted
	fullFormatted, err := c.model.ApplyChatTemplate(c.messages,
		WithChatTemplate(c.config.Template),
		WithAddAssistant(false),
	)
	if err != nil {
		c.lastFormatted = formatted + result
	} else {
		c.lastFormatted = fullFormatted
	}

	return result, nil
}

// executeTool executes a single tool call with timeout and panic recovery.
func (c *chat) executeTool(ctx context.Context, call ToolCall) ToolResult {
	// Find tool
	var tool Tool
	for _, t := range c.agentConfig.Tools {
		if t.Name() == call.Name {
			tool = t
			break
		}
	}
	if tool == nil {
		return ToolResult{
			CallID:  call.ID,
			Content: fmt.Sprintf("Error: tool %q not found", call.Name),
			IsError: true,
		}
	}

	// Apply per-tool timeout
	toolTimeout := c.agentConfig.ToolTimeout
	if toolTimeout <= 0 {
		toolTimeout = 30 * time.Second
	}
	toolCtx, cancel := context.WithTimeout(ctx, toolTimeout)
	defer cancel()

	// Execute with panic recovery
	resultCh := make(chan ToolResult, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				resultCh <- ToolResult{
					CallID:  call.ID,
					Content: fmt.Sprintf("Error: tool panicked: %v", r),
					IsError: true,
				}
			}
		}()

		content, err := tool.Execute(toolCtx, call.Arguments)
		if err != nil {
			resultCh <- ToolResult{
				CallID:  call.ID,
				Content: fmt.Sprintf("Error: %v", err),
				IsError: true,
			}
			return
		}
		resultCh <- ToolResult{
			CallID:  call.ID,
			Content: content,
			IsError: false,
		}
	}()

	select {
	case result := <-resultCh:
		return result
	case <-toolCtx.Done():
		return ToolResult{
			CallID:  call.ID,
			Content: fmt.Sprintf("Error: tool execution timed out after %v", toolTimeout),
			IsError: true,
		}
	}
}

// Tools returns the registered tools for this chat.
func (c *chat) Tools() []Tool {
	if c == nil {
		return nil
	}
	return c.agentConfig.Tools
}

// formatToolResult formats a tool result for inclusion in conversation.
func formatToolResult(call ToolCall, result ToolResult) string {
	if result.IsError {
		return fmt.Sprintf("[Tool %s returned error: %s]", call.Name, result.Content)
	}
	return fmt.Sprintf("[Tool %s result: %s]", call.Name, result.Content)
}
