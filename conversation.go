package zeus

import (
	"context"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
)

// Chat represents a conversation that tracks message history.
// Generate methods mutate the chat, appending assistant responses.
// Use Checkpoint() before generation to save state for branching.
// Note: Chat wraps a Session internally for KV cache efficiency.
type Chat interface {
	// Generate sends a user message and returns the assistant's response as a stream.
	// The user message and assistant response are added to the conversation.
	Generate(ctx context.Context, userMessage string, opts ...GenerateOption) io.ReadCloser

	// GenerateSequence sends a user message and returns tokens as an iterator.
	// The user message and assistant response are added to the conversation.
	GenerateSequence(ctx context.Context, userMessage string, opts ...GenerateOption) iter.Seq2[Token, error]

	// AddMessage adds a message to the conversation without generating.
	// Useful for adding system prompts or reconstructing conversation history.
	AddMessage(role Role, content string)

	// Checkpoint creates a snapshot of the current chat state.
	Checkpoint() Chat

	// Backtrack returns to the state before the last Generate call.
	Backtrack() (Chat, bool)

	// Messages returns a copy of the message history.
	Messages() []ChatMessage

	// MessageCount returns the number of messages in the conversation.
	MessageCount() int

	// Model returns the parent model.
	Model() Model

	// Compact summarizes older messages and replaces them with the summary, keeping the last 10% of messages (or at least 1).
	Compact(ctx context.Context) error
}

// ChatConfig holds options for creating a Chat.
type ChatConfig struct {
	ChatTemplateConfig // Embedded - Template, AddAssistant
}

// ChatOption is a functional option for NewChat.
type ChatOption func(*ChatConfig)

// DefaultChatConfig returns the default chat configuration.
func DefaultChatConfig() ChatConfig {
	return ChatConfig{
		ChatTemplateConfig: DefaultChatTemplateConfig(),
	}
}

// chat implements the Chat interface.
type chat struct {
	model    *model
	messages []ChatMessage
	parent   *chat
	config   ChatConfig

	// Session wrapping for KV cache efficiency
	session       Session // Underlying session for token management
	lastFormatted string  // Last formatted prompt (to detect what's new)
}

// NewChat creates a new empty chat for conversation.
func (m *model) NewChat(opts ...ChatOption) Chat {
	if m == nil || m.isClosed() {
		return nil
	}

	cfg := DefaultChatConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &chat{
		model:    m,
		messages: nil,
		parent:   nil,
		config:   cfg,
		session:  m.NewSession(),
	}
}

// Generate sends a user message and returns the assistant's response as a stream.
func (c *chat) Generate(ctx context.Context, userMessage string, opts ...GenerateOption) io.ReadCloser {
	pr, pw := io.Pipe()
	if c == nil {
		pw.CloseWithError(ErrChatIsNil)
		return pr
	}

	go func() {
		defer pw.Close()

		for tok, err := range c.GenerateSequence(ctx, userMessage, opts...) {
			if err != nil {
				pw.CloseWithError(err)
				return
			}
			if _, err := pw.Write([]byte(tok.Text)); err != nil {
				return
			}
		}
	}()

	return pr
}

// GenerateSequence sends a user message and returns tokens as an iterator.
func (c *chat) GenerateSequence(ctx context.Context, userMessage string, opts ...GenerateOption) iter.Seq2[Token, error] {
	return func(yield func(Token, error) bool) {
		if c == nil {
			yield(Token{}, ErrChatIsNil)
			return
		}
		if c.session == nil {
			yield(Token{}, ErrSessionIsNil)
			return
		}
		if c.model == nil {
			yield(Token{}, ErrModelIsNil)
			return
		}
		if c.model.isClosed() {
			yield(Token{}, ErrModelClosed)
			return
		}

		// Autocompact
		if c.config.AutoCompactThreshold > 0 && c.config.AutoCompactThreshold < 1 && (float32(c.session.TokenCount())/float32(c.model.ContextSize())) > c.config.AutoCompactThreshold {
			err := c.Compact(ctx)
			if err != nil {
				yield(Token{}, err)
				return
			}
		}

		// Save current state as parent for Backtrack support
		c.parent = &chat{
			model:         c.model,
			messages:      c.messages,
			parent:        c.parent,
			config:        c.config,
			session:       c.session.Checkpoint(),
			lastFormatted: c.lastFormatted,
		}

		// Add user message
		c.messages = append(slices.Clone(c.messages), ChatMessage{Role: RoleUser, Content: userMessage})

		// Format all messages with assistant prefix
		formatted, err := c.model.ApplyChatTemplate(c.messages,
			WithChatTemplate(c.config.Template),
			WithAddAssistant(true),
		)
		if err != nil {
			yield(Token{}, err)
			return
		}

		// Compute delta: new content since last formatted prompt
		var delta string
		if strings.HasPrefix(formatted, c.lastFormatted) {
			delta = formatted[len(c.lastFormatted):]
		} else {
			// Templates diverged, use full formatted prompt
			// This resets the session
			c.session = c.model.NewSession()
			delta = formatted
		}

		// Generate using the underlying session
		var responseBuilder strings.Builder
		for tok, genErr := range c.session.GenerateSequence(ctx, delta, opts...) {
			if genErr != nil {
				yield(Token{}, genErr)
				return
			}
			responseBuilder.WriteString(tok.Text)
			if !yield(tok, nil) {
				// Early termination - still save what we got
				break
			}
		}

		// Add assistant response to messages
		response := responseBuilder.String()
		c.messages = append(c.messages, ChatMessage{Role: RoleAssistant, Content: response})

		// Update lastFormatted to include the response
		// Format again without assistant prefix to get the full conversation
		fullFormatted, err := c.model.ApplyChatTemplate(c.messages,
			WithChatTemplate(c.config.Template),
			WithAddAssistant(false),
		)
		if err != nil {
			// Fall back to simple concatenation
			c.lastFormatted = formatted + response
		} else {
			c.lastFormatted = fullFormatted
		}
	}
}

// AddMessage adds a message to the conversation without generating.
func (c *chat) AddMessage(role Role, content string) {
	if c == nil {
		return
	}
	c.messages = append(c.messages, ChatMessage{Role: role, Content: content})
}

// Checkpoint creates a snapshot of the current chat state.
func (c *chat) Checkpoint() Chat {
	if c == nil {
		return nil
	}
	if c.session == nil {
		return nil
	}
	return &chat{
		model:         c.model,
		messages:      slices.Clone(c.messages),
		parent:        c.parent,
		config:        c.config,
		session:       c.session.Checkpoint(),
		lastFormatted: c.lastFormatted,
	}
}

// Backtrack returns to the state before the last Generate call.
func (c *chat) Backtrack() (Chat, bool) {
	if c == nil {
		return nil, false
	}
	if c.parent == nil {
		return c, false
	}
	return c.parent, true
}

// Messages returns a copy of the message history.
func (c *chat) Messages() []ChatMessage {
	if c == nil {
		return nil
	}
	return slices.Clone(c.messages)
}

// MessageCount returns the number of messages in the conversation.
func (c *chat) MessageCount() int {
	if c == nil {
		return -1
	}
	return len(c.messages)
}

// Model returns the parent model.
func (c *chat) Model() Model {
	if c == nil {
		return nil
	}
	return c.model
}

const compactSystemPrompt = "You are a helpful assistant that summarizes conversations. Create a concise summary that preserves key information, decisions, and context needed to continue the discussion."

// Compact summarizes older messages and replaces them with the summary,
// keeping the last 10% of messages (or at least 1).
func (c *chat) Compact(ctx context.Context) error {
	if c == nil {
		return ErrChatIsNil
	}
	if c.model == nil {
		return ErrModelIsNil
	}
	if len(c.messages) <= 1 {
		return nil
	}

	// Extract original system prompt
	var originalSystemPrompt string
	if c.messages[0].Role == RoleSystem {
		originalSystemPrompt = strings.Split(c.messages[0].Content, "\n\nsummary: ")[0]
	}

	// Keep last 10% of messages, minimum 1
	keepLast := max(len(c.messages)/10, 1)

	removeCount := len(c.messages) - keepLast
	toSummarize := c.messages[:removeCount]
	preserved := c.messages[removeCount:]

	// Format messages for summarization
	var prompt strings.Builder
	prompt.WriteString("Summarize this conversation:\n\n")
	for _, msg := range toSummarize {
		prompt.WriteString(string(msg.Role))
		prompt.WriteString(": ")
		prompt.WriteString(msg.Content)
		prompt.WriteString("\n\n")
	}

	// Generate summary using a temporary chat
	tempChat := c.model.NewChat()
	tempChat.AddMessage(RoleSystem, compactSystemPrompt)
	reader := tempChat.Generate(ctx, prompt.String())
	summaryBytes, err := io.ReadAll(reader)
	reader.Close()
	if err != nil {
		return err
	}
	summary := string(summaryBytes)

	// Build new messages: summary + preserved
	c.messages = slices.Concat(
		[]ChatMessage{{
			Role:    RoleSystem,
			Content: fmt.Sprintf("%s\n\nsummary: %s", originalSystemPrompt, summary),
		}},
		preserved,
	)

	// Reset session - prompt structure changed
	c.session = c.model.NewSession()
	c.lastFormatted = ""
	c.parent = nil

	return nil
}
