package zeus

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestNewChat(t *testing.T) {
	chat := testModel.NewChat()
	if chat == nil {
		t.Fatal("NewChat() returned nil")
	}
}

func TestChat_AddMessage(t *testing.T) {
	chat := testModel.NewChat()

	chat.AddMessage(RoleSystem, "You are helpful.")
	chat.AddMessage(RoleUser, "Hello")

	msgs := chat.Messages()
	if len(msgs) != 2 {
		t.Fatalf("Messages() len = %d, want 2", len(msgs))
	}

	if msgs[0].Role != RoleSystem {
		t.Errorf("msgs[0].Role = %s, want %s", msgs[0].Role, RoleSystem)
	}
	if msgs[0].Content != "You are helpful." {
		t.Errorf("msgs[0].Content = %q, want %q", msgs[0].Content, "You are helpful.")
	}
	if msgs[1].Role != RoleUser {
		t.Errorf("msgs[1].Role = %s, want %s", msgs[1].Role, RoleUser)
	}
}

func TestChat_MessageCount(t *testing.T) {
	chat := testModel.NewChat()

	if chat.MessageCount() != 0 {
		t.Errorf("fresh chat MessageCount() = %d, want 0", chat.MessageCount())
	}

	chat.AddMessage(RoleSystem, "System")
	if chat.MessageCount() != 1 {
		t.Errorf("after AddMessage MessageCount() = %d, want 1", chat.MessageCount())
	}

	chat.AddMessage(RoleUser, "User")
	if chat.MessageCount() != 2 {
		t.Errorf("after 2nd AddMessage MessageCount() = %d, want 2", chat.MessageCount())
	}
}

func TestChat_GenerateSequence(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "You are helpful.")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var response strings.Builder
	for tok, err := range chat.GenerateSequence(ctx, "Say hello", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
		response.WriteString(tok.Text)
	}

	if response.Len() == 0 {
		t.Error("GenerateSequence() should generate some text")
	}

	// Should now have 3 messages: system, user, assistant
	if chat.MessageCount() != 3 {
		t.Errorf("MessageCount() = %d, want 3 (system + user + assistant)", chat.MessageCount())
	}

	msgs := chat.Messages()
	if msgs[2].Role != RoleAssistant {
		t.Errorf("msgs[2].Role = %s, want %s", msgs[2].Role, RoleAssistant)
	}
}

func TestChat_MultiTurn(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "You are helpful.")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// First turn
	for _, err := range chat.GenerateSequence(ctx, "Hello", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("first GenerateSequence() error = %v", err)
		}
	}

	// Should have 3 messages
	if chat.MessageCount() != 3 {
		t.Errorf("after first turn MessageCount() = %d, want 3", chat.MessageCount())
	}

	// Second turn
	for _, err := range chat.GenerateSequence(ctx, "Thanks", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("second GenerateSequence() error = %v", err)
		}
	}

	// Should have 5 messages: system, user1, assistant1, user2, assistant2
	if chat.MessageCount() != 5 {
		t.Errorf("after second turn MessageCount() = %d, want 5", chat.MessageCount())
	}
}

func TestChat_Checkpoint(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "You are helpful.")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Generate first response
	for _, err := range chat.GenerateSequence(ctx, "Hello", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Create checkpoint
	checkpoint := chat.Checkpoint()
	checkpointCount := checkpoint.MessageCount()

	// Generate another response
	for _, err := range chat.GenerateSequence(ctx, "More", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Checkpoint should be unchanged
	if checkpoint.MessageCount() != checkpointCount {
		t.Errorf("checkpoint MessageCount changed from %d to %d", checkpointCount, checkpoint.MessageCount())
	}

	// Original should have more
	if chat.MessageCount() <= checkpointCount {
		t.Error("chat should have more messages than checkpoint")
	}
}

func TestChat_Backtrack(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "You are helpful.")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Generate first response
	for _, err := range chat.GenerateSequence(ctx, "Hello", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}
	firstCount := chat.MessageCount()

	// Generate second response
	for _, err := range chat.GenerateSequence(ctx, "More", WithMaxTokens(5)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Backtrack
	previous, ok := chat.Backtrack()
	if !ok {
		t.Fatal("Backtrack() returned ok=false")
	}

	// Should be back to first count
	if previous.MessageCount() != firstCount {
		t.Errorf("after Backtrack() MessageCount() = %d, want %d", previous.MessageCount(), firstCount)
	}
}

func TestChat_Backtrack_Initial(t *testing.T) {
	chat := testModel.NewChat()

	// Backtrack on fresh chat should return ok=false
	_, ok := chat.Backtrack()
	if ok {
		t.Error("Backtrack() on fresh chat should return ok=false")
	}
}

func TestChat_Messages_Copy(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "System")

	msgs := chat.Messages()
	msgs[0].Content = "Modified"

	// Original should be unchanged
	if chat.Messages()[0].Content == "Modified" {
		t.Error("Messages() should return a copy")
	}
}

func TestChat_Model(t *testing.T) {
	chat := testModel.NewChat()
	m := chat.Model()

	if m == nil {
		t.Fatal("Model() returned nil")
	}
	if m != testModel {
		t.Error("Model() should return parent model")
	}
}

func TestApplyChatTemplate(t *testing.T) {
	messages := []ChatMessage{
		{Role: RoleSystem, Content: "You are helpful."},
		{Role: RoleUser, Content: "Hello"},
	}

	formatted, err := testModel.ApplyChatTemplate(messages)
	if err != nil {
		t.Fatalf("ApplyChatTemplate() error = %v", err)
	}

	if formatted == "" {
		t.Error("ApplyChatTemplate() returned empty string")
	}

	// Should contain the message content
	if !strings.Contains(formatted, "Hello") {
		t.Errorf("formatted template should contain 'Hello': %s", formatted)
	}
}

func TestApplyChatTemplate_Empty(t *testing.T) {
	formatted, err := testModel.ApplyChatTemplate([]ChatMessage{})
	if err != nil {
		t.Fatalf("ApplyChatTemplate() error = %v", err)
	}

	if formatted != "" {
		t.Errorf("ApplyChatTemplate([]) = %q, want empty", formatted)
	}
}

func TestApplyChatTemplate_WithAssistant(t *testing.T) {
	messages := []ChatMessage{
		{Role: RoleUser, Content: "Hello"},
	}

	withAssistant, err := testModel.ApplyChatTemplate(messages, WithAddAssistant(true))
	if err != nil {
		t.Fatalf("ApplyChatTemplate(WithAddAssistant(true)) error = %v", err)
	}

	withoutAssistant, err := testModel.ApplyChatTemplate(messages, WithAddAssistant(false))
	if err != nil {
		t.Fatalf("ApplyChatTemplate(WithAddAssistant(false)) error = %v", err)
	}

	// With assistant should be longer (has assistant prefix)
	if len(withAssistant) <= len(withoutAssistant) {
		t.Errorf("WithAddAssistant(true) should produce longer output: with=%d, without=%d",
			len(withAssistant), len(withoutAssistant))
	}
}

func TestChat_NilReceiver(t *testing.T) {
	var c *chat = nil

	if c.Messages() != nil {
		t.Error("nil chat Messages() should return nil")
	}
	if c.MessageCount() != -1 {
		t.Errorf("nil chat MessageCount() = %d, want -1", c.MessageCount())
	}
	if c.Model() != nil {
		t.Error("nil chat Model() should return nil")
	}
	if cp := c.Checkpoint(); cp != nil {
		t.Error("nil chat Checkpoint() should return nil")
	}
	if _, ok := c.Backtrack(); ok {
		t.Error("nil chat Backtrack() should return ok=false")
	}
}

func TestChat_Compact(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "Help.")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Add minimal messages to trigger compact (3 total: keeps 1, compacts 2)
	chat.AddMessage(RoleUser, "A")
	chat.AddMessage(RoleAssistant, "B")

	initialCount := chat.MessageCount() // 3

	err := chat.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	// Should have fewer messages after compact (1 system with summary + 1 preserved = 2)
	if chat.MessageCount() >= initialCount {
		t.Errorf("MessageCount() after Compact = %d, should be less than %d", chat.MessageCount(), initialCount)
	}

	// First message should be system with summary
	msgs := chat.Messages()
	if msgs[0].Role != RoleSystem {
		t.Errorf("msgs[0].Role = %s, want %s", msgs[0].Role, RoleSystem)
	}
	if !strings.Contains(msgs[0].Content, "summary:") {
		t.Error("system message should contain 'summary:'")
	}
	if !strings.Contains(msgs[0].Content, "Help.") {
		t.Error("system message should preserve original system prompt")
	}
}

func TestChat_Compact_TooFewMessages(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "You are helpful.")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Only 1 message - nothing to compact
	err := chat.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact() with 1 message should not error, got: %v", err)
	}

	// Should still have 1 message
	if chat.MessageCount() != 1 {
		t.Errorf("MessageCount() = %d, want 1 (unchanged)", chat.MessageCount())
	}
}

func TestChat_Compact_NoSystemPrompt(t *testing.T) {
	chat := testModel.NewChat()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Add messages without system prompt (2 total, keeps 1, compacts 1)
	chat.AddMessage(RoleUser, "A")
	chat.AddMessage(RoleAssistant, "B")

	err := chat.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	// First message should be system with summary
	msgs := chat.Messages()
	if msgs[0].Role != RoleSystem {
		t.Errorf("msgs[0].Role = %s, want %s", msgs[0].Role, RoleSystem)
	}
	if !strings.Contains(msgs[0].Content, "summary:") {
		t.Error("system message should contain 'summary:'")
	}
}

func TestChat_Compact_ResetsSession(t *testing.T) {
	chat := testModel.NewChat()
	chat.AddMessage(RoleSystem, "Help.")

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Generate to populate session and create parent for backtrack
	for _, err := range chat.GenerateSequence(ctx, "Hi", WithMaxTokens(3)) {
		if err != nil {
			t.Fatalf("GenerateSequence() error = %v", err)
		}
	}

	// Backtrack should work before compact
	_, ok := chat.Backtrack()
	if !ok {
		t.Error("Backtrack() should work before Compact")
	}

	// Restore chat and add minimal messages for compact
	chat = testModel.NewChat()
	chat.AddMessage(RoleSystem, "Help.")
	chat.AddMessage(RoleUser, "A")
	chat.AddMessage(RoleAssistant, "B")

	err := chat.Compact(ctx)
	if err != nil {
		t.Fatalf("Compact() error = %v", err)
	}

	// Backtrack should not work after compact (parent cleared)
	_, ok = chat.Backtrack()
	if ok {
		t.Error("Backtrack() should return ok=false after Compact")
	}
}

func TestChat_Compact_NilReceiver(t *testing.T) {
	var c *chat = nil

	ctx := context.Background()
	err := c.Compact(ctx)
	if err != ErrChatIsNil {
		t.Errorf("nil chat Compact() error = %v, want ErrChatIsNil", err)
	}
}
