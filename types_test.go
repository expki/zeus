package zeus

import "testing"

func TestKVCacheType_String(t *testing.T) {
	tests := []struct {
		input    KVCacheType
		expected string
	}{
		{KVCacheF32, "f32"},
		{KVCacheF16, "f16"},
		{KVCacheQ8_0, "q8_0"},
		{KVCacheQ4_0, "q4_0"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.input.String(); got != tt.expected {
				t.Errorf("KVCacheType(%d).String() = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestKVCacheType_String_Invalid(t *testing.T) {
	// Invalid value should default to "f16"
	invalid := KVCacheType(99)
	if got := invalid.String(); got != "f16" {
		t.Errorf("KVCacheType(99).String() = %q, want 'f16'", got)
	}
}

func TestStopReason_String(t *testing.T) {
	tests := []struct {
		input    StopReason
		expected string
	}{
		{StopReasonEOS, "eos"},
		{StopReasonMaxTokens, "max_tokens"},
		{StopReasonStopSequence, "stop_sequence"},
		{StopReasonCancelled, "cancelled"},
		{StopReasonError, "error"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if got := tt.input.String(); got != tt.expected {
				t.Errorf("StopReason(%d).String() = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestStopReason_String_Invalid(t *testing.T) {
	// Invalid value should return "unknown"
	invalid := StopReason(99)
	if got := invalid.String(); got != "unknown" {
		t.Errorf("StopReason(99).String() = %q, want 'unknown'", got)
	}
}

func TestGPULayersAll(t *testing.T) {
	if GPULayersAll != 999 {
		t.Errorf("GPULayersAll = %d, want 999", GPULayersAll)
	}
}
