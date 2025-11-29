package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/expki/zeus"
)

//go:embed models/SmolLM2-135M-Instruct-Q2_K.gguf
var embeddedModel []byte

func main() {
	var modelPath string
	var contextSize int
	var maxTokens int
	var system string
	var query string
	var verbose bool
	var seed int
	var timeout int

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&modelPath, "m", "", "path to GGUF model file (uses embedded model if not specified)")
	flags.IntVar(&contextSize, "c", 0, "context size (0 = use model's native context)")
	flags.IntVar(&maxTokens, "n", 0, "max tokens to generate (0 = till context is full)")
	flags.StringVar(&system, "s", "You are a helpful assistant.", "system prompt")
	flags.StringVar(&query, "p", "Count from 1 to 5", "prompt/query to send to the model")
	flags.IntVar(&seed, "r", -1, "seed for reproducible output (-1 = random)")
	flags.IntVar(&timeout, "timeout", 60, "generation timeout in seconds")
	flags.BoolVar(&verbose, "v", false, "show verbose logs")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing arguments: %s\n", err)
		os.Exit(1)
	}

	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: %s -p <prompt> [options]\n", os.Args[0])
		flags.PrintDefaults()
		os.Exit(1)
	}

	// Use embedded model if no model path specified
	modelPath = getModelPath(modelPath)

	// Suppress verbose llama.cpp logging
	zeus.SetVerbose(verbose)

	// Build model options
	modelOpts := []zeus.ModelOption{
		zeus.WithContextSize(contextSize),
		zeus.WithKVCacheType(zeus.KVCacheF16),
		zeus.WithSeed(seed),
	}

	// Load model
	model, err := zeus.LoadModel(modelPath, modelOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load model: %s\n", err.Error())
		os.Exit(1)
	}
	defer model.Close()

	// Print model info
	info, err := json.MarshalIndent(model.Info(), "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to marshal model info: %s\n", err)
		os.Exit(1)
	}
	fmt.Println("model info:", string(info))
	fmt.Println("model has chat template:", model.ChatTemplate() != "")

	// Build generation options
	genOpts := []zeus.GenerateOption{
		zeus.WithMaxTokens(maxTokens),
		zeus.WithTopK(40),
		zeus.WithTopP(0.95),
		zeus.WithTemperature(0.7),
		zeus.WithGenerateSeed(seed),
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Minute)
	defer cancel()

	// Create a new chat (automatically uses model's embedded template)
	chat := model.NewChat()
	chat.AddMessage(zeus.RoleSystem, system)

	// Stream tokens using the iterator API
	fmt.Print("\ngenerating response:\n\n")
	for tok, err := range chat.GenerateSequence(ctx, query, genOpts...) {
		if err != nil {
			if ctx.Err() != nil {
				fmt.Fprintf(os.Stderr, "\nGeneration timed out or cancelled\n")
			} else {
				fmt.Fprintf(os.Stderr, "\nGeneration failed: %s\n", err.Error())
			}
			break
		}
		fmt.Print(tok.Text)
	}
	fmt.Println()
}

// getModelPath returns the path to use for loading the model
func getModelPath(modelPath string) string {
	if modelPath != "" {
		return modelPath
	}

	// extract embedded model to deterministic temporary location
	exeName := "embedded"
	if ex, err := os.Executable(); err == nil {
		exeName = filepath.Base(ex)
	}
	tmpPath := filepath.Join(os.TempDir(), exeName+"-model.gguf")

	// check if file already exists and matches size
	if info, err := os.Stat(tmpPath); err == nil && info.Size() == int64(len(embeddedModel)) {
		return tmpPath // Reuse existing file
	}

	// extract embedded model to temp file
	if err := os.WriteFile(tmpPath, embeddedModel, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing embedded model: %s\n", err)
		os.Exit(1)
	}

	return tmpPath
}
