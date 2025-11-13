#!/usr/bin/env python3
"""
Interactive script to test your trained ismAIl model.
Load a checkpoint and generate text with custom prompts.
"""

import torch
import json
from pathlib import Path
import sys
from model import ismail, ModelArgs
from generation import (
    generate_text_simple,
    generate_text_with_sampling,
    text_to_token_ids,
    token_ids_to_text,
    get_tokenizer,
    load_checkpoint
)


def interactive_generation(model, tokenizer, args):
    """Interactive mode: continuously prompt for text and generate responses."""
    print("\n" + "="*60)
    print("🎤 INTERACTIVE GENERATION MODE")
    print("="*60)
    print("Commands:")
    print("  - Type your prompt and press Enter to generate")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'params' to change generation parameters")
    print("="*60 + "\n")

    # Default generation parameters
    temperature = 0.8
    top_k = 50
    max_tokens = 50
    use_sampling = True

    while True:
        try:
            prompt = input("\n💬 Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if prompt.lower() == 'params':
                print("\n⚙️  Current parameters:")
                print(f"   Temperature: {temperature}")
                print(f"   Top-k: {top_k}")
                print(f"   Max tokens: {max_tokens}")
                print(f"   Use sampling: {use_sampling}")

                try:
                    temp_input = input(f"   New temperature (current: {temperature}): ").strip()
                    if temp_input:
                        temperature = float(temp_input)

                    topk_input = input(f"   New top-k (current: {top_k}): ").strip()
                    if topk_input:
                        top_k = int(topk_input)

                    tokens_input = input(f"   New max tokens (current: {max_tokens}): ").strip()
                    if tokens_input:
                        max_tokens = int(tokens_input)

                    sampling_input = input(f"   Use sampling? (y/n, current: {'y' if use_sampling else 'n'}): ").strip()
                    if sampling_input:
                        use_sampling = sampling_input.lower() in ['y', 'yes', 't', 'true']

                    print("✅ Parameters updated!")
                except ValueError as e:
                    print(f"❌ Invalid input: {e}")
                continue

            if not prompt:
                print("⚠️  Empty prompt, try again")
                continue

            # Tokenize
            token_ids = text_to_token_ids(prompt, tokenizer)
            print(f"📝 Input tokens: {token_ids.shape[1]}")

            # Generate
            print("🤖 Generating...", end='', flush=True)
            if use_sampling:
                generated_ids = generate_text_with_sampling(
                    model=model,
                    idx=token_ids,
                    max_new_tokens=max_tokens,
                    context_size=args.max_seq_len,
                    temperature=temperature,
                    top_k=top_k
                )
            else:
                generated_ids = generate_text_simple(
                    model=model,
                    idx=token_ids,
                    max_new_tokens=max_tokens,
                    context_size=args.max_seq_len
                )

            # Decode
            generated_text = token_ids_to_text(generated_ids, tokenizer)
            print(f"\r🤖 Generated ({generated_ids.shape[1]} tokens):")
            print(f"\n{generated_text}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def batch_generation(model, tokenizer, args, prompts):
    """Generate text for a list of prompts."""
    print("\n" + "="*60)
    print("📋 BATCH GENERATION MODE")
    print("="*60 + "\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}/{len(prompts)} ---")
        print(f"Input: {prompt}")

        token_ids = text_to_token_ids(prompt, tokenizer)

        # Generate with sampling
        generated_ids = generate_text_with_sampling(
            model=model,
            idx=token_ids,
            max_new_tokens=50,
            context_size=args.max_seq_len,
            temperature=0.8,
            top_k=50
        )

        generated_text = token_ids_to_text(generated_ids, tokenizer)
        print(f"Output: {generated_text}\n")


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <checkpoint_path> [--interactive] [--prompts \"prompt1\" \"prompt2\" ...]")
        print("\nExample:")
        print("  python test_model.py checkpoints/step_55000_expert_2.pt --interactive")
        print("  python test_model.py checkpoints/step_55000_expert_2.pt --prompts \"Merhaba\" \"Yapay zeka\"")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    interactive_mode = '--interactive' in sys.argv or '-i' in sys.argv

    # Extract prompts from command line
    custom_prompts = []
    if '--prompts' in sys.argv:
        idx = sys.argv.index('--prompts')
        custom_prompts = [arg for arg in sys.argv[idx+1:] if not arg.startswith('--')]

    print("="*60)
    print("🧠 ismAIl Model Testing Script")
    print("="*60)

    # Load config
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_path}")
        args = ModelArgs(**config["model"])
    else:
        print("❌ config.json not found!")
        sys.exit(1)

    # Initialize tokenizer
    tokenizer_name = getattr(args, "tokenizer_name", "gpt2")
    use_turkish = tokenizer_name.lower() == "turkish"

    tokenizer = get_tokenizer(
        use_turkish=use_turkish,
        tokenizer_name="gpt2" if use_turkish else tokenizer_name
    )

    # Update vocab size if using Turkish tokenizer
    if use_turkish:
        from data import TurkishTokenizerWrapper
        if isinstance(tokenizer, TurkishTokenizerWrapper):
            if args.vocab_size != tokenizer.n_vocab:
                print(f"⚠️  Updating vocab_size: {args.vocab_size:,} -> {tokenizer.n_vocab:,}")
                args.vocab_size = tokenizer.n_vocab

    # Initialize model
    print("\n🚀 Initializing model...")
    model = ismail(args)

    # Load checkpoint
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        load_checkpoint(model, checkpoint_file)
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model.eval()

    # Run appropriate mode
    if interactive_mode:
        interactive_generation(model, tokenizer, args)
    elif custom_prompts:
        batch_generation(model, tokenizer, args, custom_prompts)
    else:
        # Default: use some Turkish prompts
        default_prompts = [
            "Merhaba, ben",
            "Yapay zekanın geleceği",
            "Bir varmış bir yokmuş",
            "Türkiye'nin başkenti"
        ]
        batch_generation(model, tokenizer, args, default_prompts)


if __name__ == "__main__":
    main()
