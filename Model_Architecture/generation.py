import torch
import tiktoken
from model import ismail, ModelArgs
from data import TurkishTokenizerWrapper, TURKISH_TOKENIZER_AVAILABLE


#####################################
# TEXT GENERATION FUNCTIONS
#####################################

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using simple greedy decoding (argmax).

    Args:
        model: The transformer model
        idx: Input token indices of shape (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context size the model can handle

    Returns:
        Generated token indices of shape (batch_size, seq_len + max_new_tokens)
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_text_with_sampling(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    Generate text using sampling with temperature and optional top-k filtering.

    Args:
        model: The transformer model
        idx: Input token indices of shape (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context size the model can handle
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
        top_k: If set, only sample from the top k most likely tokens

    Returns:
        Generated token indices of shape (batch_size, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :]

        # Clamp temperature to avoid division by very small numbers
        temperature = max(temperature, 1e-8)
        logits = logits / temperature

        # Optional: apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

        # Handle edge cases: check for invalid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Fallback to uniform distribution over valid tokens
            probs = torch.ones_like(probs) / probs.size(-1)

        # Ensure probabilities sum to 1
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    """
    Convert text to token IDs.

    Args:
        text: Input text string
        tokenizer: Tokenizer instance (tiktoken or TurkishTokenizerWrapper)

    Returns:
        Tensor of token IDs with shape (1, seq_len)
    """
    # Turkish tokenizer doesn't support allowed_special parameter
    if isinstance(tokenizer, TurkishTokenizerWrapper):
        encoded = tokenizer.encode(text)
    else:
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs to text.

    Args:
        token_ids: Tensor of token IDs, can be 1D or 2D
        tokenizer: Tokenizer instance (tiktoken or TurkishTokenizerWrapper)

    Returns:
        Decoded text string
    """
    # Handle both 1D and 2D tensors
    if token_ids.dim() == 2:
        token_ids = token_ids.squeeze(0)

    # Convert to list and decode
    flat = token_ids.tolist()
    return tokenizer.decode(flat)


def get_tokenizer(use_turkish=False, tokenizer_name="gpt2"):
    """
    Get the appropriate tokenizer based on user preference.

    Args:
        use_turkish: Whether to use Turkish tokenizer
        tokenizer_name: Name of tiktoken tokenizer to use if not using Turkish

    Returns:
        Tokenizer instance (TurkishTokenizerWrapper or tiktoken tokenizer)
    """
    if use_turkish:
        if not TURKISH_TOKENIZER_AVAILABLE:
            raise ImportError(
                "Turkish tokenizer requested but not available. "
                "Install it with: pip install turkish-tokenizer"
            )
        tokenizer = TurkishTokenizerWrapper()
        print(f"🇹🇷 Using Turkish Tokenizer (vocab size: {tokenizer.n_vocab:,})")
        return tokenizer
    else:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"📚 Using tiktoken tokenizer: {tokenizer_name} (vocab size: {tokenizer.n_vocab:,})")
        return tokenizer


#####################################
# EXAMPLE USAGE
#####################################

if __name__ == "__main__":
    import json
    from pathlib import Path

    # Configuration: Set to True to use Turkish tokenizer, False for tiktoken
    USE_TURKISH_TOKENIZER = True  # Change this to False for English text generation

    # Example configuration - smaller model for testing
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"✅ Loaded config from {config_path}")
        args = ModelArgs(**config["model"])
    else:
        print("⚠️ config.json not found, using default ModelArgs")
        args = ModelArgs()

    # Initialize tokenizer
    tokenizer_name = getattr(args, "tokenizer_name", "gpt2")
    # Auto-detect Turkish tokenizer from config
    use_turkish = (tokenizer_name.lower() == "turkish") or USE_TURKISH_TOKENIZER

    tokenizer = get_tokenizer(
        use_turkish=use_turkish,
        tokenizer_name="gpt2" if use_turkish else tokenizer_name
    )

    # Update vocab size if using Turkish tokenizer
    if use_turkish and isinstance(tokenizer, TurkishTokenizerWrapper):
        if args.vocab_size != tokenizer.n_vocab:
            print(f"⚠️  Config vocab_size ({args.vocab_size:,}) doesn't match tokenizer ({tokenizer.n_vocab:,})")
            args.vocab_size = tokenizer.n_vocab
            print(f"📊 Updated vocab_size to {args.vocab_size:,} for Turkish tokenizer")

    # Initialize model
    print("Initializing model...")
    torch.manual_seed(123)
    model = ismail(args)
    model.eval()

    # Example 1: Greedy generation (argmax)
    print(f"\n{'='*60}")
    print("EXAMPLE 1: GREEDY GENERATION (ARGMAX)")
    print(f"{'='*60}")

    # Use Turkish or English prompts based on tokenizer
    if USE_TURKISH_TOKENIZER:
        start_context = "Merhaba, ben"
    else:
        start_context = "Hello, I am"
    print(f"\nInput: '{start_context}'")

    token_ids = text_to_token_ids(start_context, tokenizer)
    print(f"Token IDs shape: {token_ids.shape}")

    generated_ids = generate_text_simple(
        model=model,
        idx=token_ids,
        max_new_tokens=20,
        context_size=args.max_seq_len
    )

    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"\nGenerated: '{generated_text}'")
    print(f"Total tokens: {generated_ids.shape[1]}")

    # Example 2: Sampling with temperature
    print(f"\n{'='*60}")
    print("EXAMPLE 2: SAMPLING WITH TEMPERATURE")
    print(f"{'='*60}")

    if USE_TURKISH_TOKENIZER:
        start_context = "Bir varmış bir yokmuş"
    else:
        start_context = "Once upon a time"
    print(f"\nInput: '{start_context}'")

    token_ids = text_to_token_ids(start_context, tokenizer)

    # Generate with different temperatures
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temp} ---")
        generated_ids = generate_text_with_sampling(
            model=model,
            idx=token_ids.clone(),
            max_new_tokens=20,
            context_size=args.max_seq_len,
            temperature=temp
        )
        generated_text = token_ids_to_text(generated_ids, tokenizer)
        print(f"Generated: '{generated_text}'")

    # Example 3: Top-k sampling
    print(f"\n{'='*60}")
    print("EXAMPLE 3: TOP-K SAMPLING")
    print(f"{'='*60}")

    if USE_TURKISH_TOKENIZER:
        start_context = "Yapay zekanın geleceği"
    else:
        start_context = "The future of AI is"
    print(f"\nInput: '{start_context}'")

    token_ids = text_to_token_ids(start_context, tokenizer)

    generated_ids = generate_text_with_sampling(
        model=model,
        idx=token_ids,
        max_new_tokens=30,
        context_size=args.max_seq_len,
        temperature=0.8,
        top_k=50
    )

    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"Generated: '{generated_text}'")

    print(f"\n{'='*60}")
    print("Generation examples completed!")
    print(f"{'='*60}\n")
