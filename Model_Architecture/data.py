import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Literal, List
from pathlib import Path
from tqdm import tqdm
import mmap
import numpy as np
import os
import json

from model import ModelArgs

# Turkish Tokenizer support
try:
    from turkish_tokenizer import TurkishTokenizer as TurkishTokenizerBase
    TURKISH_TOKENIZER_AVAILABLE = True
except ImportError:
    TURKISH_TOKENIZER_AVAILABLE = False
    TurkishTokenizerBase = None

#####################################
# TURKISH TOKENIZER WRAPPER
#####################################
class TurkishTokenizerWrapper:
    def __init__(self):
        if not TURKISH_TOKENIZER_AVAILABLE:
            raise ImportError(
                "turkish-tokenizer package is not installed. "
                "Install it with: pip install turkish-tokenizer"
            )
        self.tokenizer = TurkishTokenizerBase()
        self.name = "turkish-tokenizer"

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def n_vocab(self) -> int:
        """Get vocabulary size"""
        return self.tokenizer.vocab_size

    @property
    def max_token_value(self) -> int:
        """Get maximum token value"""
        return self.n_vocab - 1


#####################################
# DATA
#####################################

class MemoryEfficientTextDataset(Dataset):
    """
    Memory-efficient dataset that tokenizes on-the-fly from disk.
    Instead of loading all data into RAM, it:
    1. Memory-maps the text file
    2. Pre-computes line offsets for fast random access
    3. Tokenizes only the required chunks during __getitem__
    """
    def __init__(self, file_path: str, tokenizer, args: ModelArgs, stride: Optional[int] = None, max_samples: Optional[int] = None):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.stride = stride if stride is not None else self.max_seq_len // 2

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        print(f"📝 Creating memory-efficient dataset from {self.file_path.name}...")

        # Get file size
        file_size = self.file_path.stat().st_size
        print(f"   File size: {file_size / 1024**2:.1f} MB")

        # Pre-tokenize to get sample count (lightweight - just count)
        self._count_samples(max_samples)

        print(f"✅ Dataset ready with {len(self.samples)} samples (using lazy loading)")

    def _count_samples(self, max_samples: Optional[int]):
        """Count how many samples we can create without loading everything"""
        # Read file in chunks to estimate token count
        chunk_size = 1024 * 1024  # 1MB chunks
        total_tokens = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                # Quick estimate: ~4 chars per token for most languages
                total_tokens += len(chunk) // 4

        # Calculate number of samples
        num_samples = max((total_tokens - self.max_seq_len - 1) // self.stride, 1)

        if max_samples:
            num_samples = min(num_samples, max_samples)

        # Store sample metadata (start positions in tokens)
        self.samples = list(range(num_samples))
        self.estimated_total_tokens = total_tokens

    def _get_text_chunk(self, token_start: int) -> str:
        """Get a chunk of text starting from approximate token position"""
        # Estimate byte position (rough: 4 chars per token, 1 byte per char for ASCII/UTF-8)
        approx_byte_pos = token_start * 4
        chunk_size = (self.max_seq_len + 1) * 8  # Read extra for safety

        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(max(0, approx_byte_pos - 100))  # Seek with small buffer
            return f.read(chunk_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize on-the-fly for this specific sample"""
        # Calculate token position
        token_start = idx * self.stride

        # Get text chunk from file
        text_chunk = self._get_text_chunk(token_start)

        # Tokenize
        try:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(text_chunk, allowed_special={"<|endoftext|>"})
            else:
                tokens = self.tokenizer.encode(text_chunk)
        except:
            # Fallback for Turkish tokenizer
            tokens = self.tokenizer.encode(text_chunk)

        # Ensure we have enough tokens
        if len(tokens) < self.max_seq_len + 1:
            # Pad with zeros if needed
            tokens = tokens + [0] * (self.max_seq_len + 1 - len(tokens))

        # Create input/target pair
        input_ids = torch.tensor(tokens[:self.max_seq_len], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:self.max_seq_len + 1], dtype=torch.long)

        return input_ids, target_ids


class TextDataset(Dataset):
    def __init__(self, txt: str, tokenizer, args: ModelArgs, stride: Optional[int] = None, max_samples: Optional[int] = None):
        self.max_seq_len = args.max_seq_len
        self.stride = stride if stride is not None else self.max_seq_len // 2
        
        # Handle file paths efficiently with memory mapping
        # Check if txt is a file path (avoid Path().exists() for long strings)
        try:
            path = Path(txt)
            if len(txt) < 4096 and path.exists():  # Reasonable path length check
                text_content = self._read_file_mmap(txt)
            else:
                text_content = txt
        except (OSError, ValueError):
            # If Path() fails or string is too long, treat as raw text
            text_content = txt
        
        # Validate input
        if not text_content or len(text_content.strip()) < self.max_seq_len:
            raise ValueError(f"Text too short. Need at least {self.max_seq_len} chars, got {len(text_content)}")
        
        print(f"📝 Tokenizing {len(text_content):,} characters...")
        
        # Tokenize with progress bar for large texts
        token_ids = self._tokenize_with_progress(tokenizer, text_content)
        
        # Create sliding windows with vectorized operations
        self.samples = self._create_sliding_windows(token_ids, max_samples)
        
        print(f"✅ Created {len(self.samples)} training samples")

    def _read_file_mmap(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().decode('utf-8', errors='ignore')
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

    def _tokenize_with_progress(self, tokenizer, text: str) -> List[int]:
        # Process in chunks for memory efficiency
        chunk_size = 10_000_000  # 10MB chunks
        tokens = []
        
        if len(text) > chunk_size:
            # Process large texts in chunks
            pbar = tqdm(total=len(text), desc="Tokenizing", unit="char")
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunk_tokens = tokenizer.encode(chunk, allowed_special={"<|endoftext|>"})
                tokens.extend(chunk_tokens)
                pbar.update(len(chunk))
            pbar.close()
        else:
            # Single pass for smaller texts
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        if not tokens:
            raise ValueError("No tokens generated from input text")
            
        return tokens

    def _create_sliding_windows(self, token_ids: List[int], max_samples: Optional[int]) -> torch.Tensor:
        if len(token_ids) < self.max_seq_len + 1:
            raise ValueError(f"Not enough tokens. Need {self.max_seq_len + 1}, got {len(token_ids)}")
        
        # Convert to numpy for faster slicing
        tokens_array = np.array(token_ids, dtype=np.int64)
        
        # Calculate number of windows
        num_windows = (len(tokens_array) - self.max_seq_len - 1) // self.stride + 1
        
        if max_samples:
            num_windows = min(num_windows, max_samples)
        
        # Pre-allocate tensors
        inputs = torch.zeros(num_windows, self.max_seq_len, dtype=torch.long)
        targets = torch.zeros(num_windows, self.max_seq_len, dtype=torch.long)
        
        # Fill tensors efficiently
        for i in range(num_windows):
            start = i * self.stride
            inputs[i] = torch.from_numpy(tokens_array[start:start + self.max_seq_len])
            targets[i] = torch.from_numpy(tokens_array[start + 1:start + self.max_seq_len + 1])
        
        # Stack into pairs (more memory efficient than separate lists)
        self.samples = torch.stack([inputs, targets], dim=1)
        
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) tuple"""
        return self.samples[idx, 0], self.samples[idx, 1]


def create_dataloader(
    txt: str,
    args: ModelArgs,
    stride: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    max_samples: Optional[int] = None,
    use_turkish_tokenizer: bool = True,
    use_memory_efficient: bool = True,  # NEW: Use memory-efficient loading by default
    is_val: bool = True 

) -> DataLoader:

    # Select tokenizer based on user preference
    if use_turkish_tokenizer:
        if not TURKISH_TOKENIZER_AVAILABLE:
            raise ImportError(
                "Turkish tokenizer requested but not available. "
                "Install it with: pip install turkish-tokenizer"
            )
        tokenizer = TurkishTokenizerWrapper()
        print(f"🇹🇷 Using Turkish Tokenizer (vocab size: {tokenizer.n_vocab:,})")
    else:
        # Use the best default tokenizer for your setup
        # tiktoken's gpt2 is fast, well-tested, and has reasonable vocab size (~50k)
        # For multilingual or code, consider "cl100k_base" or "o200k_base"
        tokenizer_name = getattr(args, "tokenizer_name", "gpt2")
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"📚 Using tiktoken tokenizer: {tokenizer_name} (vocab size: {tokenizer.n_vocab:,})")

    # Create dataset with size validation
    try:
        # Check if txt is a file path
        is_file_path = False
        try:
            path = Path(txt)
            if len(txt) < 4096 and path.exists():
                is_file_path = True
        except (OSError, ValueError):
            pass

        # Use memory-efficient dataset for file paths
        if use_memory_efficient and is_file_path:
            print(f"💾 Using memory-efficient dataset (lazy loading from disk)")
            dataset = MemoryEfficientTextDataset(
                file_path=txt,
                tokenizer=tokenizer,
                args=args,
                stride=stride,
                max_samples=max_samples
            )
        else:
            # Use original dataset (loads everything into RAM)
            print(f"⚠️  Using in-memory dataset (loads all data into RAM)")
            dataset = TextDataset(
                txt=txt,
                tokenizer=tokenizer,
                args=args,
                stride=stride,
                max_samples=max_samples
            )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {e}")

    config_path = Path("config.json")
    
    with open(config_path,"r") as f:
        config = json.load(f)
        val_batch_size = config["model"]["max_batch_size"] #* config["training"].get("val_batch_size_multiplier", 4)

    if is_val:
        batch_size = val_batch_size
    else:
        batch_size = args.max_batch_size

    # Create DataLoader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, tokenizer


# Convenience function for downloading sample data
def get_sample_data(url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt") -> str:
    """Download sample text data for testing"""
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"⚠️  Could not download sample data: {e}")
        return ""
    

if __name__ == "__main__":
    print("=" * 60)
    print("TOKENIZER TESTING")
    print("=" * 60)

    # Choose which tokenizer to test
    USE_TURKISH = True  # Set to False to test tiktoken instead

    if USE_TURKISH and TURKISH_TOKENIZER_AVAILABLE:
        print("\n🇹🇷 Testing Turkish Tokenizer")
        tokenizer = TurkishTokenizerWrapper()
        print(f"📚 Tokenizer: {tokenizer.name}")
        print(f"📊 Vocabulary Size: {tokenizer.n_vocab:,}")
        print(f"📝 Max Token Value: {tokenizer.max_token_value:,}")
    else:
        # Test different tokenizers
        tokenizer_name = "gpt2"  # Change to "cl100k_base" or "o200k_base" to test others
        tokenizer = tiktoken.get_encoding(tokenizer_name)

        print(f"\n📚 Tokenizer: {tokenizer_name}")
        print(f"📊 Vocabulary Size: {tokenizer.n_vocab:,}")
        print(f"📝 Max Token Value: {tokenizer.max_token_value:,}")
        print(f"🔤 Name: {tokenizer.name}")

    # Test encoding/decoding
    if USE_TURKISH and TURKISH_TOKENIZER_AVAILABLE:
        test_samples = [
            "Merhaba Dünya!",
            "İstanbul'da yaşıyorum ve Türkçe dilini öğreniyorum.",
            "Kitap okumak çok güzeldir ve bilgi verir.",
            "Türkiye Cumhuriyeti'nin başkenti Ankara'dır.",
            "Yapay zeka ve makine öğrenmesi teknolojileri gelişiyor.",
        ]
    else:
        test_samples = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
            "print('Hello, World!')",  # Code sample
            "日本語のテキスト",  # Non-English
        ]

    print("\n" + "=" * 60)
    print("ENCODING EXAMPLES")
    print("=" * 60)

    for text in test_samples:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\nText: {text}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Token range: [{min(tokens)}, {max(tokens)}]")
        print(f"Decoded: {decoded}")

    # Test with actual data
    print("\n" + "=" * 60)
    print("DATALOADER TESTING")
    print("=" * 60)

    sample_text = get_sample_data()
    if sample_text:
        print(f"\n📄 Sample text length: {len(sample_text):,} characters")

        # Tokenize sample
        if USE_TURKISH and TURKISH_TOKENIZER_AVAILABLE:
            full_tokens = tokenizer.encode(sample_text)
        else:
            full_tokens = tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})

        print(f"🔢 Total tokens: {len(full_tokens):,}")
        print(f"📈 Unique tokens used: {len(set(full_tokens)):,}")
        print(f"📊 Vocabulary coverage: {len(set(full_tokens)) / tokenizer.n_vocab * 100:.2f}%")

        # Create dataloader
        args = ModelArgs(max_seq_len=128, max_batch_size=16)
        dataloader = create_dataloader(
            sample_text,
            args,
            num_workers=0,
            max_samples=100,
            use_turkish_tokenizer=USE_TURKISH and TURKISH_TOKENIZER_AVAILABLE
        )

        print(f"\n⚙️  DataLoader Config:")
        print(f"   Sequence length: {args.max_seq_len}")
        print(f"   Batch size: {args.max_batch_size}")
        print(f"   Total batches: {len(dataloader)}")

        # Test first batch
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            print(f"\n🎯 Batch {batch_idx}:")
            print(f"   input_ids shape: {input_ids.shape}")
            print(f"   target_ids shape: {target_ids.shape}")
            print(f"   input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
            print(f"   Sample input (first 10 tokens): {input_ids[0, :10].tolist()}")
            print(f"   Decoded: {tokenizer.decode(input_ids[0, :10].tolist())}")
            break

    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)