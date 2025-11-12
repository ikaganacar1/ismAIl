import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Literal, List
from pathlib import Path
from tqdm import tqdm
import mmap
import numpy as np

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
    """
    Wrapper for Turkish Tokenizer to make it compatible with tiktoken interface.
    This allows seamless integration with the existing TextDataset class.
    """
    def __init__(self):
        if not TURKISH_TOKENIZER_AVAILABLE:
            raise ImportError(
                "turkish-tokenizer package is not installed. "
                "Install it with: pip install turkish-tokenizer"
            )
        self.tokenizer = TurkishTokenizerBase()
        self.name = "turkish-tokenizer"

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        """
        Encode text to token IDs (compatible with tiktoken interface).

        Args:
            text: Input text to tokenize
            allowed_special: Not used for Turkish tokenizer, kept for compatibility

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
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
class TextDataset(Dataset):
    def __init__(self, txt: str, tokenizer, args: ModelArgs, stride: Optional[int] = None, max_samples: Optional[int] = None):
        """
        Optimized text dataset with memory-mapped reading and batched tokenization.
        
        Args:
            txt: Text content or path to file
            tokenizer: Pretrained tokenizer with .encode() method
            args: ModelArgs containing max_seq_len, max_batch_size
            stride: Sliding window stride. Defaults to max_seq_len // 2
            max_samples: Limit number of samples for quick testing
        """
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
        """Memory-efficient file reading for large files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return mm.read().decode('utf-8', errors='ignore')
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path}: {e}")

    def _tokenize_with_progress(self, tokenizer, text: str) -> List[int]:
        """Tokenize with progress bar for large texts"""
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
        """Create overlapping sequences using vectorized operations"""
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
    use_turkish_tokenizer: bool = False
) -> DataLoader:
    """
    Optimized DataLoader with proper memory pinning and worker settings.

    Args:
        txt: Text content or file path
        args: ModelArgs configuration
        stride: Sliding window stride
        shuffle: Whether to shuffle samples
        drop_last: Drop incomplete batches
        num_workers: Number of data loading workers (0 = main process)
        pin_memory: Pin memory for faster GPU transfer (recommended)
        persistent_workers: Keep workers alive between epochs (if num_workers > 0)
        max_samples: Limit samples for testing
        use_turkish_tokenizer: Use Turkish morphological tokenizer instead of tiktoken
    """
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
        dataset = TextDataset(
            txt=txt,
            tokenizer=tokenizer,
            args=args,
            stride=stride,
            max_samples=max_samples
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {e}")
    
    # Create DataLoader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=args.max_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return dataloader


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