import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Literal, List
from pathlib import Path
from tqdm import tqdm
import mmap
import numpy as np

from model import ModelArgs

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
        if Path(txt).exists():
            text_content = self._read_file_mmap(txt)
        else:
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
    max_samples: Optional[int] = None
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
    """
    # Use the best default tokenizer for your setup
    # tiktoken's gpt2 is fast, well-tested, and has reasonable vocab size (~50k)
    # For multilingual or code, consider "cl100k_base" or "o200k_base"
    tokenizer_name = getattr(args, "tokenizer_name", "gpt2")
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
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