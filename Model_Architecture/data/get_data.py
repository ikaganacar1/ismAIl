#!/usr/bin/env python3
"""
Dataset Download and Preparation Script
Downloads Turkish text data from HuggingFace and prepares it for training
Compatible with the ismail model training pipeline
"""

import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
from itertools import islice
import json

# Configuration
SMALL_DATA = True  # set to False to use the full dataset
DEFAULT_DATA_DIR = Path(__file__).parent  # Save to Model_Architecture/data/
DATASET_NAME = "uonlp/CulturaX"  # HuggingFace dataset
SUBSET = "tr"  # Turkish subset


def download_and_prepare_data(
    data_dir: Path,
    use_small: bool = True,
    parquet_file: str = None,
    full_data_path: str = None,
    train_ratio: float = 0.90,
    seed: int = 2357,
    max_samples: int = None,
):

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("DATASET DOWNLOAD AND PREPARATION")
    print("="*70 + "\n")

    # Load dataset
    if use_small:
        print(f"📥 Loading small dataset...")
        if parquet_file and Path(parquet_file).exists():
            print(f"   Using local file: {parquet_file}")
            dataset = load_dataset('parquet', data_files=parquet_file)
        else:
            # Try to find cached parquet files first
            import os
            cache_dir = Path.home() / ".cache/huggingface/datasets/downloads"

            print(f"   Looking for cached parquet files...")
            parquet_files = []

            if cache_dir.exists():
                # Find all parquet files in cache
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        if file.endswith('.parquet') and 'tr_part' in file:
                            parquet_files.append(os.path.join(root, file))

            # Also check for extracted parquet files in the main cache directory
            culturax_cache = Path.home() / ".cache/huggingface/datasets/uonlp___cultura_x"
            if culturax_cache.exists():
                for root, dirs, files in os.walk(culturax_cache):
                    for file in files:
                        if file.endswith('.parquet'):
                            parquet_files.append(os.path.join(root, file))

            if parquet_files:
                # Use the cached parquet files
                parquet_files = sorted(parquet_files)[:2]  # Use first 2 cached files
                print(f"   ✅ Found {len(parquet_files)} cached parquet file(s)")
                for pf in parquet_files:
                    print(f"      - {Path(pf).name}")

                dataset = load_dataset('parquet', data_files=parquet_files)

                # Limit samples if requested
                if max_samples and max_samples < len(dataset['train']):
                    print(f"   Limiting to {max_samples:,} samples...")
                    dataset['train'] = dataset['train'].select(range(max_samples))
            else:
                # No cached files found, use streaming
                print(f"   No cached files found. Using streaming mode...")
                print(f"   Downloading from HuggingFace: {DATASET_NAME}/{SUBSET}")

                dataset = load_dataset(
                    DATASET_NAME,
                    SUBSET,
                    split="train",
                    streaming=True,
                )

                # Take limited number of samples from stream
                num_samples = max_samples if max_samples else 100_000
                print(f"   Taking {num_samples:,} samples from stream...")

                samples = []
                for sample in tqdm(islice(dataset, num_samples), total=num_samples, desc="Downloading"):
                    samples.append(sample)

                dataset = Dataset.from_list(samples)
                dataset = DatasetDict({"train": dataset})

            print(f"   ✅ Loaded {len(dataset['train']):,} samples")
    else:
        print(f"📥 Loading full dataset from: {full_data_path or 'HuggingFace'}")
        if full_data_path and Path(full_data_path).parent.exists():
            dataset = load_dataset('parquet', data_files=full_data_path)
        else:
            # Download full dataset from HuggingFace
            dataset = load_dataset(DATASET_NAME, SUBSET, split="train")
            dataset = DatasetDict({"train": dataset})

    print(f"✅ Dataset loaded: {len(dataset['train']):,} documents")

    # Remove unnecessary columns
    print(f"\n🔧 Preprocessing dataset...")
    columns_to_remove = ['timestamp', 'url', 'source']
    existing_columns = [col for col in columns_to_remove if col in dataset['train'].column_names]
    if existing_columns:
        dataset = dataset.remove_columns(existing_columns)
        print(f"   Removed columns: {existing_columns}")

    # Print dataset info
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total documents: {len(dataset['train']):,}")
    print(f"   Columns: {dataset['train'].column_names}")
    print(f"   Features: {dataset['train'].features}")

    # Split into train/val
    print(f"\n✂️  Creating train/val split (train ratio: {train_ratio:.2%})...")
    test_size = 1.0 - train_ratio
    split_dataset = dataset['train'].train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop("test")

    print(f"\n📈 Split Statistics:")
    print(f"   Training samples: {len(split_dataset['train']):,}")
    print(f"   Validation samples: {len(split_dataset['val']):,}")
    print(f"   Split ratio: {len(split_dataset['train'])/len(dataset['train']):.2%} train / {len(split_dataset['val'])/len(dataset['train']):.2%} val")

    # Save to text files for training pipeline
    print(f"\n💾 Saving processed data to {data_dir}...")

    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"

    # Save training data
    print(f"   Writing training data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in tqdm(split_dataset['train'], desc="Train"):
            text = example.get('text', '')
            if text.strip():  # Only save non-empty texts
                f.write(text + '\n')

    # Save validation data
    print(f"   Writing validation data to {val_file}...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in tqdm(split_dataset['val'], desc="Val"):
            text = example.get('text', '')
            if text.strip():
                f.write(text + '\n')

    # Save metadata
    metadata = {
        "dataset": DATASET_NAME if not parquet_file else "local_parquet",
        "subset": SUBSET,
        "use_small": use_small,
        "total_documents": len(dataset['train']),
        "train_samples": len(split_dataset['train']),
        "val_samples": len(split_dataset['val']),
        "train_ratio": train_ratio,
        "seed": seed,
        "train_file": str(train_file),
        "val_file": str(val_file),
    }

    metadata_file = data_dir / "dataset_info.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Data preparation complete!")
    print(f"\n📁 Output files:")
    print(f"   Train: {train_file} ({train_file.stat().st_size / 1024**2:.1f} MB)")
    print(f"   Val:   {val_file} ({val_file.stat().st_size / 1024**2:.1f} MB)")
    print(f"   Meta:  {metadata_file}")

    print(f"\n🚀 Ready for training! Use these files in your train.py config:")
    print(f"   train_file: {train_file}")
    print(f"   val_file:   {val_file}")

    return split_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Turkish text dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory to save processed data (default: ./Model_Architecture/data/)"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        default=SMALL_DATA,
        help="Use small dataset (default: True)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (overrides --small)"
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        help="Local parquet file for small dataset (e.g., tr_part_00000.parquet)"
    )
    parser.add_argument(
        "--full_data_path",
        type=str,
        help="Path pattern for full dataset (e.g., /path/to/tr/*.parquet)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Training data ratio (default: 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2357,
        help="Random seed (default: 2357)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # Handle full vs small dataset
    use_small = not args.full if args.full else args.small

    # Adjust train ratio based on dataset size
    train_ratio = args.train_ratio
    if not use_small:
        # For full dataset, use smaller validation set
        train_ratio = 0.999995  # ~0.0005% validation
        print(f"ℹ️  Using full dataset with adjusted train ratio: {train_ratio:.6f}")

    download_and_prepare_data(
        data_dir=Path(args.data_dir),
        use_small=use_small,
        parquet_file=args.parquet_file,
        full_data_path=args.full_data_path,
        train_ratio=train_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )


if __name__ == '__main__':
    main()


