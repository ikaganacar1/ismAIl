"""
Example usage of Turkish Tokenizer in the data pipeline.

This demonstrates how to use the Turkish morphological tokenizer
for training language models on Turkish text.
"""

from data import create_dataloader, TurkishTokenizerWrapper, TURKISH_TOKENIZER_AVAILABLE
from model import ModelArgs

def main():
    """Example of using Turkish tokenizer with the data pipeline"""

    if not TURKISH_TOKENIZER_AVAILABLE:
        print("❌ Turkish tokenizer is not installed!")
        print("Install it with: pip install turkish-tokenizer")
        return

    # Sample Turkish text
    turkish_text = """
    Merhaba! Bu bir Türkçe metin örneğidir.
    İstanbul'da yaşıyorum ve Türkçe dilini öğreniyorum.
    Kitap okumak çok güzeldir ve bilgi verir.
    Türkiye Cumhuriyeti'nin başkenti Ankara'dır.
    Yapay zeka ve makine öğrenmesi teknolojileri gelişiyor.
    """ * 100  # Repeat to have enough text for training

    print("=" * 60)
    print("TURKISH TOKENIZER EXAMPLE")
    print("=" * 60)

    # Test the tokenizer directly
    print("\n1️⃣  Testing Turkish Tokenizer Wrapper")
    tokenizer = TurkishTokenizerWrapper()
    print(f"   Tokenizer: {tokenizer.name}")
    print(f"   Vocabulary size: {tokenizer.n_vocab:,}")

    # Test encoding/decoding
    sample = "Kitapları okuyorum ve öğreniyorum."
    tokens = tokenizer.encode(sample)
    decoded = tokenizer.decode(tokens)

    print(f"\n   Original: {sample}")
    print(f"   Tokens ({len(tokens)}): {tokens[:20]}..." if len(tokens) > 20 else f"   Tokens: {tokens}")
    print(f"   Decoded: {decoded}")

    # Create dataloader with Turkish tokenizer
    print("\n2️⃣  Creating DataLoader with Turkish Tokenizer")
    args = ModelArgs(
        max_seq_len=128,
        max_batch_size=8,
        vocab_size=tokenizer.n_vocab  # Important: set vocab size for model
    )

    dataloader = create_dataloader(
        txt=turkish_text,
        args=args,
        stride=64,  # 50% overlap
        shuffle=True,
        num_workers=0,
        max_samples=50,  # Limit for testing
        use_turkish_tokenizer=True  # Enable Turkish tokenizer
    )

    print(f"\n   ✅ DataLoader created successfully!")
    print(f"   Sequence length: {args.max_seq_len}")
    print(f"   Batch size: {args.max_batch_size}")
    print(f"   Total batches: {len(dataloader)}")
    print(f"   Total samples: {len(dataloader.dataset)}")

    # Test a batch
    print("\n3️⃣  Testing First Batch")
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        print(f"\n   Batch {batch_idx}:")
        print(f"   input_ids shape: {input_ids.shape}")
        print(f"   target_ids shape: {target_ids.shape}")
        print(f"   input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        print(f"   Sample input (first 10 tokens): {input_ids[0, :10].tolist()}")
        print(f"   Decoded sample: {tokenizer.decode(input_ids[0, :30].tolist())}")
        break

    print("\n" + "=" * 60)
    print("✅ Turkish Tokenizer Example Complete!")
    print("=" * 60)

    # Usage tips
    print("\n💡 Usage Tips:")
    print("   • Set vocab_size in ModelArgs to tokenizer.n_vocab")
    print("   • Use use_turkish_tokenizer=True in create_dataloader()")
    print("   • Turkish tokenizer handles morphological analysis automatically")
    print("   • Vocabulary size is optimized for Turkish language")
    print("\n📚 To use in training:")
    print("   tokenizer = TurkishTokenizerWrapper()")
    print("   args = ModelArgs(vocab_size=tokenizer.n_vocab, ...)")
    print("   dataloader = create_dataloader(..., use_turkish_tokenizer=True)")


if __name__ == "__main__":
    main()
