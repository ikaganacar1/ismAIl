# ismail - "Is My AI Lame?"

**ismail** is a from-scratch Turkish language model implementation designed for low-end hardware, built and trained on a single RTX 5070 (12GB). This is my first LLM project, heavily inspired by [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) and built with guidance from [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).

**Language Focus**: ismail is trained exclusively on Turkish datasets using a custom morphology-aware tokenizer optimized for Turkish's agglutinative structure.

> **Status**: Pretraining is currently ongoing on Turkish text with a single 5070 GPU. This will take a while!

## Architecture Highlights

ismail implements several advanced techniques optimized for memory-constrained environments:

- **Multi-Head Latent Attention (MLA)**: DeepSeek-inspired attention mechanism with LoRA-style compression
  - KV cache compression via low-rank projection (kv_lora_rank: 512/256)
  - Separate RoPE and non-RoPE attention heads
  - Reduced memory footprint for longer sequences

- **Mixture of Experts (MoE)**: Efficient sparse expert routing
  - Routed experts: 4-6 experts with top-2 activation
  - Shared experts for common knowledge
  - Sequential expert training for limited VRAM
  - Configurable expert rotation during training

- **YaRN RoPE**: Extended context length support
  - Dynamic frequency scaling based on sequence length
  - Smooth interpolation for position embeddings
  - Support for sequences beyond training length

- **Custom Kernels**: Triton-based GPU kernels for FP8 quantization
  - Optimized matrix multiplication
  - Activation and weight quantization
  - Memory-efficient inference

- **Turkish Morphological Tokenizer**: Custom hybrid tokenizer designed for Turkish
  - Combines rule-based morphological analysis with BPE
  - Preserves linguistic structure (roots, suffixes, phonological rules)
  - Based on research: ["Tokens with Meaning"](https://arxiv.org/abs/2508.14292)
  - 32,768 vocabulary size optimized for Turkish

## Model Configuration

**Current Training Config** (512-dim model for 12GB GPU):
```json
{
  "vocab_size": 32768,
  "dim": 512,
  "n_layers": 16,
  "n_heads": 12,
  "n_routed_experts": 4,
  "n_activated_experts": 2,
  "max_seq_len": 512,
  "kv_lora_rank": 256
}
```

**Full-Scale Config** (1024-dim model):
- 1024 hidden dimensions
- 20 layers (3 dense + 17 MoE)
- 6 routed experts per MoE layer
- Support for 2048+ token sequences

## Project Structure

```
ismail/
├── Model_Architecture/
│   ├── model.py              # Core model implementation
│   ├── train.py              # Training loop with expert rotation
│   ├── generation.py         # Text generation and sampling
│   ├── data.py               # Dataset and data loading
│   ├── kernel.py             # Custom Triton kernels for FP8
│   ├── config.json           # Model and training configuration
│   └── requirements.txt      # Dependencies
├── LiteratureReview/
│   ├── Deepseek-V3/          # DeepSeek architecture analysis
│   ├── GPT-2/                # GPT-2 baseline implementations
│   ├── Llama/                # Llama 3 architecture study
│   ├── Mistral/              # Mistral architecture analysis
│   └── Qwen3/                # Qwen 3 architecture study
└── turkish_tiktokenizer/     # Custom Turkish morphological tokenizer
    ├── app.py                # Gradio demo interface
    └── README.md             # Tokenizer documentation
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (tested on RTX 5070 12GB)
- 16GB+ system RAM recommended

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ismail.git
cd ismail

# Install dependencies
cd Model_Architecture
pip install -r requirements.txt

# Optional: Install W&B for experiment tracking
pip install wandb

# Optional: Install bitsandbytes for 8-bit Adam optimizer
pip install bitsandbytes
```

## Usage

### Training

```bash
cd Model_Architecture

# Train with default config
python train.py

# Train with custom config
python train.py --config config.json

# Resume from checkpoint
python train.py --resume checkpoints/step_10000.pt
```

**Training Features**:
- Gradient accumulation for effective larger batch sizes
- Expert rotation for memory-efficient MoE training
- Mixed precision training (FP32/BF16/FP8)
- Automatic checkpointing
- W&B integration for tracking
- Validation during training

### Generation

```bash
# Generate text
python generation.py --checkpoint checkpoints/latest.pt --prompt "Your prompt here"
```

### Model Configuration

Edit [config.json](Model_Architecture/config.json) to customize:
- Model architecture (dimensions, layers, experts)
- Training hyperparameters (learning rate, batch size)
- Data paths and tokenizer
- Logging and checkpointing

## Turkish Language Support

ismail uses a custom hybrid tokenizer specifically designed for Turkish:

- **Morphological Awareness**: Understands Turkish word structure (roots + suffixes)
- **Efficient Encoding**: 32K vocabulary with ~3.5x compression ratio
- **Linguistic Preservation**: Maintains grammatical information in token boundaries
- **Research-Based**: Implements hybrid approach from [arXiv:2508.14292](https://arxiv.org/abs/2508.14292)

The tokenizer handles Turkish's rich morphology better than standard BPE, preserving linguistic meaning while maintaining vocabulary efficiency. See [turkish_tiktokenizer/README.md](turkish_tiktokenizer/README.md) for details.

## Key Features for Low-End Hardware

1. **Sequential Expert Training**: Train one expert at a time to fit in 12GB VRAM
2. **Gradient Checkpointing**: Trade compute for memory
3. **8-bit Optimizer**: bitsandbytes Adam optimizer reduces memory by ~40%
4. **Small Batch Training**: Gradient accumulation enables large effective batch sizes
5. **FP8 Inference**: Custom kernels for efficient inference
6. **Flexible Configuration**: Easy to scale down for smaller GPUs

## Inspiration & References

This project draws heavily from:

- **[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)**: MLA and MoE architecture
- **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)**: Educational foundation and best practices
- **GPT-2/3**: Transformer baseline architecture
- **Llama 3**: RoPE and normalization techniques

## Technical Details

### Multi-Head Latent Attention (MLA)
The MLA mechanism compresses KV cache using low-rank projections:
- Query: Standard multi-head projection
- Key/Value: Compressed via LoRA-style down/up projection
- Split heads: RoPE-enabled (64d) + Non-RoPE (128d)
- Memory savings: ~4x reduction in KV cache size

### Mixture of Experts (MoE)
- Top-K routing (K=2) with learned router
- Shared experts for common features
- Load balancing loss to prevent expert collapse
- Sequential training mode for VRAM constraints

### YaRN Positional Encoding
- Extends context beyond training length
- Smooth frequency interpolation
- Maintains performance on short sequences
- Configurable extrapolation factors

## Current Status & Roadmap

**Current**:
- ✅ Core architecture implemented
- ✅ Training pipeline functional
- ✅ Custom Turkish morphological tokenizer
- ✅ Turkish dataset preparation
- 🔄 Pretraining on Turkish text with single 5070 (ongoing)

**Planned**:
- [ ] Complete initial pretraining run
- [ ] Evaluation on Turkish benchmarks (TurkishBench, etc.)
- [ ] Fine-tuning pipeline for instruction following
- [ ] Model release (if not too lame!)
- [ ] Multi-GPU training support
- [ ] Inference optimization and quantization

## Performance

Training on RTX 5070 (12GB):
- **512-dim model**: ~3.5 tokens/sec with batch_size=16, grad_accum=8
- **Memory usage**: ~11.5GB during training
- **Estimated pretraining**: Several weeks for 100K steps

*Performance will improve significantly with better hardware!*


## Acknowledgments

Special thanks to:
- [DeepSeek AI](https://github.com/deepseek-ai) for the innovative MLA and MoE architectures
- [Sebastian Raschka](https://github.com/rasbt) for the excellent LLMs-from-scratch educational resource
- The broader open-source LLM community for making this possible

## Contributing

This is primarily a learning project, but suggestions and feedback are welcome! Feel free to open issues or PRs.

## Contact

For questions or discussions, please open an issue on GitHub.

---

*Built with determination and limited VRAM* 🚀
