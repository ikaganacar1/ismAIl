# GPT-2 Model Architecture Analysis

## Model Overview
GPT-2 is a transformer-based decoder-only language model featuring standard Multi-Head Attention, learned absolute positional embeddings, GELU activation, and LayerNorm. This implementation represents the canonical GPT-2 architecture from the original paper, optimized for autoregressive text generation.

---

## Model Configuration

### Core Parameters (GPT-2 124M Configuration)
- **Vocabulary Size**: 50,257 (`vocab_size`)
- **Hidden Dimension**: 768 (`emb_dim`)
- **Number of Layers**: 12 (`n_layers`)
- **Number of Attention Heads**: 12 (`n_heads`)
- **Head Dimension**: 64 (emb_dim / n_heads = 768 / 12)
- **Max Sequence Length**: 1,024 tokens (`context_length`)

### Additional Configuration
- **Dropout Rate**: 0.1 (`drop_rate`) - applied to embeddings, attention, and shortcuts
- **QKV Bias**: False (`qkv_bias`) - no bias terms in attention projections
- **Tokenizer**: BPE (Byte Pair Encoding) via tiktoken ("gpt2" encoding)

---

## Transformer Block Architecture

### Block Structure
**Architecture Type**: Pre-normalization (Pre-LN)

```
Input
  ↓
Residual Connection ──────────
  ↓                           │
LayerNorm (norm1)            │
  ↓                           │
Multi-Head Attention         │
  ↓                           │
Dropout (drop_shortcut)      │
  ↓                           │
Add Residual ←───────────────┘
  ↓
Residual Connection ──────────
  ↓                           │
LayerNorm (norm2)            │
  ↓                           │
Feed-Forward Network         │
  ↓                           │
Dropout (drop_shortcut)      │
  ↓                           │
Add Residual ←───────────────┘
  ↓
Output
```

### Component Order
1. **Attention Normalization** → Multi-Head Attention → Dropout → Residual Add
2. **FFN Normalization** → Feed-Forward Network → Dropout → Residual Add

**Code Implementation** (lines 175-188):
```python
# Attention block
shortcut = x
x = self.norm1(x)
x = self.att(x)
x = self.drop_shortcut(x)
x = x + shortcut

# FFN block
shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = self.drop_shortcut(x)
x = x + shortcut
```

### Residual Connections
- **Type**: Standard additive residual connections
- **Placement**: 
  - After attention (with dropout)
  - After FFN (with dropout)
- **Dropout**: Applied to residual path via `drop_shortcut` (0.1 rate)

---

## Attention Mechanism

### Attention Type
**Multi-Head Attention (MHA)** - Standard full attention

**Key Characteristics** (lines 71-72):
- All heads have independent K, V projections
- No parameter sharing between heads (unlike GQA or MQA)
- Each head: `head_dim = d_out / num_heads` = 768 / 12 = 64

**Configuration**:
- **124M Model**: 12 attention heads
- All heads operate in parallel with full parameters

### QKV Projection Details

**Projection Layers** (lines 77-79):
```python
W_query: Linear(d_in → d_out, bias=qkv_bias)
W_key:   Linear(d_in → d_out, bias=qkv_bias)
W_value: Linear(d_in → d_out, bias=qkv_bias)
out_proj: Linear(d_out → d_out, bias=True)
```

**Dimensions**:
- Input: `d_in` = 768 (emb_dim)
- Output: `d_out` = 768 (same as input)
- Per-head dimension: 64 (d_out / 12)

**Key Features**:
- Configurable bias via `qkv_bias` parameter (default: False)
- Output projection always has bias (line 80)
- All three projections map to same output dimension

### Attention Computation Method

**Implementation**: Standard scaled dot-product attention with causal masking

**Computation Flow**:

1. **Project** queries, keys, values (lines 85-87)
   ```python
   keys = W_key(x)      # (b, num_tokens, d_out)
   queries = W_query(x)  # (b, num_tokens, d_out)
   values = W_value(x)   # (b, num_tokens, d_out)
   ```

2. **Reshape** to separate heads (lines 90-92)
   ```python
   keys = keys.view(b, num_tokens, num_heads, head_dim)
   values = values.view(b, num_tokens, num_heads, head_dim)
   queries = queries.view(b, num_tokens, num_heads, head_dim)
   ```

3. **Transpose** for attention computation (lines 95-97)
   ```python
   keys = keys.transpose(1, 2)     # (b, num_heads, num_tokens, head_dim)
   queries = queries.transpose(1, 2)
   values = values.transpose(1, 2)
   ```

4. **Compute attention scores** (line 100)
   ```python
   attn_scores = queries @ keys.transpose(2, 3)
   # Shape: (b, num_heads, num_tokens, num_tokens)
   ```

5. **Apply causal mask** (lines 103-107)
   ```python
   mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
   attn_scores.masked_fill_(mask_bool, -torch.inf)
   ```

6. **Scale and softmax** (line 109)
   ```python
   attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
   ```
   - Scaling factor: `head_dim^(-0.5)` = `64^(-0.5)` = 0.125

7. **Apply dropout** (line 110)
   ```python
   attn_weights = self.dropout(attn_weights)
   ```

8. **Weighted sum with values** (line 113)
   ```python
   context_vec = (attn_weights @ values).transpose(1, 2)
   # Shape: (b, num_tokens, num_heads, head_dim)
   ```

9. **Combine heads and project** (lines 116-117)
   ```python
   context_vec = context_vec.reshape(b, num_tokens, self.d_out)
   context_vec = self.out_proj(context_vec)
   ```

### Positional Encoding

**Type**: Learned Absolute Positional Embeddings

**Implementation** (lines 196, 205-207):
```python
# Initialization
self.pos_emb = nn.Embedding(context_length, emb_dim)

# Application during forward pass
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
x = tok_embeds + pos_embeds
```

**Key Characteristics**:
- **Learned Parameters**: Position embeddings are trained, not fixed
- **Absolute Positions**: Each position has independent embedding
- **Maximum Length**: Limited to `context_length` (1,024 tokens)
- **Addition**: Position embeddings added to token embeddings before processing

**Comparison to RoPE**:
- Simpler implementation (just lookup + add)
- Less effective for extrapolation to longer sequences
- No relative position information
- Standard in original GPT-2 and earlier transformers

### Attention Masks/Biases

**Mask Type**: Causal mask (upper triangular)

**Mask Creation** (line 82):
```python
self.register_buffer(
    "mask", 
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

**Mask Structure**:
- Upper triangular matrix with 1s above diagonal
- Registered as buffer (not a parameter, but moves with model)
- Size: `(context_length, context_length)` = (1024, 1024)

**Application** (lines 103-107):
```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

**Masking Behavior**:
- Positions with mask value 1 → set to -inf
- Positions with mask value 0 → unchanged
- After softmax, -inf becomes 0 probability
- Ensures token i can only attend to positions ≤ i

**No Additional Biases**: Model does not use learnable attention biases

---

## Feed-Forward Network

### Architecture
**Type**: Standard FFN with GELU activation

**Structure** (lines 143-148):
```
Input (emb_dim=768)
  ↓
Linear (768 → 3072)
  ↓
GELU
  ↓
Linear (3072 → 768)
  ↓
Output (emb_dim=768)
```

**Implementation**:
```python
self.layers = nn.Sequential(
    nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
    GELU(),
    nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
)
```

### Hidden Dimension Multiplier

**Dimensions**:
- Input/Output: `emb_dim` = 768
- Intermediate: `4 * emb_dim` = 3,072
- **Multiplier**: **4.0×**

This is the standard FFN expansion used in GPT-2 and many transformer models.

### Activation Function
**GELU (Gaussian Error Linear Unit)** - Custom Implementation

**Formula** (lines 134-139):
```python
def forward(x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
    ))
```

**Mathematical Approximation**:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Characteristics**:
- **Smooth Activation**: Differentiable everywhere
- **Non-Monotonic**: Has slight negative values for x < 0
- **Approximation**: Uses tanh-based approximation instead of exact erf function
- **Better than ReLU**: Smoother gradients, proven effective in transformers

**Why This Approximation?**:
- Exact GELU uses error function (erf), which is slower
- Tanh approximation is faster and nearly identical
- Standard in GPT-2 implementation
- Computational efficiency without quality loss

### Gating Mechanism
**None** - Standard feed-forward without gating

**Properties**:
- No SwiGLU or GeGLU gating
- Simple two-layer MLP
- All parameters always active
- Deterministic computation

---

## Normalization Layers

### Type
**LayerNorm (Layer Normalization)**

**Implementation** (lines 124-132):
```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
```

**Mathematical Formula**:
```
LayerNorm(x) = γ * ((x - μ) / σ) + β

where:
  μ = mean(x) along feature dimension
  σ = sqrt(var(x) + ε)
  γ = learnable scale parameter (self.scale)
  β = learnable shift parameter (self.shift)
  ε = 1e-5 (numerical stability)
```

**Key Characteristics**:
- **Mean Centering**: Subtracts mean (unlike RMSNorm)
- **Learnable Scale and Shift**: Both γ (scale) and β (shift) are trained
- **Epsilon**: 1e-5 for numerical stability
- **Unbiased Variance**: Uses `unbiased=False` for consistency with original GPT-2

### Placement
**Pre-normalization (Pre-LN) Architecture**:
- `norm1`: Before attention in each transformer block (line 169)
- `norm2`: Before FFN in each transformer block (line 170)
- `final_norm`: After all transformer blocks, before output head (line 201)

**Order in Forward Pass**:
```python
# In TransformerBlock (lines 175-188)
x = norm1(x)      # Pre-attention normalization
x = att(x)        # Attention computation
x = x + shortcut  # Residual

x = norm2(x)      # Pre-FFN normalization
x = ff(x)         # FFN computation
x = x + shortcut  # Residual

# In GPTModel (lines 209-210)
x = final_norm(x)      # Final normalization
logits = out_head(x)   # Output projection
```

### Parameters
- **Scale (γ)**: Learnable parameter (shape: [emb_dim], initialized to 1.0)
- **Shift (β)**: Learnable parameter (shape: [emb_dim], initialized to 0.0)
- **Epsilon**: 1e-5 (for numerical stability in sqrt)
- **Has both scale and shift** (unlike RMSNorm which typically only has scale)

**Benefits of Pre-normalization**:
- Improved gradient flow during training
- Better training stability for deep models
- Reduces need for learning rate warmup
- Standard in modern transformers (though GPT-2 originally used post-norm)

**Note**: The original GPT-2 paper used **post-normalization** (norm after attention/FFN), but this implementation uses **pre-normalization** which is now considered best practice.

---

## Notable Features

### 1. Learned Absolute Positional Embeddings
**Classic Positional Encoding Approach**

**Implementation** (lines 196, 205-207):
```python
self.pos_emb = nn.Embedding(context_length, emb_dim)
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
x = tok_embeds + pos_embeds
```

**Characteristics**:
- **Learned**: Position embeddings are trained parameters
- **Absolute**: Each position has independent embedding (no relative information)
- **Fixed Length**: Cannot extrapolate beyond `context_length` (1,024 tokens)
- **Simple**: Just lookup and add to token embeddings

**Comparison to Modern Approaches**:
- **vs RoPE**: RoPE encodes relative positions, can extrapolate better
- **vs ALiBi**: ALiBi uses attention biases, no position embeddings
- **Historical**: Standard in GPT-2, BERT; replaced by RoPE in modern LLMs

**Limitations**:
- Poor extrapolation to longer sequences
- No explicit relative position information
- Memory overhead (stores 1024 × 768 parameters)

### 2. Custom GELU Implementation
**Tanh-based Approximation**

**Formula** (lines 136-139):
```python
0.5 * x * (1 + torch.tanh(
    torch.sqrt(torch.tensor(2.0 / torch.pi)) *
    (x + 0.044715 * torch.pow(x, 3))
))
```

**Why Custom Implementation?**:
- **Historical**: PyTorch didn't have native GELU when GPT-2 was released
- **Efficiency**: Tanh approximation faster than erf-based exact GELU
- **Accuracy**: Very close to exact GELU (< 0.1% max error)
- **Educational**: Shows mathematical approximation technique

**Modern Alternative**:
```python
torch.nn.functional.gelu(x)  # Native PyTorch implementation
```

### 3. Standard Multi-Head Attention
**Full MHA Without Optimization**

**Characteristics**:
- Each head has full K, V projections (12 × 2 = 24 projection matrices)
- No parameter sharing (unlike GQA with 8 KV groups)
- No KV cache optimization in this implementation
- Suitable for smaller models (124M parameters)

**Memory Profile**:
- KV Cache (if implemented): `2 × seq_len × n_heads × head_dim`
- For 1024 tokens: `2 × 1024 × 12 × 64 × 2 bytes` ≈ 3 MB per layer
- Total: 3 MB × 12 layers ≈ 36 MB

**Why Not GQA?**:
- GPT-2 predates GQA innovation
- 12 heads manageable for 124M model
- Full MHA provides maximum expressiveness

### 4. Pre-Normalization Architecture
**Modern Best Practice**

**Implementation** (lines 175-188):
```python
# Norm → Attention → Add (not Attention → Add → Norm)
x = self.norm1(x)
x = self.att(x)
x = x + shortcut
```

**Advantages**:
- **Gradient Flow**: Better gradient propagation to early layers
- **Training Stability**: Prevents gradient explosion
- **No Warmup**: Can train with full learning rate from start
- **Convergence**: Faster and more stable training

**Historical Note**:
- Original GPT-2 used **post-normalization** (norm after residual)
- This implementation modernizes to **pre-normalization**
- Pre-norm now standard in LLaMA, Mistral, GPT-3, etc.

### 5. Dropout on Residual Paths
**Additional Regularization**

**Implementation** (lines 171, 181-182):
```python
self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

x = self.att(x)
x = self.drop_shortcut(x)  # Dropout before adding residual
x = x + shortcut
```

**Purpose**:
- Regularizes residual connections
- Prevents over-reliance on skip connections
- Standard in original GPT-2
- Rate: 0.1 (10% dropout)

**Also Applied**:
- Embedding dropout (line 197): After token + position embeddings
- Attention dropout (line 110): On attention weights

### 6. Causal Masking with Upper Triangular
**Efficient Autoregressive Masking**

**Implementation** (line 82, 103-107):
```python
# Pre-computed mask (registered as buffer)
torch.triu(torch.ones(context_length, context_length), diagonal=1)

# Application
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

**Efficiency**:
- Mask computed once during initialization
- Registered as buffer (moves with model to CPU/GPU)
- Dynamic slicing for variable sequence lengths
- -inf ensures zero attention after softmax

### 7. Simple Text Generation
**Greedy Decoding Implementation**

**Implementation** (lines 213-235):
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Crop to context size
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # Last token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Features**:
- **Greedy Decoding**: Always selects highest probability token
- **Context Window**: Crops to last `context_size` tokens
- **No Sampling**: No temperature, top-k, or nucleus sampling
- **No KV Cache**: Recomputes attention for full sequence each step

**Limitations**:
- Inefficient (O(n²) for n generated tokens)
- Deterministic output (no randomness)
- No advanced decoding strategies

**Production Alternative** would include:
- KV caching for O(n) generation
- Temperature sampling for diversity
- Top-k/top-p for quality control
- Beam search for better sequences

### 8. No Bias in QKV Projections
**Simplified Attention**

**Configuration** (line 77-79, default `qkv_bias=False`):
```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
```

**Rationale**:
- **Fewer Parameters**: Eliminates 3 × d_out parameters
- **Pre-Normalization**: Normalization reduces need for biases
- **Empirical**: No quality loss observed in practice
- **Modern Standard**: Most recent LLMs omit QKV biases

**But Note**: Output projection `out_proj` still has bias (line 80)

### 9. Educational Implementation
**Clean, Readable Code**

**Design Principles**:
- From "Build a Large Language Model From Scratch" book
- Explicit implementations (no complex abstractions)
- Clear variable names and structure
- Educational comments and documentation
- No optimization tricks that obscure logic

**Trade-offs**:
- **Clarity over Speed**: Readable but not maximally optimized
- **Simplicity over Features**: No KV cache, advanced sampling, etc.
- **Learning Focus**: Shows transformer fundamentals clearly

**Production Gaps**:
- No KV caching for efficient generation
- No FlashAttention or optimized kernels
- No advanced sampling strategies
- No quantization support
- No distributed training utilities

---

## Optimizations

### 1. Memory Optimizations
- **Dropout Regularization**: 0.1 rate prevents overfitting, allows smaller models
- **Shared Embedding-Output Weights**: Code supports weight tying (though not shown in config)
- **Buffer Registration**: Causal mask stored as buffer, not parameter
- **No Bias in QKV**: Reduces parameters by ~3 × 768 = 2,304 per layer

### 2. Computational Optimizations
- **Pre-computed Mask**: Causal mask created once, reused for all batches
- **View Operations**: Reshaping uses `.view()` which doesn't copy data
- **Transpose Instead of Reshape**: Efficient dimension reordering
- **Sequential FFN**: Single `nn.Sequential` for efficient execution

### 3. Numerical Stability
- **Epsilon in LayerNorm**: 1e-5 prevents division by zero
- **Masked Fill with -inf**: Clean masking without numerical issues
- **Attention Scaling**: `head_dim^(-0.5)` prevents vanishing/exploding attention scores
- **Variance Computation**: `unbiased=False` for consistency

### 4. Inference Optimizations (Basic)
- **Context Window Cropping**: `idx_cond = idx[:, -context_size:]` in generation
- **No Grad**: Uses `torch.no_grad()` during generation
- **Eval Mode**: `model.eval()` disables dropout

### 5. Missing Optimizations (vs Modern LLMs)
- **No KV Caching**: Recomputes attention for full sequence
- **No FlashAttention**: Standard PyTorch attention implementation
- **No Mixed Precision**: No automatic mixed precision or bfloat16
- **No Quantization**: Full precision weights
- **No Distributed**: Single-GPU implementation

---

## Comparison to Modern Architectures

### vs. LLaMA 3
**Similarities**:
- Pre-normalization architecture
- Similar residual structure
- Standard transformer decoder

**Key Differences**:
- **Positional Encoding**: GPT-2 uses learned absolute; LLaMA 3 uses RoPE with advanced scaling
- **Attention**: GPT-2 uses MHA; LLaMA 3 uses GQA (3-4× KV cache reduction)
- **Normalization**: GPT-2 uses LayerNorm; LLaMA 3 uses RMSNorm
- **Activation**: GPT-2 uses GELU; LLaMA 3 uses SwiGLU
- **Context**: GPT-2 supports 1K; LLaMA 3 supports 128K
- **Precision**: GPT-2 is float32; LLaMA 3 is bfloat16 native

### vs. Mistral
**Similarities**:
- Pre-normalization
- Transformer decoder architecture
- Autoregressive masking

**Key Differences**:
- **Positional Encoding**: GPT-2 learned absolute; Mistral uses RoPE
- **Attention**: GPT-2 MHA; Mistral GQA with optional sliding window
- **Normalization**: GPT-2 LayerNorm; Mistral RMSNorm
- **Activation**: GPT-2 GELU; Mistral SwiGLU
- **Backend**: GPT-2 basic; Mistral supports FlashAttention, SDPA

### vs. GPT-3/GPT-4 (Conceptual)
**GPT-2 is the Foundation**:
- GPT-3 and GPT-4 build on GPT-2 architecture
- Main differences: scale (175B vs 124M), training data, optimizations
- Architecture largely similar: MHA, learned positions, GELU, LayerNorm

**GPT-3+ Improvements**:
- Larger models with more layers/heads
- Better training techniques (mixed precision, etc.)
- More sophisticated generation algorithms
- Likely optimizations (sparse attention, etc.)

### vs. Qwen3
**Similarities**:
- Pre-normalization
- Transformer decoder
- Similar block structure

**Key Differences**:
- **Positional Encoding**: GPT-2 learned; Qwen3 uses RoPE
- **Attention**: GPT-2 MHA; Qwen3 GQA with optional MoE
- **Normalization**: GPT-2 LayerNorm; Qwen3 RMSNorm
- **Activation**: GPT-2 GELU; Qwen3 SwiGLU
- **Flexibility**: Qwen3 supports MoE; GPT-2 is always dense

### vs. DeepSeek V3
**Massive Differences**:
- **Attention**: GPT-2 basic MHA; DeepSeek V3 uses MLA with 85% cache reduction
- **FFN**: GPT-2 dense; DeepSeek V3 uses hybrid dense-MoE
- **Positional Encoding**: GPT-2 learned; DeepSeek V3 uses YaRN
- **Quantization**: GPT-2 none; DeepSeek V3 has FP8 support
- **Scale**: GPT-2 is 124M; DeepSeek V3 designed for massive scale

---

## Historical Context

### GPT-2's Place in History
**Released**: February 2019 by OpenAI

**Significance**:
- **Scaling Demonstration**: Showed larger models (1.5B) perform much better
- **Zero-Shot Learning**: Demonstrated emergent capabilities without fine-tuning
- **Controversy**: Initially withheld due to "safety concerns"
- **Foundation**: Basis for GPT-3 and modern decoder-only LLMs

### Architectural Evolution
**Pre-GPT-2 (2017-2018)**:
- Original Transformer (2017): Post-norm, encoder-decoder
- GPT-1 (2018): First decoder-only GPT, smaller scale

**GPT-2 Era (2019)**:
- Introduced larger scale (up to 1.5B parameters)
- Proved decoder-only models sufficient for many tasks
- Demonstrated scaling laws

**Post-GPT-2 (2020-present)**:
- GPT-3 (2020): Scaled to 175B, few-shot learning
- Switch Transformer (2021): Introduced MoE at scale
- LLaMA (2023): Open-source, efficient architecture (GQA, RoPE, RMSNorm)
- Modern LLMs: Incorporate efficiency innovations (GQA, RoPE, SwiGLU, etc.)

### Why This Implementation?
**Educational Value**:
- Canonical GPT-2 architecture
- Clean, readable code
- Shows transformer fundamentals
- Modernized with pre-norm (improvement over original)
- Suitable for learning and experimentation

**Production Note**:
- Real GPT-2 implementations include many optimizations
- This version prioritizes clarity over performance
- Modern production code uses: KV cache, FlashAttention, quantization, etc.

---

## Summary

**GPT-2** represents the foundational decoder-only transformer architecture that influenced all modern LLMs:

### Core Architecture
- **Attention**: Standard Multi-Head Attention (12 heads)
- **Positional Encoding**: Learned absolute positional embeddings
- **Feed-Forward**: Standard FFN with 4× expansion and GELU
- **Normalization**: LayerNorm with pre-normalization (modernized)
- **Activation**: Custom GELU implementation (tanh approximation)

### Key Characteristics
- **Size**: 124M parameters (smallest GPT-2 variant)
- **Context**: 1,024 tokens
- **Vocabulary**: 50,257 tokens (BPE encoding)
- **Architecture**: 12 layers, 768 hidden dim, 12 attention heads
- **Simplicity**: No complex features (MoE, GQA, sliding window, etc.)

### Historical Importance
- **Foundation**: Basis for GPT-3, modern decoder-only LLMs
- **Scaling**: Demonstrated importance of model size
- **Architecture**: Proved decoder-only sufficient for language understanding
- **Influence**: Inspired LLaMA, Mistral, and other open-source models

### Modern Perspective
**What GPT-2 Got Right**:
- Decoder-only architecture (still standard)
- Pre-training + fine-tuning paradigm
- Autoregressive generation
- Basic transformer components

**What Modern LLMs Improved**:
- **Positional Encoding**: RoPE instead of learned (better extrapolation)
- **Attention**: GQA/MLA instead of MHA (memory efficiency)
- **Normalization**: RMSNorm instead of LayerNorm (stability, speed)
- **Activation**: SwiGLU instead of GELU (quality)
- **Context**: 128K+ instead of 1K (long-range understanding)
- **Efficiency**: FlashAttention, KV cache, quantization, etc.

### Educational Value
This implementation excels as a learning resource:
- **Clear Code**: Easy to understand transformer mechanics
- **Complete**: Full model from embeddings to generation
- **Documented**: From "Build a Large Language Model From Scratch" book
- **Extensible**: Good starting point for experiments
- **Historical**: Shows evolution from GPT-2 to modern architectures

**For Production**: Modern deployments should incorporate: KV caching, FlashAttention, GQA, RoPE, RMSNorm, SwiGLU, mixed precision, quantization, and distributed training support.

GPT-2 remains an excellent starting point for understanding transformer architectures, even as modern models have advanced significantly in efficiency and capability.