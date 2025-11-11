# Qwen3 Model Architecture Analysis

## Model Overview
Qwen3 is a transformer-based language model featuring Grouped Query Attention (GQA), RoPE positional encoding, and optional Mixture of Experts (MoE) layers. The model uses a pre-normalization architecture with RMSNorm and supports efficient inference through KV caching.

---

## Model Configuration

### Core Parameters
- **Vocabulary Size**: `cfg["vocab_size"]` - Size of the token vocabulary
- **Hidden Dimension**: `cfg["emb_dim"]` - Main embedding/hidden dimension
- **Number of Layers**: `cfg["n_layers"]` - Number of transformer blocks
- **Number of Attention Heads**: `cfg["n_heads"]` - Total number of query heads
- **Number of KV Groups**: `cfg["n_kv_groups"]` - Number of key-value head groups for GQA
- **Head Dimension**: `cfg["head_dim"]` - Dimension per attention head (optional, defaults to `emb_dim / n_heads`)
- **Max Sequence Length**: `cfg["context_length"]` - Maximum sequence length for RoPE computation

### Feed-Forward Network Parameters
- **Hidden Dimension (Dense)**: `cfg["hidden_dim"]` - Intermediate dimension for standard FFN
- **MoE Hidden Dimension**: `cfg["moe_hidden_dim"]` - Intermediate dimension for MoE experts (if MoE enabled)
- **Number of Experts**: `cfg["num_experts"]` - Number of expert networks (0 for standard FFN, >0 for MoE)
- **Experts Per Token**: `cfg["num_experts_per_tok"]` - Number of experts activated per token (top-k routing)

### Additional Configuration
- **Data Type**: `cfg["dtype"]` - Model dtype (e.g., torch.bfloat16, torch.float32)
- **RoPE Base**: `cfg["rope_base"]` - Base frequency for RoPE (default: 10,000)
- **QK Normalization**: `cfg["qk_norm"]` - Whether to apply RMSNorm to queries and keys

---

## Transformer Block Architecture

### Block Structure
**Architecture Type**: Pre-normalization (Pre-LN)

```
Input
  ↓
Residual Connection ─────────┐
  ↓                           │
RMSNorm (norm1)              │
  ↓                           │
Grouped Query Attention      │
  ↓                           │
Add Residual ←───────────────┘
  ↓
Residual Connection ─────────┐
  ↓                           │
RMSNorm (norm2)              │
  ↓                           │
FFN (FeedForward or MoE)     │
  ↓                           │
Add Residual ←───────────────┘
  ↓
Output
```

### Component Order
1. **Attention Normalization** → Grouped Query Attention → Residual Add
2. **FFN Normalization** → Feed-Forward Network → Residual Add

### Residual Connections
- **Type**: Standard additive residual connections
- **Placement**: 
  - After attention (pre-normalized input)
  - After FFN (pre-normalized input)

**Code Implementation**:
```python
# Attention block
shortcut = x
x = self.norm1(x)
x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
x = x + shortcut

# FFN block
shortcut = x
x = self.norm2(x)
x = self.ff(x)
x = x + shortcut
```

---

## Attention Mechanism

### Attention Type
**Grouped Query Attention (GQA)**

**Key Characteristics**:
- Multiple query heads share the same key-value heads
- Reduces KV cache size compared to Multi-Head Attention (MHA)
- More efficient than MHA, more expressive than Multi-Query Attention (MQA)
- `group_size = num_heads / num_kv_groups`

**Memory Efficiency**:
- Standard MHA: Stores `num_heads` copies of K and V
- GQA: Stores `num_kv_groups` copies of K and V
- Keys and values are repeated using `repeat_interleave(group_size)` to match query heads

### QKV Projection Details

**Projection Layers**:
```python
W_query:  Linear(d_in → num_heads × head_dim, bias=False)
W_key:    Linear(d_in → num_kv_groups × head_dim, bias=False)
W_value:  Linear(d_in → num_kv_groups × head_dim, bias=False)
out_proj: Linear(num_heads × head_dim → d_in, bias=False)
```

**Head Dimension Calculation**:
```python
if head_dim is None:
    head_dim = d_in / num_heads
# Otherwise, head_dim is explicitly specified
```

**Output Dimension**: `d_out = num_heads × head_dim`

### Attention Computation Method

**Implementation**: Standard scaled dot-product attention with GQA

**Computation Flow**:
1. **Project** queries, keys, values
   ```python
   queries = W_query(x)  # (batch, num_tokens, num_heads × head_dim)
   keys = W_key(x)       # (batch, num_tokens, num_kv_groups × head_dim)
   values = W_value(x)   # (batch, num_tokens, num_kv_groups × head_dim)
   ```

2. **Reshape** to separate heads
   ```python
   queries = queries.view(b, num_tokens, num_heads, head_dim).transpose(1, 2)
   keys = keys.view(b, num_tokens, num_kv_groups, head_dim).transpose(1, 2)
   values = values.view(b, num_tokens, num_kv_groups, head_dim).transpose(1, 2)
   ```

3. **Optional QK Normalization** (if enabled)
   ```python
   if qk_norm:
       queries = q_norm(queries)  # RMSNorm per head
       keys = k_norm(keys)        # RMSNorm per head
   ```

4. **Apply RoPE** to queries and keys
   ```python
   queries = apply_rope(queries, cos, sin, offset=start_pos)
   keys = apply_rope(keys, cos, sin, offset=start_pos)
   ```

5. **Update KV Cache** (if provided)
   ```python
   if cache is not None:
       prev_k, prev_v = cache
       keys = torch.cat([prev_k, keys_new], dim=2)
       values = torch.cat([prev_v, values_new], dim=2)
   ```

6. **Repeat KV for GQA**: Expand keys and values to match query head count
   ```python
   keys = keys.repeat_interleave(group_size, dim=1)
   values = values.repeat_interleave(group_size, dim=1)
   ```

7. **Compute attention scores**
   ```python
   attn_scores = queries @ keys.transpose(2, 3)
   ```

8. **Apply causal mask**
   ```python
   attn_scores = attn_scores.masked_fill(mask, -torch.inf)
   ```

9. **Softmax with scaling**
   ```python
   attn_weights = torch.softmax(attn_scores / (head_dim ** 0.5), dim=-1)
   ```

10. **Weighted sum with values**
    ```python
    context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, d_out)
    ```

11. **Output projection**
    ```python
    output = out_proj(context)
    ```

### Positional Encoding

**Type**: Rotary Position Embeddings (RoPE)

**Precomputation**: `compute_rope_params(head_dim, theta_base, context_length)`

**Implementation**:
```python
# Compute inverse frequencies
inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2) / head_dim))

# Generate position indices
positions = torch.arange(context_length)

# Compute angles
angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (context_length, head_dim // 2)

# Expand to full head dimension
angles = torch.cat([angles, angles], dim=1)  # (context_length, head_dim)

# Precompute sine and cosine
cos = torch.cos(angles)
sin = torch.sin(angles)
```

**Application**: `apply_rope(x, cos, sin, offset=0)`

**Rotation Method**:
```python
# Split into first and second halves
x1 = x[..., :head_dim // 2]  # First half
x2 = x[..., head_dim // 2:]  # Second half

# Adjust cos and sin for sequence position
cos = cos[offset:offset + seq_len, :]  # (1, 1, seq_len, head_dim)
sin = sin[offset:offset + seq_len, :]

# Apply rotation: (x1, x2) → (x1 × cos - x2 × sin, x2 × cos + x1 × sin)
rotated = torch.cat((-x2, x1), dim=-1)
x_rotated = (x * cos) + (rotated * sin)
```

**Key Features**:
- Position information encoded through rotation in complex space
- Relative positional relationships naturally preserved
- No learned parameters (fully deterministic)
- Works well with KV caching (offset parameter tracks position)

### Attention Masks/Biases

**Mask Type**: Causal mask (autoregressive masking)

**Mask Generation**:
```python
# For prefill (full sequence)
mask = torch.triu(
    torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), 
    diagonal=1
)

# For generation with KV cache (partial sequence)
mask = torch.triu(
    torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), 
    diagonal=1
)[pos_start:pos_end, :pos_end]

# Expand for batch and heads: (1, 1, num_tokens, num_tokens)
mask = mask[None, None, :, :]
```

**Application**: 
- Mask is applied by setting masked positions to `-inf` before softmax
- Ensures tokens can only attend to previous positions (autoregressive property)
- Efficiently handles both prefill and generation phases

**No Additional Biases**: Model does not use learned attention biases

---

## Feed-Forward Network

### Architecture

**Type 1: Standard Feed-Forward** (when `num_experts == 0`):
```
x → fc1 → SiLU → ⊙ ← fc2 ← x
              ↓
            fc3 → output
```

**Type 2: Mixture of Experts (MoE)** (when `num_experts > 0`):
```
       ┌→ Gate (Router) → Top-K Expert Selection
       │
x ─────┼→ Expert 1 (if selected) ──┐
       │                            │
       ├→ Expert 2 (if selected) ──┤
       │           ...              ├→ Weighted Sum → output
       ├→ Expert N (if selected) ──┤
       │                            │
       └→ (no shared experts)      ─┘
```

### Standard FFN Structure

**Implementation**:
```python
class FeedForward:
    fc1: Linear(emb_dim → hidden_dim, bias=False)
    fc2: Linear(emb_dim → hidden_dim, bias=False)
    fc3: Linear(hidden_dim → emb_dim, bias=False)
    
    def forward(x):
        x_fc1 = fc1(x)
        x_fc2 = fc2(x)
        x = silu(x_fc1) * x_fc2
        return fc3(x)
```

**Dimensions**:
- Input/Output: `emb_dim`
- Intermediate: `hidden_dim`
- **Hidden Dimension Multiplier**: `hidden_dim / emb_dim` (typically 4x to 8x)

### MoE FFN Structure

**Router (Gate)**:
```python
gate: Linear(emb_dim → num_experts, bias=False)

# Routing computation
scores = gate(x)  # (batch, seq_len, num_experts)
topk_scores, topk_indices = torch.topk(scores, num_experts_per_tok, dim=-1)
topk_probs = torch.softmax(topk_scores, dim=-1)  # Normalize only selected experts
```

**Individual Expert**:
```python
class Expert:
    fc1: Linear(emb_dim → moe_hidden_dim, bias=False)
    fc2: Linear(emb_dim → moe_hidden_dim, bias=False)
    fc3: Linear(moe_hidden_dim → emb_dim, bias=False)
    
    def forward(x):
        hidden = silu(fc1(x)) * fc2(x)
        return fc3(hidden)
```

**MoE Forward Pass**:
```python
def MoEFeedForward.forward(x):
    # 1. Compute routing
    scores = gate(x)
    topk_scores, topk_indices = torch.topk(scores, num_experts_per_tok, dim=-1)
    topk_probs = torch.softmax(topk_scores, dim=-1)
    
    # 2. Flatten for efficient processing
    x_flat = x.reshape(batch * seq_len, -1)
    out_flat = torch.zeros_like(x_flat)
    
    # 3. Process each expert
    for expert_id in unique_experts:
        # Select tokens routed to this expert
        mask = (topk_indices == expert_id)
        token_mask = mask.any(dim=-1)
        selected_idx = token_mask.nonzero().squeeze(-1)
        
        # Compute expert output
        expert_input = x_flat[selected_idx]
        hidden = silu(fc1[expert_id](expert_input)) * fc2[expert_id](expert_input)
        expert_out = fc3[expert_id](hidden)
        
        # Get routing weights for selected tokens
        mask_selected = mask[selected_idx]
        slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
        selected_probs = topk_probs[selected_idx].gather(dim=-1, index=slot_indices)
        
        # Accumulate weighted outputs
        out_flat.index_add_(0, selected_idx, expert_out * selected_probs.unsqueeze(-1))
    
    return out_flat.reshape(batch, seq_len, emb_dim)
```

### Activation Function
**SwiGLU** (Swish-Gated Linear Unit):
```
f(x) = SiLU(fc1(x)) ⊙ fc2(x)
     = [x × sigmoid(x)] ⊙ fc2(x)

where SiLU(x) = x × sigmoid(x)
      ⊙ = element-wise multiplication
```

**Characteristics**:
- Combines smooth activation (SiLU/Swish) with gating mechanism
- Two separate projections (fc1 for activation, fc2 for gating)
- Superior performance compared to ReLU or GELU in language models
- No learned gating parameters (unlike some variants)

### Gating Mechanism

**Standard FFN**: 
- Fixed gating through SwiGLU activation
- No learned routing

**MoE Gating**: 
- Learnable router via linear projection
- Top-K sparse routing (selects `num_experts_per_tok` experts)
- Softmax normalization over selected experts only
- Token-level routing (each token can route to different experts)

**Routing Properties**:
- **Sparse**: Only k out of N experts activated per token
- **Differentiable**: Routing weights are trained end-to-end
- **Load-balancing**: Natural load balancing through top-k selection
- **No shared experts**: Unlike some MoE designs, no experts are always active

---

## Normalization Layers

### Type
**RMSNorm (Root Mean Square Layer Normalization)**

**Implementation**:
```python
class RMSNorm:
    def __init__(emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        self.scale = Parameter(torch.ones(emb_dim))
        self.shift = Parameter(torch.zeros(emb_dim)) if bias else None
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
    
    def forward(x):
        input_dtype = x.dtype
        
        # Convert to float32 for numerical stability (if Qwen3 compatible)
        if qwen3_compatible:
            x = x.to(torch.float32)
        
        # Compute RMS
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + eps)
        
        # Apply scale
        norm_x = norm_x * self.scale
        
        # Optional bias/shift
        if self.shift is not None:
            norm_x = norm_x + self.shift
        
        # Convert back to input dtype
        return norm_x.to(input_dtype)
```

**Mathematical Formula**:
```
RMSNorm(x) = (x / RMS(x)) × γ + β

where:
  RMS(x) = sqrt(mean(x²) + ε)
  γ = learnable scale parameter (self.scale)
  β = optional learnable bias parameter (self.shift)
  ε = 1e-6 (numerical stability)
```

**Key Characteristics**:
- Simpler than LayerNorm (no mean centering by default)
- Optional bias term (disabled by default)
- Qwen3 compatibility mode forces float32 computation for stability
- More efficient than LayerNorm while maintaining effectiveness

### Placement
**Pre-normalization (Pre-LN) Architecture**:
- `norm1`: Before attention in each transformer block
- `norm2`: Before FFN in each transformer block
- `final_norm`: After all transformer blocks, before output projection
- `q_norm`, `k_norm`: Optional normalization inside attention (when `qk_norm=True`)

**Order in Forward Pass**:
```python
# In TransformerBlock
x = norm1(x)      # Pre-attention normalization
x = attention(x)  # Attention computation
x = x + shortcut  # Residual

x = norm2(x)      # Pre-FFN normalization
x = ffn(x)        # FFN computation
x = x + shortcut  # Residual

# In Qwen3Model
x = final_norm(x)  # Final normalization before output head
logits = out_head(x)
```

### Parameters
- **Scale (weight)**: Learnable parameter (shape: [emb_dim], initialized to 1.0)
- **Shift (bias)**: Optional learnable parameter (shape: [emb_dim], initialized to 0.0, disabled by default)
- **Epsilon**: 1e-6 (for numerical stability in rsqrt computation)
- **No mean subtraction** (unlike standard LayerNorm)

---

## Notable Features

### 1. Grouped Query Attention (GQA)
**Efficient Attention Mechanism**

**Advantages**:
- **Reduced KV Cache**: Stores only `num_kv_groups` instead of `num_heads` key-value pairs
- **Memory Efficiency**: Significant reduction in memory footprint during inference
- **Flexibility**: `group_size = num_heads / num_kv_groups` allows tuning memory-quality tradeoff
- **Quality Retention**: Maintains model quality better than Multi-Query Attention (MQA)

**Implementation Details**:
- Query heads: `num_heads` (full count)
- Key-Value heads: `num_kv_groups` (reduced count)
- Keys and values are expanded via `repeat_interleave(group_size, dim=1)` during attention
- Typical configuration: 8-16 query heads with 2-4 KV groups (2x-8x cache reduction)

### 2. Rotary Position Embeddings (RoPE)
**Superior Positional Encoding**

**Key Benefits**:
- **Relative Positioning**: Naturally encodes relative positions between tokens
- **No Learned Parameters**: Fully deterministic based on position and frequencies
- **Extrapolation**: Can generalize to longer sequences than seen during training
- **Efficient**: No additional memory overhead during inference

**Technical Details**:
- Uses complex number rotation in embedding space
- Precomputes cos/sin values for all positions up to `context_length`
- Applied separately to queries and keys (not values)
- Offset parameter enables efficient KV caching by tracking position
- Base frequency (theta_base) controls the rotation speed (default: 10,000)

### 3. Optional QK Normalization
**Attention Stability Enhancement**

**Feature**:
```python
if qk_norm:
    queries = q_norm(queries)  # RMSNorm on queries
    keys = k_norm(keys)        # RMSNorm on keys
```

**Purpose**:
- Stabilizes attention computation, especially in very deep models
- Normalizes query and key magnitudes before dot product
- Can help prevent attention saturation or vanishing
- Inspired by research showing benefits for training stability

### 4. Mixture of Experts (MoE) Support
**Conditional Computation for Scaling**

**Architecture**:
- Optional: Enabled when `num_experts > 0`
- Top-K routing: Activates `num_experts_per_tok` experts per token (typically 2)
- Sparse activation: Only selected experts compute for each token
- Per-expert SwiGLU gating: Each expert is a full FFN with gating

**Advantages**:
- **Increased Capacity**: More parameters without proportional compute increase
- **Specialization**: Experts can specialize for different input types
- **Scalability**: Allows models to scale to larger parameter counts efficiently
- **Conditional**: Only pay compute cost for activated experts

**Implementation Notes**:
- No shared experts (unlike some MoE architectures like DeepSeek V3)
- Simple top-k routing with softmax normalization
- Efficient expert processing through masking and index selection

### 5. KV Cache with Position Tracking
**Efficient Autoregressive Generation**

**Features**:
```python
self.current_pos = 0  # Track position in sequence

# During forward pass
if cache is not None:
    pos_start = self.current_pos
    pos_end = pos_start + num_tokens
    self.current_pos = pos_end
```

**Benefits**:
- **Memory Efficiency**: Reuses past key-value computations
- **Speed**: Avoids recomputing attention for previous tokens
- **Correctness**: Position tracking ensures proper RoPE offset and masking
- **Flexible**: Supports both prefill (full sequence) and generation (token-by-token)

**Cache Structure**:
```python
cache = {
    layer_idx: (keys_cache, values_cache),  # Tuple per layer
    ...
}
```

### 6. Numerical Stability Features
**Robust Training and Inference**

**Stability Techniques**:
1. **Float32 Normalization**: RMSNorm can compute in FP32 even with BF16/FP16 model weights
2. **Epsilon Values**: 1e-6 epsilon in normalization prevents division by zero
3. **Masked Fill**: Uses `-torch.inf` instead of large negative values for masking
4. **Dtype Conversion**: Careful handling of dtype conversions in RoPE and normalization

### 7. Flexible Configuration
**Highly Parameterizable Architecture**

**Configuration Options**:
- **Attention**: Number of heads, KV groups, head dimension, QK normalization
- **FFN**: Dense or MoE, hidden dimensions, number of experts, experts per token
- **Normalization**: RMSNorm with optional bias and FP32 mode
- **RoPE**: Configurable base frequency and context length
- **Dtype**: Flexible precision (BF16, FP16, FP32)

**Design Philosophy**:
- All configuration via dictionary (`cfg`) for easy experimentation
- Reasonable defaults with override capability
- Modular components for easy modification

### 8. Clean Implementation
**Production-Ready Code**

**Code Quality**:
- Clear module separation (Attention, FFN, Block, Model)
- No unnecessary dependencies or custom CUDA kernels
- Standard PyTorch operations throughout
- Well-commented and readable
- Efficient tensor operations (minimal loops)

**Optimizations**:
- Efficient MoE routing with vectorized operations
- Minimal tensor copies and reshapes
- Strategic use of `torch.compile` friendly patterns
- Buffer registration for RoPE parameters (no gradient tracking)

---

## Summary

**Qwen3** is a modern transformer architecture that combines proven techniques with efficiency innovations:

### Core Design Principles
1. **Efficiency First**: GQA and optional MoE reduce memory and compute costs
2. **Stability**: Pre-normalization with RMSNorm and optional QK normalization
3. **Flexibility**: Highly configurable with sensible defaults
4. **Simplicity**: Clean implementation using standard PyTorch operations

### Key Innovations
- **Grouped Query Attention**: 2x-8x KV cache reduction with minimal quality loss
- **RoPE**: Strong positional encoding without learned parameters
- **Optional MoE**: Conditional computation for parameter-efficient scaling
- **KV Caching**: Efficient autoregressive generation with position tracking

### Architecture Characteristics
- **Size**: Configurable (typically 1B-70B+ parameters)
- **Depth**: Variable number of transformer blocks
- **Width**: Flexible hidden dimensions and head counts
- **Context**: RoPE enables strong long-context capabilities
- **Precision**: Supports mixed precision training and inference

### Comparison to Reference Architectures
Compared to GptOss and DeepSeek V3:
- **Simpler than DeepSeek V3**: No MLA compression, no hybrid dense-MoE, standard GQA
- **More flexible than GptOss**: Optional MoE instead of always-on, cleaner codebase
- **Modern design**: GQA (like DeepSeek V3), pre-norm (like both), RoPE (like both)
- **Production-ready**: Clean implementation without custom kernels

### Target Use Cases
- **General Language Modeling**: Flexible architecture suitable for various scales
- **Efficient Inference**: GQA and KV caching optimize deployment
- **Research**: Clean, modular design facilitates experimentation
- **Scaling**: Optional MoE enables parameter-efficient model scaling

Qwen3 represents a pragmatic, efficient transformer design that prioritizes simplicity and effectiveness while incorporating modern architectural innovations.
