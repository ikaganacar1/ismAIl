# GptOss Model Architecture Analysis

## Model Overview
GptOss is a Mixture of Experts (MoE) transformer-based language model that extends the Mixtral architecture with several unique features including attention sinks, custom gating mechanisms, and specialized routing.

---

## Model Configuration

### Core Parameters
- **Vocabulary Size**: Defined in `GptOssConfig` (not shown in code, inherited from parent)
- **Hidden Dimension**: `config.hidden_size`
- **Number of Layers**: Defined by length of `config.layer_types` (per-layer attention type configuration)
- **Number of Attention Heads**: `config.num_attention_heads`
- **Number of Key-Value Heads**: `config.num_key_value_heads` (for Grouped Query Attention)
- **Max Sequence Length**: Determined by RoPE configuration
- **Intermediate Size (FFN)**: `config.intermediate_size`
- **Number of Experts**: `config.num_local_experts`
- **Experts Per Token**: `config.num_experts_per_tok` (top-k routing)

### Additional Configuration
- **Attention Bias**: `config.attention_bias` (QKV projections can have bias)
- **RMS Norm Epsilon**: `config.rms_norm_eps`
- **Sliding Window Support**: Configurable per-layer attention types
- **Layer Types**: `config.layer_types[layer_idx]` - supports "full_attention" and "sliding_attention"

---

## Transformer Block Architecture

### Block Structure
**Architecture Type**: Pre-normalization (Pre-LN)

```
Input
  ↓
Residual Connection ─────────┐
  ↓                           │
RMSNorm (input_layernorm)     │
  ↓                           │
Self-Attention                │
  ↓                           │
Add Residual ←────────────────┘
  ↓
Residual Connection ─────────┐
  ↓                           │
RMSNorm (post_attention_ln)   │
  ↓                           │
MoE MLP (Router + Experts)    │
  ↓                           │
Add Residual ←────────────────┘
  ↓
Output
```

### Component Order
1. **Input Layer Normalization** → Self-Attention → Residual Add
2. **Post-Attention Normalization** → MoE MLP → Residual Add

### Residual Connections
- **Type**: Standard additive residual connections
- **Placement**: 
  - After self-attention (pre-normalized)
  - After MLP/MoE layer (pre-normalized)

---

## Attention Mechanism

### Attention Type
**Grouped Query Attention (GQA)**
- Inherits from `Qwen2Attention`
- Uses separate head counts for queries vs key-values
- `num_key_value_groups = num_attention_heads / num_key_value_heads`
- Keys and values are repeated using `repeat_kv()` to match query head count

### QKV Projection Details

**Projection Layers**:
```python
q_proj: Linear(hidden_size → num_attention_heads × head_dim, bias=attention_bias)
k_proj: Linear(hidden_size → num_key_value_heads × head_dim, bias=attention_bias)
v_proj: Linear(hidden_size → num_key_value_heads × head_dim, bias=attention_bias)
o_proj: Linear(num_attention_heads × head_dim → hidden_size, bias=attention_bias)
```

**Head Dimension**: `head_dim = hidden_size / num_attention_heads`

### Attention Computation Method

**Implementation**: Custom eager attention with attention sinks

**Computation Flow**:
1. **Project** queries, keys, values
2. **Apply RoPE** to queries and keys
3. **Repeat KV** for GQA: `repeat_kv(key/value, num_key_value_groups)`
4. **Compute scores**: `Q @ K^T × scaling`
5. **Apply causal mask**
6. **Inject attention sinks**: Learnable sink parameters concatenated to attention logits
   ```python
   sinks = nn.Parameter(torch.empty(num_attention_heads))
   combined_logits = torch.cat([attn_weights, sinks], dim=-1)
   ```
7. **Stability**: Subtract max value before softmax (prevents overflow in BF16/FP16)
8. **Softmax** over combined logits (including sinks)
9. **Drop sinks**: Remove sink dimension from attention probabilities
10. **Apply dropout**
11. **Weighted sum** with values

### Positional Encoding

**Type**: Rotary Position Embeddings (RoPE) with Attention Scaling

**Implementation**: `GptOssRotaryEmbedding` (extends `Qwen2RotaryEmbedding`)

**Key Features**:
```python
# Compute rotation frequencies
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))

# Apply to position IDs
freqs = inv_freq @ position_ids
cos = freqs.cos() × attention_scaling  # ← Unique: scaled cosine
sin = freqs.sin() × attention_scaling  # ← Unique: scaled sine
```

**Application**:
```python
# Split into two halves and rotate
first_half × cos - second_half × sin
second_half × cos + first_half × sin
```

**Notable**: 
- Uses `attention_scaling` factor applied to cos/sin embeddings (not typical in standard RoPE)
- Supports dynamic RoPE updates via `@dynamic_rope_update` decorator
- Forces float32 computation for numerical stability

### Attention Masks/Biases

**Mask Types**:
1. **Causal Mask**: Standard autoregressive causal masking
2. **Sliding Window Mask**: For local attention patterns (per-layer configurable)

**Mask Selection**: Per-layer via `config.layer_types[layer_idx]`:
- `"full_attention"` → full causal mask
- `"sliding_attention"` → sliding window causal mask

**Attention Sinks**: 
- Learnable parameters (`nn.Parameter`) with shape `[num_attention_heads]`
- Appended as extra "tokens" in attention computation
- Provides model with learned attention bias mechanism
- Dropped after softmax, don't contribute to output

---

## Feed-Forward Network (MoE Architecture)

### Architecture
**Type**: Mixture of Experts (MoE) with Custom Gating

**Components**:
1. **Router** (`GptOssTopKRouter`): Top-k expert selection
2. **Experts** (`GptOssExperts`): Parallel expert networks

### Router Architecture

**Implementation**:
```python
router_logits = Linear(hidden_dim → num_experts, bias=True)
top_k_values, top_k_indices = topk(router_logits, k=num_experts_per_tok)
routing_weights = softmax(top_k_values)  # Only over selected experts
```

**Features**:
- Linear projection with bias to compute expert scores
- Top-k selection (typically k=2)
- Softmax only over selected experts (sparse routing)

### Expert Network Architecture

**Structure**: Custom gated architecture with biases

```python
# Gate-Up projection (combined)
gate_up_proj: Parameter[num_experts, hidden_size, 2 × intermediate_size]
gate_up_proj_bias: Parameter[num_experts, 2 × intermediate_size]

# Down projection  
down_proj: Parameter[num_experts, intermediate_size, hidden_size]
down_proj_bias: Parameter[num_experts, hidden_size]
```

**Forward Pass**:
```python
# 1. Combined gate-up projection
gate_up = hidden @ gate_up_proj + gate_up_proj_bias

# 2. Split into gate and up
gate = gate_up[..., ::2]  # Even indices
up = gate_up[..., 1::2]   # Odd indices

# 3. Clamp for stability
gate = gate.clamp(min=None, max=7.0)
up = up.clamp(min=-7.0, max=7.0)

# 4. Custom gating activation
glu = gate × sigmoid(gate × 1.702)  # α = 1.702
gated_output = (up + 1) × glu

# 5. Down projection
output = gated_output @ down_proj + down_proj_bias

# 6. Weight by routing scores
final_output = output × routing_weights
```

### Hidden Dimension Multiplier
- **Expert Dimension**: `intermediate_size`
- **Multiplier**: `2 × intermediate_size` in gate_up_proj (for both gate and up)
- Typical multiplier relative to hidden_size depends on config (commonly ~4x)

### Activation Function
**Custom Gated Activation**:
```
f(gate, up) = (up + 1) × [gate × σ(α × gate)]
```

Where:
- `σ` = sigmoid
- `α` = 1.702 (fixed constant)
- Offset `+1` applied to `up` before gating

**Characteristics**:
- Similar to SwiGLU but with modifications
- Uses sigmoid-based gating instead of SiLU
- Includes learned scaling factor (α) and offset
- Clamping for numerical stability (±7.0)

### Gating Mechanism
**Type**: Expert gating with sparse routing

**Properties**:
- Top-k expert selection per token
- Weighted combination of expert outputs
- Training vs Inference modes:
  - **Training/CPU**: Loop over experts (memory efficient)
  - **Inference/GPU**: Parallel computation with repeated inputs (faster)

---

## Normalization Layers

### Type
**RMSNorm (Root Mean Square Layer Normalization)**

**Implementation**: `GptOssRMSNorm` (extends `LlamaRMSNorm`)

```python
# Convert to float32 for stability
hidden_states = hidden_states.to(torch.float32)

# Compute variance (mean of squares)
variance = hidden_states.pow(2).mean(-1, keepdim=True)

# Normalize
normalized = hidden_states × rsqrt(variance + epsilon)

# Scale and convert back
output = (weight × normalized).to(input_dtype)
```

**Key Difference from LlamaRMSNorm**:
- Applies weight scaling **then** converts to output dtype
- Original: converts dtype first, then applies weight

### Placement
**Pre-normalization (Pre-LN) Architecture**:
- `input_layernorm`: Before self-attention
- `post_attention_layernorm`: Before MLP/MoE
- `norm`: Final layer normalization (after all transformer blocks)

### Parameters
- **Weight**: Learnable scale parameter (initialized to 1.0)
- **Epsilon**: `config.rms_norm_eps` (for numerical stability)
- **No bias term**

---

## Notable Features

### 1. Attention Sinks
**Innovation**: Learnable attention bias mechanism
- Implemented as learnable parameters per attention head
- Concatenated to attention logits as additional "tokens"
- Provides flexible attention bias without architectural constraints
- Dropped after attention computation (don't contribute to values)

**Purpose**: 
- May help with attention saturation
- Provides learned "null" attention targets
- Similar to "sink tokens" in streaming attention

### 2. Custom Expert Gating
**Key Features**:
- **Bias terms**: Both gate_up and down projections have biases (uncommon in MoE)
- **Custom activation**: `(up + 1) × [gate × σ(1.702 × gate)]`
- **Clamping**: Limits to ±7.0 for numerical stability
- **Offset**: `+1` offset in up branch before gating

### 3. Scaled RoPE
**Unique Feature**: Attention scaling applied to RoPE embeddings
```python
cos = emb.cos() × attention_scaling
sin = emb.sin() × attention_scaling
```
This is non-standard and may help with attention magnitude control.

### 4. Per-Layer Attention Types
**Flexibility**: Each layer can use different attention patterns
- `"full_attention"`: Standard causal attention
- `"sliding_attention"`: Local sliding window attention

Configured via `config.layer_types[layer_idx]`, enabling hybrid architectures.

### 5. Dual-Mode Expert Computation
**Training Mode** (CPU or explicitly set):
- Loop over experts individually
- Memory efficient for large batch sizes
- Uses masking to identify which tokens route to which experts

**Inference Mode** (GPU):
- Parallel expert computation
- Repeats inputs across experts
- Batch matrix multiplication for speed
- Higher memory usage

### 6. Sparse Top-K Routing
**Efficiency**: Only activates `num_experts_per_tok` experts per token
- Reduces computation by ~(num_experts / k)×
- Softmax only over selected experts
- Typical configuration: k=2 out of 8 or 16 experts

### 7. Numerical Stability Features
- **RMSNorm in FP32**: Forces float32 computation in normalization
- **Attention logit clamping**: Subtracts max before softmax
- **Expert clamping**: ±7.0 limits on gate and up projections
- **Autocast management**: Explicit float32 contexts in RoPE

---

## Inheritance Structure

```
GptOss Components:
├── GptOssRMSNorm → LlamaRMSNorm
├── GptOssRotaryEmbedding → Qwen2RotaryEmbedding
├── GptOssAttention → Qwen2Attention
├── GptOssDecoderLayer → LlamaDecoderLayer
├── GptOssPreTrainedModel → LlamaPreTrainedModel
└── GptOssModel → MixtralModel
```

**Key Insight**: Builds on proven architectures (Llama, Qwen2, Mixtral) while adding:
- Attention sinks
- Custom expert gating
- Scaled RoPE
- Flexible per-layer attention patterns

---

## Optimizations

### 1. Memory Efficiency
- Training mode expert computation (loops vs parallel)
- Sparse routing (only k experts active)
- GQA reduces KV cache size vs MHA

### 2. Computational Efficiency  
- Grouped Query Attention (shared KV across query heads)
- Top-k routing (sparse expert activation)
- Combined gate_up projection (single matmul)
- Inference mode parallel expert computation

### 3. Numerical Stability
- FP32 normalization layers
- Attention logit clamping
- Expert activation clamping
- Explicit autocast management

### 4. Flexibility
- Per-layer attention types (full vs sliding window)
- Configurable expert routing (top-k parameter)
- Optional biases in attention projections
- Dynamic RoPE support

---

## Summary

**GptOss** is a sophisticated MoE transformer that combines:
- **Grouped Query Attention** for efficient inference
- **Custom gated experts** with biases and novel activation
- **Attention sinks** for learnable attention bias
- **Scaled RoPE** with attention magnitude control  
- **Flexible architecture** with per-layer attention patterns
- **Robust numerics** with extensive stability features

The model represents an evolution of the Mixtral architecture with several experimental features aimed at improving both performance and stability, particularly for long-context and large-scale training scenarios.
