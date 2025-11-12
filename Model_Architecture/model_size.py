import sys
from pathlib import Path

# Add the Model_Architecture directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import ModelArgs

def estimate_model_size(args: ModelArgs):
    """Calculate detailed model size and parameter count"""

    print(f"\n{'='*70}")
    print(f"MODEL ARCHITECTURE ANALYSIS: ismail")
    print(f"{'='*70}\n")

    # Display configuration
    print(f"📋 CONFIGURATION:")
    print(f"   Model dimension (dim):        {args.dim}")
    print(f"   Vocabulary size:              {args.vocab_size:,}")
    print(f"   Number of layers:             {args.n_layers}")
    print(f"   Dense layers:                 {args.n_dense_layers}")
    print(f"   MoE layers:                   {args.n_layers - args.n_dense_layers}")
    print(f"   Attention heads:              {args.n_heads}")
    print(f"   Max sequence length:          {args.max_seq_len}")
    print(f"   Max batch size:               {args.max_batch_size}")
    print(f"   \nMoE Configuration:")
    print(f"   Routed experts:               {args.n_routed_experts}")
    print(f"   Shared experts:               {args.n_shared_experts}")
    print(f"   Activated experts:            {args.n_activated_experts}")
    print(f"   \nMLA Configuration:")
    print(f"   Q LoRA rank:                  {args.q_lora_rank}")
    print(f"   KV LoRA rank:                 {args.kv_lora_rank}")
    print(f"   QK nope head dim:             {args.qk_nope_head_dim}")
    print(f"   QK rope head dim:             {args.qk_rope_head_dim}")
    print(f"   V head dim:                   {args.v_head_dim}")

    # Calculate parameters by component
    print(f"\n{'='*70}")
    print(f"🔢 PARAMETER COUNT BY COMPONENT:")
    print(f"{'='*70}\n")

    # 1. Embeddings
    tok_embed_params = args.vocab_size * args.dim
    output_params = args.vocab_size * args.dim
    total_embed_params = tok_embed_params + output_params
    print(f"   Token Embeddings:             {tok_embed_params:>15,} params")
    print(f"   Output Layer:                 {output_params:>15,} params")
    print(f"   {'─' * 50}")
    print(f"   Total Embeddings:             {total_embed_params:>15,} params\n")

    # 2. Attention (per layer)
    if args.q_lora_rank == 0:
        wq_params = args.dim * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim)
        wq_norm_params = 0
    else:
        wq_params = args.dim * args.q_lora_rank + args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim)
        wq_norm_params = args.q_lora_rank

    wkv_a_params = args.dim * (args.kv_lora_rank + args.qk_rope_head_dim)
    kv_norm_params = args.kv_lora_rank
    wkv_b_params = args.kv_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.v_head_dim)
    wo_params = args.n_heads * args.v_head_dim * args.dim
    attn_norm_params = args.dim

    attn_params_per_layer = wq_params + wq_norm_params + wkv_a_params + kv_norm_params + wkv_b_params + wo_params + attn_norm_params

    print(f"   Attention (per layer):")
    if args.q_lora_rank > 0:
        print(f"      WQ (LoRA):                 {wq_params:>15,} params")
        print(f"      Q Norm:                    {wq_norm_params:>15,} params")
    else:
        print(f"      WQ:                        {wq_params:>15,} params")
    print(f"      WKV_A:                     {wkv_a_params:>15,} params")
    print(f"      KV Norm:                   {kv_norm_params:>15,} params")
    print(f"      WKV_B:                     {wkv_b_params:>15,} params")
    print(f"      WO:                        {wo_params:>15,} params")
    print(f"      Attn Norm:                 {attn_norm_params:>15,} params")
    print(f"   {'─' * 50}")
    print(f"      Subtotal:                  {attn_params_per_layer:>15,} params\n")

    # 3. Dense FFN
    dense_w1_params = args.dim * args.inter_dim
    dense_w2_params = args.inter_dim * args.dim
    dense_w3_params = args.dim * args.inter_dim
    ffn_norm_params = args.dim
    dense_ffn_per_layer = dense_w1_params + dense_w2_params + dense_w3_params + ffn_norm_params

    print(f"   Dense FFN (per layer):")
    print(f"      FC1 (W1):                  {dense_w1_params:>15,} params")
    print(f"      FC2 (W3):                  {dense_w3_params:>15,} params")
    print(f"      FC3 (W2):                  {dense_w2_params:>15,} params")
    print(f"      FFN Norm:                  {ffn_norm_params:>15,} params")
    print(f"   {'─' * 50}")
    print(f"      Subtotal:                  {dense_ffn_per_layer:>15,} params\n")

    # 4. MoE FFN
    gate_params = args.n_routed_experts * args.dim
    if args.use_routing_bias:
        gate_params += args.n_routed_experts

    expert_w1_params = args.dim * args.moe_inter_dim
    expert_w2_params = args.moe_inter_dim * args.dim
    expert_w3_params = args.dim * args.moe_inter_dim
    per_expert_params = expert_w1_params + expert_w2_params + expert_w3_params
    routed_experts_params = args.n_routed_experts * per_expert_params

    shared_w1_params = args.dim * (args.n_shared_experts * args.moe_inter_dim)
    shared_w2_params = (args.n_shared_experts * args.moe_inter_dim) * args.dim
    shared_w3_params = args.dim * (args.n_shared_experts * args.moe_inter_dim)
    shared_experts_params = shared_w1_params + shared_w2_params + shared_w3_params

    moe_ffn_per_layer = gate_params + routed_experts_params + shared_experts_params + ffn_norm_params

    print(f"   MoE FFN (per layer):")
    print(f"      Gate:                      {gate_params:>15,} params")
    print(f"      Routed Experts ({args.n_routed_experts}x):       {routed_experts_params:>15,} params")
    print(f"         Per expert:             {per_expert_params:>15,} params")
    print(f"      Shared Experts:            {shared_experts_params:>15,} params")
    print(f"      FFN Norm:                  {ffn_norm_params:>15,} params")
    print(f"   {'─' * 50}")
    print(f"      Subtotal:                  {moe_ffn_per_layer:>15,} params\n")

    # 5. Final Norm
    final_norm_params = args.dim

    # Total calculation
    dense_layer_params = attn_params_per_layer + dense_ffn_per_layer
    moe_layer_params = attn_params_per_layer + moe_ffn_per_layer

    total_dense_params = args.n_dense_layers * dense_layer_params
    total_moe_params = (args.n_layers - args.n_dense_layers) * moe_layer_params

    total_params = total_embed_params + total_dense_params + total_moe_params + final_norm_params

    print(f"   Layer Summary:")
    print(f"      Dense layers ({args.n_dense_layers}x):        {total_dense_params:>15,} params")
    print(f"      MoE layers ({args.n_layers - args.n_dense_layers}x):          {total_moe_params:>15,} params")
    print(f"      Final Norm:                {final_norm_params:>15,} params")

    print(f"\n{'='*70}")
    print(f"📊 TOTAL PARAMETERS:              {total_params:>15,} ({total_params/1e6:.2f}M)")
    print(f"{'='*70}\n")

    # Memory calculations
    print(f"{'='*70}")
    print(f"💾 MEMORY USAGE:")
    print(f"{'='*70}\n")

    bytes_per_param_bf16 = 2
    bytes_per_param_fp32 = 4

    # Model weights
    weight_memory_bf16 = total_params * bytes_per_param_bf16 / (1024**3)
    weight_memory_fp32 = total_params * bytes_per_param_fp32 / (1024**3)

    print(f"   Model Weights:")
    print(f"      BF16 (inference):          {weight_memory_bf16:>10.3f} GB")
    print(f"      FP32 (training):           {weight_memory_fp32:>10.3f} GB\n")

    # KV Cache
    kv_cache_per_layer = args.max_batch_size * args.max_seq_len * (args.kv_lora_rank + args.qk_rope_head_dim)
    total_kv_cache = kv_cache_per_layer * args.n_layers * bytes_per_param_bf16 / (1024**3)

    print(f"   KV Cache (BF16):")
    print(f"      Per layer:                 {kv_cache_per_layer * bytes_per_param_bf16 / (1024**3):>10.3f} GB")
    print(f"      Total ({args.n_layers} layers):         {total_kv_cache:>10.3f} GB\n")

    # Activations (rough estimate)
    activation_memory = (args.max_batch_size * args.max_seq_len * args.dim * args.n_layers * 4) / (1024**3)

    print(f"   Activations (estimate):       {activation_memory:>10.3f} GB\n")

    # Training overhead
    gradients_memory = weight_memory_fp32  # Same size as weights
    optimizer_states = weight_memory_fp32 * 2  # Adam: 2x for momentum + variance
    training_overhead = gradients_memory + optimizer_states

    print(f"   Training Overhead (FP32):")
    print(f"      Gradients:                 {gradients_memory:>10.3f} GB")
    print(f"      Optimizer states (Adam):   {optimizer_states:>10.3f} GB")
    print(f"      Total overhead:            {training_overhead:>10.3f} GB\n")

    # Total estimates
    inference_total = weight_memory_bf16 + total_kv_cache + activation_memory
    training_total = weight_memory_fp32 + total_kv_cache + activation_memory + training_overhead

    print(f"{'='*70}")
    print(f"   INFERENCE (BF16):             {inference_total:>10.3f} GB")
    print(f"   TRAINING (FP32 + Adam):       {training_total:>10.3f} GB")
    print(f"{'='*70}\n")

    # Memory analysis
    print(f"{'='*70}")
    print(f"🎯 MEMORY ANALYSIS:")
    print(f"{'='*70}\n")

    for threshold, name in [(8, "8GB"), (16, "16GB"), (24, "24GB"), (32, "32GB"), (40, "40GB"), (48, "48GB"), (80, "80GB")]:
        if inference_total <= threshold:
            print(f"   ✅ Inference fits in {name} GPU")
            break
    else:
        print(f"   ❌ Inference requires >80GB GPU")

    for threshold, name in [(8, "8GB"), (16, "16GB"), (24, "24GB"), (32, "32GB"), (40, "40GB"), (48, "48GB"), (80, "80GB")]:
        if training_total <= threshold:
            print(f"   ✅ Training fits in {name} GPU")
            break
    else:
        print(f"   ❌ Training requires >80GB GPU")

    print(f"\n{'='*70}\n")

    return {
        'total_params': total_params,
        'weight_memory_gb': weight_memory_bf16,
        'inference_memory_gb': inference_total,
        'training_memory_gb': training_total
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Try to load from config.json, otherwise use defaults
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        print(f"📄 Loading configuration from {config_path}")
        with open(config_path) as f:
            config = json.load(f)
        args = ModelArgs(**config["model"])
    else:
        print("⚠️  config.json not found, using default ModelArgs")
        args = ModelArgs()

    # Run estimation
    results = estimate_model_size(args)