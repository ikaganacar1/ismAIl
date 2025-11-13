import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

#####################################
# CONFIGURATION
#####################################
@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None

    vocab_size: int = 102400
    dim: int = 1024
    inter_dim: int = 4096
    moe_inter_dim: int = 1024
    n_layers: int = 20
    n_dense_layers: int = 3
    n_heads: int = 12
   
    # moe
    n_routed_experts: int = 6
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    route_scale: float = 1.
    use_routing_bias: bool = True  # Enable routing bias for fine-tuning expert selection
    
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

    tokenizer_name: str = "gpt2"  #

# others
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"




#####################################
# RoPE
#####################################
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


#####################################
# LINEAR LAYERS
#####################################

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:

    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.float32
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set dtype
        param_dtype = dtype or Linear.dtype

        # Initialize weight with proper distribution
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=param_dtype))
        # CRITICAL: Initialize weights!
        nn.init.normal_(self.weight, mean=0.0, std=0.02 / math.sqrt(in_features))

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            # Initialize scale to 1.0
            nn.init.ones_(self.scale)
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=param_dtype))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y

#####################################
# NORMALIZATION
#####################################

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Keep weight in float32 for stability
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # F.rms_norm handles dtype conversion internally
        output = F.rms_norm(x.float(), (self.dim,), self.weight, self.eps)
        return output.to(x.dtype)


#####################################
# ATTENTION
#####################################

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale


        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank, dtype=Linear.dtype), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim, dtype=Linear.dtype), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)


        wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv).detach()
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2).detach()
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)


        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


#####################################
# MOE FEEDFORWARD
#####################################

class Gate(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.route_scale = args.route_scale

        # Gate weight
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=Linear.dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02 / math.sqrt(args.dim)) 

        # Optional routing bias for fine-tuning expert selection
        if args.use_routing_bias:
            self.bias = nn.Parameter(torch.zeros(args.n_routed_experts, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute routing scores
        scores = linear(x, self.weight)

        # Apply scoring function
        scores = scores.sigmoid()

        original_scores = scores

        # Apply routing bias if available
        if self.bias is not None:
            scores = scores + self.bias

        # Select top-k experts
        indices = torch.topk(scores, self.n_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        # Normalize weights (sigmoid always needs normalization)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Apply route scaling
        weights = weights * self.route_scale

        return weights.type_as(x), indices


class Expert(nn.Module):

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, bias=False)
        self.w2 = Linear(inter_dim, dim, bias=False)
        self.w3 = Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: w2(silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.active_expert_idx = None  # None = all active (inference mode)
        
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for _ in range(args.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.ffn_norm = RMSNorm(args.dim)
        
        # Load balance loss coefficient
        self.lb_loss_coef = 0.01

    def set_active_expert(self, expert_idx: Optional[int]):
        """Freeze all but the active expert to save optimizer memory"""
        self.active_expert_idx = expert_idx
        
        for i, expert in enumerate(self.experts):
            requires_grad = (expert_idx is None) or (i == expert_idx)
            for param in expert.parameters():
                param.requires_grad = requires_grad

    def compute_load_balance_loss(self, router_probs, expert_indices):
        """Encourage uniform expert utilization"""
        # router_probs: [num_tokens, n_experts]
        # expert_indices: [num_tokens, top_k]
        
        # Token fraction per expert
        tokens_per_expert = torch.zeros(self.n_routed_experts, device=router_probs.device)
        indices_flat = expert_indices.view(-1)
        ones = torch.ones_like(indices_flat, dtype=torch.float32)
        tokens_per_expert.scatter_add_(0, indices_flat, ones)
        tokens_per_expert = tokens_per_expert / (indices_flat.numel() + 1e-8)
        
        # Average routing probability per expert
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss (minimize difference)
        loss = torch.mean(tokens_per_expert * router_prob_per_expert) * self.n_routed_experts
        return loss

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        original_shape = x.size()
        x = x.view(-1, self.dim)

        router_logits = linear(x, self.gate.weight, self.gate.bias)
        router_probs = router_logits.sigmoid()
        weights, indices = torch.topk(router_probs, self.n_activated_experts, dim=-1)
        
        # Normalize weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # Add epsilon for stability
        weights = weights * self.gate.route_scale

        # CRITICAL FIX: Check training mode AND active expert
        if self.training and self.active_expert_idx is not None:
            # Sequential training mode - only train one expert
            y = torch.zeros_like(x)
            i = self.active_expert_idx

            # Find tokens where expert i is in the top-k
            mask = (indices == i)
            idx = torch.where(mask.any(dim=1))[0]

            if idx.numel() > 0:
                top_positions = torch.argmax(mask[idx].int(), dim=1)
                expert_weights = weights[idx, top_positions].unsqueeze(-1)
                expert_out = self.experts[i](x[idx])
                y[idx] = expert_out * expert_weights

            # Load balance loss
            lb_loss = self.compute_load_balance_loss(router_probs, indices)

            # Shared experts
            z = self.shared_experts(x)
            return (y + z).view(original_shape), lb_loss
        
        else:
            # Inference mode or all-experts training mode
            y = torch.zeros_like(x)
            for i in range(self.n_routed_experts):
                mask = (indices == i)
                idx = torch.where(mask.any(dim=1))[0]

                if idx.numel() == 0:
                    continue

                top_positions = torch.argmax(mask[idx].int(), dim=1)
                expert_weights = weights[idx, top_positions].unsqueeze(-1)
                expert_out = self.experts[i](x[idx])
                y[idx] += expert_out * expert_weights

            z = self.shared_experts(x)
            output = (y + z).view(original_shape)

            # Only compute load balance loss during training
            if self.training:
                lb_loss = self.compute_load_balance_loss(router_probs, indices)
                return output, lb_loss
            else:
                return output, None



#####################################
# DENSE FEEDFORWARD (MLP)
#####################################

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.fc1 = Linear(dim, inter_dim, bias=False)
        self.fc2 = Linear(dim, inter_dim, bias=False)
        self.fc3 = Linear(inter_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU-style activation: silu(fc1(x)) * fc2(x)
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


#####################################
# TRANSFORMER BLOCKS
#####################################

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MultiHeadLatentAttention(args)
        # Use dense MLP for first n_dense_layers, then MoE for remaining layers
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)

        # Handle both MLP (returns single output) and MoE (returns output + loss)
        ffn_result = self.ffn(self.ffn_norm(x))
        if isinstance(ffn_result, tuple):
            ffn_out, lb_loss = ffn_result
        else:
            ffn_out = ffn_result
            lb_loss = None

        x = x + ffn_out
        return x, lb_loss
    


#####################################
# TRANSFORMER MODEL
#####################################

class ismail(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        # Create embedding with correct dtype
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, dtype=Linear.dtype)
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.output = Linear(args.dim, args.vocab_size, bias=False, dtype=Linear.dtype)
        self.use_checkpointing = False

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def set_active_expert(self, expert_idx: Optional[int]):
        """Set active expert for all MoE layers (for sequential training)"""
        for layer in self.layers:
            if isinstance(layer.ffn, MoE):
                layer.ffn.set_active_expert(expert_idx)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens).to(Linear.dtype)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        # CRITICAL: Always clear caches at start_pos=0, regardless of training mode
        if start_pos == 0:
            for layer in self.layers:
                if hasattr(layer.attn, 'kv_cache'):
                    layer.attn.kv_cache.zero_()
                if hasattr(layer.attn, 'pe_cache'):
                    layer.attn.pe_cache.zero_()

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device, dtype=h.dtype), mask])

        total_lb_loss = 0.0
        
        for layer in self.layers:
            h, lb_loss = layer(h, start_pos, freqs_cis, mask)
            if lb_loss is not None:
                total_lb_loss += lb_loss

        h = self.norm(h)
        output = self.output(h)

        # FIX: Only return load balance loss during training
        if self.training and total_lb_loss > 0:
            return output, total_lb_loss
        else:
            return output
