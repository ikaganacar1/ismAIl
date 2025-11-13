import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

import torch.nn.functional as F
from pathlib import Path
import json
import time
import math


# Import your model
from model import ismail, ModelArgs

# Try to import optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not installed. Run 'pip install wandb' for experiment tracking.")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("⚠️  bitsandbytes not installed. Run 'pip install bitsandbytes' for memory-efficient optimizer.")

# Configuration - matches ModelArgs defaults
DEFAULT_CONFIG = {
    "model": {
        "max_batch_size": 8,
        "max_seq_len": 2048,
        "dtype": "bf16",
        "scale_fmt": None,
        "vocab_size": 102400,
        "dim": 1024,
        "inter_dim": 4096,
        "moe_inter_dim": 1024,
        "n_layers": 20,
        "n_dense_layers": 3,
        "n_heads": 12,
        "n_routed_experts": 6,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "route_scale": 1.0,
        "use_routing_bias": True,
        "q_lora_rank": 0,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "original_seq_len": 4096,
        "rope_theta": 10000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "tokenizer_name": "gpt2",
    },
    "training": {
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "warmup_steps": 1000,
        "total_steps": 50000,
        "expert_rotation_steps": 2000,  # Rotate expert every N steps
        "gradient_accumulation_steps": 16,
        "eval_every": 1000,
        "save_every": 5000,
        "save_dir": "./checkpoints",
        "log_every": 100,
        "dtype": "bf16",
        "compile": True,  # PyTorch 2.0+ compilation
    },
    "data": {
        "train_file": "./data/train.txt",
        "val_file": "./data/val.txt",
        "stride": 512,
    },
    "logging": {
        "use_wandb": HAS_WANDB,
        "project_name": "sequential-moe",
        "run_name": "moe-12gb-gpu",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoE model with sequential experts")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--train_file", type=str, help="Training text file")
    parser.add_argument("--val_file", type=str, help="Validation text file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    return parser.parse_args()


def load_config(args):
    """Load and merge configuration"""
    config = DEFAULT_CONFIG.copy()
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_config = json.load(f)
        # Deep merge
        for key, value in user_config.items():
            if key in config and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    
    # Override from CLI args
    if args.train_file:
        config["data"]["train_file"] = args.train_file
    if args.val_file:
        config["data"]["val_file"] = args.val_file
    if args.save_dir:
        config["training"]["save_dir"] = args.save_dir
    if args.no_wandb:
        config["logging"]["use_wandb"] = False
    
    return config


def setup_model(config, device):
    from model import Linear

    args = ModelArgs(**config["model"])

    # ✅ CRITICAL: Set the global dtype for Linear layers
    training_dtype = config["training"]["dtype"].lower()
    if training_dtype == "bf16":
        Linear.dtype = torch.bfloat16
    elif training_dtype == "fp16":
        Linear.dtype = torch.float16
    else:
        Linear.dtype = torch.float32

    model = ismail(args).to(device=device, dtype=Linear.dtype)

    # Add this line to enable checkpointing
    model.use_checkpointing = config["training"].get("use_checkpointing", True)

    if config["training"]["compile"]:
        try:
            model = torch.compile(model)
            print("✅ Model compiled\n")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}\n")

    return model, args


def setup_optimizer(model, config):
    """Setup memory-efficient optimizer"""
    training_cfg = config["training"]
    
    # Separate parameter groups
    expert_params = []
    base_params = []
    router_params = []
    
    for name, param in model.named_parameters():
        if "experts" in name and "shared" not in name:
            expert_params.append(param)
        elif "gate" in name:
            router_params.append(param)
        else:
            base_params.append(param)
    
    # Use 8-bit Adam if available
    if HAS_BNB:
        optimizer_class = bnb.optim.AdamW8bit
        print("✅ Using AdamW8bit for memory efficiency")
    else:
        optimizer_class = torch.optim.AdamW
        print("⚠️  Using standard AdamW (install bitsandbytes for memory savings)")
    
    optimizer = optimizer_class(
        [
            {"params": base_params, "weight_decay": training_cfg["weight_decay"]},
            {"params": expert_params, "weight_decay": training_cfg["weight_decay"]},
            {"params": router_params, "weight_decay": 0.0},  # Usually no WD for router
        ],
        lr=training_cfg["learning_rate"],
        betas=(training_cfg["beta1"], training_cfg["beta2"]),
    )
    
    return optimizer


def get_lr(step, config):
    """Learning rate scheduler with warmup and cosine decay"""
    training_cfg = config["training"]
    warmup_steps = training_cfg["warmup_steps"]
    total_steps = training_cfg["total_steps"]
    base_lr = training_cfg["learning_rate"]
    
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def load_data(config):
    from data import create_dataloader

    data_cfg = config["data"]

    print("\n" + "="*70)
    print("DATA LOADING")
    print("="*70 + "\n")

    from model import ModelArgs
    

    args = ModelArgs(**config["model"])

    train_loader, tokenizer = create_dataloader(
        txt=str(data_cfg["train_file"]),
        use_turkish_tokenizer=True,  
        args=args,
        stride=data_cfg["stride"],
        shuffle=True,
        drop_last=True,
        use_memory_efficient=True,
        is_val=False
    )

    val_loader, tokenizer = create_dataloader(
        txt=str(data_cfg["val_file"]),
        use_turkish_tokenizer=True, 
        args=args,
        stride=data_cfg["stride"],
        shuffle=False,
        drop_last=True,
        use_memory_efficient=True,
        is_val=True
    )

    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}\n")

    return train_loader, val_loader, tokenizer  # Return tokenizer

def evaluate(model, val_loader, device, config, tokenizer, active_expert=None):
    """Evaluate model on validation set
    
    Args:
        active_expert: If not None, only evaluate with this expert active
                      (useful for sequential training to see individual expert progress)
    """
    model.eval()
    
    # Clear caches...
    for layer in model.layers:
        if hasattr(layer.attn, 'kv_cache'):
            layer.attn.kv_cache.zero_()
        if hasattr(layer.attn, 'pe_cache'):
            layer.attn.pe_cache.zero_()
    
    # Set expert mode for validation
    if hasattr(model, 'set_active_expert'):
        model.set_active_expert(active_expert)
        if active_expert is not None:
            print(f"   Validating with ONLY expert {active_expert}")
        else:
            print(f"   Validating with ALL experts")
    
    total_loss = 0.0
    total_tokens = 0
    max_batches = config["training"].get("max_val_batches", 200)
    
    from tqdm import tqdm
    pbar = tqdm(total=max_batches, desc="📊 Validating", ncols=80)
    
    val_dtype = config["training"]["dtype"]
    batch_losses = []
    
    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(val_loader):
            if i >= max_batches:
                break
                
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            
            # 🔥 VISUAL TURKISH SAMPLE: Show human-readable text
            if i == 0:  # First batch only
                sample_tokens = input_ids[0].cpu().tolist()
                # Decode first 30 tokens (skip padding zeros)
                non_zero_tokens = [t for t in sample_tokens[:30] if t > 0]
                try:
                    sample_text = tokenizer.decode(non_zero_tokens)
                    # Truncate if too long
                    if len(sample_text) > 60:
                        sample_text = sample_text[:57] + "..."
                    print(f"\n📝 Örnek Turkce metin: '{sample_text}'")
                except Exception as e:
                    print(f"\n⚠️ Decode failed: {e}\n   Tokens: {non_zero_tokens[:10]}...")
            
            with torch.amp.autocast(device_type='cuda', enabled=(val_dtype == 'bf16')):
                output = model(input_ids, start_pos=0)
                logits = output[0] if isinstance(output, tuple) else output
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=-1,
                )
            
            batch_losses.append(loss.item())
            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    pbar.close()
    model.train()
    
    final_loss = total_loss / total_tokens
    
    # Show loss variation stats
    if len(batch_losses) > 1:
        loss_std = torch.std(torch.tensor(batch_losses)).item()
        print(f"   Loss std dev: {loss_std:.6f} (should be >0.01)")
    
    return final_loss


def save_checkpoint(model, optimizer, step, config, expert_idx=None):
    """Save model checkpoint"""
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_name = f"step_{step}_expert_{expert_idx}.pt" if expert_idx is not None else f"step_{step}.pt"
    ckpt_path = save_dir / ckpt_name
    
    # 🔥 Exclude cache buffers - they should be reinitialized from config
    state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'cache' not in k.lower()}
    
    checkpoint = {
        "step": step,
        "model_state_dict": filtered_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"💾 Checkpoint saved: {ckpt_path}")


def train_step(model, input_mb, target_mb, device, config, scaler=None):
    """Process a SINGLE micro-batch (already sliced)"""

    # 🚨 Validate data with more detail
    if input_mb.size(0) == 0:
        print("🚨 Warning: Empty micro-batch received")
        return 0.0, 0.0

    vocab_size = config["model"]["vocab_size"]
    input_max = input_mb.max().item()
    target_max = target_mb.max().item()

    if input_max >= vocab_size or target_max >= vocab_size:
        print(f"🚨 Invalid token detected! "
              f"Input max: {input_max}, Target max: {target_max}, "
              f"Vocab size: {vocab_size}")
        # Clamp tokens to valid range
        input_mb = torch.clamp(input_mb, max=vocab_size-1)
        target_mb = torch.clamp(target_mb, max=vocab_size-1)

    # Check for NaN in data
    if torch.isnan(input_mb).any() or torch.isnan(target_mb).any():
        print("🚨 NaN detected in input data! Replacing with zeros")
        input_mb = torch.nan_to_num(input_mb, nan=0)
        target_mb = torch.nan_to_num(target_mb, nan=0)

    input_mb = input_mb.to(device, non_blocking=True)
    target_mb = target_mb.to(device, non_blocking=True)

    with torch.amp.autocast(device_type='cuda', enabled=(config["training"]["dtype"] == "bf16")):
        output = model(input_mb, start_pos=0)
        
        if isinstance(output, tuple):
            logits, lb_loss = output
        else:
            logits = output
            lb_loss = 0.0
        
        # 🚨 Check for NaN in logits before computing loss
        if torch.isnan(logits).any():
            print(f"🚨 NaN detected in logits! Scale: {logits.abs().max().item()}")
            print(f"    Input range: [{input_mb.min().item()}, {input_mb.max().item()}]")
            return 0.0, 0.0
        
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_mb.view(-1),
            ignore_index=-1,
        )
        
        # 🚨 Check for NaN in loss components
        if torch.isnan(lm_loss):
            print(f"🚨 NaN in lm_loss!")
            return 0.0, 0.0
        
        accum_steps = config["training"]["gradient_accumulation_steps"]
        if isinstance(lb_loss, float):
            total_loss = lm_loss / accum_steps
        else:
            if torch.isnan(lb_loss):
                print(f"🚨 NaN in lb_loss! Setting to 0")
                lb_loss = 0.0
            lb_loss_coef = config["training"].get("lb_loss_coef", 0.01)
            total_loss = (lm_loss + lb_loss_coef * lb_loss) / accum_steps

    # Backward with NaN check
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()
    
    return lm_loss.item(), lb_loss if isinstance(lb_loss, float) else lb_loss.item()

def main():
    args = parse_args()
    config = load_config(args)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    # Wandb setup
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(project=config["logging"]["project_name"], 
                  name=config["logging"]["run_name"], config=config)
    
    # Model setup
    model, model_args = setup_model(config, device)
    
    # Optimizer setup
    optimizer = setup_optimizer(model, config)
    
    # Data setup
    train_loader, val_loader, tokenizer = load_data(config)
    train_iter = iter(train_loader)
    
    # Training state
    step = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint
    if args.resume:
        print(f"📥 Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        # Create model with current config (ensures correct cache sizes)
        model, model_args = setup_model(config, device)
        
        # Load state dict but skip/resize mismatched buffers
        model_state_dict = model.state_dict()
        loaded_state_dict = ckpt["model_state_dict"]
        
        skip_count = 0
        for name, param in loaded_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].shape != param.shape:
                    if "cache" in name:  # Skip cache buffers
                        skip_count += 1
                        continue
                    else:
                        raise RuntimeError(f"Shape mismatch {name}: {param.shape} vs {model_state_dict[name].shape}")
                model_state_dict[name].copy_(param)
            else:
                print(f"⚠️  Unexpected parameter: {name}")
        
        model.load_state_dict(model_state_dict, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        print(f"✅ Resumed from step {step} (skipped {skip_count} cache buffers)\n")
    
    # ✅ FIX: Only create scaler for FP16, not BF16 or FP32
    training_dtype = config["training"]["dtype"].lower()
    use_fp16 = training_dtype == "fp16"
    use_bf16 = training_dtype == "bf16"

    if use_fp16:
        scaler = torch.amp.GradScaler(device='cuda', enabled=True)
        print("✅ FP16 mode: Using GradScaler\n")
    elif use_bf16:
        scaler = None
        print("⚠️  BF16 mode: Disabling GradScaler (not needed/supported)\n")
    else:  # FP32
        scaler = None
        print("✅ FP32 mode: No scaler needed\n")
    
    # Expert rotation
    current_expert = 0
    rotation_steps = config["training"]["expert_rotation_steps"]
    
    # Check if we should train all experts simultaneously
    train_all_experts = config["training"].get("train_all_experts", False)
    
    if train_all_experts:
        print("🎯 Training ALL experts simultaneously\n")
        model.set_active_expert(None)  # None = all experts active
    else:
        print(f"🎯 Training expert {current_expert}/{model_args.n_routed_experts - 1} (sequential mode)\n")
        model.set_active_expert(current_expert)
    
    # Define variables
    accum_steps = config["training"]["gradient_accumulation_steps"]
    total_steps = config["training"]["total_steps"]
    grad_clip = config["training"]["grad_clip"]
    
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70 + "\n")
    
    model.train()
    
    # MAIN TRAINING LOOP
    while step < total_steps:
        step_start = time.time()
        
        # Expert rotation (only in sequential mode)
        if not train_all_experts and step > 0 and step % rotation_steps == 0:
            current_expert = (current_expert + 1) % model_args.n_routed_experts
            model.set_active_expert(current_expert)
            print(f"\n🔄 Rotating to expert {current_expert}/{model_args.n_routed_experts - 1}")
            optimizer.zero_grad(set_to_none=True)
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Split batch
        input_ids, target_ids = batch
        batch_size = input_ids.size(0)
        micro_batch_size = batch_size // accum_steps
        
        # Initialize accumulators
        lm_loss_accum = 0.0
        lb_loss_accum = 0.0
        
        # Gradient accumulation loop
        for accum_step in range(accum_steps):
            # Calculate slice indices
            start_idx = micro_batch_size * accum_step
            
            # Handle last micro-batch
            if accum_step == accum_steps - 1:
                end_idx = batch_size
            else:
                end_idx = start_idx + micro_batch_size
            
            # Extract micro-batch
            input_mb = input_ids[start_idx:end_idx]
            target_mb = target_ids[start_idx:end_idx]
            
            # Process micro-batch
            lm_loss, lb_loss = train_step(
                model, input_mb, target_mb, device, config, scaler
            )
            
            # Accumulate losses
            lm_loss_accum += lm_loss / accum_steps
            lb_loss_accum += lb_loss / accum_steps
        
        # Gradient clipping (if enabled)
        if grad_clip > 0:
            # Only unscale if using FP16 scaler
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # ✅ FIX: Conditional optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        # LR scheduling
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Logging
        if step % config["training"]["log_every"] == 0:
            step_time = time.time() - step_start
            tokens_per_sec = (batch_size * model_args.max_seq_len) / step_time
            
            print(f"Step {step:6d} | "
                  f"Loss: {lm_loss_accum:.4f} | "
                  f"LB Loss: {lb_loss_accum:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Expert: {current_expert} | "
                  f"Tokens/s: {tokens_per_sec:.0f}")
            
            if config["logging"]["use_wandb"] and HAS_WANDB:
                wandb.log({
                    "step": step,
                    "loss": lm_loss_accum,
                    "load_balance_loss": lb_loss_accum,
                    "learning_rate": lr,
                    "active_expert": current_expert,
                    "tokens_per_sec": tokens_per_sec,
                    "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3,
                })
        
        # Evaluation
        if step % config["training"]["eval_every"] == 0 and step > 0:
            print(f"\n📊 Evaluating at step {step}...")
            
            if train_all_experts:
                # In all-experts mode, just validate with all experts
                val_loss = evaluate(model, val_loader, device, config, tokenizer, active_expert=None)
                print(f"Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}\n")
                
                if config["logging"]["use_wandb"] and HAS_WANDB:
                    wandb.log({"val_loss": val_loss, "val_perplexity": math.exp(val_loss)})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, step, config, expert_idx="best")
            else:
                # In sequential mode, validate both per-expert and all-experts
                val_loss_active = evaluate(model, val_loader, device, config, tokenizer, active_expert=current_expert)
                print(f"Val Loss (Expert {current_expert}): {val_loss_active:.4f} | Perplexity: {math.exp(val_loss_active):.2f}")
                
                val_loss_all = evaluate(model, val_loader, device, config, tokenizer, active_expert=None)
                print(f"Val Loss (All Experts): {val_loss_all:.4f} | Perplexity: {math.exp(val_loss_all):.2f}\n")
                
                if config["logging"]["use_wandb"] and HAS_WANDB:
                    wandb.log({
                        f"val_loss_expert_{current_expert}": val_loss_active, 
                        f"val_perplexity_expert_{current_expert}": math.exp(val_loss_active),
                        "val_loss_all_experts": val_loss_all,
                        "val_perplexity_all_experts": math.exp(val_loss_all)
                    })
                
                # Save best based on active expert performance
                if val_loss_active < best_val_loss:
                    best_val_loss = val_loss_active
                    save_checkpoint(model, optimizer, step, config, expert_idx="best")
        
        # Save checkpoint
        if step % config["training"]["save_every"] == 0 and step > 0:
            save_checkpoint(model, optimizer, step, config, expert_idx=current_expert)
        
        step += 1
    
    # Final save
    save_checkpoint(model, optimizer, step, config, expert_idx="final")
    
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.finish()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()