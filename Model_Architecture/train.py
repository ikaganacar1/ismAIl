#!/usr/bin/env python3
"""
Sequential Expert Training Script for MoE on Single GPU
Memory Usage: ~7.2GB (vs 10.9GB for full MoE)
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
import math

# Import your model
from model import ismail, ModelArgs
from model_size import estimate_model_size

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
    """Initialize model and print size estimate"""
    args = ModelArgs(**config["model"])
    
    print("\n" + "="*70)
    print("MODEL INITIALIZATION")
    print("="*70 + "\n")
    
    # Estimate size
    size_info = estimate_model_size(args)
    
    model = ismail(args).to(device)
    
    # Compile for speed (PyTorch 2.0+)
    if config["training"]["compile"]:
        try:
            model = torch.compile(model)
            print("✅ Model compiled with torch.compile()\n")
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
    """Create data loaders"""
    data_cfg = config["data"]
    
    print("\n" + "="*70)
    print("DATA LOADING")
    print("="*70 + "\n")
    
    from data import create_dataloader
    
    train_loader = create_dataloader(
        txt=Path(data_cfg["train_file"]).read_text(encoding="utf-8"),
        args=ModelArgs(**config["model"]),
        stride=data_cfg["stride"],
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = create_dataloader(
        txt=Path(data_cfg["val_file"]).read_text(encoding="utf-8"),
        args=ModelArgs(**config["model"]),
        stride=data_cfg["stride"],
        shuffle=False,
        drop_last=True,
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}\n")
    
    return train_loader, val_loader


def evaluate(model, val_loader, device, config):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # Model returns just logits in eval mode (no lb_loss)
            output = model(input_ids, start_pos=0)
            logits = output if not isinstance(output, tuple) else output[0]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=-1,
            )

            total_loss += loss.item() * target_ids.numel()
            total_tokens += target_ids.numel()

    model.train()
    return total_loss / total_tokens


def save_checkpoint(model, optimizer, step, config, expert_idx=None):
    """Save model checkpoint"""
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint name
    if expert_idx is not None:
        ckpt_name = f"step_{step}_expert_{expert_idx}.pt"
    else:
        ckpt_name = f"step_{step}.pt"
    
    ckpt_path = save_dir / ckpt_name
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"💾 Checkpoint saved: {ckpt_path}")


def train_step(model, batch, device, config, scaler=None):
    """Single training step"""
    input_ids, target_ids = batch
    input_ids = input_ids.to(device, non_blocking=True)
    target_ids = target_ids.to(device, non_blocking=True)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=(config["training"]["dtype"] == "bf16")):
        output = model(input_ids, start_pos=0)

        # Handle model output (tuple in training mode with MoE, single tensor otherwise)
        if isinstance(output, tuple):
            logits, lb_loss = output
        else:
            logits = output
            lb_loss = 0.0

        # Main language modeling loss
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-1,
        )

        # Total loss with load balancing
        total_loss = lm_loss + config["training"].get("lb_loss_coef", 0.01) * lb_loss

    return total_loss, lm_loss, lb_loss if isinstance(lb_loss, float) else lb_loss.item()


def main():
    args = parse_args()
    config = load_config(args)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Wandb setup
    if config["logging"]["use_wandb"] and HAS_WANDB:
        wandb.init(
            project=config["logging"]["project_name"],
            name=config["logging"]["run_name"],
            config=config,
        )
    
    # Model setup
    model, model_args = setup_model(config, device)
    
    # Optimizer setup
    optimizer = setup_optimizer(model, config)
    
    # Data setup
    train_loader, val_loader = load_data(config)
    train_iter = iter(train_loader)
    
    # Training state
    step = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        print(f"✅ Resumed from step {step}\n")
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(config["training"]["dtype"] == "bf16"))
    
    # Expert rotation schedule
    current_expert = 0
    rotation_steps = config["training"]["expert_rotation_steps"]
    
    # Set initial expert
    model.set_active_expert(current_expert)
    print(f"🎯 Training expert {current_expert}/{model_args.n_routed_experts - 1}")
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70 + "\n")
    
    model.train()
    
    while step < config["training"]["total_steps"]:
        step_start = time.time()
        
        # Expert rotation
        if step > 0 and step % rotation_steps == 0:
            current_expert = (current_expert + 1) % model_args.n_routed_experts
            model.set_active_expert(current_expert)
            print(f"\n🔄 Rotating to expert {current_expert}/{model_args.n_routed_experts - 1}")
            
            # Clear gradients after rotation
            optimizer.zero_grad(set_to_none=True)
        
        # Get batch with cycle handling
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Training step with gradient accumulation
        accum_steps = config["training"]["gradient_accumulation_steps"]
        total_loss_accum = 0.0
        lm_loss_accum = 0.0
        lb_loss_accum = 0.0
        
        for accum_step in range(accum_steps):
            # Split batch for micro-batching (if needed)
            # For now, process full batch
            loss, lm_loss, lb_loss = train_step(model, batch, device, config, scaler)
            
            # Normalize for accumulation
            loss = loss / accum_steps
            
            # Backward pass
            if config["training"]["dtype"] == "bf16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss_accum += loss.item()
            lm_loss_accum += lm_loss.item() / accum_steps
            lb_loss_accum += lb_loss.item() / accum_steps
        
        # Gradient clipping
        if config["training"]["grad_clip"] > 0:
            if config["training"]["dtype"] == "bf16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
        
        # Optimizer step
        if config["training"]["dtype"] == "bf16":
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
            tokens_per_sec = (model_args.max_batch_size * model_args.max_seq_len) / step_time
            
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
                    "total_loss": total_loss_accum,
                    "learning_rate": lr,
                    "active_expert": current_expert,
                    "tokens_per_sec": tokens_per_sec,
                    "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3,
                })
        
        # Evaluation
        if step % config["training"]["eval_every"] == 0 and step > 0:
            print(f"\n📊 Evaluating at step {step}...")
            val_loss = evaluate(model, val_loader, device, config)
            print(f"Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}\n")
            
            if config["logging"]["use_wandb"] and HAS_WANDB:
                wandb.log({"val_loss": val_loss, "val_perplexity": math.exp(val_loss)})
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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