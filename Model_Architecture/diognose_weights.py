def diagnose_checkpoint(checkpoint_path, config, device):
    """Diagnose if the checkpoint has actually learned anything"""
    import torch
    import numpy as np
    
    print("🔍 Diagnosing checkpoint...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create model with fixes
    from model import ismail, ModelArgs
    args = ModelArgs(**config["model"])
    model = ismail(args).to(device)
    
    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    
    # Check expert weight statistics
    print("\n📊 Expert Weight Analysis:")
    for name, param in model.named_parameters():
        if "experts" in name and "routed" in name:
            expert_idx = int(name.split("experts.")[1].split(".")[0])
            weight_std = param.std().item()
            weight_mean = param.mean().item()
            print(f"  Expert {expert_idx}: mean={weight_mean:.6f}, std={weight_std:.6f}")
    
    # Check router weights
    print("\n🎯 Router Weight Analysis:")
    for name, param in model.named_parameters():
        if "gate.weight" in name:
            weight_std = param.std().item()
            weight_range = (param.max() - param.min()).item()
            print(f"  {name}: std={weight_std:.6f}, range={weight_range:.6f}")
            
            # Check if router has learned to differentiate
            router_weights = param.detach().cpu()
            correlations = []
            for i in range(min(5, router_weights.shape[0])):
                for j in range(i+1, min(5, router_weights.shape[0])):
                    corr = torch.corrcoef(torch.stack([router_weights[i], router_weights[j]]))[0,1].item()
                    correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                print(f"  Average correlation between experts: {avg_correlation:.4f}")
                if avg_correlation < 0.9:
                    print("  ✅ Experts show differentiation (good!)")
                else:
                    print("  ⚠️ Experts are too similar (potential issue)")
    
    # Test with random input
    print("\n🎲 Testing with random input:")
    with torch.no_grad():
        test_input = torch.randint(0, config["model"]["vocab_size"], (2, 128)).to(device)
        output = model(test_input)
        if isinstance(output, tuple):
            output = output[0]
        
        # Check output statistics
        output_std = output.std().item()
        output_mean = output.mean().item()
        print(f"  Output mean: {output_mean:.6f}, std: {output_std:.6f}")
        
        if output_std > 0.1:
            print("  ✅ Model produces varied outputs")
        else:
            print("  ⚠️ Model outputs might be collapsed")
    
    return ckpt["step"]


if __name__== "__main__":
    import json

    # Load config
    with open("./config.json", "r") as f:
        config = json.load(f)

    # Run diagnostic
    current_step = diagnose_checkpoint("./checkpoints/your_latest_checkpoint.pt", config, "cuda")
    print(f"\n📍 Current step: {current_step}")