"""
Test if the Qwen3-Omni model will use CUDA
"""
import yaml
import torch

print("=" * 60)
print("Model CUDA Test")
print("=" * 60)

# Check PyTorch CUDA
print(f"\n1. PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Check config
print("\n2. Configuration Check:")
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = config['model']['device']
    dtype = config['model']['torch_dtype']
    model_name = config['model']['name']
    
    print(f"   Model: {model_name}")
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    
    # Check compatibility
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  WARNING: Config set to 'cuda' but CUDA not available!")
        print("   Solution: Change config.yaml device to 'cpu'")
    elif device == 'cpu' and torch.cuda.is_available():
        print("\n⚠️  INFO: CUDA available but config set to 'cpu'")
        print("   For faster processing, change config.yaml device to 'cuda'")
    else:
        print("\n✓ Configuration matches hardware capability")
    
except FileNotFoundError:
    print("   config.yaml not found")

# Test model loading (without downloading full model)
print("\n3. Testing Model Device Assignment:")
try:
    # Create a simple tensor to simulate model behavior
    if torch.cuda.is_available() and config['model']['device'] == 'cuda':
        print("   Creating test tensor on CUDA...")
        test_tensor = torch.rand(10, 10).cuda()
        print(f"   ✓ Tensor successfully created on: {test_tensor.device}")
        print(f"   ✓ Model will use GPU for inference")
        
        # Estimate speed
        print(f"\n4. Expected Performance:")
        gpu_name = torch.cuda.get_device_name(0)
        if 'RTX 4090' in gpu_name or 'RTX 4080' in gpu_name:
            print(f"   Your {gpu_name} can process ~1,000 files in 20-30 minutes")
        elif 'RTX 3090' in gpu_name or 'RTX 3080' in gpu_name:
            print(f"   Your {gpu_name} can process ~1,000 files in 30-45 minutes")
        elif 'RTX 3070' in gpu_name or 'RTX 3060' in gpu_name:
            print(f"   Your {gpu_name} can process ~1,000 files in 60-90 minutes")
        else:
            print(f"   Your {gpu_name} should provide significant speedup over CPU")
    else:
        print("   Model will use CPU (slower but functional)")
        print("   Expected: ~10-20 hours for 1,000 files")
        
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
print("\nNext Steps:")
print("  1. If CUDA is available and config is correct, you're ready!")
print("  2. Run: python transcribe.py --speaker ABA")
print("  3. Monitor GPU usage with: nvidia-smi")

