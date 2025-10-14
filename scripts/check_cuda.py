"""
Check CUDA availability and PyTorch installation
"""
import sys

def check_cuda():
    """Check if CUDA is available in PyTorch"""
    print("=" * 60)
    print("CUDA & PyTorch Configuration Check")
    print("=" * 60)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch not installed")
        print("  Run: setup.bat")
        return False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"✓ CUDA available: YES")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        # List all GPUs
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Compute Capability: {props.major}.{props.minor}")
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Multi-Processors: {props.multi_processor_count}")
        
        # Check current device
        print(f"\n✓ Current device: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA with a simple operation
        try:
            x = torch.rand(3, 3).cuda()
            y = torch.rand(3, 3).cuda()
            z = x + y
            print("✓ CUDA test successful: tensor operations work")
        except Exception as e:
            print(f"✗ CUDA test failed: {e}")
            return False
        
    else:
        print("✗ CUDA available: NO")
        print("\n  Possible reasons:")
        print("    1. No NVIDIA GPU in system")
        print("    2. PyTorch installed without CUDA support")
        print("    3. CUDA drivers not installed")
        print("\n  To install PyTorch with CUDA:")
        print("    For CUDA 12.1 (RTX 30xx/40xx):")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("    For CUDA 11.8 (older GPUs):")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Check config.yaml device setting
    print("\n" + "-" * 60)
    print("Configuration File Check")
    print("-" * 60)
    
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        device = config['model']['device']
        print(f"config.yaml device setting: {device}")
        
        if device == 'cuda' and not cuda_available:
            print("⚠️  WARNING: config.yaml is set to 'cuda' but CUDA is not available!")
            print("   Change 'device: cuda' to 'device: cpu' in config.yaml")
        elif device == 'cpu' and cuda_available:
            print("ℹ️  INFO: CUDA is available but config.yaml is set to 'cpu'")
            print("   Change 'device: cpu' to 'device: cuda' for faster processing")
        else:
            print("✓ Configuration matches hardware")
    except FileNotFoundError:
        print("config.yaml not found")
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
    
    print("\n" + "=" * 60)
    if cuda_available:
        print("✓ System is ready for GPU-accelerated transcription!")
    else:
        print("ℹ️  System will use CPU (slower but functional)")
    print("=" * 60)
    
    return cuda_available


if __name__ == "__main__":
    has_cuda = check_cuda()
    sys.exit(0 if has_cuda else 1)

