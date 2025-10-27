#!/usr/bin/env python3
"""
Script to clear GPU memory and check availability
"""
import os
import signal
import subprocess
import torch

def get_gpu_processes():
    """Get processes using GPU memory"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            processes = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid = parts[0].strip()
                        name = parts[1].strip()
                        memory = parts[2].strip()
                        processes.append({'pid': pid, 'name': name, 'memory': memory})
            return processes
        return []
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def clear_gpu_memory():
    """Clear GPU memory"""
    print("🧹 Clearing GPU Memory...")
    print("=" * 50)
    
    # Step 1: Check current GPU usage
    print("📊 Current GPU Usage:")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    
    # Step 2: Get processes using GPU
    processes = get_gpu_processes()
    if processes:
        print(f"\n🔍 Found {len(processes)} processes using GPU:")
        for proc in processes:
            print(f"  PID: {proc['pid']}, Name: {proc['name']}, Memory: {proc['memory']} MiB")
        
        print(f"\n💀 Killing GPU processes...")
        for proc in processes:
            try:
                pid = int(proc['pid'])
                print(f"  Killing PID {pid} ({proc['name']})...")
                os.kill(pid, signal.SIGTERM)
            except Exception as e:
                print(f"  Failed to kill PID {proc['pid']}: {e}")
    else:
        print("\n✓ No processes found using GPU")
    
    # Step 3: Clear PyTorch cache if available
    try:
        if torch.cuda.is_available():
            print(f"\n🔄 Clearing PyTorch CUDA cache...")
            torch.cuda.empty_cache()
            print("✓ PyTorch cache cleared")
        else:
            print("\n! CUDA not available for PyTorch")
    except Exception as e:
        print(f"! Failed to clear PyTorch cache: {e}")
    
    # Step 4: Check GPU usage after cleanup
    print(f"\n📊 GPU Usage After Cleanup:")
    os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    
    print(f"\n✅ GPU memory cleanup completed!")

def check_memory_requirements():
    """Check memory requirements for different configurations"""
    print("\n💾 Memory Requirements:")
    print("=" * 50)
    print("Qwen3-Omni-30B Model:")
    print("  • Unquantized (bfloat16): ~60GB GPU memory")
    print("  • 4-bit quantized: ~15GB GPU memory") 
    print("  • CPU (float32): ~120GB RAM")
    print("  • CPU (float16): ~60GB RAM")
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🎮 Your GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 60:
            print("  ✅ Sufficient for unquantized model")
        elif gpu_memory >= 15:
            print("  ⚠️  Only sufficient for quantized model")
        else:
            print("  ❌ Insufficient GPU memory - use CPU")
    else:
        print("\n! CUDA not available")
    
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"\n💻 Your RAM: {ram_gb:.1f}GB")
        
        if ram_gb >= 120:
            print("  ✅ Sufficient for CPU inference (float32)")
        elif ram_gb >= 60:
            print("  ⚠️  May work with CPU (float16)")
        else:
            print("  ❌ Insufficient RAM for this model")
    except ImportError:
        print("\n! psutil not available - cannot check RAM")

if __name__ == "__main__":
    print("🔧 GPU Memory Management Tool")
    print("=" * 50)
    
    clear_gpu_memory()
    check_memory_requirements()
    
    print("\n🚀 Recommended Next Steps:")
    print("1. Try GPU again: python debug_step_by_step.py")
    print("2. Use CPU config: python transcribe.py --config config_cpu.yaml --test_single datasets/ABA/wav/arctic_b0519.wav")
    print("3. Check processes: nvidia-smi")
