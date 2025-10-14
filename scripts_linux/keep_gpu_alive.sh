#!/bin/bash

# Keep GPU alive with dummy process to prevent machine shutdown
# Useful for remote servers or systems that kill inactive GPU processes

set -e

echo "================================================================================"
echo "GPU Keep-Alive Script"
echo "================================================================================"
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Go to parent directory
cd ..

echo "Checking CUDA availability..."
python -c "
import torch
if not torch.cuda.is_available():
    print('CUDA not available - cannot run GPU keep-alive')
    exit(1)
print(f'CUDA available with {torch.cuda.device_count()} GPU(s)')
"

echo
echo "Starting GPU keep-alive process..."
echo "This will run a minimal dummy process to keep GPU active"
echo "Press Ctrl+C to stop"
echo

# Create a simple GPU keep-alive script
cat > gpu_keepalive.py << 'EOF'
import torch
import time
import signal
import sys

def signal_handler(sig, frame):
    print('\nStopping GPU keep-alive...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("GPU Keep-Alive Started")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("Press Ctrl+C to stop")
print()

# Create a small tensor on GPU and keep it active
try:
    while True:
        # Create a small tensor on GPU
        dummy_tensor = torch.randn(100, 100, device='cuda')
        
        # Do a minimal computation to keep GPU busy
        result = torch.matmul(dummy_tensor, dummy_tensor.T)
        
        # Clear the tensors
        del dummy_tensor, result
        
        # Wait a bit before next iteration
        time.sleep(1)
        
        # Print status every 60 seconds
        if int(time.time()) % 60 == 0:
            memory_used = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            print(f"GPU Memory: {memory_used:.1f}MB / {memory_total:.1f}MB")
            
except KeyboardInterrupt:
    print("\nStopping GPU keep-alive...")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU keep-alive stopped")
EOF

# Run the keep-alive script
python gpu_keepalive.py

# Clean up
rm -f gpu_keepalive.py
