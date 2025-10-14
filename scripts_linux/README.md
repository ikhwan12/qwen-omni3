# Linux Scripts

This folder contains bash scripts for Linux/Unix systems (including WSL on Windows).

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional but recommended)
- Git

## Quick Start

1. **Setup environment:**
   ```bash
   chmod +x *.sh
   ./setup.sh
   ```

2. **Update transformers for Qwen3-Omni support:**
   ```bash
   ./update_transformers.sh
   ```

3. **Download the model:**
   ```bash
   ./download_model.sh
   ```

4. **Run the pipeline:**
   ```bash
   ./run_pipeline.sh
   ```

5. **Keep GPU alive (optional - for remote servers):**
   ```bash
   # Interactive mode (Ctrl+C to stop)
   ./keep_gpu_alive.sh
   
   # Daemon mode (runs in background)
   ./keep_gpu_alive_daemon.sh start
   ./keep_gpu_alive_daemon.sh status
   ./keep_gpu_alive_daemon.sh stop
   ```

## Individual Scripts

### `setup.sh`
- Creates virtual environment
- Detects NVIDIA GPU and installs appropriate PyTorch version
- Installs all required dependencies
- Sets up FlashAttention 2

### `update_transformers.sh`
- Updates transformers library from GitHub to get Qwen3-Omni support
- Installs qwen-omni-utils
- Installs FlashAttention 2

### `download_model.sh`
- Downloads Qwen3-Omni-30B-A3B-Instruct model (~75GB)
- Uses hf_transfer for faster downloads
- Sets cache directory to avoid system drive space issues
- Supports resume on interruption

### `run_pipeline.sh`
- Runs the complete transcription and evaluation pipeline
- Prepares dataset, transcribes audio, evaluates WER
- Saves results to outputs/ directory

### `check_cuda.sh`
- Checks CUDA availability and GPU details
- Shows PyTorch CUDA information
- Displays NVIDIA driver status
- Tests if the model configuration will use CUDA

### `keep_gpu_alive.sh`
- Keeps GPU active with dummy process to prevent machine shutdown
- Useful for remote servers that kill inactive GPU processes
- Minimal GPU usage (small tensor operations)
- Shows GPU memory status every minute

### `keep_gpu_alive_daemon.sh`
- Runs GPU keep-alive as a background daemon process
- Can be controlled with start/stop/restart/status commands
- Logs activity to file
- Perfect for long-running sessions

## Troubleshooting

### Permission Denied
Make scripts executable:
```bash
chmod +x *.sh
```

### CUDA Issues
Check CUDA installation:
```bash
./check_cuda.sh
```

### Download Issues
- Ensure you have enough disk space (75GB+)
- Check internet connection
- The download script supports resume on interruption

### Virtual Environment Issues
Activate manually:
```bash
source ../.venv/bin/activate
```

## Notes

- Scripts automatically activate the virtual environment
- Model cache is set to `$HOME/huggingface_cache` to avoid system drive issues
- All scripts include error handling and will exit on failure
- GPU detection is automatic - CPU fallback is available
