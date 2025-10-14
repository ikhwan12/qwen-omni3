#!/bin/bash

# Run GPU keep-alive as a daemon process
# This version runs in the background and can be controlled

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/gpu_keepalive.pid"
LOG_FILE="$SCRIPT_DIR/gpu_keepalive.log"

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "GPU keep-alive is already running (PID: $(cat "$PID_FILE"))"
            exit 1
        fi
        
        echo "Starting GPU keep-alive daemon..."
        
        # Create the keep-alive script
        cat > "$SCRIPT_DIR/gpu_keepalive_daemon.py" << 'EOF'
import torch
import time
import signal
import sys
import os

def signal_handler(sig, frame):
    print(f'\nReceived signal {sig}, stopping GPU keep-alive...')
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    # Also write to log file if specified
    log_file = os.environ.get('LOG_FILE')
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

log_message("GPU Keep-Alive Daemon Started")
log_message(f"GPU: {torch.cuda.get_device_name(0)}")
log_message(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

try:
    iteration = 0
    while True:
        # Create a small tensor on GPU
        dummy_tensor = torch.randn(100, 100, device='cuda')
        
        # Do a minimal computation to keep GPU busy
        result = torch.matmul(dummy_tensor, dummy_tensor.T)
        
        # Clear the tensors
        del dummy_tensor, result
        
        # Wait a bit before next iteration
        time.sleep(5)
        
        iteration += 1
        
        # Log status every 12 iterations (60 seconds)
        if iteration % 12 == 0:
            memory_used = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            log_message(f"GPU Memory: {memory_used:.1f}MB / {memory_total:.1f}MB (iteration {iteration})")
            
except KeyboardInterrupt:
    log_message("Received keyboard interrupt, stopping...")
except Exception as e:
    log_message(f"Error: {e}")
finally:
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_message("GPU keep-alive daemon stopped")
EOF
        
        # Start the daemon
        cd "$SCRIPT_DIR/.."
        source .venv/bin/activate
        LOG_FILE="$LOG_FILE" nohup python "$SCRIPT_DIR/gpu_keepalive_daemon.py" > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        
        echo "GPU keep-alive daemon started (PID: $(cat "$PID_FILE"))"
        echo "Log file: $LOG_FILE"
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "GPU keep-alive is not running"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "GPU keep-alive process not found, cleaning up PID file"
            rm -f "$PID_FILE"
            exit 1
        fi
        
        echo "Stopping GPU keep-alive daemon (PID: $PID)..."
        kill -TERM "$PID"
        
        # Wait for process to stop
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing GPU keep-alive daemon..."
            kill -KILL "$PID"
        fi
        
        rm -f "$PID_FILE"
        rm -f "$SCRIPT_DIR/gpu_keepalive_daemon.py"
        echo "GPU keep-alive daemon stopped"
        ;;
        
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            PID=$(cat "$PID_FILE")
            echo "GPU keep-alive daemon is running (PID: $PID)"
            if [ -f "$LOG_FILE" ]; then
                echo "Last few log entries:"
                tail -5 "$LOG_FILE"
            fi
        else
            echo "GPU keep-alive daemon is not running"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo
        echo "Commands:"
        echo "  start   - Start GPU keep-alive daemon"
        echo "  stop    - Stop GPU keep-alive daemon"
        echo "  restart - Restart GPU keep-alive daemon"
        echo "  status  - Check daemon status"
        echo
        echo "This script keeps your GPU active to prevent machine shutdown."
        echo "Useful for remote servers or systems that kill inactive GPU processes."
        exit 1
        ;;
esac
