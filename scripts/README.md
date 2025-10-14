# 🛠️ Scripts Folder

This folder contains all installation, testing, and utility scripts for the Qwen3-Omni transcription system.

## 📋 Installation Scripts

### `setup.bat` ⭐ MAIN SETUP
**Purpose:** Complete environment setup with automatic CUDA detection

**Usage:**
```batch
cd scripts
setup.bat
```

**What it does:**
- Creates Python virtual environment
- Detects NVIDIA GPU
- Installs PyTorch with CUDA 12.1/11.8 or CPU version
- Installs all dependencies (transformers, werpy, etc.)

**Options during setup:**
1. CUDA 12.1 (RTX 30xx/40xx, Quadro RTX 8000) - **RECOMMENDED**
2. CUDA 11.8 (older GPUs)
3. CPU only (no GPU)

---

### `fix_cuda.bat` 🔧 CUDA FIX
**Purpose:** Fix PyTorch if CPU version was installed by mistake

**Usage:**
```batch
cd scripts
fix_cuda.bat
```

**When to use:**
- If `test_cuda_quick.bat` shows "CUDA Available: False"
- If you accidentally selected CPU during setup
- If PyTorch version shows "+cpu" instead of "+cu121"

---

## 🧪 Testing Scripts

### `test_cuda_quick.bat` ⚡ QUICK CUDA TEST
**Purpose:** Fast 5-test check if CUDA is working

**Usage:**
```batch
cd scripts
test_cuda_quick.bat
```

**Tests:**
1. Is CUDA available?
2. What CUDA version?
3. What GPU name?
4. Can create CUDA tensors?
5. What PyTorch version?

**Expected output:**
```
CUDA Available: True
CUDA Version: 12.1
GPU: Quadro RTX 8000
PyTorch version: 2.x.x+cu121
```

---

### `check_cuda.py` 📊 FULL CUDA DIAGNOSTIC
**Purpose:** Comprehensive CUDA and configuration check

**Usage:**
```batch
cd scripts
python check_cuda.py
```

**What it shows:**
- PyTorch version and CUDA availability
- GPU name, memory, compute capability
- CUDA tensor operation test
- config.yaml device setting verification
- Compatibility check between config and hardware

---

### `test_model_cuda.py` 🎯 MODEL CUDA TEST
**Purpose:** Test if the Qwen3-Omni model will use CUDA

**Usage:**
```batch
cd scripts
python test_model_cuda.py
```

**What it checks:**
- Model configuration from config.yaml
- Device compatibility (CUDA vs CPU)
- Expected processing speed for your GPU

---

### `test_dataset.py` 📁 DATASET VALIDATION
**Purpose:** Quick test of dataset and configuration

**Usage:**
```batch
cd scripts
python test_dataset.py
```

**What it tests:**
- config.yaml loads correctly
- Dataset directory exists
- Speakers and files are accessible
- Sample files can be read
- Dependencies are installed
- Output directory can be created

---

## 🔧 Utility Scripts

### `prepare_dataset.py` 📋 DATASET PREPARATION
**Purpose:** Validate and analyze L2-ARCTIC dataset structure

**Usage:**
```batch
cd scripts

:: Validate dataset
python prepare_dataset.py --validate

:: Check sample files from specific speaker
python prepare_dataset.py --check_samples --speaker ABA

:: Export dataset info to JSON
python prepare_dataset.py --export_info dataset_info.json
```

**What it does:**
- Scans L2-ARCTIC dataset structure
- Counts audio and transcript files per speaker
- Validates file matching
- Shows sample file contents
- Exports statistics to JSON

---

### `run_pipeline.bat` ▶️ FULL PIPELINE
**Purpose:** Run complete transcription and evaluation pipeline

**Usage:**
```batch
cd scripts
run_pipeline.bat
```

**What it does:**
1. Activates virtual environment
2. Checks dataset exists
3. Runs transcription on all speakers
4. Runs WER evaluation
5. Saves results to outputs/

**Warning:** This can take 12-24 hours for full L2-ARCTIC dataset!

---

## 🚀 Quick Start Workflow

### First Time Setup
```batch
cd scripts
setup.bat                    # Select option 1 for CUDA 12.1
test_cuda_quick.bat         # Verify CUDA is working
python test_dataset.py      # Verify dataset
```

### If CUDA Not Working
```batch
cd scripts
fix_cuda.bat                # Fix PyTorch installation
test_cuda_quick.bat         # Verify it's fixed
```

### Before Transcription
```batch
cd scripts
python check_cuda.py        # Full diagnostic
python prepare_dataset.py --validate    # Validate dataset
```

### Run Transcription
```batch
cd ..                       # Go back to project root
.venv\Scripts\activate.bat
python transcribe.py --speaker ABA      # Test with one speaker
python evaluate_wer.py                  # Evaluate results
```

---

## 📂 File Organization

```
scripts/
├── README.md                    ← You are here!
│
├── Installation:
│   ├── setup.bat               ← Main setup (run this first!)
│   └── fix_cuda.bat            ← Fix CUDA if needed
│
├── Testing:
│   ├── test_cuda_quick.bat     ← Quick CUDA test
│   ├── check_cuda.py           ← Full CUDA diagnostic
│   ├── test_model_cuda.py      ← Model CUDA test
│   └── test_dataset.py         ← Dataset validation
│
├── Utilities:
│   ├── prepare_dataset.py      ← Dataset analysis
│   └── run_pipeline.bat        ← Full pipeline automation
```

---

## 💡 Tips

1. **Always test CUDA after setup:**
   ```batch
   test_cuda_quick.bat
   ```

2. **If CUDA shows False:**
   ```batch
   fix_cuda.bat
   ```

3. **Before long transcription:**
   ```batch
   python check_cuda.py        # Verify CUDA
   python test_dataset.py      # Verify dataset
   ```

4. **Monitor GPU during transcription:**
   ```batch
   nvidia-smi -l 1             # Updates every 1 second
   ```

---

## 🆘 Troubleshooting

**Problem:** Scripts won't run from root directory

**Solution:** Either:
- Run from scripts folder: `cd scripts && setup.bat`
- Or use the launchers in root directory (created automatically)

**Problem:** "File not found" errors

**Solution:** Make sure you're in the right directory:
```batch
cd scripts      # Go into scripts folder
dir             # List files to verify
```

**Problem:** CUDA not detected

**Solution:**
1. Check GPU: `nvidia-smi`
2. Run: `fix_cuda.bat`
3. Verify: `test_cuda_quick.bat`

---

## 📖 See Also

- **Main Documentation:** `../README.md`
- **Quick Start Guide:** `../START_HERE.md`
- **CUDA Installation:** `../INSTALL_PYTORCH_CUDA.md`
- **Configuration:** `../config.yaml`

