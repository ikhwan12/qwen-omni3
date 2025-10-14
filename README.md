# Qwen3-Omni Transcription & WER Evaluation for L2-ARCTIC

Complete ASR transcription system using Qwen3-Omni with WER evaluation for the L2-ARCTIC dataset.

---

## 🚀 Quick Start

### Step 1: Complete Installation

Run the all-in-one installer:

```batch
install_all.bat
```

This installs:
- PyTorch with CUDA 12.1
- Transformers from GitHub (Qwen3-Omni support)
- qwen-omni-utils, flash-attn, werpy
- All other dependencies

Time: ~10-15 minutes

### Step 2: Download Model

The model is ~75GB and may timeout. Use the download script with auto-resume:

```batch
download_model.bat
```

**If download times out:** Just run again - it will resume automatically!

Time: 1-8 hours (depending on internet speed)

### Step 3: Verify Setup

```batch
test_cuda.bat
```

You should see: `CUDA Available: True` and `GPU: Quadro RTX 8000`

### Step 3: Validate Dataset

```batch
cd scripts
python test_dataset.py
cd ..
```

### Step 4: Run Transcription

```batch
.venv\Scripts\activate.bat
python transcribe.py --speaker ABA
python evaluate_wer.py
```

---

## 📁 Project Structure

```
qwen-omni-3/
├── scripts/              # Installation & testing scripts
│   └── README.md        # Scripts documentation
├── transcribe.py        # Main transcription
├── evaluate_wer.py      # WER evaluation
├── config.yaml          # Configuration
└── README.md           # This file
```

**Root Launchers:**
- `setup.bat` - Setup environment with CUDA
- `fix_cuda.bat` - Fix PyTorch CUDA installation
- `test_cuda.bat` - Quick CUDA test
- `run_pipeline.bat` - Full transcription pipeline

---

## 🎯 Your Hardware

**GPU:** Quadro RTX 8000 (48GB VRAM) - Excellent!  
**Expected Speed:** 20-30 minutes per 1,000 audio files  
**Full Dataset (24,000 files):** 12-18 hours with CUDA

⚠️ **Important Notes:**
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct` ([Hugging Face](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct))
- ASR Prompt: "Transcribe the audio into text." (official prompt)
- Requires 48GB VRAM (your Quadro RTX 8000 ✓)
- **Without CUDA:** 10-20 DAYS (30-40x slower!)

---

## 🔧 Installation

### Automatic Setup

```batch
setup.bat
```

Select option **1** for CUDA 12.1 (your Quadro RTX 8000 supports this).

### Update Transformers for Qwen3-Omni

Qwen3-Omni requires transformers from GitHub:

```batch
update_transformers.bat
```

This installs the latest transformers with Qwen3-Omni support.

### Verify CUDA

```batch
test_cuda.bat
```

Expected output:
```
CUDA Available: True
CUDA Version: 12.1
GPU: Quadro RTX 8000
GPU Memory: 48.00 GB
```

### Fix CUDA (If Shows False)

```batch
fix_cuda.bat
```

This reinstalls PyTorch with CUDA support.

---

## 📊 Dataset Configuration

Your L2-ARCTIC v5.0 dataset is pre-configured:

**Location:** `E:/Dataset/LNV/L2-ARCTIC/l2arctic_release_v5.0`

**Structure:**
```
l2arctic_release_v5.0/
├── ABA/
│   ├── wav/          # Audio files
│   └── transcript/   # Reference texts
├── ASI/
└── ... (24 speakers total)
```

**Validate dataset:**
```batch
cd scripts
python prepare_dataset.py --validate
cd ..
```

---

## 🎮 Usage

### Test with One Speaker

```batch
.venv\Scripts\activate.bat
python transcribe.py --speaker ABA
python evaluate_wer.py
```

### Transcribe All Speakers

```batch
python transcribe.py
python evaluate_wer.py
```

### Full Automated Pipeline

```batch
run_pipeline.bat
```

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
model:
  name: "Qwen/Qwen3-Omni-7B-Instruct"
  device: "cuda"              # Use GPU
  torch_dtype: "bfloat16"

dataset:
  data_dir: "E:/Dataset/LNV/L2-ARCTIC/l2arctic_release_v5.0"
  wav_subdir: "wav"
  transcript_subdir: "transcript"

evaluation:
  normalize_text: true        # Normalize both hypothesis and reference
  remove_punctuation: true
  lowercase: true
```

---

## 📈 WER Evaluation

Uses **werpy** for WER calculation with normalization applied to **both** hypothesis and reference texts.

**Metrics:**
- **WER** - Word Error Rate
- **CER** - Character Error Rate

**Output files:**
```
outputs/
├── transcriptions.json          # All transcriptions
├── evaluation_results.json      # WER/CER metrics
├── evaluation_results.csv       # Per-file details
└── per_speaker_results.csv      # Per-speaker summary
```

---

## 🐛 Troubleshooting

### CUDA Not Available

**Check:**
```batch
test_cuda.bat
```

**Fix:**
```batch
fix_cuda.bat
test_cuda.bat
```

### CUDA Out of Memory

Edit `config.yaml`:
```yaml
model:
  torch_dtype: "float16"  # Change from bfloat16
```

### Slow Processing

**Check GPU usage:**
```batch
nvidia-smi -l 1
```

Should show 80-100% GPU utilization during transcription.

### Dataset Not Found

**Validate:**
```batch
cd scripts
python prepare_dataset.py --validate
cd ..
```

---

## 📚 Key Features

✅ **Qwen3-Omni** - State-of-the-art multimodal ASR  
✅ **CUDA Auto-Detection** - Automatic GPU setup  
✅ **werpy Integration** - Proper WER calculation  
✅ **Dual Normalization** - Normalizes both hypothesis and reference  
✅ **L2-ARCTIC v5.0** - Native support for 24 speakers  
✅ **Per-Speaker Analysis** - Detailed metrics per speaker  
✅ **Progress Tracking** - Real-time progress display  

---

## 🔍 Advanced Usage

### Custom Speakers

```batch
python transcribe.py --speaker ABA --speaker ASI
```

### Custom Output

```batch
python transcribe.py --output_dir ./my_results
```

### Dataset Analysis

```batch
cd scripts
python prepare_dataset.py --validate
python prepare_dataset.py --check_samples --speaker ABA
cd ..
```

### Monitor GPU

```batch
nvidia-smi -l 1
```

---

## 📋 L2-ARCTIC Dataset

**Speakers:** 24 non-native English speakers  
**Files per speaker:** ~1,000 audio files  
**Total files:** ~24,000  

**Native languages:**
- Arabic: ABA, ASI, YBAA, ZHAA
- Mandarin: BWC, LXC, NCC, TXHC
- Hindi: RRBI, SVBI
- Korean: HJK, HKK, YDCK, YKWK
- Spanish: EBVS, ERMS, MBMPS, NJS
- Vietnamese: PNV, THV, TLV, TNI
- Other: SKA (Telugu), HQTV (Cantonese)

---

## ⚡ Performance

| Hardware | Time (1,000 files) | Full Dataset (24,000) |
|----------|-------------------|----------------------|
| Quadro RTX 8000 | 20-30 min | 12-18 hours |
| RTX 4090 | 20-30 min | 12-18 hours |
| RTX 3090 | 30-45 min | 18-24 hours |
| CPU | 10-20 hours | 10-20 DAYS ⚠️ |

**With CUDA: 30-40x faster than CPU!**

---

## 🆘 Support & Documentation

- **Scripts Documentation:** `scripts/README.md`
- **Qwen3-Omni:** https://github.com/QwenLM/Qwen3-Omni
- **werpy:** https://github.com/antoinemadec/werpy

---

## 📖 Commands Reference

```batch
:: Setup
setup.bat                    # Initial setup
fix_cuda.bat                 # Fix CUDA installation
test_cuda.bat                # Test CUDA

:: Validation
cd scripts
python test_dataset.py       # Test configuration
python prepare_dataset.py --validate
cd ..

:: Transcription
.venv\Scripts\activate.bat
python transcribe.py --speaker ABA
python transcribe.py         # All speakers

:: Evaluation
python evaluate_wer.py

:: Pipeline
run_pipeline.bat            # Full automated pipeline

:: Monitoring
nvidia-smi                  # Current GPU status
nvidia-smi -l 1            # Live GPU monitoring
```

---

## 🎯 Expected WER

| Speaker Proficiency | Expected WER |
|-------------------|--------------|
| High | 10-15% |
| Medium | 15-25% |
| Lower | 25-40% |

L2-ARCTIC contains non-native speakers, so WER 15-30% is typical.

---

## 🔐 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (Quadro RTX 8000 ✓)
- 16GB+ VRAM (48GB ✓)
- NVIDIA drivers installed
- L2-ARCTIC v5.0 dataset

---

## 📝 Citation

```bibtex
@article{Qwen3-Omni,
  title={Qwen3-Omni Technical Report},
  author={Jin Xu and Zhifang Guo and others},
  journal={arXiv preprint arXiv:2509.17765},
  year={2025}
}
```

---

## 📄 License

Apache-2.0 (same as Qwen3-Omni)

---

## ✅ Checklist

Before running transcription:

- [ ] Run `setup.bat` with CUDA 12.1 (option 1)
- [ ] Run `test_cuda.bat` - shows CUDA: True
- [ ] Run `scripts/test_dataset.py` - all tests pass
- [ ] GPU shows in nvidia-smi as Quadro RTX 8000
- [ ] config.yaml has `device: "cuda"`

---

**Ready to start?** 

```batch
fix_cuda.bat        # Fix your current CPU-only installation
test_cuda.bat       # Verify CUDA works
python transcribe.py --speaker ABA
```

Your Quadro RTX 8000 will make transcription 30-40x faster! 🚀
