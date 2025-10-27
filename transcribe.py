"""
Transcription script using Qwen3-Omni model for L2-ARCTIC dataset
"""
import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List
import torch
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    QWEN3_AVAILABLE = True
except ImportError:
    from transformers import AutoModel as Qwen3OmniMoeForConditionalGeneration
    from transformers import AutoProcessor as Qwen3OmniMoeProcessor
    QWEN3_AVAILABLE = False
    print("Warning: Qwen3-Omni not available in transformers. Please run: update_transformers.bat")

try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    QUANTIZATION_AVAILABLE = False
from tqdm import tqdm
import soundfile as sf


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: Dict):
    """Initialize Qwen3-Omni model and processor"""
    model_name = config['model']['name']
    device = config['model']['device']
    dtype_str = config['model']['torch_dtype']
    
    # Map dtype string to torch dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    # Check quantization settings
    quantization_config = config['model'].get('quantization', {})
    quantization_enabled = quantization_config.get('enabled', False)
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}, Dtype: {dtype_str}")
    
    if quantization_enabled:
        print("Quantization enabled:")
        if quantization_config.get('load_in_4bit', False):
            print("  - 4-bit quantization")
        elif quantization_config.get('load_in_8bit', False):
            print("  - 8-bit quantization")
        print(f"  - Quantization type: {quantization_config.get('bnb_4bit_quant_type', 'nf4')}")
    
    if not QWEN3_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: Qwen3-Omni models not available in current transformers!")
        print("="*70)
        print("\nPlease run: update_transformers.bat")
        print("\nThis will install transformers from GitHub with Qwen3-Omni support.")
        print("="*70)
        raise ImportError("Qwen3-Omni not available. Run update_transformers.bat")
    
    # Setup quantization config if enabled
    bnb_config = None
    if quantization_enabled and QUANTIZATION_AVAILABLE:
        print("Setting up BitsAndBytes quantization...")
        
        # Get compute dtype for quantization
        compute_dtype_str = quantization_config.get('bnb_4bit_compute_dtype', dtype_str)
        compute_dtype = dtype_map.get(compute_dtype_str, torch_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', True),
            load_in_8bit=quantization_config.get('load_in_8bit', False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4')
        )
    elif quantization_enabled and not QUANTIZATION_AVAILABLE:
        print("Warning: Quantization enabled but bitsandbytes not available. Install with: pip install bitsandbytes")
        print("Proceeding without quantization...")
    
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Prepare model loading arguments
    model_kwargs = {
        'trust_remote_code': True
    }
    
    # Add quantization config if available
    if bnb_config is not None:
        model_kwargs['quantization_config'] = bnb_config
        # When using quantization, let transformers handle device placement
        model_kwargs['device_map'] = 'auto'
    else:
        model_kwargs['torch_dtype'] = torch_dtype
        model_kwargs['device_map'] = device
    
    # Use Qwen3OmniMoeForConditionalGeneration for Qwen3-Omni-30B-A3B-Instruct
    # Requires transformers from GitHub: pip install git+https://github.com/huggingface/transformers
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    return model, processor


def find_audio_files_l2arctic(data_dir: str, wav_subdir: str = "wav") -> List[Dict]:
    """
    Find all audio files in the L2-ARCTIC dataset
    Structure: {speaker}/wav/arctic_aXXXX.wav
    Returns list of dicts with audio_path and metadata
    """
    data_path = Path(data_dir)
    audio_files = []
    
    # Get all speaker directories (exclude non-speaker folders)
    speaker_dirs = [d for d in data_path.iterdir() 
                   if d.is_dir() and not d.name.startswith('.') 
                   and d.name not in ['l2arctic_audio']]
    
    print(f"Found {len(speaker_dirs)} speakers")
    
    for speaker_dir in speaker_dirs:
        wav_dir = speaker_dir / wav_subdir
        if not wav_dir.exists():
            continue
        
        # Get all wav files for this speaker
        for wav_file in sorted(wav_dir.glob("*.wav")):
            audio_files.append({
                'audio_path': str(wav_file),
                'relative_path': str(wav_file.relative_to(data_path)),
                'speaker': speaker_dir.name,
                'filename': wav_file.stem
            })
    
    print(f"Found {len(audio_files)} audio files across all speakers")
    return audio_files


def transcribe_audio(audio_path: str, model, processor, config: Dict, debug: bool = False) -> str:
    """Transcribe a single audio file"""
    if debug:
        print(f"  Processing: {audio_path}")
    
    language = config['dataset']['language']
    
    # Select appropriate prompt based on official Qwen3-Omni documentation
    # https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
    if language.lower() == 'chinese':
        prompt = "请将这段中文语音转换为纯文本。"
    else:
        prompt = "Transcribe the audio into text."
    
    if debug:
        print(f"    Language: {language}, Prompt: {prompt}")
        
    # Check if audio file exists and is readable
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if debug:
        print(f"    Audio file exists: {os.path.getsize(audio_path)} bytes")
    
    # Prepare message in Qwen3-Omni format (audio first, then text)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    if debug:
        print(f"    Prepared messages with audio and text")
    
    # Process messages
    try:
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        if debug:
            print(f"    Applied chat template successfully")
    except Exception as e:
        if debug:
            print(f"    Failed to apply chat template: {str(e)}")
        raise
    
    audios = []
    for message in messages:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(ele["audio"])
    
    if debug:
        print(f"    Found {len(audios)} audio files to process")
    
    # Prepare inputs
    try:
        inputs = processor(
            text=[text],
            audios=audios,
            return_tensors="pt",
            padding=True
        )
        if debug:
            print(f"    Processor inputs prepared successfully")
    except Exception as e:
        if debug:
            print(f"    Failed to prepare inputs: {str(e)}")
        raise
    
    # Move to device
    device = config['model']['device']
    try:
        inputs = inputs.to(device)
        if debug:
            print(f"    Inputs moved to device: {device}")
    except Exception as e:
        if debug:
            print(f"    Failed to move inputs to device: {str(e)}")
        raise
    
    # Generate transcription
    try:
        with torch.no_grad():
            if debug:
                print(f"    Starting model generation...")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256
            )
            if debug:
                print(f"    Model generation completed")
    except Exception as e:
        if debug:
            print(f"    Failed during model generation: {str(e)}")
        raise
    
    # Trim input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen3-Omni")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override dataset directory from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Process only specific speaker (e.g., ABA, ASI)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )
    parser.add_argument(
        "--test_single",
        type=str,
        default=None,
        help="Test transcription on a single audio file first"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir
    if args.output_dir:
        config['dataset']['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = Path(config['dataset']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable debug mode if requested
    debug_mode = args.debug
    if debug_mode:
        print("Debug mode enabled - verbose output will be shown")
    
    # Setup model
    print("Setting up model...")
    try:
        model, processor = setup_model(config)
        print("✓ Model loaded successfully")
    except Exception as e:
        import traceback
        print(f"✗ Failed to load model: {str(e)}")
        print(f"Full error traceback:\n{traceback.format_exc()}")
        return
    
    # Find audio files
    print(f"\nSearching for audio files in: {config['dataset']['data_dir']}")
    wav_subdir = config['dataset'].get('wav_subdir', 'wav')
    audio_files = find_audio_files_l2arctic(config['dataset']['data_dir'], wav_subdir)
    
    # Filter by speaker if specified
    if args.speaker:
        audio_files = [af for af in audio_files if af['speaker'] == args.speaker]
        print(f"Filtering for speaker: {args.speaker} ({len(audio_files)} files)")
    
    if not audio_files:
        print("No audio files found! Please check the data_dir path.")
        return
    
    # Test single file if requested
    if args.test_single:
        test_file = args.test_single
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return
        
        print(f"\nTesting transcription on: {test_file}")
        try:
            test_result = transcribe_audio(test_file, model, processor, config, debug=debug_mode)
            print(f"✓ Test transcription successful: {test_result}")
            return
        except Exception as e:
            import traceback
            print(f"✗ Test transcription failed: {str(e)}")
            print(f"Full error traceback:\n{traceback.format_exc()}")
            return
    
    # Transcribe all audio files
    results = []
    print(f"\nTranscribing {len(audio_files)} audio files...")
    
    for audio_info in tqdm(audio_files, desc="Transcribing"):
        try:
            transcription = transcribe_audio(
                audio_info['audio_path'],
                model,
                processor,
                config,
                debug=debug_mode
            )
            
            result = {
                'audio_path': audio_info['audio_path'],
                'relative_path': audio_info['relative_path'],
                'speaker': audio_info['speaker'],
                'filename': audio_info['filename'],
                'transcription': transcription
            }
            results.append(result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\nError processing {audio_info['audio_path']}: {str(e)}")
            print(f"Full error traceback:\n{error_details}")
            results.append({
                'audio_path': audio_info['audio_path'],
                'relative_path': audio_info['relative_path'],
                'speaker': audio_info['speaker'],
                'filename': audio_info['filename'],
                'transcription': '',
                'error': str(e),
                'error_traceback': error_details
            })
    
    # Save results
    output_file = output_dir / "transcriptions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranscriptions saved to: {output_file}")
    print(f"Successfully transcribed: {sum(1 for r in results if 'error' not in r)} / {len(results)} files")
    
    # Save per-speaker summary
    speakers = {}
    for r in results:
        speaker = r['speaker']
        if speaker not in speakers:
            speakers[speaker] = {'total': 0, 'success': 0}
        speakers[speaker]['total'] += 1
        if 'error' not in r:
            speakers[speaker]['success'] += 1
    
    print("\nPer-speaker summary:")
    for speaker, stats in sorted(speakers.items()):
        print(f"  {speaker}: {stats['success']}/{stats['total']} files")


if __name__ == "__main__":
    main()
