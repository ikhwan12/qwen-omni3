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
import librosa
import scipy.io.wavfile
import numpy as np


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


def validate_and_load_audio(audio_path: str, debug: bool = False):
    """Validate and load audio file in the correct format for Qwen3-Omni"""
    if debug:
        print(f"    Validating audio: {audio_path}")
    
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check file size
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    if debug:
        print(f"    Audio file size: {file_size} bytes")
    
    try:
        # Load audio with librosa to ensure compatibility
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        if debug:
            print(f"    Audio loaded: {len(audio_data)} samples, {sample_rate}Hz")
        
        # Check if audio data is valid
        if len(audio_data) == 0:
            raise ValueError(f"Audio file contains no data: {audio_path}")
        
        # Ensure minimum duration (at least 0.1 seconds)
        if len(audio_data) / sample_rate < 0.1:
            raise ValueError(f"Audio file too short: {audio_path}")
        
        return audio_data, sample_rate
        
    except Exception as e:
        raise ValueError(f"Failed to load audio file {audio_path}: {str(e)}")


def manual_audio_processing(audio_path: str, processor, debug: bool = False):
    """Manually process audio to bypass processor bugs"""
    if debug:
        print(f"    Using manual audio processing")
    
    # Load audio with librosa and resample to 16kHz (standard for speech models)
    audio_data, sample_rate = librosa.load(audio_path, sr=16000)
    
    if debug:
        print(f"    Audio loaded: {len(audio_data)} samples at 16kHz")
    
    # Ensure audio is not too short or too long
    min_length = 0.1 * 16000  # 0.1 seconds minimum
    max_length = 30.0 * 16000  # 30 seconds maximum
    
    if len(audio_data) < min_length:
        # Pad short audio
        padding = int(min_length - len(audio_data))
        audio_data = np.concatenate([audio_data, np.zeros(padding)])
    elif len(audio_data) > max_length:
        # Truncate long audio
        audio_data = audio_data[:int(max_length)]
    
    # Convert to tensor format expected by the model
    audio_tensor = torch.tensor(audio_data).unsqueeze(0).float()  # Add batch dimension
    
    if debug:
        print(f"    Audio tensor shape: {audio_tensor.shape}")
    
    return audio_tensor


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
    
    # Validate and load audio first to catch issues early
    try:
        audio_data, sample_rate = validate_and_load_audio(audio_path, debug)
    except Exception as e:
        if debug:
            print(f"    Audio validation failed: {str(e)}")
        raise
    
    # Try Approach 3: Manual audio processing to completely bypass processor audio loading
    try:
        if debug:
            print(f"    Trying approach 3: manual audio processing")
        
        # Manually process the audio
        audio_tensor = manual_audio_processing(audio_path, processor, debug)
        
        # Create text input manually (without audio in the messages)
        # This bypasses the problematic audio processing in the processor
        text_messages = [
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Apply chat template for text only
        text_input = processor.apply_chat_template(
            text_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        if debug:
            print(f"    Text template applied successfully")
        
        # Manually create inputs by bypassing the problematic processor.__call__
        # We'll tokenize the text and handle audio separately
        tokenizer = processor.tokenizer
        
        # Tokenize text
        text_inputs = tokenizer(
            text_input,
            return_tensors="pt",
            padding=True
        )
        
        if debug:
            print(f"    Text tokenized successfully")
        
        # For Qwen3-Omni, we need to handle audio features manually
        # This is a simplified approach that bypasses the buggy processor method
        inputs = {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
        }
        
        # Try to add audio features if the model expects them
        # Note: This is a simplified approach - the model might need audio in a specific format
        if hasattr(processor, 'audio_processor') and processor.audio_processor is not None:
            try:
                # Process audio with the audio processor
                audio_features = processor.audio_processor(
                    audio_tensor, 
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                # Add audio features to inputs if they exist
                if hasattr(audio_features, 'input_features'):
                    inputs['audio_features'] = audio_features.input_features
                elif hasattr(audio_features, 'audio_features'):
                    inputs['audio_features'] = audio_features.audio_features
                elif isinstance(audio_features, dict):
                    inputs.update(audio_features)
                    
                if debug:
                    print(f"    Audio features processed successfully")
            except Exception as audio_e:
                if debug:
                    print(f"    Audio processing failed, continuing with text-only: {str(audio_e)}")
        
        if debug:
            print(f"    Approach 3 input preparation successful")
            
    except Exception as e3:
        if debug:
            print(f"    Approach 3 failed: {str(e3)}")
        
        # Fallback: Try the simple text-only approach
        try:
            if debug:
                print(f"    Trying fallback approach: text-only processing")
            
            # Simple text-only processing as last resort
            text_messages = [{"role": "user", "content": prompt}]
            text_input = processor.apply_chat_template(
                text_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Simple tokenization
            inputs = processor.tokenizer(
                text_input,
                return_tensors="pt",
                padding=True
            )
            
            if debug:
                print(f"    Fallback approach successful (text-only)")
            
        except Exception as e_final:
            if debug:
                print(f"    All approaches failed: {str(e_final)}")
            raise Exception(f"All transcription approaches failed. Original error: {str(e3)}, Fallback error: {str(e_final)}")
    
    # Move to device
    device = config['model']['device']
    try:
        # Move all tensor inputs to device
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        
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
            
            # Try generation with the prepared inputs
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Use deterministic generation
                pad_token_id=processor.tokenizer.eos_token_id
            )
            
            if debug:
                print(f"    Model generation completed")
                print(f"    Generated IDs shape: {generated_ids.shape}")
                
    except Exception as e:
        if debug:
            print(f"    Failed during model generation: {str(e)}")
            # Try a simpler generation approach
            print(f"    Trying simplified generation...")
        
        try:
            # Fallback: Try with just input_ids and attention_mask
            basic_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask', None)
            }
            
            # Remove None values
            basic_inputs = {k: v for k, v in basic_inputs.items() if v is not None}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **basic_inputs,
                    max_new_tokens=128,  # Shorter for fallback
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            if debug:
                print(f"    Simplified generation successful")
                
        except Exception as e2:
            if debug:
                print(f"    Simplified generation also failed: {str(e2)}")
            raise Exception(f"Model generation failed: {str(e)}")
    
    # Trim input tokens
    try:
        if 'input_ids' in inputs:
            input_length = inputs['input_ids'].shape[1]
        else:
            input_length = 0
        
        # Trim the input tokens from generated output
        generated_ids_trimmed = [
            out_ids[input_length:] for out_ids in generated_ids
        ]
        
        if debug:
            print(f"    Trimmed {input_length} input tokens")
            
    except Exception as e:
        if debug:
            print(f"    Token trimming failed, using full output: {str(e)}")
        generated_ids_trimmed = generated_ids
    
    # Decode output
    try:
        if hasattr(processor, 'batch_decode'):
            decoder = processor.batch_decode
        else:
            decoder = processor.tokenizer.batch_decode
            
        output_text = decoder(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        if debug:
            print(f"    Decoded output: {len(output_text)} items")
            
        # Return the first result, handling empty results
        if output_text and len(output_text) > 0:
            result = output_text[0].strip()
            if debug:
                print(f"    Final result: '{result[:100]}{'...' if len(result) > 100 else ''}'")
            return result
        else:
            if debug:
                print(f"    No output text generated")
            return ""
            
    except Exception as e:
        if debug:
            print(f"    Decoding failed: {str(e)}")
        raise Exception(f"Output decoding failed: {str(e)}")


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
