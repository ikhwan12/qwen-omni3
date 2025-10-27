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
from qwen_omni_utils import process_mm_info


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


def transcribe_audio_official_method(audio_path: str, model, processor, config: Dict, debug: bool = False) -> str:
    """Transcribe audio using the official Qwen3-Omni demo approach"""
    if debug:
        print(f"  Processing with official method: {audio_path}")
    
    language = config['dataset']['language']
    
    # Select appropriate prompt based on official Qwen3-Omni documentation
    if language.lower() == 'chinese':
        prompt = "请将这段中文语音转换为纯文本。"
    else:
        prompt = "Transcribe the audio into text."
    
    if debug:
        print(f"    Language: {language}, Prompt: {prompt}")
    
    # Validate audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Create messages in the official format
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
            print(f"    Messages formatted successfully")
        
        # Apply chat template (official method)
    text = processor.apply_chat_template(
        messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        if debug:
            print(f"    Chat template applied successfully")
        
        # Process multimodal info using official utility
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        
        if debug:
            print(f"    Multimodal info processed: audios={audios is not None}, images={images is not None}, videos={videos is not None}")
        
        # Create inputs using official method
        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=True
        )
        
        # Move to device as in official demo
        inputs = inputs.to(model.device).to(model.dtype)
        
        if debug:
            print(f"    Inputs prepared successfully")
            print(f"    Input keys: {list(inputs.keys())}")
            print(f"    Moved to device: {model.device}, dtype: {model.dtype}")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    print(f"    {key} shape: {value.shape}")
                    print(f"    {key} device: {value.device if hasattr(value, 'device') else 'N/A'}")
        
        return inputs, text
        
    except Exception as e:
        if debug:
            print(f"    Official method failed: {str(e)}")
        raise e


def transcribe_audio(audio_path: str, model, processor, config: Dict, debug: bool = False) -> str:
    """Transcribe a single audio file using official Qwen3-Omni approach"""
    if debug:
        print(f"  Processing: {audio_path}")
    
    language = config['dataset']['language']
    
    # Select appropriate prompt based on official Qwen3-Omni documentation
    if language.lower() == 'chinese':
        prompt = "请将这段中文语音转换为纯文本。"
    else:
        prompt = "Transcribe the audio into text."
    
    if debug:
        print(f"    Language: {language}, Prompt: {prompt}")
    
    # Try the official method first
    try:
        if debug:
            print(f"    Trying official Qwen3-Omni method")
        
        inputs, original_text = transcribe_audio_official_method(audio_path, model, processor, config, debug)
        
        if debug:
            print(f"    Official method successful - proceeding with generation")
            
    except Exception as e1:
        if debug:
            print(f"    Official method failed: {str(e1)}")
        
        # Fallback to text-only approach
        try:
            if debug:
                print(f"    Trying fallback text-only approach")
            
            # Simple text-only processing as fallback
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
            raise Exception(f"All transcription approaches failed. Official error: {str(e1)}, Fallback error: {str(e_final)}")
    
    # Note: Inputs are already moved to device in official method
    # Skip device movement for official method, do it for fallback only
    if not hasattr(inputs, 'input_ids') or inputs.get('input_ids') is None or inputs['input_ids'].device.type == 'cpu':
        # This is likely the fallback method, so move to device manually
    device = config['model']['device']
        try:
            # Move all tensor inputs to device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(device)
            
            if debug:
                print(f"    Fallback inputs moved to device: {device}")
        except Exception as e:
            if debug:
                print(f"    Failed to move inputs to device: {str(e)}")
            raise
    else:
        if debug:
            print(f"    Inputs already on device (official method)")
    
    # Generate transcription using official Qwen3-Omni parameters
    try:
        with torch.no_grad():
            if debug:
                print(f"    Starting model generation with official parameters...")
            
            # Use official Qwen3-Omni generation parameters
            generated_result = model.generate(
                **inputs,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=32768,
                thinker_do_sample=True,
                thinker_temperature=0.7,
                thinker_top_p=0.8,
                thinker_top_k=20,
                use_audio_in_video=True
            )
            
            if debug:
                print(f"    Model generation completed")
                print(f"    Generated result type: {type(generated_result)}")
                
            # Handle different return formats from official method
            if isinstance(generated_result, tuple) and len(generated_result) >= 2:
                # Official method returns (text_ids, audio) tuple
                generated_ids, audio = generated_result[0], generated_result[1]
                if debug:
                    print(f"    Extracted text_ids and audio from tuple")
                    print(f"    Text IDs type: {type(generated_ids)}")
                    print(f"    Audio result: {audio is not None}")
            elif hasattr(generated_result, 'sequences'):
                # Result is a generation output object
                generated_ids = generated_result.sequences
                audio = None
                if debug:
                    print(f"    Extracted sequences from generation output")
            else:
                # Simple tensor result
                generated_ids = generated_result
                audio = None
                if debug:
                    print(f"    Using direct tensor result")
            
            if hasattr(generated_ids, 'shape'):
                if debug:
                    print(f"    Generated IDs shape: {generated_ids.shape}")
            
    except Exception as e:
        if debug:
            print(f"    Official generation failed: {str(e)}")
            print(f"    Trying simplified generation...")
        
        try:
            # Fallback: Try simpler generation parameters
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            if debug:
                print(f"    Simplified generation successful")
                # Handle tuple case for simplified generation
                if isinstance(generated_ids, tuple):
                    generated_ids = generated_ids[0]
                    if debug:
                        print(f"    Extracted tensor from tuple (simplified)")
                print(f"    Generated IDs shape (simplified): {generated_ids.shape}")
                
        except Exception as e2:
            if debug:
                print(f"    Simplified generation also failed: {str(e2)}")
            
            # Last resort: Try with minimal inputs
            try:
                basic_inputs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask', None)
                }
                basic_inputs = {k: v for k, v in basic_inputs.items() if v is not None}
                
                with torch.no_grad():
                    generated_ids = model.generate(
                        **basic_inputs,
                        max_new_tokens=128,
                        do_sample=False
                    )
                
                if debug:
                    print(f"    Minimal generation successful")
                
            except Exception as e3:
                if debug:
                    print(f"    All generation approaches failed: {str(e3)}")
                raise Exception(f"Model generation failed: Original: {str(e)}, Simplified: {str(e2)}, Minimal: {str(e3)}")
    
    # Decode output using official Qwen3-Omni method
    try:
        if 'input_ids' in inputs:
            input_length = inputs['input_ids'].shape[1]
        else:
            input_length = 0
        
        if debug:
            print(f"    Input length to trim: {input_length}")
            if hasattr(generated_ids, 'sequences'):
                print(f"    Generated IDs sequences shape: {generated_ids.sequences.shape}")
            else:
                print(f"    Generated IDs shape: {generated_ids.shape}")
        
        # Use official decoding method: processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], ...)
        if hasattr(generated_ids, 'sequences'):
            # If it's a generation output object, use sequences attribute
            sequences_to_decode = generated_ids.sequences[:, input_length:]
            if debug:
                print(f"    Using sequences attribute from GenerateDecoderOnlyOutput")
        else:
            # If it's already a tensor, use it directly
            sequences_to_decode = generated_ids[:, input_length:]
            if debug:
                print(f"    Using tensor directly")
        
        if debug:
            print(f"    Sequences to decode shape: {sequences_to_decode.shape}")
            # Show some token IDs for debugging
            if sequences_to_decode.shape[1] > 0:
                sample_tokens = sequences_to_decode[0][:min(10, sequences_to_decode.shape[1])]
                print(f"    Sample token IDs: {sample_tokens.tolist()}")
        
        # Use processor.batch_decode as in official demo
        response = processor.batch_decode(
            sequences_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if debug:
            print(f"    Decoded response length: {len(response)}")
            if response.strip():
                print(f"    Decoded response: '{response[:200]}{'...' if len(response) > 200 else ''}'")
            else:
                print(f"    Decoded response is empty")
        
        # Clean up and return the result
        result = response.strip()
        
        if result:
            if debug:
                print(f"    ✅ Successful transcription: '{result[:100]}{'...' if len(result) > 100 else ''}'")
            return result
        else:
            if debug:
                print(f"    ⚠️  Empty transcription result")
                
                # Try decoding the full sequence as fallback
                try:
                    if hasattr(generated_ids, 'sequences'):
                        full_sequences = generated_ids.sequences
                    else:
                        full_sequences = generated_ids
                    
                    full_response = processor.batch_decode(
                        full_sequences,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0].strip()
                    
                    if full_response:
                        if debug:
                            print(f"    Using full sequence decode: '{full_response[:100]}{'...' if len(full_response) > 100 else ''}'")
                        return full_response
                    else:
                        if debug:
                            print(f"    Full sequence decode also empty")
                
                except Exception as fallback_e:
                    if debug:
                        print(f"    Full sequence decode failed: {str(fallback_e)}")
            
            return "No transcription generated"
            
    except Exception as e:
        if debug:
            print(f"    Decoding failed: {str(e)}")
            import traceback
            print(f"    Traceback:\n{traceback.format_exc()}")
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
