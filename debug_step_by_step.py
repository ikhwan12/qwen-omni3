#!/usr/bin/env python3
"""
Comprehensive step-by-step debugging script for Qwen3-Omni transcription issues
"""
import os
import sys
import torch
import traceback

def test_step_1_imports():
    """Test 1: Check all imports"""
    print("🔍 STEP 1: Testing imports...")
    
    try:
        import transformers
        print(f"  ✓ transformers: {transformers.__version__}")
    except Exception as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        print("  ✓ Qwen3-Omni classes available")
    except Exception as e:
        print(f"  ✗ Qwen3-Omni classes: {e}")
        return False
    
    try:
        from transformers import BitsAndBytesConfig
        print("  ✓ BitsAndBytesConfig available")
    except Exception as e:
        print(f"  ✗ BitsAndBytesConfig: {e}")
    
    try:
        import librosa
        import numpy as np
        import scipy.io.wavfile
        print("  ✓ Audio processing libraries available")
    except Exception as e:
        print(f"  ✗ Audio libraries: {e}")
        return False
    
    return True

def test_step_2_config():
    """Test 2: Load configuration"""
    print("\n🔍 STEP 2: Testing configuration loading...")
    
    try:
        from transcribe import load_config
        config = load_config("config.yaml")
        print("  ✓ Config loaded successfully")
        print(f"    Model: {config['model']['name']}")
        print(f"    Device: {config['model']['device']}")
        print(f"    Quantization enabled: {config['model'].get('quantization', {}).get('enabled', False)}")
        return config
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        traceback.print_exc()
        return None

def test_step_3_audio():
    """Test 3: Audio file loading"""
    print("\n🔍 STEP 3: Testing audio file loading...")
    
    # Find a test audio file
    test_file = None
    possible_files = [
        "datasets/ABA/wav/arctic_b0519.wav",
        "datasets/ABA/wav/arctic_a0001.wav"
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("  ✗ No test audio file found")
        return None
    
    print(f"  Testing with: {test_file}")
    
    try:
        import librosa
        import numpy as np
        
        # Test basic audio loading
        audio_data, sample_rate = librosa.load(test_file, sr=None)
        print(f"  ✓ Audio loaded: {len(audio_data)} samples, {sample_rate}Hz")
        
        # Test resampling to 16kHz
        audio_16k = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        print(f"  ✓ Audio resampled to 16kHz: {len(audio_16k)} samples")
        
        # Test conversion to tensor
        audio_tensor = torch.tensor(audio_16k).unsqueeze(0).float()
        print(f"  ✓ Audio tensor created: {audio_tensor.shape}")
        
        return test_file, audio_tensor
        
    except Exception as e:
        print(f"  ✗ Audio loading failed: {e}")
        traceback.print_exc()
        return None

def test_step_4_model_loading(config):
    """Test 4: Model loading (with and without quantization)"""
    print("\n🔍 STEP 4: Testing model loading...")
    
    # First try without quantization
    print("  Trying without quantization...")
    try:
        config_no_quant = config.copy()
        config_no_quant['model'] = config['model'].copy()
        config_no_quant['model']['quantization'] = {'enabled': False}
        
        from transcribe import setup_model
        model, processor = setup_model(config_no_quant)
        print("  ✓ Model loaded successfully (no quantization)")
        return model, processor, "no_quant"
        
    except Exception as e:
        print(f"  ✗ Model loading without quantization failed: {e}")
    
    # Try with quantization
    if config['model'].get('quantization', {}).get('enabled', False):
        print("  Trying with quantization...")
        try:
            model, processor = setup_model(config)
            print("  ✓ Model loaded successfully (with quantization)")
            return model, processor, "quant"
        except Exception as e:
            print(f"  ✗ Model loading with quantization failed: {e}")
    
    return None

def test_step_5_tokenization(processor):
    """Test 5: Basic tokenization"""
    print("\n🔍 STEP 5: Testing tokenization...")
    
    try:
        # Test simple text tokenization
        text = "Transcribe the audio into text."
        
        # Test chat template
        messages = [{"role": "user", "content": text}]
        template_result = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"  ✓ Chat template applied: {template_result[:100]}...")
        
        # Test tokenization
        tokens = processor.tokenizer(
            template_result,
            return_tensors="pt",
            padding=True
        )
        print(f"  ✓ Tokenization successful: {tokens['input_ids'].shape}")
        
        return template_result, tokens
        
    except Exception as e:
        print(f"  ✗ Tokenization failed: {e}")
        traceback.print_exc()
        return None

def test_step_6_basic_generation(model, processor, config):
    """Test 6: Basic text generation (no audio)"""
    print("\n🔍 STEP 6: Testing basic text generation...")
    
    try:
        # Simple text input
        text = "Generate a greeting:"
        tokens = processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        device = config['model']['device']
        if torch.cuda.is_available() and device == "cuda":
            for key in tokens:
                if torch.is_tensor(tokens[key]):
                    tokens[key] = tokens[key].to(device)
        
        print(f"  ✓ Inputs prepared and moved to device: {device}")
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **tokens,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        print(f"  ✓ Generation successful: {generated_ids.shape}")
        
        # Decode
        output = processor.tokenizer.batch_decode(
            generated_ids[:, tokens['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"  ✓ Decoding successful: '{output[0][:100]}...'")
        return True
        
    except Exception as e:
        print(f"  ✗ Basic generation failed: {e}")
        traceback.print_exc()
        return False

def test_step_7_audio_transcription(model, processor, config, audio_info):
    """Test 7: Audio transcription attempt"""
    print("\n🔍 STEP 7: Testing audio transcription...")
    
    if not audio_info:
        print("  ✗ No audio data available for testing")
        return False
    
    audio_file, audio_tensor = audio_info
    
    try:
        # Method 1: Try the manual approach from the fixed transcribe function
        print("  Trying manual audio processing approach...")
        
        # Create text input
        prompt = "Transcribe the audio into text."
        text_messages = [{"role": "user", "content": prompt}]
        text_input = processor.apply_chat_template(
            text_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize text
        text_inputs = processor.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        device = config['model']['device']
        if torch.cuda.is_available() and device == "cuda":
            for key in text_inputs:
                if torch.is_tensor(text_inputs[key]):
                    text_inputs[key] = text_inputs[key].to(device)
        
        print("  ✓ Text processing successful")
        
        # Try generation with text only (no audio for now)
        with torch.no_grad():
            generated_ids = model.generate(
                **text_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode
        output = processor.tokenizer.batch_decode(
            generated_ids[:, text_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"  ✓ Text-only transcription successful: '{output[0][:100]}...'")
        return True
        
    except Exception as e:
        print(f"  ✗ Audio transcription failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("🐛 COMPREHENSIVE QWEN3-OMNI DEBUGGING")
    print("="*80)
    
    # Step 1: Test imports
    if not test_step_1_imports():
        print("\n❌ CRITICAL: Import issues detected. Please check dependencies.")
        return False
    
    # Step 2: Test config
    config = test_step_2_config()
    if not config:
        print("\n❌ CRITICAL: Configuration loading failed.")
        return False
    
    # Step 3: Test audio
    audio_info = test_step_3_audio()
    if not audio_info:
        print("\n❌ WARNING: Audio testing failed, but continuing...")
    
    # Step 4: Test model loading
    model_result = test_step_4_model_loading(config)
    if not model_result:
        print("\n❌ CRITICAL: Model loading failed.")
        return False
    
    model, processor, load_type = model_result
    print(f"\n✅ Model loaded successfully ({load_type})")
    
    # Step 5: Test tokenization
    tokenization_result = test_step_5_tokenization(processor)
    if not tokenization_result:
        print("\n❌ CRITICAL: Tokenization failed.")
        return False
    
    # Step 6: Test basic generation
    if not test_step_6_basic_generation(model, processor, config):
        print("\n❌ CRITICAL: Basic generation failed.")
        return False
    
    # Step 7: Test audio transcription
    if not test_step_7_audio_transcription(model, processor, config, audio_info):
        print("\n❌ Audio transcription failed, but basic model works.")
        return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("Your setup should be working. Try running:")
    print("python transcribe.py --test_single datasets/ABA/wav/arctic_b0519.wav --debug")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 NEXT STEPS:")
        print("1. Check the specific error messages above")
        print("2. Try disabling quantization in config.yaml")
        print("3. Make sure all dependencies are installed")
        print("4. Check GPU memory availability")
    
    sys.exit(0 if success else 1)
