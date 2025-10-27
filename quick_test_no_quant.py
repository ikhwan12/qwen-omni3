#!/usr/bin/env python3
"""
Quick test with quantization disabled to isolate the issue
"""
import os
import sys

def test_without_quantization():
    """Test transcription with quantization completely disabled"""
    print("Testing Qwen3-Omni WITHOUT quantization...")
    print("="*50)
    
    try:
        # Import necessary modules
        from transcribe import load_config, setup_model
        import yaml
        
        # Load config and disable quantization
        config = load_config("config.yaml")
        
        # Force disable quantization
        config['model']['quantization'] = {
            'enabled': False,
            'load_in_4bit': False,
            'load_in_8bit': False
        }
        
        print("✓ Quantization disabled in config")
        
        # Try to load model
        print("Loading model without quantization (this may take a while)...")
        model, processor = setup_model(config)
        print("✓ Model loaded successfully!")
        
        # Test basic text generation first
        print("\nTesting basic text generation...")
        
        # Simple text input
        text_input = "Hello, how are you?"
        inputs = processor.tokenizer(
            text_input,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device if CUDA available
        if config['model']['device'] == 'cuda' and inputs['input_ids'].device.type == 'cpu':
            inputs = {k: v.cuda() for k, v in inputs.items() if hasattr(v, 'cuda')}
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode
        response = processor.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0]
        
        print(f"✓ Basic generation works: '{response}'")
        
        # Now test with a simple audio transcription approach
        print("\nTesting simplified audio transcription...")
        
        # Find test audio file
        audio_file = None
        test_files = [
            "datasets/ABA/wav/arctic_b0519.wav",
            "datasets/ABA/wav/arctic_a0001.wav"
        ]
        
        for f in test_files:
            if os.path.exists(f):
                audio_file = f
                break
        
        if not audio_file:
            print("✗ No test audio file found")
            return False
        
        print(f"Using audio file: {audio_file}")
        
        # Simple text-only approach (ignore audio for now)
        prompt = "Transcribe the following audio: [AUDIO]"
        
        # Create message format but without actually processing audio
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        if config['model']['device'] == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items() if hasattr(v, 'cuda')}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode
        response = processor.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        print(f"✓ Text-only 'transcription' works: '{response}'")
        
        print("\n" + "="*50)
        print("✅ SUCCESS! Model works without quantization.")
        print("\nTry running transcribe.py with quantization disabled:")
        print("1. Edit config.yaml and set quantization.enabled: false")
        print("2. Run: python transcribe.py --test_single " + audio_file + " --debug")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ Test failed: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_without_quantization()
    
    if not success:
        print("\n💡 Suggestions:")
        print("1. Check if you have enough GPU memory")
        print("2. Try running on CPU by setting device: 'cpu' in config.yaml")
        print("3. Make sure the model is properly downloaded")
        print("4. Run the comprehensive debug: python debug_step_by_step.py")
    
    sys.exit(0 if success else 1)
