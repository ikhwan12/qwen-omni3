"""
Quick test script to verify dataset and configuration
"""
import yaml
from pathlib import Path
import json


def test_config():
    """Test configuration file"""
    print("🔧 Testing configuration...")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ Config loaded")
        print(f"  Model: {config['model']['name']}")
        print(f"  Device: {config['model']['device']}")
        print(f"  Dataset: {config['dataset']['data_dir']}")
        return config
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def test_dataset(config):
    """Test dataset accessibility"""
    print("\n📁 Testing dataset...")
    data_dir = Path(config['dataset']['data_dir'])
    
    if not data_dir.exists():
        print(f"  ✗ Dataset directory not found: {data_dir}")
        return False
    
    print(f"  ✓ Dataset directory exists: {data_dir}")
    
    # Count speakers
    speakers = [d for d in data_dir.iterdir() 
               if d.is_dir() and not d.name.startswith('.') 
               and d.name not in ['l2arctic_audio']]
    
    print(f"  ✓ Found {len(speakers)} speakers: {', '.join([s.name for s in speakers[:5]])}...")
    
    # Check first speaker
    if speakers:
        speaker = speakers[0]
        wav_dir = speaker / config['dataset'].get('wav_subdir', 'wav')
        transcript_dir = speaker / config['dataset'].get('transcript_subdir', 'transcript')
        
        if wav_dir.exists():
            wav_count = len(list(wav_dir.glob('*.wav')))
            print(f"  ✓ Speaker {speaker.name} has {wav_count} WAV files")
        else:
            print(f"  ✗ WAV directory not found: {wav_dir}")
            return False
        
        if transcript_dir.exists():
            txt_count = len(list(transcript_dir.glob('*.txt')))
            print(f"  ✓ Speaker {speaker.name} has {txt_count} transcript files")
        else:
            print(f"  ✗ Transcript directory not found: {transcript_dir}")
            return False
    
    return True


def test_sample_files(config):
    """Test reading sample files"""
    print("\n📄 Testing sample files...")
    data_dir = Path(config['dataset']['data_dir'])
    
    # Try first speaker
    speakers = [d for d in data_dir.iterdir() 
               if d.is_dir() and not d.name.startswith('.') 
               and d.name not in ['l2arctic_audio']]
    
    if not speakers:
        print("  ✗ No speakers found")
        return False
    
    speaker = speakers[0]
    wav_subdir = config['dataset'].get('wav_subdir', 'wav')
    transcript_subdir = config['dataset'].get('transcript_subdir', 'transcript')
    
    # Try to read a sample file
    sample_file = "arctic_a0001"
    wav_file = speaker / wav_subdir / f"{sample_file}.wav"
    txt_file = speaker / transcript_subdir / f"{sample_file}.txt"
    
    if wav_file.exists():
        size = wav_file.stat().st_size / 1024  # KB
        print(f"  ✓ Sample WAV readable: {wav_file.name} ({size:.1f} KB)")
    else:
        print(f"  ✗ Sample WAV not found: {wav_file}")
        return False
    
    if txt_file.exists():
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        print(f"  ✓ Sample transcript readable: {txt_file.name}")
        print(f"    Content: \"{content[:60]}...\"")
    else:
        print(f"  ✗ Sample transcript not found: {txt_file}")
        return False
    
    return True


def test_dependencies():
    """Test required packages"""
    print("\n📦 Testing dependencies...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('werpy', 'WERpy'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('pandas', 'Pandas')
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} not installed")
            all_ok = False
    
    return all_ok


def test_output_dir(config):
    """Test output directory"""
    print("\n💾 Testing output directory...")
    output_dir = Path(config['dataset']['output_dir'])
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory ready: {output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Cannot create output directory: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 L2-ARCTIC Dataset & Configuration Test")
    print("=" * 60)
    
    results = {}
    
    # Test configuration
    config = test_config()
    results['config'] = config is not None
    
    if config:
        # Test dataset
        results['dataset'] = test_dataset(config)
        
        # Test sample files
        results['sample_files'] = test_sample_files(config)
        
        # Test output directory
        results['output_dir'] = test_output_dir(config)
    else:
        results['dataset'] = False
        results['sample_files'] = False
        results['output_dir'] = False
    
    # Test dependencies
    results['dependencies'] = test_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    all_pass = all(results.values())
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 All tests passed! Ready to run transcription.")
        print("\nNext steps:")
        print("  1. Activate venv: .\.venv\Scripts\activate.ps1")
        print("  2. Test with one speaker: python transcribe.py --speaker ABA")
        print("  3. Run evaluation: python evaluate_wer.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Run setup.ps1 to install dependencies")
        print("  - Check dataset path in config.yaml")
        print("  - Ensure dataset has correct structure")
    print("=" * 60)
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

