#!/usr/bin/env python3
"""
Install qwen-omni-utils with fallback options
"""
import subprocess
import sys

def install_qwen_utils():
    """Try different methods to install qwen-omni-utils"""
    
    print("📦 Installing qwen-omni-utils...")
    
    # Method 1: Direct pip install
    try:
        print("  Method 1: pip install qwen-omni-utils")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "qwen-omni-utils"], 
                              capture_output=True, text=True, check=True)
        print("✅ Successfully installed qwen-omni-utils")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Method 1 failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
    
    # Method 2: Try with --upgrade
    try:
        print("  Method 2: pip install --upgrade qwen-omni-utils")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "qwen-omni-utils"], 
                              capture_output=True, text=True, check=True)
        print("✅ Successfully installed qwen-omni-utils (with upgrade)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try with --force-reinstall
    try:
        print("  Method 3: pip install --force-reinstall qwen-omni-utils")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "qwen-omni-utils"], 
                              capture_output=True, text=True, check=True)
        print("✅ Successfully installed qwen-omni-utils (force reinstall)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Method 4: Try installing from PyPI with specific index
    try:
        print("  Method 4: pip install qwen-omni-utils --index-url https://pypi.org/simple/")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "qwen-omni-utils", 
                               "--index-url", "https://pypi.org/simple/"], 
                              capture_output=True, text=True, check=True)
        print("✅ Successfully installed qwen-omni-utils (from PyPI)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Method 4 failed: {e}")
    
    print("❌ All installation methods failed")
    return False

def test_import():
    """Test if qwen-omni-utils can be imported"""
    
    print("\n🧪 Testing qwen-omni-utils import...")
    
    try:
        from qwen_omni_utils import process_mm_info
        print("✅ Successfully imported qwen_omni_utils.process_mm_info")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🔧 Qwen-Omni Utils Installation")
    print("=" * 40)
    
    # Test if it's already installed
    if test_import():
        print("✅ qwen-omni-utils is already working!")
        return True
    
    # Try to install
    if install_qwen_utils():
        # Test again after installation
        if test_import():
            print("\n🎉 Installation successful!")
            return True
        else:
            print("\n⚠️  Installation completed but import still fails")
            return False
    else:
        print("\n❌ Installation failed")
        print("\n💡 Manual steps:")
        print("1. Try: pip install qwen-omni-utils")
        print("2. If that fails, check if the package exists on PyPI")
        print("3. You may need to install from source or use an alternative")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
