#!/usr/bin/env python3
"""
Setup script for the Improved NSFW Censoring System
Run this script to install all dependencies automatically.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages from requirements.txt"""
    print("🔧 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        
        # Check for GPU support
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🚀 Apple Silicon MPS detected")
            else:
                print("💻 Using CPU (consider installing GPU support for better performance)")
        except ImportError:
            print("❌ PyTorch installation may have failed")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True

def test_installation():
    """Test if the installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        from improved_nsfw_censor import NSFWCensor
        print("✅ NSFWCensor import successful")
        
        # Test model loading (this will download models on first run)
        print("📥 Loading models (this may take a while on first run)...")
        censor = NSFWCensor()
        print("✅ Models loaded successfully!")
        
        print("🎉 Installation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    print("🔥 Improved NSFW Censoring System Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during dependency installation")
        return
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📚 Quick Usage:")
    print("```python")
    print("from improved_nsfw_censor import NSFWCensor")
    print("")
    print("censor = NSFWCensor()")
    print('censor.process_image("input.jpg", "output.jpg", method="blur")')
    print("```")
    print("\n📖 See README_NSFW_CENSOR.md for detailed usage instructions")

if __name__ == "__main__":
    main()