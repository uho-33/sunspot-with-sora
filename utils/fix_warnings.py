"""
Utility script to fix common warnings in the Open-Sora codebase.

This script addresses:
1. The deprecated Transformer2DModelOutput import
2. The missing Apex library warnings 
3. TensorNVMe warnings

Usage:
    python utils/fix_warnings.py --fix-all
"""

import os
import sys
import argparse
import importlib
import re
from pathlib import Path

def fix_diffusers_import_warning():
    """Fix the deprecated import in diffusers if installed"""
    try:
        import diffusers
        diffusers_path = Path(diffusers.__file__).parent
        transformer_2d_path = diffusers_path / "models" / "transformers" / "transformer_2d.py"
        
        if transformer_2d_path.exists():
            print(f"Found transformer_2d.py at {transformer_2d_path}")
            
            with open(transformer_2d_path, 'r') as f:
                content = f.read()
            
            if "from diffusers.models.transformer_2d import Transformer2DModelOutput" in content:
                new_content = content.replace(
                    "from diffusers.models.transformer_2d import Transformer2DModelOutput",
                    "from diffusers.models.modeling_outputs import Transformer2DModelOutput"
                )
                
                with open(transformer_2d_path, 'w') as f:
                    f.write(new_content)
                print("Fixed deprecated import in transformer_2d.py")
            else:
                print("No deprecated import found or already fixed")
        else:
            print("Could not find transformer_2d.py")
    except ImportError:
        print("Diffusers package not installed")

def install_tensornvme():
    """Install or update TensorNVMe"""
    try:
        import subprocess
        print("Installing/updating TensorNVMe...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/hpcaitech/TensorNVMe.git"
        ])
        print("TensorNVMe installed successfully")
    except Exception as e:
        print(f"Failed to install TensorNVMe: {e}")

def install_apex():
    """Install NVIDIA Apex from source"""
    try:
        import subprocess
        print("Installing NVIDIA Apex (this may take a while)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-v",
            "--disable-pip-version-check", "--no-cache-dir", "--no-build-isolation",
            "--config-settings", "--build-option=--cpp_ext",
            "--config-settings", "--build-option=--cuda_ext",
            "git+https://github.com/NVIDIA/apex.git"
        ])
        print("NVIDIA Apex installed successfully")
    except Exception as e:
        print(f"Failed to install Apex: {e}")
        print("Note: Apex installation can fail for various reasons including CUDA version mismatches.")
        print("You may need to install it manually following the instructions at:")
        print("https://github.com/NVIDIA/apex#from-source")

def main():
    parser = argparse.ArgumentParser(description="Fix common warnings in Open-Sora")
    parser.add_argument("--fix-diffusers", action="store_true", help="Fix diffusers deprecation warning")
    parser.add_argument("--install-tensornvme", action="store_true", help="Install TensorNVMe")
    parser.add_argument("--install-apex", action="store_true", help="Install NVIDIA Apex")
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    
    args = parser.parse_args()
    
    if args.fix_all or args.fix_diffusers:
        fix_diffusers_import_warning()
    
    if args.fix_all or args.install_tensornvme:
        install_tensornvme()
    
    if args.fix_all or args.install_apex:
        install_apex()
        
    if not any([args.fix_all, args.fix_diffusers, args.install_tensornvme, args.install_apex]):
        parser.print_help()

if __name__ == "__main__":
    main()
