#!/usr/bin/env python3
"""
GenAI Repository Setup Script

This script helps set up the development environment for the GenAI repository.
It installs dependencies, configures the environment, and runs initial tests.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description or command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment for the project."""
    venv_path = "genai_env"
    
    if os.path.exists(venv_path):
        print(f"✓ Virtual environment '{venv_path}' already exists")
        return True
    
    print(f"Creating virtual environment: {venv_path}")
    success = run_command(f"python -m venv {venv_path}", 
                         "Creating virtual environment")
    
    if success:
        print(f"✓ Virtual environment created: {venv_path}")
        print(f"To activate: source {venv_path}/bin/activate (Linux/Mac) or {venv_path}\\Scripts\\activate (Windows)")
    
    return success

def install_dependencies():
    """Install required dependencies."""
    requirements_files = [
        "requirements.txt",
        "requirements-dev.txt"  # If exists
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            success = run_command(f"pip install -r {req_file}", 
                                f"Installing dependencies from {req_file}")
            if not success:
                return False
    
    # Install additional packages that might not be in requirements
    additional_packages = [
        "jupyter",
        "ipywidgets",
        "notebook"
    ]
    
    for package in additional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    return True

def setup_jupyter():
    """Setup Jupyter notebook extensions and kernels."""
    commands = [
        ("python -m ipykernel install --user --name genai --display-name 'GenAI'", 
         "Setting up Jupyter kernel"),
        ("jupyter nbextension enable --py widgetsnbextension", 
         "Enabling Jupyter widgets")
    ]
    
    for command, description in commands:
        run_command(command, description)

def create_config_files():
    """Create default configuration files."""
    configs = {
        ".env.example": """# Environment Variables for GenAI Project
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
""",
        "config/default.json": {
            "model_cache_dir": "./models",
            "data_dir": "./data",
            "output_dir": "./outputs",
            "log_level": "INFO",
            "default_device": "auto",
            "max_memory_gb": 16
        },
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
genai_env/
venv/
env/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env

# Model files and data
models/
data/
*.pth
*.bin
*.safetensors

# Outputs
outputs/
results/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    }
    
    for file_path, content in configs.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not path.exists():
            if isinstance(content, dict):
                with open(path, 'w') as f:
                    json.dump(content, f, indent=2)
            else:
                with open(path, 'w') as f:
                    f.write(content)
            print(f"✓ Created {file_path}")
        else:
            print(f"✓ {file_path} already exists")

def run_initial_tests():
    """Run initial tests to verify setup."""
    test_script = """
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt

print("Testing core dependencies...")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")

# Test basic functionality
x = torch.randn(2, 3)
print(f"✓ PyTorch tensor creation: {x.shape}")

print("✓ All tests passed!")
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_setup.py", "Running initial tests")
    
    # Clean up
    if os.path.exists("test_setup.py"):
        os.remove("test_setup.py")
    
    return success

def display_next_steps():
    """Display next steps for the user."""
    next_steps = """
🎉 GenAI Repository Setup Complete!

Next Steps:
1. Activate the virtual environment:
   - Linux/Mac: source genai_env/bin/activate
   - Windows: genai_env\\Scripts\\activate

2. Set up environment variables:
   - Copy .env.example to .env
   - Add your API keys (OpenAI, HuggingFace, etc.)

3. Start exploring:
   - jupyter notebook (to run the example notebooks)
   - python -m streamlit run projects/demo_app.py (for web demos)

4. Check out the examples:
   - finetuning/mistral-lora.ipynb (LoRA fine-tuning)
   - diffusion/sdxl-controlnet.ipynb (Image generation)
   - speech/voice_assistant.ipynb (Voice assistant)
   - projects/medical_rag.ipynb (RAG system)

📚 Documentation:
   - README.md - Project overview
   - docs/ - Detailed documentation
   - Each directory has its own README

🔧 Development:
   - Use the genai_utils.py for common functions
   - Follow the coding guidelines in docs/
   - Run tests before committing changes

Happy coding with GenAI! 🚀
"""
    print(next_steps)

def main():
    """Main setup function."""
    print("🤖 GenAI Repository Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install. Please check manually.")
    
    # Setup Jupyter
    print("\n📓 Setting up Jupyter...")
    setup_jupyter()
    
    # Create config files
    print("\n⚙️  Creating configuration files...")
    create_config_files()
    
    # Run tests
    print("\n🧪 Running initial tests...")
    if run_initial_tests():
        print("✓ All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check your installation.")
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()
