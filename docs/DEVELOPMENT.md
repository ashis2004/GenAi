# Development Guidelines

This document outlines the development guidelines and best practices for contributing to the GenAI repository.

## 📋 Table of Contents

- [Code Style](#code-style)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)
- [Git Workflow](#git-workflow)
- [Model Guidelines](#model-guidelines)
- [Performance Guidelines](#performance-guidelines)

## 🎨 Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) guidelines with these specifications:

```python
# Line length: 88 characters (Black formatter default)
# Indentation: 4 spaces
# Quote style: Double quotes preferred

# Good example
def generate_text(
    model: torch.nn.Module,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7
) -> str:
    """Generate text using the provided model.
    
    Args:
        model: The language model to use
        prompt: Input text prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        
    Returns:
        Generated text string
    """
    # Implementation here
    pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union

# Third-party imports
import torch
import numpy as np
import transformers
from PIL import Image

# Local imports
from utils.genai_utils import ModelManager
from evaluation.metrics import calculate_bleu
```

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union, Tuple, Any

def process_batch(
    items: List[str], 
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """Process items in batches."""
    pass
```

## 🏗️ Project Structure

### Directory Organization

```
GenAi/
├── category_name/           # Main category (e.g., llm, diffusion)
│   ├── README.md           # Category overview
│   ├── basic_example.py    # Simple implementation
│   ├── advanced_example.ipynb  # Complex tutorial
│   ├── utils.py           # Category-specific utilities
│   └── tests/             # Category tests
├── utils/                 # Shared utilities
├── evaluation/            # Evaluation frameworks
├── docs/                  # Documentation
└── tests/                # Global tests
```

### File Naming Conventions

- **Scripts**: `snake_case.py`
- **Notebooks**: `descriptive_name.ipynb`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_CASE`

## 📖 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    epochs: int,
    learning_rate: float = 1e-3
) -> Dict[str, List[float]]:
    """Train a model on the provided dataset.
    
    Args:
        model: PyTorch model to train
        dataset: Training dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is not positive
        RuntimeError: If CUDA is not available but required
        
    Example:
        >>> model = MyModel()
        >>> dataset = MyDataset()
        >>> metrics = train_model(model, dataset, epochs=10)
        >>> print(metrics['loss'][-1])  # Final loss
    """
```

### Notebook Documentation

- Start with a clear title and overview
- Include learning objectives
- Add markdown cells explaining each step
- Provide code comments for complex operations
- End with a conclusion and next steps

### README Guidelines

Each directory should have a README.md containing:

- Purpose and scope
- Quick start guide
- Examples and usage
- Dependencies
- References to related work

## 🧪 Testing

### Test Structure

```python
import unittest
import torch
from your_module import YourClass

class TestYourClass(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = YourClass()
        self.sample_data = torch.randn(1, 3, 224, 224)
    
    def test_forward_pass(self):
        """Test forward pass of the model."""
        output = self.model(self.sample_data)
        self.assertEqual(output.shape, (1, 1000))
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 2, 224, 224))  # Wrong channels

if __name__ == '__main__':
    unittest.main()
```

### Testing Guidelines

- Write tests for all public functions
- Test edge cases and error conditions
- Use meaningful test names
- Mock external dependencies (APIs, large models)
- Include integration tests for complete workflows

## 🔄 Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow the conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(llm): add LoRA fine-tuning implementation
fix(diffusion): resolve memory leak in batch processing
docs(readme): update installation instructions
test(evaluation): add BLEU score unit tests
```

### Pull Request Process

1. Create a feature branch from `main`
2. Implement changes with tests
3. Update documentation
4. Run all tests locally
5. Submit pull request with clear description
6. Address review feedback
7. Squash commits before merge

## 🤖 Model Guidelines

### Model Loading

```python
from utils.genai_utils import ModelManager

# Use the model manager for consistent loading
model_manager = ModelManager()

model = model_manager.load_model(
    "microsoft/DialoGPT-medium",
    AutoModelForCausalLM,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Memory Management

```python
# Clear cache after heavy operations
import torch
import gc

def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Use context managers for temporary models
@contextmanager
def temporary_model(model_name):
    model = load_model(model_name)
    try:
        yield model
    finally:
        del model
        cleanup_memory()
```

### Error Handling

```python
def generate_text(model, prompt):
    """Generate text with proper error handling."""
    try:
        # Validate inputs
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Generate with timeout
        with torch.no_grad():
            output = model.generate(
                prompt,
                max_length=100,
                timeout=30  # Prevent infinite loops
            )
        
        return output
        
    except torch.cuda.OutOfMemoryError:
        cleanup_memory()
        raise RuntimeError("Insufficient GPU memory")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
```

## ⚡ Performance Guidelines

### Optimization Checklist

- [ ] Use appropriate data types (float16 vs float32)
- [ ] Implement batch processing where possible
- [ ] Cache expensive computations
- [ ] Use GPU acceleration when available
- [ ] Profile code to identify bottlenecks
- [ ] Implement early stopping for training
- [ ] Use memory-efficient attention mechanisms

### Batch Processing

```python
def process_batch(items: List[str], batch_size: int = 32):
    """Process items in batches for efficiency."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch efficiently
        batch_results = model.process_batch(batch)
        results.extend(batch_results)
        
        # Optional: clear intermediate results
        torch.cuda.empty_cache()
    
    return results
```

### Profiling

```python
from utils.genai_utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Time operations
monitor.start_timer("model_inference")
output = model(input_data)
inference_time = monitor.end_timer("model_inference")

print(f"Inference took: {inference_time:.3f} seconds")
```

## 🔧 Development Tools

### Recommended VS Code Extensions

- Python
- Pylance
- Black Formatter
- autoDocstring
- GitLens
- Jupyter

### Pre-commit Hooks

Install pre-commit hooks for code quality:

```bash
pip install pre-commit
pre-commit install
```

### Code Formatting

Use Black for consistent formatting:

```bash
pip install black
black --line-length 88 .
```

## 📊 Logging and Monitoring

### Logging Setup

```python
import logging
from utils.genai_utils import setup_logging

# Setup logging
setup_logging(level="INFO", log_file="genai.log")
logger = logging.getLogger(__name__)

def my_function():
    logger.info("Starting operation")
    try:
        # Operation code
        logger.info("Operation completed successfully")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

### Monitoring Training

```python
import wandb  # Optional: for experiment tracking

def train_with_monitoring(model, dataset):
    # Initialize monitoring
    wandb.init(project="genai-experiments")
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, dataset)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        logger.info(f"Epoch {epoch}: loss={loss:.4f}")
```

## 🤝 Contributing

### Before Contributing

1. Read the project README
2. Check existing issues and pull requests
3. Set up the development environment
4. Run the test suite
5. Familiarize yourself with the codebase

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Ensure all tests pass
6. Submit a pull request

### Review Criteria

- Code follows style guidelines
- Tests are included and passing
- Documentation is updated
- Performance implications considered
- Security best practices followed

---

Following these guidelines helps maintain code quality, consistency, and collaboration efficiency across the GenAI repository. Thank you for contributing! 🚀
