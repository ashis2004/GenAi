"""
GenAI Utilities - Common functions and helpers for generative AI projects.

This module provides utility functions for data processing, model management,
visualization, and other common tasks across GenAI projects.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Utility class for managing AI models."""
    
    def __init__(self):
        self.loaded_models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_model(self, model_name: str, model_class, **kwargs):
        """Load and cache a model."""
        if model_name not in self.loaded_models:
            logger.info(f"Loading model: {model_name}")
            model = model_class.from_pretrained(model_name, **kwargs)
            if hasattr(model, 'to'):
                model = model.to(self.device)
            self.loaded_models[model_name] = model
        return self.loaded_models[model_name]
    
    def get_model(self, model_name: str):
        """Get a previously loaded model."""
        return self.loaded_models.get(model_name, None)
    
    def clear_cache(self):
        """Clear model cache and free memory."""
        self.loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")

class DataProcessor:
    """Utility class for data processing tasks."""
    
    @staticmethod
    def preprocess_text(text: str, max_length: int = None) -> str:
        """Basic text preprocessing."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if needed
        if max_length and len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
        
        return text
    
    @staticmethod
    def preprocess_image(image: Union[str, Image.Image], size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """Basic image preprocessing."""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(size, Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def batch_process(items: List[Any], process_func, batch_size: int = 32) -> List[Any]:
        """Process items in batches."""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
        return results

class Visualizer:
    """Utility class for visualization tasks."""
    
    @staticmethod
    def plot_images(images: List[Image.Image], titles: List[str] = None, 
                   cols: int = 3, figsize: Tuple[int, int] = (15, 5)):
        """Plot multiple images in a grid."""
        rows = (len(images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            
            if titles and i < len(titles):
                axes[row, col].set_title(titles[i])
        
        # Hide empty subplots
        for i in range(len(images), rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics(metrics: Dict[str, List[float]], title: str = "Training Metrics"):
        """Plot training metrics over time."""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.title(title)
        plt.xlabel('Epoch/Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_attention_weights(attention_weights: np.ndarray, tokens: List[str] = None):
        """Visualize attention weights as a heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=tokens if tokens else False,
                   yticklabels=tokens if tokens else False,
                   cmap='Blues', 
                   cbar=True)
        plt.title('Attention Weights')
        plt.show()

class ConfigManager:
    """Utility class for configuration management."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Configuration loaded from {path}")
    
    def save_config(self, path: str = None):
        """Save configuration to JSON file."""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {save_path}")
    
    def get(self, key: str, default: Any = None):
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)

class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.add_metric(name, duration)
            del self.start_times[name]
            return duration
        return 0.0
    
    def add_metric(self, name: str, value: float):
        """Add a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name in self.metrics:
            return np.mean(self.metrics[name])
        return 0.0
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return summary
    
    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_times.clear()

def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['memory_allocated'] = torch.cuda.memory_allocated()
        info['memory_reserved'] = torch.cuda.memory_reserved()
    
    return info

def save_model_artifacts(model, tokenizer, save_path: str, config: Dict[str, Any] = None):
    """Save model artifacts including model, tokenizer, and configuration."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), save_path / 'model.pt')
    
    # Save tokenizer
    if tokenizer and hasattr(tokenizer, 'save_pretrained'):
        tokenizer.save_pretrained(save_path)
    
    # Save configuration
    if config:
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"Model artifacts saved to {save_path}")

def estimate_model_size(model) -> Dict[str, int]:
    """Estimate model size in terms of parameters and memory."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Rough memory estimation (in bytes)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_bytes': param_size,
        'buffer_memory_bytes': buffer_size,
        'total_memory_mb': (param_size + buffer_size) / (1024 * 1024)
    }

# Global instances for convenience
model_manager = ModelManager()
data_processor = DataProcessor()
visualizer = Visualizer()
performance_monitor = PerformanceMonitor()

# Example usage and testing
if __name__ == "__main__":
    # Test device info
    device_info = get_device_info()
    print("Device Info:", json.dumps(device_info, indent=2))
    
    # Test performance monitor
    performance_monitor.start_timer("test_operation")
    import time
    time.sleep(0.1)  # Simulate work
    duration = performance_monitor.end_timer("test_operation")
    print(f"Test operation took: {duration:.3f} seconds")
    
    print("Utilities module loaded successfully!")
