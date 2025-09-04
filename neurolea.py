"""
Neurolea - The Ultimate AI Framework with Zero Dependencies
===========================================================

This is the main framework file that ties everything together.
"""

import math
import random
import json
import os
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Import the actual working components
try:
    from core_tensor import Tensor, Adam, SGD
    from neural_networks import (
        Module, Linear, ReLU, Sigmoid, Tanh, GELU, Softmax, Dropout,
        LayerNorm, Embedding, MultiHeadAttention, TransformerBlock,
        Sequential, GPTModel, SimpleTokenizer, Trainer
    )
except ImportError:
    # If running as standalone, assume modules are in same directory
    import sys
    sys.path.append(os.path.dirname(__file__))
    from core_tensor import Tensor, Adam, SGD
    from neural_networks import (
        Module, Linear, ReLU, Sigmoid, Tanh, GELU, Softmax, Dropout,
        LayerNorm, Embedding, MultiHeadAttention, TransformerBlock,
        Sequential, GPTModel, SimpleTokenizer, Trainer
    )

class UltimateAIFramework:
    """
    The main framework class that provides a simplified interface
    to all the AI capabilities.
    """
    
    def __init__(self):
        """Initialize the Ultimate AI Framework"""
        
        # Framework metadata
        self.name = "Neurolea"
        self.version = "1.0.0"
        self.author = "Zero Dependencies Team"
        
        # Components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.trainer = None
        
        # Training data
        self.training_data = []
        self.vocab_size = 1000
        
        # Model configuration (small for demo purposes)
        self.model_config = {
            'd_model': 64,      # Model dimension
            'num_layers': 2,    # Number of transformer layers
            'num_heads': 4,     # Number of attention heads
            'd_ff': 128,        # Feed-forward dimension
            'max_seq_len': 128, # Maximum sequence length
            'dropout': 0.1      # Dropout rate
        }
        
        # Capabilities list (what we claim to support)
        self.capabilities = [
            "Transformer Architecture",
            "Multi-Head Attention",
            "Gradient-Based Training",
            "Text Tokenization",
            "Neural Network Layers",
            "Automatic Differentiation",
            "Adam & SGD Optimizers",
            "Cross-Entropy Loss",
            "Layer Normalization",
            "Dropout Regularization",
            "Embedding Layers",
            "Feed-Forward Networks",
            "Activation Functions (ReLU, GELU, Sigmoid, Tanh)",
            "Backpropagation",
            "Parameter Initialization",
            "Text Generation",
            "Model Training Pipeline"
        ]
        
        # Calculate total parameters (estimated)
        self.total_parameters = self._estimate_parameters()
        
        print(f"🧠 {self.name} v{self.version} initialized")
        print(f"✨ Zero dependencies required!")
    
    def _estimate_parameters(self) -> int:
        """Estimate total parameter count"""
        vocab_size = self.vocab_size
        d_model = self.model_config['d_model']
        num_layers = self.model_config['num_layers']
        d_ff = self.model_config['d_ff']
        
        # Rough estimation
        embedding_params = vocab_size * d_model * 2  # Token + position
        attention_params = num_layers * (4 * d_model * d_model)  # Q,K,V,O projections
        ff_params = num_layers * (2 * d_model * d_ff)  # Two linear layers
        output_params = d_model * vocab_size  # Output projection
        
        return embedding_params + attention_params + ff_params + output_params
    
    def initialize_all_components(self, training_data: List[str]) -> Dict[str, Any]:
        """
        Initialize all components of the framework
        
        Args:
            training_data: List of text strings for training
            
        Returns:
            Dictionary with initialization status
        """
        self.training_data = training_data
        
        # Initialize tokenizer
        print("🔤 Initializing tokenizer...")
        self.tokenizer = SimpleTokenizer(vocab_size=self.vocab_size)
        self.tokenizer.train(training_data)
        
        # Initialize model
        print("🏗️ Building neural network model...")
        self.model = GPTModel(
            vocab_size=len(self.tokenizer.vocab),
            d_model=self.model_config['d_model'],
            num_layers=self.model_config['num_layers'],
            num_heads=self.model_config['num_heads'],
            d_ff=self.model_config['d_ff'],
            max_seq_len=self.model_config['max_seq_len'],
            dropout=self.model_config['dropout']
        )
        
        # Initialize optimizer
        print("⚙️ Setting up optimizer...")
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        
        # Initialize trainer
        print("🎯 Creating training pipeline...")
        self.trainer = Trainer(self.model, self.optimizer)
        
        # Update parameter count
        self.total_parameters = self._count_model_parameters()
        
        print(f"✅ All components initialized successfully!")
        print(f"📊 Total parameters: {self.total_parameters:,}")
        
        return {
            'status': 'success',
            'components': {
                'tokenizer': True,
                'model': True,
                'optimizer': True,
                'trainer': True
            },
            'parameters': self.total_parameters,
            'vocab_size': len(self.tokenizer.vocab),
            'capabilities': len(self.capabilities)
        }
    
    def _count_model_parameters(self) -> int:
        """Count actual model parameters"""
        if self.model is None:
            return self._estimate_parameters()
        
        total = 0
        for param in self.model.parameters():
            total += param.size
        return total
    
    def train(self, epochs: int = 5, batch_size: int = 4) -> Dict[str, List[float]]:
        """
        Train the model on the provided data
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Framework not initialized. Call initialize_all_components first.")
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Prepare training data
        train_data = []
        for text in self.training_data:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 1:
                # Create input-output pairs
                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]
                train_data.append((input_tokens, target_tokens))
        
        if len(train_data) == 0:
            print("⚠️ No valid training data found")
            return {'train_losses': [], 'val_losses': []}
        
        # Train model
        history = self.trainer.train(
            train_data,
            val_data=None,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print(f"✅ Training complete!")
        return history
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generate text continuation from a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Framework not initialized. Call initialize_all_components first.")
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Generate
        self.model.eval()
        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=50
        )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text
    
    def save_model(self, path: str):
        """Save model parameters to file"""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        # Simple parameter saving (just the raw data)
        params_data = {}
        for i, param in enumerate(self.model.parameters()):
            params_data[f'param_{i}'] = {
                'shape': param.shape,
                'data': param.data
            }
        
        with open(path, 'w') as f:
            json.dump(params_data, f)
        
        print(f"💾 Model saved to {path}")
    
    def demo(self):
        """Run a simple demonstration"""
        print("\n" + "="*60)
        print("🎭 NEUROLEA DEMONSTRATION")
        print("="*60)
        
        # Initialize with demo data
        demo_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning enables computers to learn from data.",
            "Neural networks are inspired by the human brain.",
            "Deep learning uses multiple layers of processing."
        ]
        
        print("\n📚 Initializing with demo data...")
        self.initialize_all_components(demo_texts)
        
        print("\n🏋️ Training model (this will take a moment)...")
        history = self.train(epochs=2, batch_size=2)
        
        if history['train_losses']:
            print(f"\n📈 Final training loss: {history['train_losses'][-1]:.4f}")
        
        print("\n✨ Generating text samples:")
        test_prompts = ["the", "artificial", "machine"]
        
        for prompt in test_prompts:
            try:
                generated = self.generate_text(prompt, max_length=10)
                print(f"  Prompt: '{prompt}' -> Generated: '{generated}'")
            except Exception as e:
                print(f"  Prompt: '{prompt}' -> Generation failed: {e}")
        
        print("\n✅ Demonstration complete!")
        print(f"🎯 Framework capabilities: {len(self.capabilities)}")
        print(f"📊 Model parameters: {self.total_parameters:,}")
        print(f"📦 Dependencies required: ZERO!")
        print("="*60)


# Convenience functions
def create_model(vocab_size: int = 1000, d_model: int = 64, 
                num_layers: int = 2, num_heads: int = 4) -> GPTModel:
    """
    Create a GPT model with specified configuration
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        
    Returns:
        GPTModel instance
    """
    return GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_model * 4,
        max_seq_len=128
    )


def quick_train(texts: List[str], epochs: int = 5) -> UltimateAIFramework:
    """
    Quick training function for demonstrations
    
    Args:
        texts: List of training texts
        epochs: Number of training epochs
        
    Returns:
        Trained framework instance
    """
    framework = UltimateAIFramework()
    framework.initialize_all_components(texts)
    framework.train(epochs=epochs)
    return framework


if __name__ == "__main__":
    # Run demonstration when executed directly
    print("🚀 Neurolea Framework - Direct Execution Demo")
    framework = UltimateAIFramework()
    framework.demo()
