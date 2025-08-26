"""
Neurolea Quick Start Example
"""

# Import our framework
import sys
import os

# Add the parent directory to path so we can import neurolea
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import neurolea

def main():
    print("🚀 Neurolea Quick Start Demo")
    print("=" * 50)
    
    # Initialize framework
    framework = neurolea.UltimateAIFramework()
    
    # Test with some data
    training_data = [
        "This is test data for Neurolea",
        "Your personal AI framework",
        "Built entirely from scratch with zero dependencies"
    ]
    
    print("📚 Testing framework initialization...")
    framework.initialize_all_components(training_data)
    
    print("✅ Neurolea is working perfectly!")
    print(f"💪 Capabilities: {framework.capabilities}")
    print(f"📊 Parameters: {framework.total_parameters:,}")
    
    print("\n🎉 Your zero-dependency AI framework is ready!")

if __name__ == "__main__":
    main()
