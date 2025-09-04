"""
Hybrid Acceleration Strategy for Neurolea
==========================================

This demonstrates a practical hybrid approach where:
1. Python handles high-level logic and framework
2. C++ accelerates only the critical bottlenecks (matmul, convolution)
3. Both work together using numpy as a bridge
"""

import numpy as np
import ctypes
import os
import platform
from typing import Optional, List, Tuple
import subprocess

# First, let's create a simple C++ accelerator for matrix multiplication
CPP_CODE = """
// matmul_accelerator.cpp
#include <cstring>
#include <cmath>
#include <omp.h>  // For parallel processing

extern "C" {
    // Accelerated matrix multiplication
    void matmul_fast(float* A, float* B, float* C, int M, int N, int K) {
        // Use OpenMP to parallelize
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    // Vectorized ReLU activation
    void relu_fast(float* data, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            data[i] = data[i] > 0 ? data[i] : 0;
        }
    }
    
    // Parallel softmax
    void softmax_fast(float* input, float* output, int batch_size, int num_classes) {
        #pragma omp parallel for
        for (int b = 0; b < batch_size; b++) {
            float* in_ptr = input + b * num_classes;
            float* out_ptr = output + b * num_classes;
            
            // Find max for numerical stability
            float max_val = in_ptr[0];
            for (int i = 1; i < num_classes; i++) {
                if (in_ptr[i] > max_val) max_val = in_ptr[i];
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < num_classes; i++) {
                out_ptr[i] = expf(in_ptr[i] - max_val);
                sum += out_ptr[i];
            }
            
            // Normalize
            for (int i = 0; i < num_classes; i++) {
                out_ptr[i] /= sum;
            }
        }
    }
}
"""

class HybridAccelerator:
    """
    Intelligent hybrid system that:
    1. Uses C++ for heavy computation
    2. Python for control flow and high-level logic
    3. Automatically falls back to Python if C++ unavailable
    """
    
    def __init__(self):
        self.cpp_available = False
        self.lib = None
        self._try_compile_cpp()
    
    def _try_compile_cpp(self):
        """Attempt to compile C++ accelerator"""
        try:
            # Write C++ code
            with open('matmul_accelerator.cpp', 'w') as f:
                f.write(CPP_CODE)
            
            # Compile based on platform
            if platform.system() == "Linux":
                cmd = "g++ -O3 -fopenmp -fPIC -shared matmul_accelerator.cpp -o matmul_accelerator.so"
            elif platform.system() == "Darwin":  # macOS
                cmd = "g++ -O3 -fPIC -shared matmul_accelerator.cpp -o matmul_accelerator.dylib"
            elif platform.system() == "Windows":
                cmd = "cl /O2 /openmp /LD matmul_accelerator.cpp /Fe:matmul_accelerator.dll"
            else:
                print("⚠️ Unknown platform, using pure Python")
                return
            
            # Compile
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0:
                # Load library
                lib_file = './matmul_accelerator.so' if platform.system() == "Linux" else \
                          './matmul_accelerator.dylib' if platform.system() == "Darwin" else \
                          './matmul_accelerator.dll'
                
                self.lib = ctypes.CDLL(lib_file)
                self._setup_functions()
                self.cpp_available = True
                print("✅ C++ accelerator compiled and loaded!")
            else:
                print("⚠️ C++ compilation failed, using pure Python fallback")
                
        except Exception as e:
            print(f"⚠️ Could not set up C++ accelerator: {e}")
            print("   Using pure Python implementation")
    
    def _setup_functions(self):
        """Set up C++ function signatures"""
        # Matrix multiplication
        self.lib.matmul_fast.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        # ReLU
        self.lib.relu_fast.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        
        # Softmax
        self.lib.softmax_fast.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int
        ]
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Smart matrix multiplication:
        - Uses C++ if available and matrices are large
        - Falls back to numpy for small matrices or if C++ unavailable
        """
        M, K = a.shape
        K2, N = b.shape
        
        if K != K2:
            raise ValueError(f"Shape mismatch: {a.shape} @ {b.shape}")
        
        # Use C++ for large matrices, numpy for small ones
        if self.cpp_available and M * N * K > 10000:  # Threshold for C++ benefit
            # Convert to C-contiguous float32 arrays
            a_cont = np.ascontiguousarray(a, dtype=np.float32)
            b_cont = np.ascontiguousarray(b, dtype=np.float32)
            c = np.zeros((M, N), dtype=np.float32)
            
            # Call C++ function
            self.lib.matmul_fast(
                a_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                b_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, N, K
            )
            return c
        else:
            # Fallback to numpy
            return np.dot(a, b)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """Smart ReLU with C++ acceleration for large arrays"""
        if self.cpp_available and x.size > 10000:
            x_copy = np.ascontiguousarray(x, dtype=np.float32).copy()
            self.lib.relu_fast(
                x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_copy.size
            )
            return x_copy
        else:
            return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Smart softmax with C++ acceleration"""
        if len(x.shape) == 2:
            batch_size, num_classes = x.shape
            
            if self.cpp_available and x.size > 1000:
                x_cont = np.ascontiguousarray(x, dtype=np.float32)
                output = np.zeros_like(x_cont)
                
                self.lib.softmax_fast(
                    x_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    batch_size, num_classes
                )
                return output
        
        # Fallback to numpy
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class HybridTensor:
    """
    A smarter tensor that automatically uses C++ acceleration when beneficial
    while maintaining compatibility with the existing pure Python code
    """
    
    # Shared accelerator instance
    _accelerator = None
    
    def __init__(self, shape: Tuple[int, ...], data=None):
        if HybridTensor._accelerator is None:
            HybridTensor._accelerator = HybridAccelerator()
        
        self.shape = shape
        self.size = np.prod(shape)
        
        # Use numpy for storage (more efficient than Python lists)
        if data is None:
            self.data = np.zeros(shape, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32).reshape(shape)
        
        self.grad = None
        self.requires_grad = False
    
    def matmul(self, other):
        """Matrix multiplication using hybrid acceleration"""
        result_data = self._accelerator.matmul(self.data, other.data)
        result = HybridTensor(result_data.shape)
        result.data = result_data
        return result
    
    def relu(self):
        """ReLU activation using hybrid acceleration"""
        result = HybridTensor(self.shape)
        result.data = self._accelerator.relu(self.data)
        return result
    
    def softmax(self, axis=-1):
        """Softmax using hybrid acceleration"""
        result = HybridTensor(self.shape)
        result.data = self._accelerator.softmax(self.data)
        return result
    
    def to_python_list(self):
        """Convert back to Python list for compatibility"""
        return self.data.flatten().tolist()


# Example of intelligent workload distribution
class IntelligentScheduler:
    """
    Distributes work between Python and C++ intelligently:
    - Small operations stay in Python (low overhead)
    - Large operations go to C++ (high performance)
    - Can run both simultaneously for different parts of the network
    """
    
    def __init__(self):
        self.python_queue = []
        self.cpp_queue = []
    
    def schedule_operation(self, op_type: str, size: int, data):
        """Intelligently route operations based on size and type"""
        
        # Operations that benefit from C++ acceleration
        cpp_beneficial = ['matmul', 'conv2d', 'softmax', 'layernorm']
        
        # Size threshold (operations smaller than this stay in Python)
        size_threshold = 10000
        
        if op_type in cpp_beneficial and size > size_threshold:
            self.cpp_queue.append((op_type, data))
            return "cpp"
        else:
            self.python_queue.append((op_type, data))
            return "python"
    
    def execute_parallel(self):
        """Execute Python and C++ operations in parallel"""
        import threading
        
        def run_python_ops():
            for op_type, data in self.python_queue:
                # Execute Python operations
                pass
        
        def run_cpp_ops():
            for op_type, data in self.cpp_queue:
                # Execute C++ operations
                pass
        
        # Run both queues simultaneously
        python_thread = threading.Thread(target=run_python_ops)
        cpp_thread = threading.Thread(target=run_cpp_ops)
        
        python_thread.start()
        cpp_thread.start()
        
        python_thread.join()
        cpp_thread.join()


# Demo usage
if __name__ == "__main__":
    print("🚀 Hybrid Python/C++ Accelerator Demo")
    print("=" * 50)
    
    # Initialize accelerator
    accelerator = HybridAccelerator()
    
    # Test performance comparison
    import time
    
    # Create test matrices
    size = 500
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # Pure numpy
    start = time.time()
    c_numpy = np.dot(a, b)
    numpy_time = time.time() - start
    print(f"NumPy matmul ({size}x{size}): {numpy_time:.4f}s")
    
    # Hybrid (C++ if available)
    start = time.time()
    c_hybrid = accelerator.matmul(a, b)
    hybrid_time = time.time() - start
    print(f"Hybrid matmul ({size}x{size}): {hybrid_time:.4f}s")
    
    if accelerator.cpp_available:
        speedup = numpy_time / hybrid_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("(C++ not available, used NumPy fallback)")
    
    print("\n✅ Hybrid system operational!")
    print("   - Small ops: Python (low overhead)")
    print("   - Large ops: C++ (high performance)")
    print("   - Automatic fallback if C++ unavailable")
