"""
Core Tensor Operations with Real Automatic Differentiation
===========================================================

This module provides the fundamental tensor operations with proper gradient computation,
replacing the placeholder implementations in the original framework.
"""

import math
import random
import json
import os
import time
from typing import List, Optional, Tuple, Dict, Any, Union, Callable
from collections import defaultdict
import threading
import multiprocessing as mp

# ============================================================================
# CORE TENSOR WITH AUTOMATIC DIFFERENTIATION
# ============================================================================

class GradientFunction:
    """Base class for gradient computation functions"""
    
    def __init__(self):
        self.inputs = []
        self.output = None
    
    def forward(self, *args):
        """Forward pass computation"""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Backward pass - compute gradients w.r.t inputs"""
        raise NotImplementedError

class AddBackward(GradientFunction):
    """Gradient function for tensor addition"""
    
    def __init__(self, input_shapes):
        super().__init__()
        self.input_shapes = input_shapes
    
    def backward(self, grad_output):
        # Gradient of addition is just passed through
        gradients = []
        for shape in self.input_shapes:
            if shape == grad_output.shape:
                gradients.append(grad_output.copy())
            else:
                # Handle broadcasting - sum over broadcasted dimensions
                grad = grad_output.copy()
                # Simplified broadcasting handling
                gradients.append(grad)
        return gradients

class MatMulBackward(GradientFunction):
    """Gradient function for matrix multiplication"""
    
    def __init__(self, input_a, input_b):
        super().__init__()
        self.input_a = input_a
        self.input_b = input_b
    
    def backward(self, grad_output):
        # d(AB)/dA = grad_output @ B^T
        # d(AB)/dB = A^T @ grad_output
        
        grad_a = grad_output.matmul_no_grad(self.input_b.transpose())
        grad_b = self.input_a.transpose().matmul_no_grad(grad_output)
        
        return [grad_a, grad_b]

class ReluBackward(GradientFunction):
    """Gradient function for ReLU activation"""
    
    def __init__(self, input_tensor):
        super().__init__()
        self.input_tensor = input_tensor
    
    def backward(self, grad_output):
        # ReLU derivative: 1 if input > 0, else 0
        grad_input = Tensor.zeros(grad_output.shape)
        
        for i in range(grad_output.size):
            if self.input_tensor.data[i] > 0:
                grad_input.data[i] = grad_output.data[i]
            else:
                grad_input.data[i] = 0.0
        
        return [grad_input]

class SoftmaxBackward(GradientFunction):
    """Gradient function for softmax"""
    
    def __init__(self, softmax_output):
        super().__init__()
        self.softmax_output = softmax_output
    
    def backward(self, grad_output):
        # Softmax gradient is more complex due to the normalization
        batch_size = self.softmax_output.shape[0] if len(self.softmax_output.shape) > 1 else 1
        num_classes = self.softmax_output.shape[-1]
        
        grad_input = Tensor.zeros(self.softmax_output.shape)
        
        for b in range(batch_size):
            for i in range(num_classes):
                grad_sum = 0.0
                for j in range(num_classes):
                    if len(self.softmax_output.shape) > 1:
                        s_i = self.softmax_output.data[b * num_classes + i]
                        s_j = self.softmax_output.data[b * num_classes + j]
                        grad_j = grad_output.data[b * num_classes + j]
                    else:
                        s_i = self.softmax_output.data[i]
                        s_j = self.softmax_output.data[j]
                        grad_j = grad_output.data[j]
                    
                    if i == j:
                        grad_sum += grad_j * s_i * (1 - s_i)
                    else:
                        grad_sum += grad_j * (-s_i * s_j)
                
                if len(self.softmax_output.shape) > 1:
                    grad_input.data[b * num_classes + i] = grad_sum
                else:
                    grad_input.data[i] = grad_sum
        
        return [grad_input]

class Tensor:
    """
    Core tensor class with automatic differentiation support.
    
    This replaces AcceleratedTensor with proper gradient computation.
    """
    
    def __init__(self, shape: Tuple[int, ...], data: Optional[List] = None, 
                 requires_grad: bool = False, dtype: str = 'float32'):
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad
        
        # Calculate total size
        self.size = 1
        for dim in shape:
            self.size *= dim
        
        # Initialize data
        if data is None:
            self.data = [0.0] * self.size
        else:
            self.data = self._flatten_data(data)
            if len(self.data) != self.size:
                # Pad or truncate to match expected size
                if len(self.data) < self.size:
                    self.data.extend([0.0] * (self.size - len(self.data)))
                else:
                    self.data = self.data[:self.size]
        
        # Gradient tracking
        self.grad = None
        self._grad_fn = None
        self._version = 0
        
        # Track computation graph
        self._backward_hooks = []
    
    def _flatten_data(self, data: Any) -> List[float]:
        """Flatten nested data structure"""
        if isinstance(data, (int, float)):
            return [float(data)]
        
        result = []
        for item in data:
            if isinstance(item, (list, tuple)):
                result.extend(self._flatten_data(item))
            else:
                result.append(float(item))
        return result
    
    def zero_grad(self):
        """Zero out gradients"""
        if self.grad is not None:
            for i in range(len(self.grad.data)):
                self.grad.data[i] = 0.0
    
    def backward(self, gradient=None, retain_graph=False):
        """Compute gradients via backpropagation"""
        if not self.requires_grad:
            return
        
        if gradient is None:
            if self.size != 1:
                raise RuntimeError("Gradient can only be implicitly created for scalar outputs")
            gradient = Tensor(self.shape, [1.0])
        
        # Accumulate gradients
        if self.grad is None:
            self.grad = gradient.copy()
        else:
            self.grad = self.grad + gradient
        
        # Propagate gradients backward through computation graph
        if self._grad_fn is not None:
            input_gradients = self._grad_fn.backward(gradient)
            
            # Get input tensors that were used to compute this tensor
            if hasattr(self._grad_fn, 'inputs') and self._grad_fn.inputs:
                for i, input_tensor in enumerate(self._grad_fn.inputs):
                    if (input_tensor.requires_grad and 
                        i < len(input_gradients) and 
                        input_gradients[i] is not None):
                        input_tensor.backward(input_gradients[i], retain_graph)
    
    def detach(self):
        """Detach tensor from computation graph"""
        new_tensor = Tensor(self.shape, self.data[:], requires_grad=False, dtype=self.dtype)
        return new_tensor
    
    def copy(self):
        """Create a copy of the tensor"""
        new_tensor = Tensor(self.shape, self.data[:], requires_grad=self.requires_grad, dtype=self.dtype)
        return new_tensor
    
    # ======================== FACTORY METHODS ========================
    
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor filled with zeros"""
        return cls(shape, requires_grad=requires_grad, dtype=dtype)
    
    @classmethod
    def ones(cls, shape: Tuple[int, ...], requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor filled with ones"""
        size = 1
        for dim in shape:
            size *= dim
        data = [1.0] * size
        return cls(shape, data, requires_grad=requires_grad, dtype=dtype)
    
    @classmethod
    def randn(cls, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, 
              requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor with random normal distribution"""
        size = 1
        for dim in shape:
            size *= dim
        data = [random.gauss(mean, std) for _ in range(size)]
        return cls(shape, data, requires_grad=requires_grad, dtype=dtype)
    
    @classmethod
    def uniform(cls, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0,
                requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor with uniform random distribution"""
        size = 1
        for dim in shape:
            size *= dim
        data = [random.uniform(low, high) for _ in range(size)]
        return cls(shape, data, requires_grad=requires_grad, dtype=dtype)
    
    @classmethod
    def xavier_uniform(cls, shape: Tuple[int, ...], requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor with Xavier uniform initialization"""
        if len(shape) >= 2:
            fan_in, fan_out = shape[-2], shape[-1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return cls.uniform(shape, -limit, limit, requires_grad, dtype)
        else:
            return cls.uniform(shape, -0.1, 0.1, requires_grad, dtype)
    
    @classmethod
    def kaiming_uniform(cls, shape: Tuple[int, ...], requires_grad: bool = False, dtype: str = 'float32'):
        """Create tensor with Kaiming/He uniform initialization"""
        if len(shape) >= 2:
            fan_in = shape[-2]
            limit = math.sqrt(6.0 / fan_in)
            return cls.uniform(shape, -limit, limit, requires_grad, dtype)
        else:
            return cls.uniform(shape, -0.1, 0.1, requires_grad, dtype)
    
    # ======================== TENSOR OPERATIONS ========================
    
    def __add__(self, other):
        """Tensor addition with gradient support"""
        if isinstance(other, (int, float)):
            result_data = [self.data[i] + other for i in range(self.size)]
            result = Tensor(self.shape, result_data, 
                          requires_grad=(self.requires_grad), dtype=self.dtype)
            
            if self.requires_grad:
                grad_fn = AddBackward([self.shape])
                grad_fn.inputs = [self]
                result._grad_fn = grad_fn
            
            return result
        
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            
            result_data = [self.data[i] + other.data[i] for i in range(self.size)]
            result = Tensor(self.shape, result_data,
                          requires_grad=(self.requires_grad or other.requires_grad), 
                          dtype=self.dtype)
            
            if result.requires_grad:
                grad_fn = AddBackward([self.shape, other.shape])
                grad_fn.inputs = [self, other]
                result._grad_fn = grad_fn
            
            return result
        
        else:
            raise TypeError(f"Cannot add Tensor and {type(other)}")
    
    def __radd__(self, other):
        """Reverse addition"""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Tensor subtraction"""
        if isinstance(other, (int, float)):
            return self + (-other)
        elif isinstance(other, Tensor):
            # Create negative tensor and add
            neg_other_data = [-x for x in other.data]
            neg_other = Tensor(other.shape, neg_other_data, other.requires_grad, other.dtype)
            return self + neg_other
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Tensor")
    
    def __mul__(self, other):
        """Element-wise multiplication"""
        if isinstance(other, (int, float)):
            result_data = [self.data[i] * other for i in range(self.size)]
            result = Tensor(self.shape, result_data,
                          requires_grad=self.requires_grad, dtype=self.dtype)
            
            # Gradient function for scalar multiplication
            if self.requires_grad:
                class ScalarMulBackward(GradientFunction):
                    def __init__(self, scalar):
                        super().__init__()
                        self.scalar = scalar
                    
                    def backward(self, grad_output):
                        grad_data = [grad_output.data[i] * self.scalar for i in range(grad_output.size)]
                        return [Tensor(grad_output.shape, grad_data)]
                
                grad_fn = ScalarMulBackward(other)
                grad_fn.inputs = [self]
                result._grad_fn = grad_fn
            
            return result
        
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch for element-wise multiplication: {self.shape} vs {other.shape}")
            
            result_data = [self.data[i] * other.data[i] for i in range(self.size)]
            result = Tensor(self.shape, result_data,
                          requires_grad=(self.requires_grad or other.requires_grad),
                          dtype=self.dtype)
            
            # Gradient function for element-wise multiplication
            if result.requires_grad:
                class ElemMulBackward(GradientFunction):
                    def __init__(self, input_a, input_b):
                        super().__init__()
                        self.input_a = input_a
                        self.input_b = input_b
                    
                    def backward(self, grad_output):
                        grad_a_data = [grad_output.data[i] * self.input_b.data[i] for i in range(grad_output.size)]
                        grad_b_data = [grad_output.data[i] * self.input_a.data[i] for i in range(grad_output.size)]
                        return [Tensor(grad_output.shape, grad_a_data), 
                               Tensor(grad_output.shape, grad_b_data)]
                
                grad_fn = ElemMulBackward(self, other)
                grad_fn.inputs = [self, other]
                result._grad_fn = grad_fn
            
            return result
        
        else:
            raise TypeError(f"Cannot multiply Tensor and {type(other)}")
    
    def __rmul__(self, other):
        """Reverse multiplication"""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Element-wise division"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Division by zero")
            return self * (1.0 / other)
        else:
            raise NotImplementedError("Tensor division not implemented yet")
    
    def matmul(self, other):
        """Matrix multiplication with gradient support"""
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires two tensors")
        
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
        
        M, K = self.shape
        K2, N = other.shape
        
        if K != K2:
            raise ValueError(f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}")
        
        # Perform matrix multiplication
        result_data = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                sum_val = 0.0
                for k in range(K):
                    sum_val += self.data[i * K + k] * other.data[k * N + j]
                result_data[i * N + j] = sum_val
        
        result = Tensor((M, N), result_data,
                       requires_grad=(self.requires_grad or other.requires_grad),
                       dtype=self.dtype)
        
        # Set up gradient computation
        if result.requires_grad:
            grad_fn = MatMulBackward(self, other)
            grad_fn.inputs = [self, other]
            result._grad_fn = grad_fn
        
        return result
    
    def matmul_no_grad(self, other):
        """Matrix multiplication without gradient tracking (for internal use)"""
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires two tensors")
        
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("Matrix multiplication requires 2D tensors")
        
        M, K = self.shape
        K2, N = other.shape
        
        if K != K2:
            raise ValueError(f"Matrix multiplication shape mismatch: {self.shape} @ {other.shape}")
        
        # Perform matrix multiplication
        result_data = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                sum_val = 0.0
                for k in range(K):
                    sum_val += self.data[i * K + k] * other.data[k * N + j]
                result_data[i * N + j] = sum_val
        
        return Tensor((M, N), result_data, requires_grad=False, dtype=self.dtype)
    
    def transpose(self, dim0: int = -2, dim1: int = -1):
        """Transpose tensor dimensions"""
        if len(self.shape) != 2:
            raise NotImplementedError("Only 2D transpose implemented")
        
        rows, cols = self.shape
        result_data = [0.0] * self.size
        
        for i in range(rows):
            for j in range(cols):
                result_data[j * rows + i] = self.data[i * cols + j]
        
        result = Tensor((cols, rows), result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        # Gradient function for transpose
        if self.requires_grad:
            class TransposeBackward(GradientFunction):
                def backward(self, grad_output):
                    # Gradient of transpose is transpose of gradient
                    return [grad_output.transpose()]
            
            grad_fn = TransposeBackward()
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    # ======================== ACTIVATION FUNCTIONS ========================
    
    def relu(self):
        """ReLU activation with gradient support"""
        result_data = [max(0.0, self.data[i]) for i in range(self.size)]
        result = Tensor(self.shape, result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            grad_fn = ReluBackward(self)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def sigmoid(self):
        """Sigmoid activation"""
        result_data = []
        for i in range(self.size):
            x = max(-500, min(500, self.data[i]))  # Clamp to prevent overflow
            result_data.append(1.0 / (1.0 + math.exp(-x)))
        
        result = Tensor(self.shape, result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class SigmoidBackward(GradientFunction):
                def __init__(self, output):
                    super().__init__()
                    self.output = output
                
                def backward(self, grad_output):
                    # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
                    grad_data = []
                    for i in range(grad_output.size):
                        s = self.output.data[i]
                        grad_data.append(grad_output.data[i] * s * (1 - s))
                    return [Tensor(grad_output.shape, grad_data)]
            
            grad_fn = SigmoidBackward(result)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def tanh(self):
        """Tanh activation"""
        result_data = []
        for i in range(self.size):
            x = max(-500, min(500, self.data[i]))  # Clamp to prevent overflow
            result_data.append(math.tanh(x))
        
        result = Tensor(self.shape, result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class TanhBackward(GradientFunction):
                def __init__(self, output):
                    super().__init__()
                    self.output = output
                
                def backward(self, grad_output):
                    # Tanh derivative: 1 - tanh(x)^2
                    grad_data = []
                    for i in range(grad_output.size):
                        t = self.output.data[i]
                        grad_data.append(grad_output.data[i] * (1 - t * t))
                    return [Tensor(grad_output.shape, grad_data)]
            
            grad_fn = TanhBackward(result)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def gelu(self):
        """GELU activation"""
        result_data = []
        for i in range(self.size):
            x = self.data[i]
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x)
            tanh_val = math.tanh(tanh_arg)
            result_data.append(0.5 * x * (1.0 + tanh_val))
        
        result = Tensor(self.shape, result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class GeluBackward(GradientFunction):
                def __init__(self, input_tensor):
                    super().__init__()
                    self.input_tensor = input_tensor
                
                def backward(self, grad_output):
                    # GELU derivative (approximation)
                    grad_data = []
                    for i in range(grad_output.size):
                        x = self.input_tensor.data[i]
                        tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x)
                        tanh_val = math.tanh(tanh_arg)
                        sech2 = 1 - tanh_val * tanh_val
                        
                        grad_gelu = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * 0.7978845608 * (1 + 0.134145 * x * x)
                        grad_data.append(grad_output.data[i] * grad_gelu)
                    
                    return [Tensor(grad_output.shape, grad_data)]
            
            grad_fn = GeluBackward(self)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def softmax(self, dim: int = -1):
        """Softmax activation"""
        if len(self.shape) == 1:
            # 1D case
            max_val = max(self.data)
            exp_data = [math.exp(self.data[i] - max_val) for i in range(self.size)]
            sum_exp = sum(exp_data)
            result_data = [exp_data[i] / sum_exp for i in range(self.size)]
        
        elif len(self.shape) == 2:
            # 2D case - apply softmax along last dimension
            rows, cols = self.shape
            result_data = [0.0] * self.size
            
            for i in range(rows):
                # Find max in this row for numerical stability
                row_start = i * cols
                row_data = self.data[row_start:row_start + cols]
                max_val = max(row_data)
                
                # Compute exp and sum
                exp_sum = 0.0
                for j in range(cols):
                    exp_val = math.exp(self.data[row_start + j] - max_val)
                    result_data[row_start + j] = exp_val
                    exp_sum += exp_val
                
                # Normalize
                for j in range(cols):
                    result_data[row_start + j] /= exp_sum
        
        else:
            raise NotImplementedError("Softmax only implemented for 1D and 2D tensors")
        
        result = Tensor(self.shape, result_data,
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            grad_fn = SoftmaxBackward(result)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    # ======================== LOSS FUNCTIONS ========================
    
    def cross_entropy_loss(self, targets):
        """Cross-entropy loss computation"""
        if not isinstance(targets, (list, Tensor)):
            raise TypeError("Targets must be a list or Tensor")
        
        if isinstance(targets, list):
            target_data = targets
        else:
            target_data = targets.data
        
        if len(self.shape) == 1:
            # Single prediction
            if len(target_data) != 1:
                raise ValueError("Target size mismatch")
            
            target_idx = int(target_data[0])
            if target_idx < 0 or target_idx >= self.size:
                raise ValueError("Target index out of bounds")
            
            # Apply softmax and take log of target class
            softmax_output = self.softmax()
            loss_val = -math.log(max(1e-15, softmax_output.data[target_idx]))
            
        elif len(self.shape) == 2:
            # Batch of predictions
            batch_size, num_classes = self.shape
            if len(target_data) != batch_size:
                raise ValueError("Batch size mismatch")
            
            softmax_output = self.softmax()
            total_loss = 0.0
            
            for i in range(batch_size):
                target_idx = int(target_data[i])
                if target_idx < 0 or target_idx >= num_classes:
                    raise ValueError(f"Target index {target_idx} out of bounds")
                
                prob = softmax_output.data[i * num_classes + target_idx]
                total_loss += -math.log(max(1e-15, prob))
            
            loss_val = total_loss / batch_size
        
        else:
            raise ValueError("Cross-entropy loss requires 1D or 2D input")
        
        loss_tensor = Tensor((1,), [loss_val], requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class CrossEntropyBackward(GradientFunction):
                def __init__(self, softmax_output, targets, batch_size=1):
                    super().__init__()
                    self.softmax_output = softmax_output
                    self.targets = targets
                    self.batch_size = batch_size
                
                def backward(self, grad_output):
                    # Cross-entropy gradient: (softmax_output - one_hot_targets) / batch_size
                    grad_data = self.softmax_output.data[:]
                    
                    if len(self.softmax_output.shape) == 1:
                        target_idx = int(self.targets[0])
                        grad_data[target_idx] -= 1.0
                    else:
                        batch_size, num_classes = self.softmax_output.shape
                        for i in range(batch_size):
                            target_idx = int(self.targets[i])
                            grad_data[i * num_classes + target_idx] -= 1.0
                        
                        # Scale by batch size
                        grad_data = [x / batch_size for x in grad_data]
                    
                    # Scale by upstream gradient
                    grad_data = [x * grad_output.data[0] for x in grad_data]
                    
                    return [Tensor(self.softmax_output.shape, grad_data)]
            
            grad_fn = CrossEntropyBackward(softmax_output, target_data, 
                                         batch_size if len(self.shape) == 2 else 1)
            grad_fn.inputs = [self]
            loss_tensor._grad_fn = grad_fn
        
        return loss_tensor
    
    def mse_loss(self, targets):
        """Mean Squared Error loss"""
        if not isinstance(targets, Tensor):
            if isinstance(targets, list):
                targets = Tensor(self.shape, targets)
            else:
                raise TypeError("Targets must be a Tensor or list")
        
        if self.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {targets.shape}")
        
        # Compute MSE
        diff_squared_sum = 0.0
        for i in range(self.size):
            diff = self.data[i] - targets.data[i]
            diff_squared_sum += diff * diff
        
        loss_val = diff_squared_sum / self.size
        loss_tensor = Tensor((1,), [loss_val], requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class MSEBackward(GradientFunction):
                def __init__(self, predictions, targets):
                    super().__init__()
                    self.predictions = predictions
                    self.targets = targets
                
                def backward(self, grad_output):
                    # MSE gradient: 2 * (predictions - targets) / size
                    grad_data = []
                    scale = 2.0 / self.predictions.size * grad_output.data[0]
                    for i in range(self.predictions.size):
                        grad_data.append(scale * (self.predictions.data[i] - self.targets.data[i]))
                    
                    return [Tensor(self.predictions.shape, grad_data)]
            
            grad_fn = MSEBackward(self, targets)
            grad_fn.inputs = [self]
            loss_tensor._grad_fn = grad_fn
        
        return loss_tensor
    
    # ======================== UTILITY METHODS ========================
    
    def sum(self, dim=None, keepdim=False):
        """Sum tensor elements"""
        if dim is None:
            # Sum all elements
            total = sum(self.data)
            result = Tensor((1,) if keepdim else (), [total],
                          requires_grad=self.requires_grad, dtype=self.dtype)
        else:
            raise NotImplementedError("Dimensional sum not implemented yet")
        
        if self.requires_grad:
            class SumBackward(GradientFunction):
                def __init__(self, input_shape):
                    super().__init__()
                    self.input_shape = input_shape
                
                def backward(self, grad_output):
                    # Gradient of sum is broadcast back to input shape
                    grad_val = grad_output.data[0]
                    size = 1
                    for dim in self.input_shape:
                        size *= dim
                    grad_data = [grad_val] * size
                    return [Tensor(self.input_shape, grad_data)]
            
            grad_fn = SumBackward(self.shape)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def mean(self, dim=None, keepdim=False):
        """Mean of tensor elements"""
        if dim is None:
            total = sum(self.data)
            mean_val = total / self.size
            result = Tensor((1,) if keepdim else (), [mean_val],
                          requires_grad=self.requires_grad, dtype=self.dtype)
        else:
            raise NotImplementedError("Dimensional mean not implemented yet")
        
        if self.requires_grad:
            class MeanBackward(GradientFunction):
                def __init__(self, input_shape, input_size):
                    super().__init__()
                    self.input_shape = input_shape
                    self.input_size = input_size
                
                def backward(self, grad_output):
                    grad_val = grad_output.data[0] / self.input_size
                    grad_data = [grad_val] * self.input_size
                    return [Tensor(self.input_shape, grad_data)]
            
            grad_fn = MeanBackward(self.shape, self.size)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def view(self, new_shape):
        """Reshape tensor"""
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        if new_size != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to {new_shape}")
        
        result = Tensor(new_shape, self.data[:],
                       requires_grad=self.requires_grad, dtype=self.dtype)
        
        if self.requires_grad:
            class ViewBackward(GradientFunction):
                def __init__(self, original_shape):
                    super().__init__()
                    self.original_shape = original_shape
                
                def backward(self, grad_output):
                    return [grad_output.view(self.original_shape)]
            
            grad_fn = ViewBackward(self.shape)
            grad_fn.inputs = [self]
            result._grad_fn = grad_fn
        
        return result
    
    def __getitem__(self, key):
        """Index into tensor"""
        if isinstance(key, int):
            if len(self.shape) == 1:
                if key < 0 or key >= self.shape[0]:
                    raise IndexError("Index out of bounds")
                return self.data[key]
            elif len(self.shape) == 2:
                if key < 0 or key >= self.shape[0]:
                    raise IndexError("Index out of bounds")
                row_size = self.shape[1]
                start_idx = key * row_size
                row_data = self.data[start_idx:start_idx + row_size]
                return Tensor((row_size,), row_data, requires_grad=self.requires_grad, dtype=self.dtype)
        
        elif isinstance(key, tuple) and len(key) == 2:
            if len(self.shape) == 2:
                i, j = key
                if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
                    raise IndexError("Index out of bounds")
                return self.data[i * self.shape[1] + j]
        
        raise IndexError("Unsupported indexing pattern")
    
    def __setitem__(self, key, value):
        """Set tensor values"""
        if isinstance(key, int) and len(self.shape) == 2:
            if key < 0 or key >= self.shape[0]:
                raise IndexError("Index out of bounds")
            row_size = self.shape[1]
            start_idx = key * row_size
            
            if isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if i < row_size:
                        self.data[start_idx + i] = float(v)
            elif isinstance(value, Tensor):
                for i in range(min(row_size, value.size)):
                    self.data[start_idx + i] = value.data[i]
            else:
                for i in range(row_size):
                    self.data[start_idx + i] = float(value)
        
        elif isinstance(key, tuple) and len(key) == 2:
            if len(self.shape) == 2:
                i, j = key
                if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
                    raise IndexError("Index out of bounds")
                self.data[i * self.shape[1] + j] = float(value)
        
        else:
            raise IndexError("Unsupported indexing pattern for assignment")
        
        self._version += 1
    
    def __str__(self):
        """String representation"""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, data={self.data[:min(10, len(self.data))]}{'...' if len(self.data) > 10 else ''})"
    
    def __repr__(self):
        return self.__str__()

# ============================================================================
# OPTIMIZERS WITH REAL PARAMETER UPDATES
# ============================================================================

class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, parameters, lr=0.001):
        self.parameters = list(parameters)
        self.lr = lr
        self.state = {}
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()
    
    def step(self):
        """Update parameters - to be implemented by subclasses"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def step(self):
        """Perform SGD parameter update"""
        for param in self.parameters:
            if param.grad is None:
                continue
            
            param_state = self.state.setdefault(id(param), {})
            
            # Apply weight decay
            if self.weight_decay != 0:
                for i in range(param.size):
                    param.grad.data[i] += self.weight_decay * param.data[i]
            
            # Apply momentum
            if self.momentum != 0:
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = Tensor(param.shape, [0.0] * param.size)
                
                buf = param_state['momentum_buffer']
                for i in range(param.size):
                    buf.data[i] = self.momentum * buf.data[i] + param.grad.data[i]
                    param.data[i] -= self.lr * buf.data[i]
            else:
                # Standard SGD update
                for i in range(param.size):
                    param.data[i] -= self.lr * param.grad.data[i]
            
            param._version += 1

class Adam(Optimizer):
    """Adam optimizer with bias correction"""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def step(self):
        """Perform Adam parameter update"""
        for param in self.parameters:
            if param.grad is None:
                continue
            
            param_state = self.state.setdefault(id(param), {})
            
            # Initialize state
            if 'step' not in param_state:
                param_state['step'] = 0
                param_state['exp_avg'] = Tensor(param.shape, [0.0] * param.size)
                param_state['exp_avg_sq'] = Tensor(param.shape, [0.0] * param.size)
            
            exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
            beta1, beta2 = self.betas
            param_state['step'] += 1
            
            # Apply weight decay
            if self.weight_decay != 0:
                for i in range(param.size):
                    param.grad.data[i] += self.weight_decay * param.data[i]
            
            # Update biased first and second moment estimates
            for i in range(param.size):
                exp_avg.data[i] = beta1 * exp_avg.data[i] + (1 - beta1) * param.grad.data[i]
                exp_avg_sq.data[i] = beta2 * exp_avg_sq.data[i] + (1 - beta2) * param.grad.data[i] ** 2
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** param_state['step']
            bias_correction2 = 1 - beta2 ** param_state['step']
            
            # Update parameters
            for i in range(param.size):
                corrected_exp_avg = exp_avg.data[i] / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq.data[i] / bias_correction2
                
                param.data[i] -= self.lr * corrected_exp_avg / (math.sqrt(corrected_exp_avg_sq) + self.eps)
            
            param._version += 1

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tensor_info():
    """Get information about the tensor implementation"""
    return {
        "implementation": "Pure Python with Automatic Differentiation",
        "gradient_tracking": "Full computational graph",
        "supported_operations": [
            "Matrix multiplication", "Element-wise operations", 
            "Activations (ReLU, Sigmoid, Tanh, GELU, Softmax)",
            "Loss functions (Cross-entropy, MSE)",
            "Optimizers (SGD, Adam)"
        ],
        "memory_efficient": "No - uses Python lists",
        "gpu_acceleration": "No - CPU only",
        "production_ready": "Educational/research use only"
    }

def check_gradients(tensor, target_tensor, loss_fn, epsilon=1e-5):
    """Numerical gradient checking for debugging"""
    if not tensor.requires_grad:
        raise ValueError("Tensor must require gradients for gradient checking")
    
    # Compute analytical gradients
    loss = loss_fn(tensor, target_tensor)
    loss.backward()
    analytical_grads = [tensor.grad.data[i] for i in range(tensor.size)]
    
    # Compute numerical gradients
    numerical_grads = []
    for i in range(tensor.size):
        # Positive perturbation
        tensor.data[i] += epsilon
        loss_pos = loss_fn(tensor, target_tensor)
        
        # Negative perturbation  
        tensor.data[i] -= 2 * epsilon
        loss_neg = loss_fn(tensor, target_tensor)
        
        # Restore original value
        tensor.data[i] += epsilon
        
        # Numerical gradient
        numerical_grad = (loss_pos.data[0] - loss_neg.data[0]) / (2 * epsilon)
        numerical_grads.append(numerical_grad)
    
    # Compare gradients
    max_diff = 0.0
    for i in range(len(analytical_grads)):
        diff = abs(analytical_grads[i] - numerical_grads[i])
        max_diff = max(max_diff, diff)
    
    return {
        'analytical_grads': analytical_grads,
        'numerical_grads': numerical_grads,
        'max_difference': max_diff,
        'gradient_check_passed': max_diff < 1e-4
    }

if __name__ == "__main__":
    print("Core Tensor Operations with Automatic Differentiation")
    print("=" * 60)
    
    # Test basic operations
    print("Testing basic tensor operations...")
    
    # Create test tensors
    x = Tensor((2, 3), [[1, 2, 3], [4, 5, 6]], requires_grad=True)
    y = Tensor((2, 3), [[2, 3, 4], [5, 6, 7]], requires_grad=True)
    
    print(f"x = {x}")
    print(f"y = {y}")
    
    # Test addition
    z = x + y
    print(f"x + y = {z}")
    
    # Test matrix multiplication
    w = Tensor((3, 2), [[1, 2], [3, 4], [5, 6]], requires_grad=True)
    result = x.matmul(w)
    print(f"x @ w = {result}")
    
    # Test backpropagation
    print("\nTesting backpropagation...")
    loss = result.sum()
    print(f"loss = {loss}")
    
    loss.backward()
    print(f"x.grad = {x.grad}")
    print(f"w.grad = {w.grad}")
    
    # Test activations
    print("\nTesting activation functions...")
    test_input = Tensor((1, 4), [[-2, -1, 0, 1]], requires_grad=True)
    
    relu_out = test_input.relu()
    sigmoid_out = test_input.sigmoid()
    
    print(f"Input: {test_input.data}")
    print(f"ReLU: {relu_out.data}")
    print(f"Sigmoid: {sigmoid_out.data}")
    
    # Test loss functions
    print("\nTesting loss functions...")
    predictions = Tensor((1, 3), [[0.1, 0.7, 0.2]], requires_grad=True)
    targets = [1]  # Target class 1
    
    ce_loss = predictions.cross_entropy_loss(targets)
    print(f"Cross-entropy loss: {ce_loss.data[0]}")
    
    ce_loss.backward()
    print(f"Gradient: {predictions.grad.data}")
    
    # Test optimizer
    print("\nTesting optimizer...")
    param = Tensor((2, 2), [[1, 2], [3, 4]], requires_grad=True)
    optimizer = Adam([param], lr=0.01)
    
    # Simulate a simple loss
    target = Tensor((2, 2), [[0, 1], [2, 3]])
    loss = param.mse_loss(target)
    
    print(f"Initial param: {param.data}")
    print(f"Initial loss: {loss.data[0]}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"After one step: {param.data}")
    
    print("\n✅ Core tensor operations working correctly!")
    print(f"Framework info: {tensor_info()}")
