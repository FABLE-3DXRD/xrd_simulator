#!/usr/bin/env python
"""
Simple test script for GPU control functionality.
Tests all three methods of controlling GPU usage.
"""

import os
import sys

print("="*70)
print("Testing GPU Control Implementation")
print("="*70)
print()

# Test the cuda module directly
print("Test 1: Testing cuda.configure_device() directly")
print("-"*70)

try:
    import torch
    from xrd_simulator.cuda import configure_device
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # Test programmatic control - CPU
    print("Testing: configure_device(use_gpu=False)")
    device = configure_device(use_gpu=False, verbose=True)
    print(f"Returned device: {device}")
    test_tensor = torch.tensor([1.0])
    print(f"Default device after config: {test_tensor.device}")
    print()
    
    # Test programmatic control - GPU (if available)
    if torch.cuda.is_available():
        print("Testing: configure_device(use_gpu=True)")
        device = configure_device(use_gpu=True, verbose=True)
        print(f"Returned device: {device}")
        test_tensor = torch.tensor([1.0])
        print(f"Default device after config: {test_tensor.device}")
        print()
    
    # Test environment variable simulation
    print("Testing: configure_device() with XRD_USE_GPU env var simulation")
    # Simulate environment variable
    original_env = os.environ.get('XRD_USE_GPU')
    os.environ['XRD_USE_GPU'] = 'false'
    device = configure_device(use_gpu=None, verbose=True)
    print(f"Returned device: {device}")
    if original_env:
        os.environ['XRD_USE_GPU'] = original_env
    else:
        os.environ.pop('XRD_USE_GPU', None)
    print()
    
    print("✓ Test 1 PASSED")
    
except ImportError as e:
    print(f"✗ Test 1 FAILED: Import error - {e}")
    print("This is expected if xrd_simulator is not installed.")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("Test 2: Testing xrd_simulator.set_device()")
print("-"*70)

try:
    # Reset to CPU first
    import torch
    torch.set_default_device("cpu")
    
    from xrd_simulator.utils import set_device
    
    # Test CPU
    print("Testing: set_device(use_gpu=False)")
    device = set_device(use_gpu=False, verbose=True)
    print(f"Returned device: {device}")
    test_tensor = torch.tensor([1.0])
    print(f"Default device after config: {test_tensor.device}")
    print()
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("Testing: set_device(use_gpu=True)")
        device = set_device(use_gpu=True, verbose=True)
        print(f"Returned device: {device}")
        test_tensor = torch.tensor([1.0])
        print(f"Default device after config: {test_tensor.device}")
        print()
    
    print("✓ Test 2 PASSED")
    
except ImportError as e:
    print(f"✗ Test 2 FAILED: Import error - {e}")
    print("This is expected if xrd_simulator is not installed.")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("All basic tests completed!")
print()
print("To test with actual simulations, run:")
print("  python docs/source/examples/gpu_usage_example.py")
print()
print("To test with environment variables, run:")
print("  XRD_USE_GPU=true python test_gpu_control.py")
print("  XRD_USE_GPU=false python test_gpu_control.py")
print("="*70)
