# GPU Usage in xrd_simulator

The `xrd_simulator` package supports both CPU and GPU (CUDA) computing. This document explains how to control which device is used for computations.

## Quick Start

### Method 1: Environment Variable (Recommended for Scripts)

Set the `XRD_USE_GPU` environment variable before running your script:

```bash
# Use GPU
XRD_USE_GPU=true python my_script.py

# Use CPU
XRD_USE_GPU=false python my_script.py
```

Accepted values for `XRD_USE_GPU`:
- **GPU**: `true`, `1`, `yes`, `y` (case-insensitive)
- **CPU**: `false`, `0`, `no`, `n` (case-insensitive)

You can also set it in your shell profile for a default behavior:

```bash
# In ~/.bashrc or ~/.zshrc
export XRD_USE_GPU=true
```

### Method 2: Programmatic Control (Recommended for Library Usage)

Control the device directly in your Python code:

```python
import xrd_simulator

# Force GPU usage
xrd_simulator.set_device(use_gpu=True)

# Force CPU usage
xrd_simulator.set_device(use_gpu=False)

# Silent mode (no status messages)
xrd_simulator.set_device(use_gpu=True, verbose=False)
```

### Method 3: Interactive Prompt (Default)

If neither the environment variable is set nor `set_device()` is called, and CUDA is available, you will be prompted:

```
CUDA is available and GPUs are found.
Do you want to run on GPU? [y/n] (default: y):
```

This is useful for interactive sessions but not suitable for batch scripts or automated workflows.

## Examples

### Example 1: Batch Processing Script

```python
#!/usr/bin/env python
"""
Run this script with:
  XRD_USE_GPU=true python process_samples.py
"""
import xrd_simulator
from xrd_simulator.beam import Beam
from xrd_simulator.detector import Detector
# ... rest of your imports

# Device is automatically configured from XRD_USE_GPU environment variable
# No need to call set_device() if using environment variable

# Your simulation code here
# ...
```

### Example 2: Library with Explicit Device Control

```python
import xrd_simulator

# Configure device at the start of your script
device = xrd_simulator.set_device(use_gpu=True, verbose=True)
print(f"Using device: {device}")

# Rest of your code
from xrd_simulator.beam import Beam
# ...
```

### Example 3: Conditional GPU Usage

```python
import torch
import xrd_simulator

# Use GPU only if available, otherwise fall back to CPU
use_gpu = torch.cuda.is_available()
xrd_simulator.set_device(use_gpu=use_gpu)

# Your simulation code
# ...
```

### Example 4: Multiple Simulations with Different Devices

```python
import xrd_simulator

# Run first simulation on GPU
xrd_simulator.set_device(use_gpu=True)
result_gpu = run_simulation()

# Run second simulation on CPU for comparison
xrd_simulator.set_device(use_gpu=False)
result_cpu = run_simulation()

# Compare results
# ...
```

## Environment Setup

### Using conda

```bash
# Create environment with CUDA support
conda create -n xrd_gpu python=3.10
conda activate xrd_gpu
conda install -c conda-forge xrd_simulator pytorch-cuda -c pytorch -c nvidia

# Set default to use GPU
export XRD_USE_GPU=true
```

### Using pip

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install xrd_simulator
pip install xrd-simulator

# Set default to use GPU
export XRD_USE_GPU=true
```

## Troubleshooting

### CUDA Not Available

If you see "CUDA is not available", it means:
1. PyTorch was installed without CUDA support, or
2. No compatible NVIDIA GPU is detected, or
3. CUDA drivers are not properly installed

To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### Force CPU Even When GPU Available

Sometimes you may want to use CPU even when GPU is available (e.g., for debugging, memory constraints):

```bash
XRD_USE_GPU=false python my_script.py
```

or in Python:

```python
import xrd_simulator
xrd_simulator.set_device(use_gpu=False)
```

### Verify Device Being Used

```python
import torch

# Check default device
test_tensor = torch.tensor([1.0])
print(f"Current device: {test_tensor.device}")

# For xrd_simulator specifically
import xrd_simulator
device = xrd_simulator.set_device(use_gpu=None)  # None = auto-detect
print(f"xrd_simulator using: {device}")
```

## Performance Considerations

- **GPU is faster** for large-scale simulations with many elements and detector pixels
- **CPU may be faster** for very small simulations due to GPU overhead
- **Memory**: GPU memory is typically more limited than system RAM
- **Batch processing**: GPU excels at parallel operations

Benchmark your specific use case to determine which device is optimal.

## API Reference

### `xrd_simulator.set_device(use_gpu=None, verbose=True)`

Configure the computing device for xrd_simulator.

**Parameters:**
- `use_gpu` (bool, optional): 
  - `True`: Force GPU usage
  - `False`: Force CPU usage  
  - `None`: Auto-detect from `XRD_USE_GPU` environment variable, or prompt if not set
- `verbose` (bool): Print status messages (default: `True`)

**Returns:**
- `str`: Device being used (`'cuda'` or `'cpu'`)

**Raises:**
- No exceptions raised; falls back to CPU if GPU requested but unavailable

## See Also

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [GPU Usage Example Script](gpu_usage_example.py)
