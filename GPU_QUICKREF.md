# GPU Control Quick Reference

## One-Line Commands

```bash
# Force GPU
XRD_USE_GPU=true python my_script.py

# Force CPU
XRD_USE_GPU=false python my_script.py
```

## Python Quick Start

```python
import xrd_simulator

# Force GPU
xrd_simulator.set_device(use_gpu=True)

# Force CPU  
xrd_simulator.set_device(use_gpu=False)
```

## Environment Variable Values

| Value | Device |
|-------|--------|
| `true`, `1`, `yes`, `y` | GPU |
| `false`, `0`, `no`, `n` | CPU |
| (not set) | Interactive prompt or programmatic control |

## Check Current Device

```python
import torch
print(torch.tensor([1.0]).device)  # Shows: cpu or cuda:0
```

## Full Documentation
See: `docs/source/examples/GPU_USAGE.md`
