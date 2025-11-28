import os
import torch

# Default
torch.set_default_device("cpu")
_device = "cpu"


def configure_device(use_gpu=None, verbose=True):
    """
    Configure PyTorch device for xrd_simulator.

    Priority:
        1. Explicit argument use_gpu=True/False
        2. Environment variable XRD_USE_GPU
        3. Auto-detect (use GPU if available)

    Returns:
        str: 'cuda' or 'cpu'
    """
    global _device

    # 1. Explicit argument
    if use_gpu is not None:
        want_gpu = bool(use_gpu)

    # 2. Environment variable
    else:
        env = os.getenv("XRD_USE_GPU", "").lower()
        if env in ("1", "true", "yes", "y"):
            want_gpu = True
        elif env in ("0", "false", "no", "n"):
            want_gpu = False
        else:
            want_gpu = None  # Will be auto-detected below

    # 3. Auto-detect based on availability
    if want_gpu is None:
        want_gpu = torch.cuda.is_available()

    # Final decision
    if want_gpu and torch.cuda.is_available():
        _device = "cuda"
    else:
        if want_gpu and verbose:
            print("CUDA requested but not available. Falling back to CPU.")
        _device = "cpu"

    torch.set_default_device(_device)

    if verbose:
        print(f"Using device: {_device}")

    return _device


def get_selected_device():
    return _device
