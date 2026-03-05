import torch

# Default
torch.set_default_device("cpu")
_device = "cpu"


def configure_device(device=None, verbose=True):
    """Configure PyTorch device for xrd_simulator.

    Parameters
    ----------
    device : str, optional
        Device to use. Options:

        - ``"cpu"``: Force CPU
        - ``"gpu"`` or ``"cuda"``: Use GPU if available
        - ``None`` or ``"auto"``: Auto-detect (use GPU if available)
    verbose : bool, optional
        Print device selection message. Default is ``True``.

    Returns
    -------
    str
        ``'cuda'`` or ``'cpu'``.

    Examples
    --------
    >>> configure_device("gpu")      # Use GPU if available
    >>> configure_device("cpu")      # Force CPU
    >>> configure_device()           # Auto-detect
    """
    global _device

    # Normalize input
    if device is not None:
        device = str(device).lower().strip()
    
    # Determine if GPU is wanted
    if device is None or device == "auto":
        # Auto-detect: use GPU if available
        want_gpu = torch.cuda.is_available()
    elif device in ("gpu", "cuda"):
        want_gpu = True
    elif device == "cpu":
        want_gpu = False
    else:
        raise ValueError(
            f"Invalid device '{device}'. Must be one of: 'cpu', 'gpu', 'cuda', 'auto', or None"
        )

    # Set device based on availability
    if want_gpu and torch.cuda.is_available():
        _device = "cuda"
    else:
        if want_gpu and not torch.cuda.is_available() and verbose:
            print("CUDA requested but not available. Falling back to CPU.")
        _device = "cpu"

    torch.set_default_device(_device)

    if verbose:
        print(f"Using device: {_device}")

    return _device


def get_selected_device():
    """Get the currently configured device."""
    return _device