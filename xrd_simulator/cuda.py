import torch
import os

# Default to CPU
torch.set_default_device("cpu")
device = "cpu"
torch.no_grad()


def configure_device(use_gpu=None, verbose=True):
    """
    Configure PyTorch device for xrd_simulator.
    
    Args:
        use_gpu (bool, optional): If True, use GPU. If False, use CPU. 
                                   If None, check environment variable XRD_USE_GPU.
        verbose (bool): Print device configuration messages.
    
    Returns:
        str: Device being used ('cuda' or 'cpu')
    
    Examples:
        >>> # Force GPU usage
        >>> configure_device(use_gpu=True)
        'cuda'
        
        >>> # Force CPU usage
        >>> configure_device(use_gpu=False)
        'cpu'
        
        >>> # Auto-detect from environment variable XRD_USE_GPU
        >>> configure_device()  # Reads XRD_USE_GPU env var
        
        >>> # Silent mode
        >>> configure_device(use_gpu=True, verbose=False)
        'cuda'
    """
    global device
    if use_gpu is None:
        # Check environment variable
        env_var = os.environ.get('XRD_USE_GPU', '').lower()
        if env_var in ('true', '1', 'yes', 'y'):
            use_gpu = True
        elif env_var in ('false', '0', 'no', 'n'):
            use_gpu = False
        else:
            # Fall back to interactive prompt only if not set
            use_gpu = None
    
    try:
        if torch.cuda.is_available():
            if use_gpu is None:
                # Interactive mode (only if not specified via env or argument)
                if verbose:
                    print("CUDA is available and GPUs are found.")
                try:
                    response = input("Do you want to run on GPU? [y/n] (default: y): ").strip().lower()
                    use_gpu = response != 'n'  # Default to yes
                except EOFError:
                    # Non-interactive environment (e.g., piped input, subprocess)
                    # Default to GPU if available
                    use_gpu = True
                    if verbose:
                        print("Non-interactive environment detected. Defaulting to GPU.")
            
            if use_gpu:
                torch.set_default_device("cuda")
                device = "cuda"
                if verbose:
                    print("Running on GPU (CUDA)...")
                return device
            else:
                torch.set_default_device("cpu")
                device = "cpu"
                if verbose:
                    print("Running on CPU...")
                return device
        else:
            if verbose:
                if use_gpu:
                    print("CUDA requested but not available. Falling back to CPU.")
                else:
                    print("CUDA is not available. Using CPU.")
            torch.set_default_device("cpu")
            device = "cpu"
            return device
    except Exception as e:
        if verbose:
            print(f"An error occurred while configuring device: {e}")
            print("Falling back to CPU.")
        torch.set_default_device("cpu")
        device = "cpu"
        return device


def get_selected_device() -> str:
    """Return the currently selected device ('cuda' or 'cpu')."""
    return device
