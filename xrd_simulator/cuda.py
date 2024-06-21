import torch

# Default to False
use_cuda = False

# ===============================================
try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        use_cuda = True
        print("CUDA is available and GPUs are found.")
    else:
        print("CUDA is not available.")
except Exception as e:
    print("An error occurred while checking for CUDA. Exception:", e)

# Print final status
print("use_cuda =", use_cuda)
