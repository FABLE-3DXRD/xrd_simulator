import torch
import numpy as np
import pandas as pd
# Default to False
frame = np


# ===============================================
try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        frame = torch
        print("CUDA is available and GPUs are found.")
    else:
        print("CUDA is not available.")
except Exception as e:
    print("An error occurred while checking for CUDA. Exception:", e)

