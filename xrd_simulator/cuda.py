use_cuda = False    # Default to False

# ===============================================
try:
    import cupy as cp
    # Try to allocate a small array on GPU
    cp.zeros((1,), dtype=cp.float32)
    use_cuda = True
    print("CUDA is available")
except Exception as e:
    print("CUDA is not available. Switching to NumPy.")
# ===============================================