import tensorflow as tf

# Default to False
use_cuda = False

# ===============================================
try:
    # Check if TensorFlow can see any GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        use_cuda = True
        print("CUDA is available and GPUs are found.")
    else:
        print("CUDA is available but no GPUs are found.")
except Exception as e:
    print("CUDA is not available. Switching to CPU. Exception:", e)

# Print final status
print("use_cuda =", use_cuda)