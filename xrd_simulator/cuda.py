import torch
# Default to False

# ===============================================
torch.set_default_device('cpu')
try:
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available and GPUs are found.")
        gpu = input("Do you want to run in GPU? [y/n]").strip().lower() or 'y'
        if gpu == 'y':
            torch.set_default_device('cuda')
            print("Running in GPU...") 
        else:
            print("Running in CPU...")      
    else:
        print("CUDA is not available.")
except Exception as e:
    print("An error occurred while checking for CUDA. Exception:", e)

