import torch
import platform

#Display system and hardware information
print("System Information:")
print(f"Operating System: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")

#Check for M4 availability(MPS)
print("\n--- Pytorch Info ---")
is_mps_available = torch.backends.mps.is_available()
print(f"Is M4 (MPS) available: {is_mps_available}")

if is_mps_available:
    device = torch.device("mps")
    data = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor on device:{data.device}")
    print("M4 (MPS) is working correctly.")
else:
    print("M4 (MPS) is not available on this system.")