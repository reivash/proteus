"""Quick GPU availability test"""
import torch

print("=" * 60)
print("GPU AVAILABILITY TEST")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Quick tensor test
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("\n[SUCCESS] GPU tensor operations working!")
    print(f"Test result shape: {z.shape}")
else:
    print("\n[WARNING] No GPU detected - will use CPU")
    print("Check NVIDIA drivers and CUDA installation")

print("=" * 60)
