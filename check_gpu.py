import torch
import paddle

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

print(f"Paddle CUDA compiled: {paddle.is_compiled_with_cuda()}")
