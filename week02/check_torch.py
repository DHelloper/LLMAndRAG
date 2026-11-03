import torch, platform

print("아키텍처 :", platform.machine()) # arm64여야 정상
print("torch  :", torch.__version__)
print("MPS 가능:", torch.backends.mps.is_available()) # True 면 GPU 사용 가능

