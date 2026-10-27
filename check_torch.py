"""PyTorch가 M4 GPU를 인식하는지 확인합니다."""
import torch
print("torch :", torch.__version__)
# is_available(): 지금 이 기기에서 MPS(GPU)를 쓸 수 있는가
print("MPS 사용 가능 :", torch.backends.mps.is_available())
# is_built(): 설치된 torch가 MPS 지원으로 빌드됐는가
print("MPS 빌드됨 :", torch.backends.mps.is_built())
# 쓸 수 있으면 GPU(mps), 아니면 CPU를 선택
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 1000x1000 난수 행렬을 그 장치에 올려 행렬곱 1회 — 실제 연산까지 되는지 검증
x = torch.randn(1000, 1000, device=dev)
print("행렬곱 결과 shape :", (x @ x).shape, "| device :", dev)


# 기대 출력
# torch : 2.5.1
# MPS 사용 가능 : True
# MPS 빌드됨 : True
# 행렬곱 결과 shape : torch.Size([1000, 1000]) | device : mps