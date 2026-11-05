"""
앞의 manual_gd.py와 똑같은 계산을 PyTorch로 하기
달라지는 것은 미분을 손으로 유도하지 않고 autograd 에게 맡긴다는 점

[autograd 가 하는 일]
requires_grad=True 인 텐서가 끼어든 연산을 PyTorch가 전부 기록해 두었다가,
loss.backward() 를 부르면 기록을 거꾸로 따라가며 미분을 자동 계산한다.
이것이 '역전파'다. 층이 수백 개여도 원리는 같다.
"""

import torch

#입력과 정답. 학습 대상이 아니므로 requires_grad를 켜지 않는다.
x = torch.tensor(2.0)
t = torch.tensor(10.0)

# requires_grad=True : "이 텐서에 대한 미분을 추적하라" 는 표시.
# 이게 있어야 w.grad에 값이 채워진다.
w = torch.tensor(1.0,requires_grad=True)
lr = 0.1

for step in range(5):
    # 1. 순전파. 이 줄이 실행되는 순간 PyTorch가 "y는 w에 x를 곱해 만들었다" 를 기록한다.
    y = w * x

    # 2. 손실. 역시 "loss는 y에서 t를 빼고 제곱해 만들었다."가 기록된다.
    loss = (y - t) ** 2

    # 3. 역전파. 계산 그래프를 거꾸로 따라가며 미분을 구해 w.grad에 넣는다.
    loss.backward()

    # .item() : 텐서 안의 값 하나를 보통 파이썬 숫자로 꺼낸다 (출력용)
    print(f"step {step}: w={w.item():.4f} loss={loss.item():.6f} "
          f"grad={w.grad.item():.4f}")

    # 4. 갱신
    # torch.no_grad()로 감싸는 이유 :
    #       이 블록 안의 연산은 '학습 과정'이지 '모델 계산'이 아니다.
    #       감싸지 않으면 갱신 연산까지 계산 그래프에 기록되어
    #       그래프가 끝없이 길어지고 메모리를 잡아먹는다.
    with torch.no_grad():
        w -= lr * w.grad

    # * gradient 초기화
    # PyTorch 는 .gard를 덮어쓰지 않고 '더한다'
    # 초기화하지 않으면 2회차 grad = 1회차 grad + 2gㅚ차 grad가 되어
    # 보폭이 회차마다 커지고 결국 발산한다.
    # 왜 더하도록 만들었나 : 배치를 여러 조각으로 나눠 gradient를 누적한 뒤 한번에 갱신하는 기법(gradient accumulation)을 쓰기 위함이다.
    # 그런 걸 안 쓸 때는 매번 직접 0으로 비워 줘야 한다.
    w.grad.zero_()