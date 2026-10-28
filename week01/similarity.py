import numpy as np

def softmax(x):
    """점수 배열 -> 합이 1인 확률 배열

    [동작 순서]
    입력 x = [1,2]
    1. exp를 씌워 전부 양수로 만든다 -> [2.718, 7.389]
    2. 전체 합으로 나눠 합이 1이 되게 -> [0.269, 0.731]

    [왜 exp 인가]
    그냥 합으로 나누면 음수 정수가 음수 확률 이된다. exp는 어떤 수를 넣어도 양수가 나오고,
    점수의 '차이'를 곱셈 비율로 바꿔 준다.
    점수가 1 차이면 확률은 항상 e배(약 2.72배) 차이가 난다.
    """
    # asarray : 리스트가 들어와도 numpy 배열로 바꿔준다.
    # dtype=float : 정수 배열이면 나눗셈 결과가 0으로 잘릴 수 있으므로 실수로 고정.
    x = np.asarray(x, dtype=float)
    
    # 1. exp. 단 그냥 np.exp(x)를 쓰면 x가 클 때 터진다.
    # exp(1000) = inf가 되고 inf/inf = nan 이 되어 결과가 전부 망가진다.
    # x.max()를 빼도 결과는 수학적으로 완전히 동일하다.
    # exp(a-m) / Sigma exp(b-m) = (exp(a)/exp(m)) / (Sigma exp(b)/exp(m)) = exp(a) / Sigma exp(b)
    # 분자 분모에서 exp(m)이 약분되기 때문이다.
    # 빼고 나면 가장 큰 값이 0이 되어 exp가 최대 1이므로 절대 넘치지 않는다.
    e = np.exp(x - x.max())

    # 2. 전체 합으로 나누기 -> 합이 1
    return e / e.sum()

def cosine(a,b) :
    """
    두 벡터의 코사인 유사도 = 두 벡터가 이루는 각도의 cos 값

    [동작 순서]
    1. 내적 : 같은 자리끼리 곱해서 전부 더한다.
    2. 각 길이 : sqrt(제곱의 합)
    3. 내적을 (길이 x 길이)로 나눈다
    
    [결과 해석]
    1  : 방향이 완전히 같음(매우 비슷)
    0  : 직각 (무관)
    -1 : 정반대

    [핵심]
    길이를 나눠 없애므로 '방향'만 본다.
    그래서 긴 문서와 짧은 문서도 내용이 같으면 높게 나온다.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # a @ b : 내적 (numpy 에서 @는 행렬곱/내적)
    # np.linalg.norm(a) : 벡터 길이 = sqrt(a[0]^2 + a[1]^2 + ...)
    # float(...) : numpy 스칼라를 보통 파이썬 실수로 바꿔 출력이 깔끔해진다.
    return float(a@b / (np.linalg.norm(a) * np.linalg.norm(b)))

if __name__ == "__main__":
    print("cos([1,0], [0.8, 0.2]) =", round(cosine([1,0], [0.8,0.2]), 4))

    # [8,2]는 [0.8, 0.2]를 10배 늘린 것 = 길이만 10배, 방향은 동일
    print("cos([1,0], [8,2] =)", round(cosine([1,0],[8,2]),4))
    print(" -> 길이가 10배여도 값이 같다. 코사인은 방향만 본다.\n")

    print("softmax([1,2]) = ", np.round(softmax([1,2]),4))

    # [101,102]는 [1,2]에 100을 더한 것. 두 점수의 '차이'는 여전히 1이다.
    # softmax 는 차이만 보므로 결과가 완전히 같아야 한다.
    print("softmax([101,102]) = ", np.round(softmax([101,102]),4))
    print(" -> 절대값이 아니라 '차이'만 영향을 준다.\n")

    # softmax에 넣기 전에 점수를 T로 나누면 분포의 뾰족함이 바뀐다.
    # 4주차 생성과 5주차 Ollama의 temperature 옵션이 이것.
    scores = np.array([1.0,2.0])

    for T in (0.5, 1.0, 2.0) :
        # T가 작으면 -> 나눈 값의 차이가 커짐 -> 1등에 확률이 쏠림 (보수적)
        # T가 크면 -> 차이가 줄어듦 -> 고르게 퍼짐 (창의적)
        print(f"T={T}: {np.round(softmax(scores / T),3)}")