"""
[핵심 아이디어]
문서를 '각 글자가 몇 번 나왔는가' 라는 숫자 묶음(벡터)으로 바꾸면,
문서끼리 코사인 유사돌르 계산할 수 있다.

[전체 흐름]
문서들 + 질문 ->
1. 등장 글자 목록(축) 정하기
2. 각 문서를 그 축 기준 벡터로 변환
3. 질문 벡터와 각 문서 벡터의 코사인 유사도 계산
4. 유사도 내림차순 정렬 = 검색 결과

[이 방식의 한계 - 6주차를 배우는 이유]
글자가 겹ㅊ벼야만 비슷하다고 판단한다.
"강아지"와 "반려견"은 뜻이 같아도 겹치는 글자가 업성 유사도가 낮다.
"""

import numpy as np

# 같은 폴더의 similarity.pydㅔ서 cosine 함수를 가져온다.
from similarity import cosine

def to_vector(text: str, vocab: list[str]) -> np.ndarray:
    """
    문서 -> 벡터 (Bag of Characters, 글자 가방)

    vocab의 각 글자가 text에 몇 번 나왔는지 세어 그 횟수를 늘어놓는다.
    예) vocab = ['가','나','다'], text = "가나가"
    '가' 2번, '나' 1번, '다' 0번 -> [2.0, 1.0, 0.0]
    vocab이 '축'이고, 각 숫자가 그 축 방향의 크기다.
    모든 문서가 같은 vocab을 써야 서로 비교할 수 있다.
    """
    # text.count(ch : text안에 ch가 몇 번 나오는지)
    return np.array([text.count(ch) for ch in vocab], dtype=float)

# 검색 대상 문서들
docs = [
"파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
"자바는 기업용 서버 개발에 많이 쓰입니다.",
"파이썬으로 데이터 분석과 머신러닝을 합니다.",
]
question = "파이썬으로 머신러닝 하려면?"

# 1. 축 정하기
# "".join(docs) : 문서 3개를 하나의 긴 문자열로 합침
# + question : 질문에만 있는 글자도 축에 포함시켜야 한다
# set(...) : 중복 제거
# sorted(...) : 순서 고정 (안 하면 실행할 때마다 축 순서가 바뀜)
vocab = sorted(set("".join(docs) + question))

# 2. 질문을 벡터로
q_vec = to_vector(question, vocab)

# 3. 각 문서마다 (유사도, 문서) 쌍을 만든다.
# 리스트 컴프리헨션이 docs를 하나씩 돌며 d에 넣고 유사도를 계산한다.

scores = [(cosine(q_vec, to_vector(d, vocab)),d) for d in docs]

# 4. 정렬. 튜플은 첫 번째 원소(유사도)부터 비교하므로 유사도 기준으로 정렬된다.
# reverse = True 로 큰 값이 앞에 오게 한다 = 유사도 내림차순
scores.sort(reverse=True)

print(f"질문: {question}")
print(f"벡터 차원: {len(vocab)}\n") # 축의 개수 = 등장한 서로 다른 글자 수

#enumerate(scores, 1) : 1부터 번호를 붙여준다. (1위, 2위 ...)
for rank, (score, doc) in enumerate(scores, 1):
    print(f"{rank}위 (유사도 {score:.4f}) {doc}")