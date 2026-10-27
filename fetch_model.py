"""임베딩 모델을 미리 내려받습니다. 캐시는 ~/.cache/huggingface 에 저장되어 다음부터는 바로 로드됩니다."""
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('intfloat/multilingual-e5-small')
print("차원 :", m.encode(['테스트']).shape) # (1, 384) 이면 정상
print("임베딩 모델 준비 완료")