"""문자 단위 토크나이저 - 4주차 Mini GPT에서 그대로 재사용"""

class CharTokenizer:
    def __init__(self, text: str):
        # 등장하는 문자를 정렬해 사전을 만든다 (정렬해야 재실행 시 ID가 고정 됨)
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)} #문자 -> ID
        self.itos = {i: ch for ch, i in self.stoi.items()} # ID -> 문자

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) ->  list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

if __name__ == "__main__":
    corpus = "나는 학교에 간다. 나는 병원에 간다."
    tok = CharTokenizer(corpus)

    print("사전 크기:", tok.vocab_size)
    print("사전 :",tok.stoi)

    ids = tok.encode("나는 학교에")
    print("encode :",ids)
    print("decode :",tok.decode(ids))

    # 왕복 검증 - 원문이 그대로 복원 되어야 함
    assert tok.decode(tok.encode(corpus)) == corpus
    print("왕복 검증 통과")
