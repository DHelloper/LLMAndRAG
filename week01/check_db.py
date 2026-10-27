"""Python에서 PostgreSQL에 붙고 vector 타입이 동작하는지 확인합니다."""
import psycopg

# with 블록을 나가면 연결이 자동으로 닫힙니다
with psycopg.connect("postgresql://localhost/ragdb") as conn:
# <=> 는 pgvector의 코사인 거리 연산자.
# 같은 벡터끼리는 거리가 0 이어야 정상입니다.
    v = conn.execute("SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector").fetchone()
    print("자기 자신과의 코사인 거리 :", v[0]) # 0.0 이면 정상