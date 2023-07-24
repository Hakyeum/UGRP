from transformers import AutoTokenizer
from sklearn.cluster import KMeans
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_korean_sentence(sentence): ## 한국어 문장(sentence)를 입력받아 토큰화하여 토큰리스트를 반환하는 함수
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-discriminator") # 원하는 토크나이저 모델 로드 --> 우리가 원하는 것에 맞춰서 바꿀 수 있나?
    tokens = tokenizer.tokenize(sentence) # 토큰리스트 저장
    return tokens # 토큰리스트 반환

def cluster_tokens(tokens): ## 토큰화된 텍스트를 입력받고 토큰을 클러스트링하여 비슷한 토큰을 묶어줌
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-discriminator") # 토크나이저 모델 로드
    token_ids = [tokenizer.encode(token, add_special_tokens=False) for token in tokens] # 토큰을 인코딩?? --> 정확히 무슨 일이 발생하는 거지?

    # 패딩해서 토큰 벡터들의 크기 맞추기
    max_length = max(len(token) for token in token_ids) # --> 이 부분에서 왜 패딩을 해야하는지 무슨일이 발생하는지 벡터를 어떻게 확인할 수 있는지 잘 모르겠음
    # 아마도 토큰의 벡터 차원을 일정하게 만드는 것 같음
    padded_token_ids = pad_sequences(token_ids, maxlen=max_length, padding='post', truncating='post') # 설정한 max_length로 패딩
    # padding : 패딩되는 위치 / truncating : 자르는 위치 pre와 post로 구분

    token_vectors = np.array(padded_token_ids) # 패딩이 완료된 토큰 벡터들을 담는 리스트
    kmeans = KMeans(n_clusters=5)  # 원하는 클러스터 수로 설정 --> KMeans라는 알고리즘이 어떤 식으로 구성되어 있는지 살펴볼 필요가 있음
    kmeans.fit(token_vectors) # 본격적인 클러스팅 부분 kmeans는 클러스터로 나누는데 중심이 되는 center를 설정한다
    clustered_tokens = [[] for _ in range(kmeans.n_clusters)] # 클러스터 개수만큼 빈리스트를 만들어냄
    for i, token in enumerate(tokens):
        cluster_id = kmeans.labels_[i]
        clustered_tokens[cluster_id].append(token)
        # 어떤 클러스터 그룹에 어떤 토큰이 들어가는지를 결정
    return clustered_tokens # 완성된 클러스터 그룹 리스트를 반환

if __name__ == '__main__': # 스크립트가 시작되면 start
    file_path = input("텍스트 파일 경로를 입력하세요: ") # 텍스트 파일 경로 입력
    with open(file_path, 'r', encoding='utf-8') as file: # 경로 처리
        content = file.read()

    tokens = tokenize_korean_sentence(content) # 토큰화 함수 사용
    print("토큰 분리 결과:", tokens) # 출력
    clustered_tokens = cluster_tokens(tokens) # 클러스터 함수 사용
    print("클러스터링 결과:", clustered_tokens) # 출력
