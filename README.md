# AI 기반 의류 이미지 유사도 및 스타일 검색 시스템

사용자가 업로드한 의류 이미지를 기반으로 데이터베이스에서 시각적으로 유사한 의류를 검색하는 시스템입니다.

CLIP ViT-B/32 이미지 인코더를 사용하여 의류 이미지를 512차원 특징 벡터로 변환하고, Cosine Similarity를 이용하여 유사도가 높은 Top-5 이미지를 검색합니다.

## 주요 기능

- 의류 이미지 업로드
- rembg 기반 배경 제거
- CLIP ViT-B/32 기반 이미지 특징 추출
- L2 Normalization
- Cosine Similarity 기반 벡터 검색
- Top-5 유사 의류 검색 결과 제공
- 검색 단계별 처리시간 측정

## 시스템 구조

### 1. Vector DB 구축

```text
의류 이미지
    ↓
배경 제거 (rembg)
    ↓
CLIP ViT-B/32
    ↓
512차원 특징 벡터
    ↓
L2 Normalization
    ↓
Vector DB
```

`build_db.py`를 실행하여 데이터베이스 이미지의 특징 벡터를 사전에 추출합니다.

생성되는 파일:

- `db_vectors.npy`: 의류 이미지 특징 벡터
- `db_paths.npy`: 원본 이미지 경로

### 2. 이미지 검색

```text
Query 이미지
    ↓
배경 제거 (rembg)
    ↓
CLIP ViT-B/32
    ↓
512차원 특징 벡터
    ↓
L2 Normalization
    ↓
Cosine Similarity
    ↓
유사도 내림차순 정렬
    ↓
Top-5 결과
```

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 이미지 데이터 준비

프로젝트 루트에 `images` 디렉터리를 생성하고 검색 대상 의류 이미지를 저장합니다.

```text
images/
├── cloth_001.jpg
├── cloth_002.jpg
├── ...
└── cloth_100.jpg
```

### 3. Vector DB 구축

```bash
python build_db.py
```

실행 후 다음 파일이 생성됩니다.

```text
db_vectors.npy
db_paths.npy
```

### 4. 검색 시스템 실행

```bash
streamlit run app.py
```

웹 브라우저에서 Query 이미지를 업로드한 후 유사 의류 검색을 실행할 수 있습니다.

## 사용 기술

- Python
- PyTorch
- Hugging Face Transformers
- CLIP ViT-B/32
- rembg
- NumPy
- Streamlit
- Pillow

## 현재 구현 범위

현재 시스템은 100개의 의류 이미지로 구성된 소규모 데이터베이스를 대상으로 구현되었습니다.

검색 대상 이미지의 특징 벡터는 사전에 구축하며, Query 이미지와 데이터베이스 특징 벡터 간 Cosine Similarity를 계산하여 Top-5 결과를 반환합니다.

현재 버전에서는 별도의 의류 카테고리 분류를 적용하지 않기 때문에 서로 다른 카테고리의 이미지가 검색 결과에 포함될 수 있습니다.

## 향후 개선 방향

- 의류 카테고리 분류 및 검색 범위 제한
- 색상 특징을 결합한 Hybrid Similarity
- 배경 제거 과정의 처리속도 개선
- 대규모 데이터베이스를 위한 ANN 기반 검색 적용
