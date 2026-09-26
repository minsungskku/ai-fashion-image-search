import os
import time
import numpy as np
import torch
import streamlit as st
from PIL import Image
from rembg import remove
from transformers import CLIPProcessor, CLIPModel


st.set_page_config(
    page_title="AI 의류 이미지 유사도 및 스타일 검색 시스템",
    layout="wide"
)


@st.cache_resource
def load_resources():
    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    db_vectors = np.load("db_vectors.npy")
    db_paths = np.load("db_paths.npy")

    if db_vectors.ndim != 2:
        raise ValueError(
            f"잘못된 DB vector shape입니다: {db_vectors.shape}"
        )

    if db_vectors.shape[1] != 512:
        raise ValueError(
            f"CLIP vector dimension이 512가 아닙니다: "
            f"{db_vectors.shape}"
        )

    if len(db_vectors) != len(db_paths):
        raise ValueError(
            f"DB vector 수({len(db_vectors)})와 "
            f"path 수({len(db_paths)})가 일치하지 않습니다."
        )

    return model, processor, db_vectors, db_paths


def extract_clip_feature(image, model, processor):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        elif hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds

        image_features = image_features / torch.norm(
            image_features,
            p=2,
            dim=-1,
            keepdim=True
        )

    return image_features.cpu().numpy().flatten()


st.title("AI 기반 의류 이미지 유사도 및 스타일 검색 시스템")

st.caption(
    "rembg 기반 배경 제거 및 "
    "CLIP 특징 벡터 기반 유사 의류 검색"
)


try:
    model, processor, db_vectors, db_paths = load_resources()

    st.sidebar.success(
        f"DB 상태: 총 {len(db_paths)}개 의류 벡터 인덱싱 완료"
    )

    st.sidebar.caption(
        f"Vector DB Shape: {db_vectors.shape}"
    )

except FileNotFoundError:
    st.error(
        "Vector DB 파일을 찾을 수 없습니다. "
        "먼저 `python build_db.py`를 실행해주세요."
    )
    st.stop()

except Exception as e:
    st.error(
        f"리소스 로드 중 오류가 발생했습니다: {e}"
    )
    st.stop()


uploaded_file = st.file_uploader(
    "검색할 의류 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    try:
        query_raw = Image.open(uploaded_file).convert("RGB")

    except Exception as e:
        st.error(f"이미지를 읽을 수 없습니다: {e}")
        st.stop()

    col_input1, col_input2 = st.columns(2)

    with col_input1:
        st.image(
            query_raw,
            caption="1. 원본 입력 이미지",
            use_container_width=True
        )

    if st.button(
        "유사 의류 검색 실행",
        type="primary"
    ):
        total_start = time.perf_counter()

        try:
            preprocess_start = time.perf_counter()

            query_nobg = remove(query_raw)

            preprocess_latency = (
                time.perf_counter() - preprocess_start
            ) * 1000

            with col_input2:
                st.image(
                    query_nobg,
                    caption=(
                        "2. [전처리 결과] "
                        "배경 제거 (rembg)"
                    ),
                    use_container_width=True
                )

            feature_start = time.perf_counter()

            query_vec = extract_clip_feature(
                query_nobg,
                model,
                processor
            )

            feature_latency = (
                time.perf_counter() - feature_start
            ) * 1000

            search_start = time.perf_counter()

            similarities = np.dot(
                db_vectors,
                query_vec
            )

            top_k = min(
                5,
                len(db_paths)
            )

            top_indices = np.argsort(
                similarities
            )[::-1][:top_k]

            search_latency = (
                time.perf_counter() - search_start
            ) * 1000

            total_latency = (
                time.perf_counter() - total_start
            ) * 1000

            st.divider()

            st.subheader(
                f"유사 의류 검색 결과 (Top-{top_k})"
            )

            res_cols = st.columns(top_k)

            for rank, idx in enumerate(top_indices):
                similarity = float(
                    similarities[idx]
                )

                with res_cols[rank]:
                    st.image(
                        db_paths[idx],
                        use_container_width=True
                    )

                    st.metric(
                        label=f"Top-{rank + 1}",
                        value=f"{similarity:.3f}"
                    )

                    st.caption(
                        "Cosine Similarity"
                    )

                    st.caption(
                        f"파일명: "
                        f"{os.path.basename(db_paths[idx])}"
                    )

            st.divider()

            st.subheader(
                "Top-5 검색 결과 상세"
            )

            for rank, idx in enumerate(
                top_indices,
                start=1
            ):
                filename = os.path.basename(
                    db_paths[idx]
                )

                similarity = float(
                    similarities[idx]
                )

                st.write(
                    f"**{rank}. {filename}** "
                    f"— Cosine Similarity: "
                    f"`{similarity:.4f}`"
                )

            st.divider()

            st.subheader(
                "검색 처리 시간 분석"
            )

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            with m_col1:
                st.metric(
                    "배경 제거",
                    f"{preprocess_latency:.1f} ms"
                )

            with m_col2:
                st.metric(
                    "CLIP 특징 추출",
                    f"{feature_latency:.1f} ms"
                )

            with m_col3:
                st.metric(
                    "Vector Search",
                    f"{search_latency:.3f} ms"
                )

            with m_col4:
                st.metric(
                    "전체 응답 시간",
                    f"{total_latency:.1f} ms"
                )

            st.info(
                "Precision@K는 여러 평가용 Query 이미지의 "
                "Top-K 검색 결과를 기준으로 별도로 계산합니다."
            )

        except Exception as e:
            st.error(
                f"검색 처리 중 오류가 발생했습니다: {e}"
            )