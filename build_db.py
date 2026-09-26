import os
import glob
import time
import numpy as np
import torch
from PIL import Image
from rembg import remove
from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = "images"

DB_VECTOR_FILE = "db_vectors.npy"
DB_PATH_FILE = "db_paths.npy"


print("[1/3] CLIP 모델 및 이미지 프로세서 로드 중...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model.eval()

print("CLIP 모델 로드 완료.")
print()


def extract_clip_feature(image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        image_features = model.get_image_features(
            **inputs
        )

        if hasattr(
            image_features,
            "pooler_output"
        ):
            image_features = (
                image_features.pooler_output
            )

        elif hasattr(
            image_features,
            "image_embeds"
        ):
            image_features = (
                image_features.image_embeds
            )

        image_features = (
            image_features
            / torch.norm(
                image_features,
                p=2,
                dim=-1,
                keepdim=True
            )
        )

    vector = (
        image_features
        .cpu()
        .numpy()
        .flatten()
    )

    return vector


image_paths = (
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpg"
        )
    )
    +
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpeg"
        )
    )
    +
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)

image_paths = sorted(image_paths)


if not image_paths:
    print(
        f"[오류] '{IMAGE_DIR}' 폴더에 "
        "의류 이미지 파일이 없습니다."
    )

    raise SystemExit(1)


print(
    f"[2/3] 총 {len(image_paths)}개의 "
    "의류 이미지 전처리 및 특징 추출 시작..."
)

print()


db_vectors = []
valid_paths = []

start_time = time.perf_counter()


for idx, path in enumerate(
    image_paths,
    start=1
):
    try:
        raw_img = Image.open(
            path
        ).convert("RGB")

        nobg_img = remove(
            raw_img
        )

        vector = extract_clip_feature(
            nobg_img
        )

        if vector.ndim != 1:
            raise ValueError(
                f"잘못된 Vector Shape: "
                f"{vector.shape}"
            )

        if len(vector) != 512:
            raise ValueError(
                f"CLIP Vector Dimension이 "
                f"512가 아닙니다: {len(vector)}"
            )

        db_vectors.append(
            vector
        )

        valid_paths.append(
            path
        )

        print(
            f"[{idx:03d}/{len(image_paths):03d}] "
            f"성공: {os.path.basename(path)}"
        )

    except Exception as e:
        print(
            f"[{idx:03d}/{len(image_paths):03d}] "
            f"실패: {path}"
        )

        print(
            f"    오류: {e}"
        )


if not db_vectors:
    print()
    print(
        "[오류] 저장 가능한 특징 벡터가 없습니다."
    )

    raise SystemExit(1)


print()
print(
    "[3/3] 추출된 Vector DB 저장 중..."
)


db_vectors = np.asarray(
    db_vectors,
    dtype=np.float32
)

db_paths = np.asarray(
    valid_paths
)


if db_vectors.ndim != 2:
    raise ValueError(
        f"잘못된 최종 DB Shape: "
        f"{db_vectors.shape}"
    )


if db_vectors.shape[1] != 512:
    raise ValueError(
        f"잘못된 Vector Dimension: "
        f"{db_vectors.shape}"
    )


if len(db_vectors) != len(db_paths):
    raise ValueError(
        "Vector 개수와 Path 개수가 "
        "일치하지 않습니다."
    )


np.save(
    DB_VECTOR_FILE,
    db_vectors
)

np.save(
    DB_PATH_FILE,
    db_paths
)


total_time = (
    time.perf_counter()
    - start_time
)

average_time = (
    total_time
    / len(db_vectors)
)


norms = np.linalg.norm(
    db_vectors,
    axis=1
)


print()
print("=" * 60)
print("Vector DB 구축 완료")
print("=" * 60)

print(
    f"전체 이미지 수     : "
    f"{len(image_paths)}"
)

print(
    f"성공 이미지 수     : "
    f"{len(db_vectors)}"
)

print(
    f"실패 이미지 수     : "
    f"{len(image_paths) - len(db_vectors)}"
)

print(
    f"Vector DB Shape   : "
    f"{db_vectors.shape}"
)

print(
    f"Vector Data Type  : "
    f"{db_vectors.dtype}"
)

print(
    f"총 구축 시간       : "
    f"{total_time:.2f}초"
)

print(
    f"이미지당 평균 시간 : "
    f"{average_time:.2f}초"
)

print(
    f"L2 Norm 최소값    : "
    f"{norms.min():.6f}"
)

print(
    f"L2 Norm 최대값    : "
    f"{norms.max():.6f}"
)

print(
    f"Vector DB 저장    : "
    f"{DB_VECTOR_FILE}"
)

print(
    f"Image Path 저장   : "
    f"{DB_PATH_FILE}"
)

print("=" * 60)


if not np.allclose(
    norms,
    1.0,
    atol=1e-5
):
    print(
        "[경고] 일부 특징 벡터의 "
        "L2 Norm이 1이 아닙니다."
    )

else:
    print(
        "L2 Normalization 검증 완료."
    )