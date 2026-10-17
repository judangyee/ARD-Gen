# ARD-Gen closed-loop Judge 데이터셋 (v1)

peg-in-hole 폐루프(ARD-VLA, closed-loop-structure)의 Judge 헤드
(성공 여부·실패 유형·진행도) 학습/평가용 데이터셋. 서브태스크 경계
하나 = 판정 샘플 하나.

## Colab에서 받기 (얕은 clone)

```bash
git clone --branch data/closed-loop-v1 --single-branch --depth 1 \
    https://github.com/judangyee/robotwin.git closed-loop-v1
cd closed-loop-v1
```

## 구조

```
manifest.json       # 완료된 청크 목록(청크별 해시/개수)
index.jsonl         # 전체 판정 샘플(모든 청크 합친 것) -- 이걸 읽으면 됨
meta.json           # 실패 유형 인덱스 표, 분할별/유형별 개수, 생성 설정
chunks/{chunk_id}/
    images/*.jpg      # top_cam/wide_cam x start/end
    episodes_shard.npz  # 그 청크 에피소드들의 매 스텝 원본 시계열
    judge_samples.jsonl # index.jsonl의 원본 조각(청크 단위)
```

## 읽기 예시

```python
import json, os
from PIL import Image

root = "closed-loop-v1"
samples = [json.loads(l) for l in open(os.path.join(root, "index.jsonl"))]
s = samples[0]
img = Image.open(os.path.join(root, "chunks", s["chunk_id"], s["images"]["top_cam_end"]))
print(s["instruction_end"], s["success"], s["failure_type"], s["progress"])
```

## 메타데이터

`meta.json`의 `failure_type_index`가 `failure_type_idx` 필드의 공식
인덱스 표다(`none`=-1, 그 외 `FAILURE_TYPES` 순서 -- F1a/F1b/F2/F3/
timeout). 자세한 생성 설정/실측값은 `closed-loop-data` 브랜치의
`ard-gen/PIPELINE.md` "Judge 학습용 중규모 데이터셋" 절 참고.
