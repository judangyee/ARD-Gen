"""생성이 끝난 뒤(`generate_judge_dataset.py`가 모든 청크를 완료한 뒤) 한
번 돌려서 통합 `index.jsonl`/`meta.json`/`README.md`를 만들고 검증한다
-- 사용자 지시 "마지막에 전체 인덱스·meta.json·README.md를 만든다"
+ "검증(마지막에)" 절 그대로. 데이터 자체는 안 건드리고(청크 디렉터리는
read-only로 취급) 루트에 세 파일만 새로 쓰거나 덮어쓴다.

사용법:
    python pipeline/finalize_judge_dataset.py --data-dir /home/user/robotwin-data
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from closedloop.peg_in_hole_adapter import FAILURE_TYPES
from pipeline.generate_closed_loop_episodes import load_disturbance_config


def _code_commit_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_index(data_dir: str) -> list[dict]:
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    samples = []
    for chunk_id in sorted(manifest["completed_chunks"]):
        jsonl_path = os.path.join(data_dir, "chunks", chunk_id, "judge_samples.jsonl")
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples, manifest


def validate_index(data_dir: str, samples: list[dict]) -> dict:
    """사용자 지시 "검증" 절 그대로 -- 이미지 경로 존재+열림, 라벨/16dof/
    instruction 전부 있는지. 결과 요약 dict를 돌려준다(실패하면 assert)."""
    from PIL import Image

    n_images_checked = 0
    for s in samples:
        for key in ("sample_id", "episode_id", "subtask", "success", "failure_type", "progress"):
            assert key in s and s[key] is not None or key == "progress", f"{s.get('sample_id')}: {key} 없음"
        assert len(s["robot_state_16dof_start"]) == 16
        assert len(s["robot_state_16dof_end"]) == 16
        assert s["instruction_start"] and s["instruction_end"]
        for cam_key, rel_path in s["images"].items():
            full_path = os.path.join(data_dir, "chunks", s["chunk_id"], rel_path)
            assert os.path.exists(full_path), f"{s['sample_id']}: 이미지 없음 {full_path}"
            with Image.open(full_path) as im:
                im.verify()
            n_images_checked += 1
    return {"n_samples": len(samples), "n_images_checked": n_images_checked}


def distribution_tables(samples: list[dict]) -> dict:
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_source_disturbance: dict[tuple, int] = Counter()
    n_slipped_but_succeeded = 0
    for s in samples:
        by_split[s["split"]]["total"] += 1
        by_split[s["split"]]["success" if s["success"] else "fail"] += 1
        by_split[s["split"]][f"failure_type:{s['failure_type']}"] += 1
        by_source_disturbance[(s["source"], s["disturbance_type"])] += 1
        if s["slipped_but_succeeded"]:
            n_slipped_but_succeeded += 1
    return {
        "by_split": {k: dict(v) for k, v in by_split.items()},
        "by_source_disturbance": {f"{k[0]}:{k[1]}": v for k, v in by_source_disturbance.items()},
        "n_slipped_but_succeeded": n_slipped_but_succeeded,
        "slipped_but_succeeded_rate": n_slipped_but_succeeded / len(samples) if samples else 0.0,
    }


def build_meta(data_dir: str, samples: list[dict], manifest: dict, disturbance_cfg: dict) -> dict:
    dist = distribution_tables(samples)
    return {
        "failure_type_index": {ft: i for i, ft in enumerate(FAILURE_TYPES)} | {"none": -1},
        "code_commit": _code_commit_hash(),
        "render": {"width": 320, "height": 240, "image_format": "JPEG", "jpeg_quality": 90},
        "n_chunks": len(manifest["completed_chunks"]),
        "n_episodes": sum(c["n_episodes"] for c in manifest["completed_chunks"].values()),
        "n_samples": len(samples),
        "splits": sorted({s["split"] for s in samples}),
        "sources": sorted({s["source"] for s in samples}),
        "distribution": dist,
        "disturbance_config": disturbance_cfg,
        "seed_rule": {
            "formula": "seed = SOURCE_SEED_BASE[source] + episode_idx",
            "source_seed_base": {"A": 0, "B": 1_000_000, "C": 2_000_000},
            "note": "출처별로 겹치지 않는 시드 공간(각 최대 100만 에피소드) -- 어느 워커가 돌려도 (source,episode_idx) 하나는 항상 같은 물리/이미지 결과.",
        },
        "split_rule": {
            "A": "전부 test_A", "C": "전부 test_C",
            "B": "episode_idx < b_train(=600) -> train, < b_train+b_val(=700) -> val, 나머지 -> test_B",
        },
        "instruction_format_note": (
            "ARD-VLA decision_logic.instruction_text()의 형식(\"목표 | current subtask: ... | "
            "completed: ...\")을 사용자 지시대로 재현했다 -- 이 세션에는 ARD-VLA 소스가 없어서 "
            "실제 구현과 토씨까지 같다는 보장은 없다(정직하게 기록)."
        ),
    }


_README_TEMPLATE = """# ARD-Gen closed-loop Judge 데이터셋 (v1)

peg-in-hole 폐루프(ARD-VLA, closed-loop-structure)의 Judge 헤드
(성공 여부·실패 유형·진행도) 학습/평가용 데이터셋. 서브태스크 경계
하나 = 판정 샘플 하나.

## Colab에서 받기 (얕은 clone)

```bash
git clone --branch data/closed-loop-v1 --single-branch --depth 1 \\
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--disturbance-config", type=str, default=os.path.join(os.path.dirname(__file__), "disturbance_config.yaml"))
    args = parser.parse_args()

    samples, manifest = build_index(args.data_dir)
    print(f"인덱스 {len(samples)}개 샘플, {len(manifest['completed_chunks'])}개 청크")

    validation = validate_index(args.data_dir, samples)
    print(f"[OK] 검증: {validation}")

    disturbance_cfg = load_disturbance_config(args.disturbance_config)
    meta = build_meta(args.data_dir, samples, manifest, disturbance_cfg)

    with open(os.path.join(args.data_dir, "index.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
    with open(os.path.join(args.data_dir, "README.md"), "w") as f:
        f.write(_README_TEMPLATE)

    print(json.dumps(meta["distribution"], indent=2, ensure_ascii=False))
    print(f"[OK] index.jsonl/meta.json/README.md 작성 완료 ({args.data_dir})")


if __name__ == "__main__":
    main()
