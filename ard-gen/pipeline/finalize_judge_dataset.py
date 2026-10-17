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
        assert s["instruction"]
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
    episodes_by_id: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_split[s["split"]]["total"] += 1
        by_split[s["split"]]["success" if s["success"] else "fail"] += 1
        by_split[s["split"]][f"failure_type:{s['failure_type']}"] += 1
        by_source_disturbance[(s["source"], s["disturbance_type"])] += 1
        episodes_by_id[s["episode_id"]].append(s)
        if s["slipped_but_succeeded"]:
            n_slipped_but_succeeded += 1

    # 조건부(injected_slip_F2/F3) hard-negative 비율 -- per-sample 전체
    # 대비(5.68%, A/C 포함 전체 희석) 수치와 per-episode(≡insert 서브태스크
    # success-only per-sample과 동치, episode당 insert 경계는 하나뿐이라서)
    # 수치를 둘 다 남긴다. v1에서 "35.8%~39.2%"로 보고됐던 수치는 후자와
    # 같은 단위(정의 자체는 안 바뀜, 집계 단위만 다름) -- PIPELINE.md
    # "Judge 데이터 v2" 절 참고.
    slip_types = {"injected_slip_F2", "injected_slip_F3"}
    slip_episode_ids = {eid for eid, ss in episodes_by_id.items() if any(s["disturbance_type"] in slip_types for s in ss)}
    n_slip_episodes = len(slip_episode_ids)
    n_slip_hard_neg_episodes = sum(1 for eid in slip_episode_ids if any(s["slipped_but_succeeded"] for s in episodes_by_id[eid]))

    return {
        "by_split": {k: dict(v) for k, v in by_split.items()},
        "by_source_disturbance": {f"{k[0]}:{k[1]}": v for k, v in by_source_disturbance.items()},
        "n_slipped_but_succeeded": n_slipped_but_succeeded,
        "slipped_but_succeeded_rate": n_slipped_but_succeeded / len(samples) if samples else 0.0,
        "slipped_but_succeeded_rate_note": (
            "위 rate는 전체 판정 샘플(모든 서브태스크·모든 source A/B/C) 대비 -- injected_slip이 "
            "없는 A/C 대량 표본에 희석됨. injected_slip_F2/F3가 걸린 B 에피소드만 조건부로 보면 "
            "(아래 conditional_*) 훨씬 높다 -- 두 수치 다 같은 정의(slipped_but_succeeded)의 "
            "다른 집계 단위일 뿐, 서로 다른 버그가 아니다."
        ),
        "slipped_but_succeeded_rate_conditional_on_injected_slip": {
            "n_episodes": n_slip_episodes,
            "n_hard_negative_episodes": n_slip_hard_neg_episodes,
            "rate_per_episode": n_slip_hard_neg_episodes / n_slip_episodes if n_slip_episodes else None,
        },
    }


def build_meta(data_dir: str, samples: list[dict], manifest: dict, disturbance_cfg: dict) -> dict:
    dist = distribution_tables(samples)
    return {
        "dataset_version": "v2",
        "failure_type_index": {ft: i for i, ft in enumerate(FAILURE_TYPES)} | {"none": -100},
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
            "ARD-VLA decision_logic.py의 실제 구현(사용자 제공)을 그대로 재현: "
            "completed_str = ', '.join(state.completed) if state.completed else 'none'; "
            "f'{goal} | current subtask: {node.id} | completed: {completed_str}'. "
            "completed는 지금 서브태스크 '시작 전'에 끝난 서브태스크 id만 담고, 지금 시도 "
            "자신의 성공/실패는 절대 포함하지 않는다(v1의 라벨 누수 버그 수정 -- v1은 "
            "completed 필드에 '지금 서브태스크 자신의 성공 여부'를 넣어서 텍스트만으로도 "
            "정답을 알 수 있었다)."
        ),
        "progress_definition_note": (
            "judge_progress(=index.jsonl의 progress 필드)는 성공하면 항상 1.0, 실패하면 "
            "성공 판정 '기준'까지 얼마나 남았는지의 연속값이다: "
            "align 실패 = 1 - clip(cur_xy_err / outer_half, 0, 1) (outer_half = align 성공 "
            "판정의 박스 허용오차), insert 실패 = clip(raw_depth_unclipped / target_depth, 0, 1) "
            "(raw_depth_unclipped = xy 정렬 여부로 0으로 깎지 않은 원시 삽입 깊이). "
            "v1은 align을 '초기 오차 대비 비율'로 정의해서, 교란 때문에 애초에 거의 정렬된 "
            "채로 시작하는 씬(outer_half > 시작 오차)에서는 성공해도 progress가 0 근처에 "
            "눌러붙었다(B_000004/bad_gains 재현으로 확인). insert는 current_depth()가 xy 정렬 "
            "실패 시 깊이를 0으로 깎는 성공 판정용 게이팅을 진행도 라벨에도 그대로 써서, 깊이 "
            "삽입된 채 jam된 실패 사례까지 progress=0으로 보였다 -- 둘 다 v2에서 수정."
        ),
    }


_README_TEMPLATE = """# ARD-Gen closed-loop Judge 데이터셋 (v2)

peg-in-hole 폐루프(ARD-VLA, closed-loop-structure)의 Judge 헤드
(성공 여부·실패 유형·진행도) 학습/평가용 데이터셋. 서브태스크 경계
하나 = 판정 샘플 하나.

v1(`data/closed-loop-v1`)에서 외부 검토로 발견된 3가지를 고쳤다 -- 자세한
근거/실측은 `closed-loop-data` 브랜치 `ard-gen/PIPELINE.md`의 "Judge
데이터 v2" 절 참고:
1. **지시문 라벨 누수 제거**: `instruction_start`/`instruction_end` 두
   필드(완료 여부가 지금 서브태스크 자신의 성공/실패를 그대로 노출)를
   단일 `instruction` 필드로 통합 -- ARD-VLA decision_logic.py 그대로,
   완료 목록은 "지금 서브태스크 시작 전" 기준이라 성공/실패와 무관하게
   항상 같은 문자열.
2. **근접 카메라 추가**: 기존 top_cam/wide_cam은 유지하고, `images`의
   정확한 카메라 목록은 실제 생성된 `index.jsonl` 샘플을 보고 확인할 것
   (경계마다 캡처된 카메라가 다를 수 있음).
3. **진행도 라벨 재정의**: `progress` 필드가 실패해도 연속값(align=성공
   판정 기준 대비 오차, insert=xy 게이팅 없는 원시 깊이)을 가지도록 수정
   -- v1은 align 성공의 상당수가 progress=0, insert 실패의 90%가
   progress=0으로 사실상 이진값이었다.

## Colab에서 받기 (얕은 clone)

```bash
git clone --branch data/closed-loop-v2 --single-branch --depth 1 \\
    https://github.com/judangyee/robotwin.git closed-loop-v2
cd closed-loop-v2
```

## 구조

```
manifest.json       # 완료된 청크 목록(청크별 해시/개수)
index.jsonl         # 전체 판정 샘플(모든 청크 합친 것) -- 이걸 읽으면 됨
meta.json           # 실패 유형 인덱스 표, 분할별/유형별 개수, 생성 설정
chunks/{chunk_id}/
    images/*.jpg      # 카메라별 start/end (meta.json 참고)
    episodes_shard.npz  # 그 청크 에피소드들의 매 스텝 원본 시계열
    judge_samples.jsonl # index.jsonl의 원본 조각(청크 단위)
```

## 읽기 예시

```python
import json, os
from PIL import Image

root = "closed-loop-v2"
samples = [json.loads(l) for l in open(os.path.join(root, "index.jsonl"))]
s = samples[0]
cam = sorted(s["images"])[0]  # 실제 카메라 목록은 meta.json/샘플마다 다를 수 있음
img = Image.open(os.path.join(root, "chunks", s["chunk_id"], s["images"][cam]))
print(s["instruction"], s["success"], s["failure_type"], s["progress"])
```

## 메타데이터

`meta.json`의 `failure_type_index`가 `failure_type_idx` 필드의 공식
인덱스 표다(`none`=-100, ARD-VLA judge_failure_type 손실의 ignore_index와
맞춤 -- 그 외 `FAILURE_TYPES` 순서로 F1a/F1b/F2/F3/timeout). `instruction_
format_note`/`progress_definition_note`에 v1->v2 변경 내역이 그대로
적혀있다. 자세한 생성 설정/실측값은 `closed-loop-data` 브랜치의
`ard-gen/PIPELINE.md` "Judge 데이터 v2" 절 참고.
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
