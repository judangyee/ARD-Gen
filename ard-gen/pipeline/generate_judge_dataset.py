"""ARD-Gen closed-loop 후속 작업 -- ARD-VLA Judge 헤드 학습/평가용 중규모
데이터셋을 생성한다. 코드는 이 브랜치(`closed-loop-data`)에 있고, 산출물은
별도 고아 브랜치 `data/closed-loop-v1`(이 저장소, git worktree로
`/home/user/robotwin-data`에 체크아웃)에 쌓는다 -- 코드 히스토리를
무겁게 만들지 않으려고 분리했다(사용자 지시).

## 왜 run_one_episode()를 그대로 재사용했나

`pipeline/generate_closed_loop_episodes.py`의 `run_one_episode(...,
render=True, disturbance_config=..., visual_disturbance=...)`가 이미
이 작업에 필요한 전부를 한다 -- 서브태스크 경계마다 top_cam/wide_cam을
생성 중 바로 캡처(작업1 "방법 A"), B의 재조정된 교란 분포, C의 시각
교란, 복구(R1~R3, 실패 유형 재정의 작업). 이 스크립트는 그 결과를
1) 에피소드 단위로 결정론적 시드/분할에 매핑하고, 2) 이미지를 JPEG로
풀어 저장하고, 3) 서브태스크 경계마다 Judge 학습용 판정 샘플 레코드를
만들고, 4) 청크 단위로 커밋+푸시하는 오케스트레이션만 담당한다 -- 물리
롤아웃 자체를 다시 구현하지 않는다(최소 구현).

## 결정론적 시드/분할 (어느 워커가 돌려도 같은 결과)

`episode_seed(source, episode_idx) = _SOURCE_SEED_BASE[source] +
episode_idx`로 출처별로 겹치지 않는 시드 공간을 쓴다(A=0대, B=100만대,
C=200만대 -- 출처당 최대 100만 에피소드까지 안전). 매 에피소드마다
`np.random.default_rng(seed)`를 새로 만들어서 쓰므로(기존 생성
스크립트들처럼 하나의 rng를 여러 에피소드에 걸쳐 이어 쓰지 않음),
어느 프로세스에서 어떤 순서로 돌든 (source, episode_idx) 하나는 항상
같은 물리 결과를 낸다 -- 멀티프로세싱 병렬화와 "이어서 생성"(청크
재개)이 성립하려면 필수 전제.

## instruction_text 포맷 (ARD-VLA decision_logic.instruction_text() 재현)

이 저장소에는 ARD-VLA 소스가 없다(사용자 지시 "ARD-VLA 저장소 수정은
범위 밖" -- 수정은커녕 체크아웃도 안 했다). 사용자 지시에 적힌 형식
("목표 | current subtask: … | completed: …")을 그대로 문자열로
재현했다 -- 실제 ARD-VLA 구현과 토씨까지 같다는 보장은 없다(정직하게
기록, README/meta.json에도 남김).

## 이미지: JPEG, 해상도는 기존 렌더러 그대로(320x240)

mm 단위 상태 변화가 이 해상도에서 실제로 보이는지는 확인 안 했다
(한계로 명시, 완료 보고 참고) -- 기존 render_frames()/Renderer(height=
240, width=320) 설정을 그대로 썼다(바꾸는 건 범위 밖).

사용법:
    MUJOCO_GL=osmesa python pipeline/generate_judge_dataset.py \\
        --data-dir /home/user/robotwin-data --chunk-size 50 \\
        --n-a 300 --n-b 800 --n-c 300 --workers 4
    (--pilot N 으로 출처당 N개만 돌리는 파일럿 모드)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# 이 스크립트는 항상 렌더링한다(Judge 데이터셋 자체가 이미지 필요) --
# generate_closed_loop_episodes.py처럼 --render 유무로 조건부 설정할
# 필요 없이 맨 위에서 바로 고정한다. mujoco가 자기 모듈 import 시점에만
# 이 값을 읽으므로(실측 확인, PIPELINE.md 참고) 다른 import보다 먼저.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from typing import Any

import numpy as np
from PIL import Image

from pipeline.generate_closed_loop_episodes import (
    VisualDisturbanceConfig,
    assert_no_torch_imported,
    load_disturbance_config,
    run_one_episode,
)
from closedloop.peg_in_hole_adapter import FAILURE_TYPES

_GOAL_TEXT = "insert the peg into the hole"
_SOURCE_SEED_BASE = {"A": 0, "B": 1_000_000, "C": 2_000_000}
_SLIP_THRESHOLD_M = 0.003
_JPEG_QUALITY = 90
_FAILURE_TYPE_IDX = {ft: i for i, ft in enumerate(FAILURE_TYPES)}  # "none"은 포함 안 됨 -- idx=-1로 취급(아래).


def episode_seed(source: str, episode_idx: int) -> int:
    return _SOURCE_SEED_BASE[source] + episode_idx


def episode_split(source: str, episode_idx: int, b_train: int, b_val: int) -> str:
    """에피소드 번호 범위로 결정론적으로 분할한다(사용자 지시 "분할은
    에피소드 단위"). B만 train/val/test_B로 더 쪼갠다(범위: [0,b_train)
    =train, [b_train,b_train+b_val)=val, 나머지=test_B) -- A/C는 전부
    test_*."""
    if source == "A":
        return "test_A"
    if source == "C":
        return "test_C"
    if episode_idx < b_train:
        return "train"
    if episode_idx < b_train + b_val:
        return "val"
    return "test_B"


def episode_id(source: str, episode_idx: int) -> str:
    return f"{source}_{episode_idx:06d}"


def instruction_text(subtask: str, completed: bool) -> str:
    """사용자 지시 형식 그대로("목표 | current subtask: … | completed: …")
    -- ARD-VLA 소스를 못 봐서 토씨까지 같다는 보장은 없다(모듈 docstring
    참고)."""
    return f"{_GOAL_TEXT} | current subtask: {subtask} | completed: {completed}"


def _episode_shard_kwargs(ep_id: str, episode: dict[str, Any]) -> dict[str, np.ndarray]:
    """에피소드 하나의 매 스텝 시계열(판정 레코드가 아니라 원본 로그)을
    `{ep_id}__{field}` 접두사로 평탄화한다 -- 여러 에피소드를 한 npz에
    합쳐서(청크 샤드) 저장하려고(사용자 지시 "에피소드 원본은 청크별
    샤드로"). pipeline/episode_io.py의 단일-에피소드 스키마를 그대로
    안 쓴 이유는 그 함수가 파일 하나=에피소드 하나를 전제하기 때문 --
    여기서는 이미지도 별도 JPEG로 빠지므로(boundary_images 제외) 스키마가
    약간 다르다."""
    cl = episode["closed_loop"]
    right = episode["right_arm"]
    p = f"{ep_id}__"
    out = {
        f"{p}joint_16dof": cl["joint_16dof"],
        f"{p}progress": cl["progress"],
        f"{p}subtask_id": cl["subtask_id"],
        f"{p}right_action": right["action"],
        f"{p}right_force": right["force"],
        f"{p}right_torque": right["torque"],
        f"{p}scene_config": np.array(json.dumps(episode["scene_config"])),
        f"{p}subtask_boundaries": np.array(json.dumps(episode["subtask_boundaries"])),
        f"{p}disturbance": np.array(json.dumps(episode.get("disturbance"))),
        f"{p}success": np.array(bool(episode["success"])),
        f"{p}loop_mode": np.array(episode["loop_mode"]),
        f"{p}split": np.array(episode["split"]),
        f"{p}seed": np.array(int(episode["seed"])),
    }
    for side in ("right", "left"):
        grasp = cl.get(f"grasp_{side}")
        if grasp is None:
            continue
        out[f"{p}grasp_{side}_weld_active"] = grasp["weld_active"]
        out[f"{p}grasp_{side}_weld_force"] = grasp["weld_force"]
        out[f"{p}grasp_{side}_slipped"] = grasp["slipped"]
        out[f"{p}grasp_{side}_cumulative_shift"] = grasp["cumulative_shift"]
    return out


def _make_judge_samples(
    ep_id: str, source: str, episode_idx: int, split: str, chunk_id: str, episode: dict[str, Any], images_rel_dir: str
) -> list[dict[str, Any]]:
    """subtask_boundaries 하나당 판정 샘플 하나(사용자 지시 "서브태스크
    경계 하나 = 판정 샘플 하나")."""
    cl = episode["closed_loop"]
    disturbance = episode.get("disturbance")
    samples = []
    for idx, b in enumerate(episode["subtask_boundaries"]):
        failure_type = b["failure_type"]
        failure_type_idx = _FAILURE_TYPE_IDX.get(failure_type, -1)  # "none"은 -1(실패 없음).
        start_step, end_step = b["start_step"], b["end_step"]
        progress_idx = min(max(end_step - 1, 0), len(cl["progress"]) - 1)

        slipped_but_succeeded = False
        if b["finished_naturally"]:
            # grasp_{side}의 해당 서브태스크 구간 누적이동량이 임계값을
            # 넘었으면 "미끄러짐은 났지만 성공"(hard negative, 사용자
            # 지시) -- 양팔 다 확인한다(어느 쪽이 교란 대상이든).
            for side in ("right", "left"):
                grasp = cl.get(f"grasp_{side}")
                if grasp is None:
                    continue
                shift = grasp["cumulative_shift"][max(start_step, 0):end_step]
                if len(shift) and float(np.linalg.norm(shift, axis=1).max()) > _SLIP_THRESHOLD_M:
                    slipped_but_succeeded = True
                    break

        sample = {
            "sample_id": f"{ep_id}__b{idx}",
            "episode_id": ep_id,
            "chunk_id": chunk_id,
            "split": split,
            "source": source,
            "disturbance_type": (disturbance or {}).get("type"),
            "disturbance_params": {k: v for k, v in (disturbance or {}).items() if k != "type"} or None,
            "loop_mode": episode["loop_mode"],
            "attempt": b["attempt"],
            "applied_recovery": b["recovery_applied"],
            "subtask": b["node"],
            "instruction_start": instruction_text(b["node"], False),
            "instruction_end": instruction_text(b["node"], bool(b["finished_naturally"])),
            "images": {
                "top_cam_start": f"{images_rel_dir}/{ep_id}__boundary_{idx}_top_cam_start.jpg",
                "wide_cam_start": f"{images_rel_dir}/{ep_id}__boundary_{idx}_wide_cam_start.jpg",
                "top_cam_end": f"{images_rel_dir}/{ep_id}__boundary_{idx}_top_cam_end.jpg",
                "wide_cam_end": f"{images_rel_dir}/{ep_id}__boundary_{idx}_wide_cam_end.jpg",
            },
            "robot_state_16dof_start": cl["joint_16dof"][start_step].tolist(),
            "robot_state_16dof_end": cl["joint_16dof"][end_step].tolist(),
            "success": bool(b["finished_naturally"]),
            "failure_type": failure_type,
            "failure_type_idx": failure_type_idx,
            "progress": float(cl["progress"][progress_idx]),
            "slipped_but_succeeded": slipped_but_succeeded,
        }
        samples.append(sample)
    return samples


def generate_one_episode(
    source: str, episode_idx: int, b_train: int, b_val: int, disturbance_cfg: dict | None, visual_cfg: VisualDisturbanceConfig | None
) -> dict[str, Any]:
    """에피소드 하나를 끝까지 생성(물리+렌더)하고, 이 스크립트가 필요한
    형태(episode dict + split)로 돌려준다. 멀티프로세싱 워커가 직접
    부르는 단위 함수(피클 가능해야 해서 모듈 최상위 함수로 뺐다)."""
    assert_no_torch_imported()
    seed = episode_seed(source, episode_idx)
    rng = np.random.default_rng(seed)
    kwargs: dict[str, Any] = {"render": True}
    if source == "B":
        kwargs["disturbance_config"] = disturbance_cfg
    if source == "C":
        kwargs["visual_disturbance"] = visual_cfg
    episode = run_one_episode(rng, source, seed, **kwargs)
    episode["split"] = episode_split(source, episode_idx, b_train, b_val)
    return episode


def _worker(args: tuple) -> tuple[str, int, dict[str, Any]]:
    source, episode_idx, b_train, b_val, disturbance_cfg, visual_cfg_kwargs = args
    visual_cfg = VisualDisturbanceConfig(**visual_cfg_kwargs) if visual_cfg_kwargs is not None else None
    episode = generate_one_episode(source, episode_idx, b_train, b_val, disturbance_cfg, visual_cfg)
    return source, episode_idx, episode


def generate_chunk(
    chunk_dir: str,
    chunk_id: str,
    source: str,
    episode_range: range,
    b_train: int,
    b_val: int,
    disturbance_cfg: dict | None,
    visual_cfg: VisualDisturbanceConfig | None,
    n_workers: int,
) -> dict[str, Any]:
    """청크 하나(한 출처의 연속 에피소드 번호 구간)를 생성해서 디스크에
    쓴다(images/, episodes_shard.npz, judge_samples.jsonl) -- 커밋은
    호출부(main)가 한다(이 함수는 git을 모른다, 관심사 분리)."""
    os.makedirs(os.path.join(chunk_dir, "images"), exist_ok=True)
    visual_cfg_kwargs = {"enabled": visual_cfg.enabled} if visual_cfg is not None else None

    tasks = [(source, i, b_train, b_val, disturbance_cfg, visual_cfg_kwargs) for i in episode_range]
    t0 = time.perf_counter()
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]
    t_sim = time.perf_counter() - t0

    shard_kwargs: dict[str, np.ndarray] = {}
    all_samples: list[dict[str, Any]] = []
    for source_, episode_idx, episode in results:
        ep_id = episode_id(source_, episode_idx)
        boundary_images = episode.pop("boundary_images", {})
        for key, img in boundary_images.items():
            # key: "boundary_{idx}_{camera}_{start|end}" -> 파일명 "{ep_id}__{key}.jpg"
            Image.fromarray(img).save(
                os.path.join(chunk_dir, "images", f"{ep_id}__{key}.jpg"), format="JPEG", quality=_JPEG_QUALITY
            )
        shard_kwargs.update(_episode_shard_kwargs(ep_id, episode))
        all_samples.extend(_make_judge_samples(ep_id, source_, episode_idx, episode["split"], chunk_id, episode, "images"))

    np.savez_compressed(os.path.join(chunk_dir, "episodes_shard.npz"), **shard_kwargs)
    samples_path = os.path.join(chunk_dir, "judge_samples.jsonl")
    with open(samples_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    return {
        "n_episodes": len(results),
        "n_samples": len(all_samples),
        "t_sim_total_s": t_sim,
        "t_sim_per_episode_s": t_sim / max(len(results), 1),
    }


def _sha256_of_dir(path: str) -> str:
    h = hashlib.sha256()
    for root, _dirs, files in os.walk(path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            h.update(os.path.relpath(fpath, path).encode())
            with open(fpath, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
    return h.hexdigest()


def _load_manifest(data_dir: str) -> dict[str, Any]:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return {"completed_chunks": {}}
    with open(path) as f:
        return json.load(f)


def _save_manifest(data_dir: str, manifest: dict[str, Any]) -> None:
    with open(os.path.join(data_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _git(data_dir: str, *args: str) -> str:
    import subprocess

    result = subprocess.run(["git", "-C", data_dir, *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stdout}\n{result.stderr}")
    return result.stdout


def plan_chunks(source: str, n_episodes: int, chunk_size: int) -> list[tuple[str, range]]:
    chunks = []
    for start in range(0, n_episodes, chunk_size):
        end = min(start + chunk_size, n_episodes)
        chunk_id = f"{source}_{start:06d}_{end:06d}"
        chunks.append((chunk_id, range(start, end)))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True, help="data/closed-loop-v1 worktree 경로")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-a", type=int, default=300)
    parser.add_argument("--n-b", type=int, default=800)
    parser.add_argument("--n-c", type=int, default=300)
    parser.add_argument("--b-train", type=int, default=600)
    parser.add_argument("--b-val", type=int, default=100)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--disturbance-config", type=str, default=os.path.join(os.path.dirname(__file__), "disturbance_config.yaml"))
    parser.add_argument("--visual-disturbance", action="store_true", default=True)
    parser.add_argument("--no-push", action="store_true", help="디버깅용 -- 커밋/푸시 생략(파일럿 등)")
    args = parser.parse_args()

    assert_no_torch_imported()
    disturbance_cfg = load_disturbance_config(args.disturbance_config)
    visual_cfg = VisualDisturbanceConfig(enabled=args.visual_disturbance)

    chunks: list[tuple[str, str, range]] = []
    for source, n in (("A", args.n_a), ("B", args.n_b), ("C", args.n_c)):
        for chunk_id, episode_range in plan_chunks(source, n, args.chunk_size):
            chunks.append((source, chunk_id, episode_range))

    manifest = _load_manifest(args.data_dir)
    completed = manifest["completed_chunks"]
    print(f"총 {len(chunks)}개 청크 중 {len(completed)}개 이미 완료 -- 나머지를 돈다.")

    for source, chunk_id, episode_range in chunks:
        if chunk_id in completed:
            print(f"[skip] {chunk_id} (이미 완료)")
            continue
        chunk_dir = os.path.join(args.data_dir, "chunks", chunk_id)
        print(f"[run] {chunk_id} ({len(episode_range)}개 에피소드, {args.workers} workers)")
        stats = generate_chunk(
            chunk_dir, chunk_id, source, episode_range, args.b_train, args.b_val,
            disturbance_cfg if source == "B" else None, visual_cfg if source == "C" else None, args.workers,
        )
        chunk_hash = _sha256_of_dir(chunk_dir)
        completed[chunk_id] = {
            "source": source, "episode_range": [episode_range.start, episode_range.stop],
            "sha256": chunk_hash, "committed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **stats,
        }
        _save_manifest(args.data_dir, manifest)

        if not args.no_push:
            _git(args.data_dir, "add", "-A")
            _git(args.data_dir, "commit", "-m", f"청크 {chunk_id} 추가 ({stats['n_episodes']}개 에피소드, {stats['n_samples']}개 판정 샘플)")
            for attempt in range(4):
                try:
                    _git(args.data_dir, "push")
                    break
                except RuntimeError as e:
                    if attempt == 3:
                        raise
                    wait = 2 ** (attempt + 1)
                    print(f"push 실패, {wait}s 뒤 재시도: {e}")
                    time.sleep(wait)
        print(f"[done] {chunk_id}: {stats}")

    print("모든 청크 완료.")


if __name__ == "__main__":
    main()
