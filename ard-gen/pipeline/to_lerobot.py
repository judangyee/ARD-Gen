"""4/5단계 완성 에피소드(data/episodes/{task}/*.npz)를 LeRobotDataset
호환 포맷으로 변환한다.

## "호환"이지 "동일"이 아니다 (정직하게 밝힘)

이 환경에는 `lerobot`/`pandas`/`pyarrow` 패키지가 없다(오프라인 시뮬레이션
전용 환경이라 무거운 의존성을 새로 설치하지 않음). 그래서 실제
LeRobotDataset이 쓰는 parquet 대신 **같은 정보를 담은 npz**로 프레임
데이터를 저장한다 -- 디렉터리 레이아웃(`meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl`, `data/chunk-*/episode_*.*`)과
컬럼/필드 이름(episode_index, frame_index, timestamp, task_index,
observation.state, action)은 LeRobotDataset v2.x 스키마를 최대한
그대로 따랐다. pyarrow/lerobot이 있는 환경으로 옮기면, 이 npz들을 읽어서
그대로 데이터프레임으로 만들고 parquet으로 다시 쓰기만 하면 된다
(아래 `_frame_arrays_from_npz()`가 그 변환에 필요한 딕셔너리를 그대로
내놓는다).

## 태스크별로 별도 데이터셋인 이유

peg_in_hole과 cap_twist는 관측/행동 차원 자체가 다르다(전자는 위치+
손목각, 후자는 뚜껑 각도 하나) -- 실제 로봇 데이터셋도 로봇/센서 구성이
다르면 별도 리포지토리로 두는 것과 같은 이유로, `--task`마다
`data/lerobot/{task}/`에 독립된 데이터셋을 만든다.

## state/action 구성

- observation.state = 오른팔(right_arm.traj)의 관측 + 왼팔(left_arm.traj,
  있으면)의 관측을 이어붙인 벡터. 왼팔 궤적은 접근(approach) 단계가 있어
  오른팔보다 몇 프레임 더 길다(sim/stabilizer.py의 run_approach_phase()
  docstring 참고) -- 그래서 왼팔 궤적의 **뒤쪽**(유지 단계, 오른팔 제어와
  실제로 동시에 기록된 구간)에서 오른팔과 같은 길이만큼만 잘라 맞춘다.
- action = right_arm.action (매 스텝 실제 제어 명령). 프레임 수는
  action 배열 길이를 기준으로 삼는다(traj가 action보다 정확히 1 프레임
  더 긴 것이 정상 -- 초기 관측 1개 + 스텝마다 관측 1개).
- (공식 그리퍼 반영) right_arm/left_arm에 joint_pos/joint_action/
  gripper_action이 있으면(현재 peg_in_hole만) build_frames()가 그 뒤에
  이어붙인다 -- 기존 Cartesian state/action 차원은 그대로 유지, 조인트
  공간(팔당 7)+그리퍼(팔당 1)가 뒤에 추가되는 구조다(sim/
  peg_in_hole_openarm_env.py 모듈 docstring 참고). 없는 태스크(cap_twist)
  는 전과 동일하게 동작한다.

사용 예:
    python pipeline/to_lerobot.py --task peg_in_hole --episodes-dir ./data/episodes/peg_in_hole \
        --out-dir ./data/lerobot/peg_in_hole
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from pipeline.episode_io import load_episode
from sim.task_registry import list_tasks

FPS = 100  # DT = N_SUBSTEPS(5) * timestep(0.002s) = 0.01s -- peg_in_hole/cap_twist 둘 다 동일(실측 확인)
CODEBASE_VERSION = "v2.1-approx"  # 실제 lerobot 패키지로 만든 게 아니라는 걸 버전 문자열에도 남겨둔다

# 태스크별 state/action 컬럼 이름(사람이 읽을 수 있는 메타데이터용, 물리적
# 의미는 각 sim 모듈의 docstring 참고 -- 여기서 하드코딩하는 이유는
# LeRobotDataset의 features.names가 "그 데이터셋 하나의 고정 스키마"라서다).
_RIGHT_JOINT_NAMES = [f"right_joint{i}_pos" for i in range(1, 8)]
_LEFT_JOINT_NAMES = [f"left_joint{i}_pos" for i in range(1, 8)]
_RIGHT_JOINT_TORQUE_NAMES = [f"right_joint{i}_torque" for i in range(1, 8)]
_LEFT_JOINT_TORQUE_NAMES = [f"left_joint{i}_torque" for i in range(1, 8)]

_STATE_NAMES = {
    # 공식 그리퍼 반영(액션 공간을 팔당 7관절+그리퍼로 확장) 이후:
    # 기존 Cartesian ee_pos/wrist_rotate는 그대로 유지하고(제거 안 함),
    # 그 뒤에 조인트 공간 관측(팔당 7)을 추가했다 -- build_frames()의
    # "조인트 공간 state/action 추가" 절 참고. 관절 토크 센서(팔당 7)도
    # 같은 방식으로 그 뒤에 추가했다("observation에 joint_torque 필드
    # 포함" 요청). 없는 태스크(cap_twist 등)는 그대로 예전 차원을 쓴다.
    "peg_in_hole": ["right_ee_x", "right_ee_y", "right_ee_z", "right_wrist_rotate",
                    "left_ee_x", "left_ee_y", "left_ee_z"] + _RIGHT_JOINT_NAMES + _LEFT_JOINT_NAMES
    + _RIGHT_JOINT_TORQUE_NAMES + _LEFT_JOINT_TORQUE_NAMES,
    "cap_twist": ["cap_angle_rad", "left_ee_x", "left_ee_y", "left_ee_z"],
}
_ACTION_NAMES = {
    "peg_in_hole": ["delta_x", "delta_y", "delta_z"] + _RIGHT_JOINT_NAMES
    + ["right_gripper_ctrl", "left_gripper_ctrl"],
    "cap_twist": ["omega_command_rad_s"],
}


def _align_trailing(arr: np.ndarray | None, n_frames: int, dim: int) -> np.ndarray:
    """arr(있으면)를 n_frames 길이로 맞춘다 -- 뒤쪽(실제 제어와 동시에
    기록된 구간)만 쓴다(_align_left_arm의 일반화, 왼팔 EE 3차원 전용이던
    것을 임의 차원에 쓸 수 있게 했다). 없으면 0으로 채운다."""
    if arr is None or len(arr) == 0:
        return np.zeros((n_frames, dim), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) >= n_frames:
        return arr[-n_frames:]
    pad = np.repeat(arr[-1:], n_frames - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def build_frames(episode: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """에피소드 하나에서 (state (T,S), action (T,A), task_instruction) 을 만든다."""
    right = episode["right_arm"]
    left = episode.get("left_arm") or {}
    action = np.asarray(right["action"], dtype=np.float32)
    n_frames = len(action)

    right_state = np.asarray(right["traj"], dtype=np.float32)[:n_frames]
    left_state = _align_trailing(left.get("traj"), n_frames, 3)
    state = np.concatenate([right_state, left_state], axis=1)
    action_parts = [action]

    # 조인트 공간 state/action 추가(공식 그리퍼 반영, 있는 태스크만 -- 현재
    # peg_in_hole) -- 기존 Cartesian 필드는 그대로 두고 뒤에 이어붙인다.
    if right.get("joint_pos") is not None:
        state = np.concatenate([state, _align_trailing(right["joint_pos"], n_frames, 7)], axis=1)
    if left.get("joint_pos") is not None:
        state = np.concatenate([state, _align_trailing(left["joint_pos"], n_frames, 7)], axis=1)
    # 관절 토크 센서 관측 추가(있는 태스크만) -- joint_pos와 같은 자리에서
    # 이어붙인다(state의 "그 뒤" 순서는 _STATE_NAMES의 순서와 일치시켜야
    # 한다 -- joint_pos 다음에 joint_torque).
    if right.get("joint_torque") is not None:
        state = np.concatenate([state, _align_trailing(right["joint_torque"], n_frames, 7)], axis=1)
    if left.get("joint_torque") is not None:
        state = np.concatenate([state, _align_trailing(left["joint_torque"], n_frames, 7)], axis=1)
    if right.get("joint_action") is not None:
        action_parts.append(_align_trailing(right["joint_action"], n_frames, 7))
    if right.get("gripper_action") is not None:
        action_parts.append(_align_trailing(right["gripper_action"], n_frames, 1))
    if left.get("gripper_action") is not None:
        action_parts.append(_align_trailing(left["gripper_action"], n_frames, 1))
    action = np.concatenate(action_parts, axis=1) if len(action_parts) > 1 else action

    language = episode.get("language") or {}
    task_instruction = language.get("task_instruction", f"{episode.get('task', 'unknown')} 태스크 수행")
    return state, action, task_instruction


def convert_task(task_name: str, episodes_dir: str, out_dir: str) -> dict:
    paths = sorted(glob.glob(os.path.join(episodes_dir, "episode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"{episodes_dir}에 episode_*.npz가 없음 -- 4/5단계를 먼저 실행하세요.")

    meta_dir = os.path.join(out_dir, "meta")
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(data_dir, "episode_*.npz")):
        os.remove(stale)

    task_to_index: dict[str, int] = {}
    episodes_meta = []
    total_frames = 0
    state_dim = action_dim = None

    for ep_idx, path in enumerate(paths):
        episode = load_episode(path)
        state, action, task_instruction = build_frames(episode)
        n_frames = len(action)

        if state_dim is None:
            state_dim, action_dim = state.shape[1], action.shape[1]
        elif state.shape[1] != state_dim or action.shape[1] != action_dim:
            raise ValueError(
                f"{path}: state/action 차원이 다른 에피소드와 다르다 "
                f"(got {state.shape[1]}/{action.shape[1]}, expected {state_dim}/{action_dim}) -- "
                "한 데이터셋 안의 모든 에피소드는 같은 관측/행동 차원이어야 한다."
            )

        if task_instruction not in task_to_index:
            task_to_index[task_instruction] = len(task_to_index)
        task_index = task_to_index[task_instruction]

        frame_index = np.arange(n_frames, dtype=np.int64)
        episode_index_col = np.full(n_frames, ep_idx, dtype=np.int64)
        timestamp = (frame_index / FPS).astype(np.float32)
        task_index_col = np.full(n_frames, task_index, dtype=np.int64)

        np.savez(
            os.path.join(data_dir, f"episode_{ep_idx:06d}.npz"),
            **{
                "observation.state": state,
                "action": action,
                "episode_index": episode_index_col,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "task_index": task_index_col,
            },
        )

        language = episode.get("language") or {}
        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_instruction],
                "length": n_frames,
                # 아래는 표준 LeRobotDataset 필드가 아니라 ARD-Gen이 추가한
                # 확장 메타데이터다(ARD-VLA의 role classifier/언어 라벨용) --
                # 표준을 따르는 로더는 모르는 키를 무시하면 되므로 호환에
                # 문제가 없다.
                "role_labels": language.get("role_labels", {}),
                "quantity_target": language.get("quantity_target"),
                "direction": language.get("direction"),
                "scene_config": episode.get("scene_config", {}),
                "success": bool(episode.get("success", True)),
            }
        )
        total_frames += n_frames

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        for task_str, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task_str}, ensure_ascii=False) + "\n")

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": f"ard_vla_bimanual_sim_{task_name}",
        "total_episodes": len(paths),
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "fps": FPS,
        "chunks_size": 1000,
        "data_path": "data/chunk-000/episode_{episode_index:06d}.npz",
        "video_path": None,  # 시뮬레이션이라 카메라 없음 -- 실물 데이터와 필드 호환을 위해 자리만 남겨둠
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim], "names": _STATE_NAMES.get(task_name)},
            "action": {"dtype": "float32", "shape": [action_dim], "names": _ACTION_NAMES.get(task_name)},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
        "note": (
            "이 환경에 lerobot/pandas/pyarrow가 없어 data/chunk-*/episode_*.npz가 "
            "parquet의 자리를 대신한다 -- 컬럼 이름/메타데이터 스키마는 LeRobotDataset "
            "v2.x를 따랐다. 자세한 내용은 이 파일(pipeline/to_lerobot.py) 상단 docstring 참고."
        ),
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return info


def load_lerobot_episode(out_dir: str, episode_index: int) -> dict:
    """검증/로딩용: 변환된 데이터셋에서 에피소드 하나를 읽어온다(진짜
    LeRobotDataset 파이썬 클래스는 아니지만, meta/info.json + 개별
    episode 파일만으로 필요한 정보를 전부 복원할 수 있다는 걸 보여준다)."""
    with open(os.path.join(out_dir, "meta", "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(out_dir, "meta", "episodes.jsonl")) as f:
        episodes_meta = [json.loads(line) for line in f]
    with open(os.path.join(out_dir, "meta", "tasks.jsonl")) as f:
        tasks = {json.loads(line)["task_index"]: json.loads(line)["task"] for line in f}

    ep_meta = episodes_meta[episode_index]
    npz_path = os.path.join(out_dir, "data", "chunk-000", f"episode_{episode_index:06d}.npz")
    data = np.load(npz_path)
    return {
        "info": info,
        "episode_meta": ep_meta,
        "task": tasks[data["task_index"][0]] if len(data["task_index"]) else None,
        "observation.state": data["observation.state"],
        "action": data["action"],
        "timestamp": data["timestamp"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=str, default="peg_in_hole", choices=list_tasks())
    parser.add_argument("--episodes-dir", type=str, default=None, help="생략하면 ./data/episodes/{task}")
    parser.add_argument("--out-dir", type=str, default=None, help="생략하면 ./data/lerobot/{task}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes_dir = args.episodes_dir or os.path.join("./data/episodes", args.task)
    out_dir = args.out_dir or os.path.join("./data/lerobot", args.task)

    info = convert_task(args.task, episodes_dir, out_dir)
    print(
        f"[to_lerobot] task={args.task} {info['total_episodes']}개 에피소드, "
        f"{info['total_frames']}프레임, state_dim={info['features']['observation.state']['shape'][0]}, "
        f"action_dim={info['features']['action']['shape'][0]}, task 종류={info['total_tasks']} -> {out_dir}"
    )

    sample = load_lerobot_episode(out_dir, 0)
    print(
        f"[to_lerobot] 검증 로드(episode 0): task={sample['task']!r} "
        f"state.shape={sample['observation.state'].shape} action.shape={sample['action'].shape} "
        f"length={sample['episode_meta']['length']}"
    )


if __name__ == "__main__":
    main()
