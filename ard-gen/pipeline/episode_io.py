"""4/5단계 에피소드 저장 포맷(data/episodes/*.npz) 공통 입출력.

에피소드는 논리적으로 다음 구조를 가진다 (지금은 Stabilizer가 없으므로
right_arm만 채워진다 -- PIPELINE.md의 "지금은 Actuator 단일 팔" 참고):

    {
        "task": "peg_in_hole",           # sim.task_registry.TASK_REGISTRY 키
        "right_arm": {
            "traj": (T+1, D) ndarray,    # ee_poses
            "gain_names": ["Kp_xy", "Kd_xy"],  # 태스크마다 개수/이름이 다름
            "gains": (len(gain_names),) ndarray,
            "force": (T, 3) ndarray,
            "torque": (T, 3) ndarray,
            "role": "actuator",
        },
        "scene_config": {...},          # sim.task_registry.TaskConfig.sample_scene_config 스키마
        "success": True,
        "insertion_depth": float,        # 있으면 저장(태스크마다 없을 수도 있음)
        "force_max": float,
        "language": str | None,         # 5단계에서 채워짐, 그 전엔 없음
        # 그 외 태스크별 스칼라 필드(예: 회전 태스크의 "direction", "quantity")도
        # 최상위에 그대로 넣으면 자동으로 저장/복원된다 -- 아래 참고.
    }

리팩토링 이전에는 right_arm_gains가 항상 [Kp_xy, Kd_xy] 고정 2원소
배열이었다(태스크가 하나뿐이라 이름 없이 순서로만 구분해도 됐음) -- 이제
태스크마다 게인 개수/이름이 다르므로(예: 나사 조이기의 torque_limit
1개), gain_names를 함께 저장해서 어떤 태스크로 읽어도 게인 벡터의 의미를
복원할 수 있게 했다.

## 그 외 태스크별 스칼라 필드 (direction/quantity 등)

peg-in-hole은 insertion_depth/force_max만 있으면 됐지만, 회전 태스크(예:
cap_twist)는 "돌린 방향"/"회전수" 같은 필드가 언어 라벨링(5단계)에
필요하다(pipeline/language_labeling.py의 direction_words/quantity_words
참고). 이런 필드를 이 파일이 미리 알 필요는 없다 -- episode dict의
최상위에 있는, 고정 키(_FIXED_TOP_LEVEL_KEYS)가 아닌 모든 int/float/bool/str
값을 `extra_{key}` 접두사로 자동 저장하고, load_episode()가 다시 원래
키로 복원한다.

npz는 중첩 dict를 그대로 못 담으므로 "right_arm_" 접두사로 평탄화해서
저장하고, load_episode()가 다시 위 구조로 복원한다.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

_FIXED_TOP_LEVEL_KEYS = {"task", "right_arm", "scene_config", "success", "insertion_depth", "force_max", "language"}


def save_episode(path: str, episode: dict[str, Any]) -> None:
    right = episode["right_arm"]
    gain_names = right.get("gain_names")
    if gain_names is None:
        # 옛 호출자 호환: 이름 없이 gains만 준 경우 자리표시자를 만든다.
        gain_names = [f"gain_{i}" for i in range(len(right["gains"]))]
    kwargs: dict[str, Any] = {
        "task": np.array(episode.get("task", "peg_in_hole")),
        "right_arm_traj": np.asarray(right["traj"], dtype=np.float32),
        "right_arm_gain_names": np.array(json.dumps(list(gain_names))),
        "right_arm_gains": np.asarray(right["gains"], dtype=np.float32),
        "right_arm_force": np.asarray(right["force"], dtype=np.float32),
        "right_arm_torque": np.asarray(right["torque"], dtype=np.float32),
        "right_arm_role": np.array(right["role"]),
        "scene_config": np.array(json.dumps(episode["scene_config"])),
        "success": np.array(bool(episode["success"])),
    }
    if episode.get("insertion_depth") is not None:
        kwargs["insertion_depth"] = np.array(episode["insertion_depth"], dtype=np.float32)
    if episode.get("force_max") is not None:
        kwargs["force_max"] = np.array(episode["force_max"], dtype=np.float32)
    if episode.get("language") is not None:
        kwargs["language"] = np.array(episode["language"])
    for key, value in episode.items():
        if key in _FIXED_TOP_LEVEL_KEYS or value is None:
            continue
        if not isinstance(value, (int, float, bool, str)):
            raise TypeError(f"episode[{key!r}] must be a scalar (int/float/bool/str) to auto-save, got {type(value)}")
        kwargs[f"extra_{key}"] = np.array(value)
    np.savez(path, **kwargs)


def load_episode(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    gain_names = (
        json.loads(str(data["right_arm_gain_names"])) if "right_arm_gain_names" in data.files else None
    )
    episode: dict[str, Any] = {
        "task": str(data["task"]) if "task" in data.files else "peg_in_hole",
        "right_arm": {
            "traj": data["right_arm_traj"],
            "gain_names": gain_names,
            "gains": data["right_arm_gains"],
            "force": data["right_arm_force"],
            "torque": data["right_arm_torque"],
            "role": str(data["right_arm_role"]),
        },
        "scene_config": json.loads(str(data["scene_config"])),
        "success": bool(data["success"]),
    }
    if "insertion_depth" in data.files:
        episode["insertion_depth"] = float(data["insertion_depth"])
    if "force_max" in data.files:
        episode["force_max"] = float(data["force_max"])
    if "language" in data.files:
        episode["language"] = str(data["language"])
    for key in data.files:
        if not key.startswith("extra_"):
            continue
        value = data[key]
        episode[key[len("extra_"):]] = value.item() if value.dtype.kind != "U" else str(value)
    return episode


def save_language(path: str, language: str) -> None:
    """기존 에피소드 npz를 읽어서 language 필드만 채워 다시 저장한다."""
    episode = load_episode(path)
    episode["language"] = language
    save_episode(path, episode)
