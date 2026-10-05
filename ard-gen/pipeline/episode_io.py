"""4/5단계 에피소드 저장 포맷(data/episodes/*.npz) 공통 입출력.

에피소드는 논리적으로 다음 구조를 가진다 (3단계 완료 후 left_arm이
추가됐다 -- tasks/*.yaml에 `stabilizer` 절이 있는 태스크는 env.run_episode()
가 "left_arm_traj"를 돌려주고, pipeline/filter_episodes.py가 이를
left_arm으로 실어 나른다. stabilizer 절이 없는 태스크는 여전히 right_arm만
있는 단일팔 에피소드가 나온다 -- 하위호환):

    {
        "task": "peg_in_hole",           # sim.task_registry.TASK_REGISTRY 키
        "left_arm": {                    # 있으면만(Stabilizer 지원 태스크)
            "traj": (T', 3) ndarray,     # world-frame EE 위치(접근+유지 전체)
            "role": "stabilizer",
        },
        "right_arm": {
            "traj": (T+1, D) ndarray,    # ee_poses (관측/state)
            "action": (T, A) ndarray,    # 매 스텝 실제 제어 명령 (LeRobotDataset의 action)
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
        "language": {                    # 5단계에서 채워짐, 그 전엔 없음(dict, JSON으로 저장)
            "task_instruction": str,       # 템플릿으로 생성한 자연어 문장
            "role_labels": {"right_arm": "actuator", "left_arm": "stabilizer"},  # left_arm 있을 때만
            "quantity_target": float | None,
            "direction": str | None,
        },
        # 그 외 태스크별 스칼라 필드(예: 회전 태스크의 "direction", "quantity")도
        # 최상위에 그대로 넣으면 자동으로 저장/복원된다 -- 아래 참고.
    }

    left_arm.traj와 right_arm.traj의 길이(T', T+1)가 다를 수 있다 -- 왼팔은
    approach 단계가 있고 오른팔은 처음부터 제어를 시작하기 때문이다(자세한
    이유는 sim/stabilizer.py의 run_approach_phase() docstring 참고). 둘 다
    "같은 물리 시뮬레이션 안에서 동시에" 기록됐다는 사실은 변하지 않는다
    -- 시간 정렬이 필요하면 오른팔 쪽 스텝 수(step_count)를 기준으로 삼는다.

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

_FIXED_TOP_LEVEL_KEYS = {
    "task", "right_arm", "left_arm", "scene_config", "success", "insertion_depth", "force_max", "language",
    "visual_config",
}


def save_episode(path: str, episode: dict[str, Any]) -> None:
    right = episode["right_arm"]
    gain_names = right.get("gain_names")
    if gain_names is None:
        # 옛 호출자 호환: 이름 없이 gains만 준 경우 자리표시자를 만든다.
        gain_names = [f"gain_{i}" for i in range(len(right["gains"]))]
    kwargs: dict[str, Any] = {
        "task": np.array(episode.get("task", "peg_in_hole")),
        "right_arm_traj": np.asarray(right["traj"], dtype=np.float32),
        "right_arm_action": np.asarray(right.get("action", np.zeros((0, 1), dtype=np.float32)), dtype=np.float32),
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
    if episode.get("visual_config") is not None:
        # RoboTwin 2.0 이식 Part 1: scene_config(물리)와 완전히 분리된
        # 시각 randomization 설정 -- 같은 JSON 직렬화 패턴(위
        # scene_config 참고). render_episode.py가 저장된 외형을 그대로
        # 재생하려면 이게 있어야 한다(없으면 default_visual_config()로
        # 대체 -- 하위호환, 이 필드가 없던 옛 npz도 그대로 읽힌다).
        kwargs["visual_config"] = np.array(json.dumps(episode["visual_config"]))
    if episode.get("language") is not None:
        # 5단계(ARD-VLA role classifier 학습용 role_labels 포함)부터
        # language는 dict다 -- JSON 문자열로 직렬화해서 저장한다(scene_config와
        # 같은 패턴). 옛 호출자가 plain string을 넘기는 경우도 그대로
        # json.dumps하면 문자열의 JSON 인코딩이 되므로 load_episode()가
        # json.loads로 똑같이 복원할 수 있어 하위호환된다.
        kwargs["language"] = np.array(json.dumps(episode["language"]))
    # 3단계(Stabilizer, 왼팔): 있으면만 저장한다(단일팔 전용으로 생성된
    # 옛 에피소드/태스크와도 호환). right_arm과 달리 게인/force/torque가
    # 없다 -- Stabilizer는 게인 탐색도 힘 측정도 안 하는 순수 기하 제어라서.
    if episode.get("left_arm") is not None:
        left = episode["left_arm"]
        kwargs["left_arm_traj"] = np.asarray(left["traj"], dtype=np.float32)
        kwargs["left_arm_role"] = np.array(left.get("role", "stabilizer"))
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
            "action": data["right_arm_action"] if "right_arm_action" in data.files else np.zeros((0, 1), dtype=np.float32),
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
    if "visual_config" in data.files:
        episode["visual_config"] = json.loads(str(data["visual_config"]))
    if "language" in data.files:
        episode["language"] = json.loads(str(data["language"]))
    if "left_arm_traj" in data.files:
        episode["left_arm"] = {
            "traj": data["left_arm_traj"],
            "role": str(data["left_arm_role"]) if "left_arm_role" in data.files else "stabilizer",
        }
    for key in data.files:
        if not key.startswith("extra_"):
            continue
        value = data[key]
        episode[key[len("extra_"):]] = value.item() if value.dtype.kind != "U" else str(value)
    return episode


def save_language(path: str, language: dict[str, Any]) -> None:
    """기존 에피소드 npz를 읽어서 language 필드(dict, 위 모듈 docstring
    스키마)만 채워 다시 저장한다."""
    episode = load_episode(path)
    episode["language"] = language
    save_episode(path, episode)
