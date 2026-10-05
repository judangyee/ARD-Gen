"""RoboTwin 2.0 이식 Part 3-1 회귀 테스트: patterns/ 추출이 기존 3개
태스크의 reward/success/step_count를 조금도 바꾸지 않았는지 확인한다.

여기 하드코딩된 값은 리팩토링(sim/peg_in_hole_openarm_env.py,
sim/cap_twist_env.py, sim/tacker_openarm_env.py의 compute_reward/
is_success를 patterns/position_correction.py, patterns/torque_reactive.py,
patterns/impact_recoil.py 호출로 교체) 적용 **전**에 같은 코드로 직접
실행해서 뽑은 실측값이다(리팩토링 전/후를 코드 리뷰가 아니라 숫자로
비교) -- 이 값이 하나라도 바뀌면 "패턴 추출이 공식을 몰래 바꿨다"는
뜻이다.

사용법:
    python tests/test_patterns_identity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from sim.task_registry import load_task_config

# task_name -> [(scene_overrides, expected_reward, expected_success, expected_step_count), ...]
_BASELINES = {
    "peg_in_hole": [
        ({"peg_init_offset_xy": (0.0, 0.0)}, 48.86939367440357, True, 155),
        ({"peg_init_offset_xy": (0.014, 0.0)}, 48.85105734049234, True, 157),
        ({"peg_init_offset_xy": (0.0099, 0.0099)}, 48.79649224315119, True, 162),
    ],
    "cap_twist": [
        ({"resistance_torque": 0.05}, 71.99145398613344, True, 377),
        ({"resistance_torque": 0.3}, 72.7820314089343, True, 386),
        ({"resistance_torque": 0.8}, 73.44755114421935, True, 395),
    ],
    "tacker": [
        ({"recoil_strength": 5.0}, 77.90405977952918, True, 209),
        ({"recoil_strength": 15.0}, 77.90405977952918, True, 209),
        ({"recoil_strength": 30.0}, 24.157577404861655, False, 211),
    ],
}

_GAINS = {
    "peg_in_hole": (0.124630, 0.001125),
    "cap_twist": {"Kp_tau": 0.02},
    "tacker": [0.03],
}


def _run(task_name: str, overrides: dict, env=None):
    task = load_task_config(task_name)
    if env is None:
        env = task.make_env(use_stabilizer=False) if task_name == "cap_twist" else task.make_env()
    gains_spec = _GAINS[task_name]
    gains = task.gains_from_vector(gains_spec) if isinstance(gains_spec, (tuple, list)) else gains_spec
    cfg = task.default_scene_config()
    cfg.update(overrides)
    return env.run_episode(gains, cfg), env


@pytest.mark.parametrize("task_name", list(_BASELINES.keys()))
def test_reward_identical_to_pre_refactor_baseline(task_name: str) -> None:
    env = None
    for overrides, expected_reward, expected_success, expected_step_count in _BASELINES[task_name]:
        result, env = _run(task_name, overrides, env)
        assert result["success"] == expected_success, f"{task_name}/{overrides}: success {result['success']} != {expected_success}"
        assert result["step_count"] == expected_step_count, (
            f"{task_name}/{overrides}: step_count {result['step_count']} != {expected_step_count}"
        )
        assert abs(result["reward"] - expected_reward) < 1e-9, (
            f"{task_name}/{overrides}: reward {result['reward']!r} != {expected_reward!r} "
            f"(patterns/ 추출이 공식을 바꿨을 가능성)"
        )
    print(f"[OK] {task_name}: patterns/ 추출 전후 reward/success/step_count bit-identical ({len(_BASELINES[task_name])}개 씬)")


if __name__ == "__main__":
    for task_name in _BASELINES:
        test_reward_identical_to_pre_refactor_baseline(task_name)
    print()
    print("ALL TESTS PASSED (patterns identity)")
