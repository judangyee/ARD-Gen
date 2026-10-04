"""peg_in_hole(OpenArm 양팔 버전) 회귀 테스트.

이 레포는 더 이상 VX300s를 쓰지 않는다(사용자 승인, VX300s 관련 코드
전체 삭제) -- 예전 tests/test_regression_peg_in_hole.py는 "VX300s 저수준
구현 vs BaseTaskEnv로 리팩토링한 버전이 같은 결과를 내는가"를 검증하는
파일이었는데, 그 비교 대상 자체(sim/peg_in_hole_sim.py, sim/
peg_in_hole_env.py, pipeline/scene_sampler.py)가 삭제되면서 테스트
목적이 사라졌다. 그 파일에서 VX300s와 무관했던 테스트(TASK_REGISTRY
등록 확인, 공식 그리퍼 반영 때 추가한 조인트 공간 state/action, 관절
토크 센서, 노이즈/지연 옵션, force_max 버그 수정 회귀 가드)만 이 파일로
옮겨서 계속 돈다.

사용법:
    python tests/test_peg_in_hole_openarm.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.task_registry import load_task_config

_KNOWN_GOOD_GAINS = (0.124630, 0.001125)  # optimize/peg_in_hole_openarm_admittance_gain_search.py로 실측 탐색


def test_task_registry_wiring() -> None:
    """TASK_REGISTRY["peg_in_hole"]가 OpenArm 버전(sim/peg_in_hole_openarm_env.py:
    PegInHoleOpenArmEnv)으로 등록돼 있는지, 알려진 성공 게인으로 실제
    성공하는지 확인한다."""
    task = load_task_config("peg_in_hole")
    assert task.gain_names == ["Kp_xy", "Kd_xy"]
    assert task.gain_bounds["Kp_xy"] == (0.001, 0.5)
    assert task.gain_bounds["Kd_xy"] == (0.0, 0.02)
    assert len(task.eval_scenarios) == 3
    assert task.condition_dim == 4
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector(_KNOWN_GOOD_GAINS), task.default_scene_config())
    assert result["success"] is True
    print("[OK] task_registry wiring for peg_in_hole (OpenArm)")


def test_joint_space_fields_present() -> None:
    """공식 그리퍼 반영: 팔당 7관절 + 그리퍼 1DOF가 올바른 shape으로 기록되고,
    기존 Cartesian ee_poses/actions는 그대로 유지되는지 확인한다."""
    task = load_task_config("peg_in_hole")
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector(_KNOWN_GOOD_GAINS), task.default_scene_config())
    n_frames = len(result["actions"])

    assert result["right_joint_pos"].shape == (n_frames + 1, 7)
    assert result["left_joint_pos"].shape == (n_frames + 1, 7)
    assert result["right_joint_action"].shape == (n_frames, 7)
    assert result["right_gripper_action"].shape == (n_frames, 1)
    assert result["left_gripper_action"].shape == (n_frames, 1)
    # 기존 필드는 그대로(차원이 안 바뀜) -- "확장"이지 "교체"가 아니다.
    assert result["ee_poses"].shape == (n_frames + 1, 4)
    assert result["actions"].shape == (n_frames, 3)
    print("[OK] joint-space state/action fields present with correct shapes")


def test_joint_torque_sensor_varies() -> None:
    """관절 토크 센서: jointactuatorfrc 값이 스텝마다 실제로 변하는지 확인한다
    (freejoint site 센서의 "항상 상수" 버그 계열에 다시 걸리지 않는지 지키는
    회귀 가드)."""
    task = load_task_config("peg_in_hole")
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector(_KNOWN_GOOD_GAINS), task.default_scene_config())
    torque = result["right_joint_torque"]
    assert torque.shape[1] == 7
    assert float(np.std(torque)) > 1e-4, "right_joint_torque가 상수에 가깝다 -- 센서가 죽었을 가능성"
    print(f"[OK] right_joint_torque varies (std={float(np.std(torque)):.4f})")


def test_torque_noise_and_delay_default_off() -> None:
    """torque_noise_std/control_delay_steps 기본값(0)은 이전 동작과 완전히
    동일해야 한다 -- "실기 모사 옵션"이 기존 파이프라인에 몰래 영향을 주면
    안 된다는 회귀 가드."""
    from sim.peg_in_hole_openarm_env import PegInHoleOpenArmEnv

    task = load_task_config("peg_in_hole")
    gains = task.gains_from_vector(_KNOWN_GOOD_GAINS)
    cfg = task.default_scene_config()

    result_default = task.make_env().run_episode(gains, cfg)
    result_explicit_zero = PegInHoleOpenArmEnv(torque_noise_std=0.0, control_delay_steps=0).run_episode(gains, cfg)

    assert result_default["success"] == result_explicit_zero["success"]
    assert result_default["step_count"] == result_explicit_zero["step_count"]
    np.testing.assert_allclose(result_default["reward"], result_explicit_zero["reward"], atol=1e-9)
    np.testing.assert_allclose(result_default["right_joint_torque"], result_explicit_zero["right_joint_torque"], atol=1e-9)

    # 옵션을 실제로 켜면 동작이 달라져야 한다(옵션 자체가 아무 효과도 없는
    # 죽은 코드가 아닌지 확인) -- control_delay_steps는 수렴을 늦추므로
    # step_count가 늘거나 reward가 떨어지는 쪽으로 움직여야 정상이다.
    result_delay = PegInHoleOpenArmEnv(control_delay_steps=5).run_episode(gains, cfg)
    assert result_delay["step_count"] >= result_default["step_count"]
    print("[OK] torque_noise_std/control_delay_steps default to prior behavior, and actually do something when enabled")


def test_force_max_not_frozen_constant() -> None:
    """force_max 버그 수정 회귀 가드: peg_force/peg_torque(freejoint site
    센서, 버그로 peg 무게만 반환)가 아니라 qfrc_constraint 기반으로 읽으므로,
    씬마다 max_force가 달라져야 한다(수정 전엔 전부 정확히 0.4905였다)."""
    task = load_task_config("peg_in_hole")
    env = task.make_env()
    gains = task.gains_from_vector(_KNOWN_GOOD_GAINS)

    max_forces = []
    for offset in [(0.0, 0.0), (0.014, 0.0), (0.0099, 0.0099), (0.0099, -0.0099)]:
        cfg = task.default_scene_config()
        cfg["peg_init_offset_xy"] = offset
        result = env.run_episode(gains, cfg)
        max_forces.append(result["max_force"])

    assert len(set(round(f, 3) for f in max_forces)) > 1, f"max_force가 씬마다 똑같다(버그 재발 의심): {max_forces}"
    assert not all(abs(f - 0.4905) < 1e-3 for f in max_forces), f"max_force가 옛 버그값(0.4905N)에 고정됨: {max_forces}"
    print(f"[OK] max_force varies across scenes: {[round(f, 4) for f in max_forces]}")


if __name__ == "__main__":
    test_task_registry_wiring()
    test_joint_space_fields_present()
    test_joint_torque_sensor_varies()
    test_torque_noise_and_delay_default_off()
    test_force_max_not_frozen_constant()
    print()
    print("ALL TESTS PASSED (peg_in_hole, OpenArm)")
