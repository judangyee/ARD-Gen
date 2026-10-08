"""insertion 태스크(gym-aloha AlohaInsertionTask 포팅, OpenArm 양팔 버전,
sim/insertion_openarm_env.py) smoke test.

이 태스크는 **아직 성공하는 게인을 못 찾았다**(sim/insertion_openarm_env.py
모듈 docstring "현재 상태" 절, tasks/insertion.yaml 참고) -- 그래서 이
파일은 tests/test_tacker_task.py처럼 "알려진 좋은 게인으로 success=True"를
확인하지 않는다. 대신 사용자가 검증 항목으로 명시한 "reset/step smoke
test + 몇 개의 무작위 액션 에피소드"만 확인한다: reset()이 충돌/관절
한계 위반 없이 수렴하는지, step()이 NaN 없이 MAX_STEPS까지 안 죽고
도는지, TASK_REGISTRY 연결이 깨지지 않았는지.

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

사용법:
    python tests/test_insertion_task.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.insertion_openarm_env import BimanualInsertionOpenArmSim, InsertionOpenArmEnv, default_scene_config
from sim.task_registry import load_task_config

_LIMIT_FREEZE_MARGIN = 0.08
_RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]


def test_reset_converges_without_collision_or_joint_limit_violation() -> None:
    sim = BimanualInsertionOpenArmSim()
    cfg = default_scene_config()
    outer_half = sim.reset(cfg)
    assert outer_half > 0.0

    assert sim.data.ncon == 0, f"reset() 직후 예기치 않은 접촉 {sim.data.ncon}개"

    for name in _RIGHT_ARM_JOINTS:
        qadr = sim._right_arm_qposadr[name]
        lo, hi = sim.model.jnt_range[sim.model.joint(name).id]
        frac = (sim.data.qpos[qadr] - lo) / (hi - lo)
        assert 0.0 <= frac <= 1.0, f"{name} 관절 한계 위반 (frac={frac:.3f})"

    peg_tip = sim.get_peg_tip_pos()
    socket_center = sim.get_socket_center_pos()
    assert np.all(np.isfinite(peg_tip)) and np.all(np.isfinite(socket_center))
    print(f"[OK] reset() converges: outer_half={outer_half*1000:.1f}mm, ncon=0, joints within range")


def test_step_runs_without_nan_for_random_actions() -> None:
    rng = np.random.default_rng(0)
    for trial in range(3):
        sim = BimanualInsertionOpenArmSim()
        cfg = default_scene_config()
        sim.reset(cfg)
        for _ in range(200):
            action = rng.uniform(-0.001, 0.001, size=3)
            sim.step(action)
            assert np.all(np.isfinite(sim.data.qpos)), "qpos에 NaN/Inf 발생"
            assert np.all(np.isfinite(sim.data.qvel)), "qvel에 NaN/Inf 발생"
        print(f"[OK] random-action trial {trial}: 200 steps, no NaN/Inf")


def test_env_run_episode_smoke() -> None:
    """InsertionOpenArmEnv.run_episode()가 MAX_STEPS까지 안 죽고 끝나는지,
    반환 스키마가 BaseTaskEnv 계약(left_arm_traj 포함)을 지키는지 확인한다
    -- success 여부는 확인하지 않는다(위 모듈 docstring 참고, 아직 미해결)."""
    env = InsertionOpenArmEnv()
    cfg = default_scene_config()
    result = env.run_episode({"Kp_xy": 0.124630, "Kd_xy": 0.001125}, cfg)
    assert np.all(np.isfinite(result["reward"]))
    assert result["ee_poses"].shape[0] == result["actions"].shape[0] + 1
    assert result["left_arm_traj"].shape[0] == result["ee_poses"].shape[0]
    print(
        f"[OK] run_episode smoke: success={result['success']} "
        f"peg_touching_socket={result['peg_touching_socket']} step_count={result['step_count']}"
    )


def test_task_registry_wiring() -> None:
    task = load_task_config("insertion")
    assert task.gain_names == ["Kp_xy", "Kd_xy"]
    assert len(task.eval_scenarios) == 3
    assert task.env_class is InsertionOpenArmEnv
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector([0.124630, 0.001125]), task.default_scene_config())
    assert np.all(np.isfinite(result["reward"]))
    print(f"[OK] task_registry wiring for insertion (success={result['success']}, not yet expected to be True)")


if __name__ == "__main__":
    test_reset_converges_without_collision_or_joint_limit_violation()
    test_step_runs_without_nan_for_random_actions()
    test_env_run_episode_smoke()
    test_task_registry_wiring()
    print()
    print("ALL INSERTION SMOKE TESTS PASSED (success=True not yet achieved -- see module docstring)")
