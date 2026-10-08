"""insertion 태스크 검증(sim/insertion_openarm_env.py, OpenArm 양팔 버전).

gym-aloha(huggingface) InsertionTask를 "소켓/peg 물체 + 단계형 보상"만
이식한 신규 태스크다(이전 버전과 비교할 회귀 대상 없음, peg_in_hole처럼
팔/그리퍼/home 자세/admittance 컨트롤러는 그대로 재사용). 검증 목표는
(1) reset/step이 정상 동작하는지(스모크), (2) 랜덤 액션으로 몇 에피소드를
돌려도 안 죽는지, (3) peg_in_hole에서 재사용한 게인으로 eval_scenarios
전부가 실제로 stage=4(pin 접촉, 완전 삽입)에 도달하는지(이 태스크의 성공
기준 자체가 성립하는지).

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 태스크 테스트(tests/
test_tacker_task.py)와 같은 스타일.

사용법:
    python tests/test_insertion_task.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.insertion_openarm_env import InsertionOpenArmEnv
from sim.task_registry import load_task_config

# peg_in_hole에서 실측 검증된 게인 -- 팔 기구학/home 자세/grasp anchor가
# 전부 동일해서 그대로 재사용 가능함을 실측 확인했다(tasks/insertion.yaml
# 주석, sim/insertion_bimanual_openarm_sim.py __main__ 참고).
_SEED_GAINS = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}


def test_task_registry_wiring() -> None:
    task = load_task_config("insertion")
    assert task.gain_names == ["Kp_xy", "Kd_xy"]
    assert task.gain_bounds["Kp_xy"] == (0.001, 0.5)
    assert len(task.eval_scenarios) == 3
    assert task.env_class is InsertionOpenArmEnv
    assert task.success_reward_threshold == 4.0
    env = task.make_env()
    result = env.run_episode(_SEED_GAINS, task.default_scene_config())
    assert result["success"] is True
    assert result["stage"] == 4
    print("[OK] task_registry wiring for insertion (OpenArm)")


def test_reset_step_smoke() -> None:
    """reset()이 유한한 outer_half를 반환하고, step() 몇 번이 에러 없이
    돌면서 peg/socket 상태가 전부 유한한 값으로 유지되는지."""
    env = InsertionOpenArmEnv()
    cfg = load_task_config("insertion").default_scene_config()
    outer_half = env.reset(cfg)
    assert np.isfinite(outer_half) and outer_half > 0

    sim = env._sim
    for _ in range(10):
        env.step(np.array([0.0, 0.0, -0.0001]))
        assert np.isfinite(sim.get_peg_tip_pos()).all()
        assert np.isfinite(sim.get_socket_center_pos()).all()
        assert np.isfinite(sim.get_ee_pose()).all()
        force, torque = sim.get_force_torque()
        assert np.isfinite(force).all() and np.isfinite(torque).all()
    print("[OK] reset/step 스모크 테스트 통과")


def test_random_actions_do_not_crash() -> None:
    """작은 무작위 액션으로 몇 에피소드를 돌려도(성공 여부와 무관하게)
    크래시 없이 끝까지 돌고, 결과 필드가 전부 유한한 값인지."""
    task = load_task_config("insertion")
    env = task.make_env()
    rng = np.random.default_rng(3)
    for episode_i in range(3):
        cfg = task.sample_scene_config(rng)
        outer_half = env.reset(cfg)
        assert np.isfinite(outer_half)
        for _ in range(50):
            action = rng.uniform(-0.0005, 0.0005, size=3)
            env.step(action)
            assert np.isfinite(env._sim.get_peg_tip_pos()).all(), f"episode {episode_i}에서 비정상 상태"
    print("[OK] 무작위 액션 3 에피소드 모두 크래시 없이 통과")


def test_eval_scenarios_reach_pin_contact() -> None:
    """이 태스크의 핵심 검증: tasks/insertion.yaml의 eval_scenarios(peg_in_hole
    에서 그대로 가져온 오프셋) 전부가, 재사용한 게인으로 실제 stage=4(pin
    접촉)까지 도달하는지. 안 되면 pin 높이/clearance를 다시 조정해야 한다는
    뜻이다(sim 모듈 TARGET_INSERTION_DEPTH 주석 참고)."""
    task = load_task_config("insertion")
    env = task.make_env()
    for i, scenario in enumerate(task.eval_scenarios):
        cfg = task.default_scene_config()
        cfg.update(scenario)
        result = env.run_episode(_SEED_GAINS, cfg)
        assert result["stage"] == 4, (
            f"eval_scenarios[{i}]={scenario}가 stage=4(pin 접촉)에 도달하지 못함 "
            f"(도달한 최고 단계={result['stage']}, step={result['step_count']})"
        )
        assert result["success"] is True
        assert result["step_count"] < 300, "성공까지 너무 오래 걸림(원본 peg_in_hole은 157~162스텝)"
    print("[OK] eval_scenarios 3개 전부 stage=4(pin 접촉) 도달 확인")


def test_clearance_looser_than_peg_in_hole_default() -> None:
    """지시대로 gym-aloha의 느슨한 clearance를 기본값으로 채택했는지 --
    원본 peg_in_hole의 _NOMINAL_CLEARANCE_M(0.003m)보다 커야 한다."""
    from sim.insertion_bimanual_openarm_sim import _NOMINAL_CLEARANCE_M as insertion_clearance
    from sim.peg_in_hole_bimanual_openarm_sim import _NOMINAL_CLEARANCE_M as peg_in_hole_clearance

    assert insertion_clearance > peg_in_hole_clearance
    print(
        f"[OK] clearance 기본값이 peg_in_hole({peg_in_hole_clearance * 1000:.1f}mm)보다 "
        f"느슨함(insertion={insertion_clearance * 1000:.1f}mm, gym-aloha 기준)"
    )


if __name__ == "__main__":
    test_task_registry_wiring()
    test_reset_step_smoke()
    test_random_actions_do_not_crash()
    test_eval_scenarios_reach_pin_contact()
    test_clearance_looser_than_peg_in_hole_default()
    print()
    print("ALL INSERTION (OPENARM) TESTS PASSED")
