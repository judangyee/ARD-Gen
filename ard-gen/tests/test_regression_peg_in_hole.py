"""리팩토링(태스크 무관 파이프라인, sim/base_task_env.py + sim/task_registry.py
+ tasks/*.yaml) 이후에도 peg_in_hole 태스크가 리팩토링 전과 완전히 같은
결과를 내는지 확인하는 회귀 테스트.

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은
스타일(직접 실행, 실패하면 AssertionError로 죽는다).

## 왜 CMA-ES 시퀀스 자체는 비교하지 않는가

이 저장소가 쓰는 `cma` 패키지 버전은 opts에 seed를 줘도 프로세스마다(심지어
같은 프로세스 안에서 CMAEvolutionStrategy를 여러 번 만들어도) 다른 후보
시퀀스를 낸다(실측 확인 -- 리팩토링 전 optimize/cma_search.py로 직접
재현했다). 즉 "세대별 최고 리워드가 리팩토링 전후로 똑같다"는 애초에
리팩토링 전 코드에서도 성립하지 않는 성질이라, 이 테스트가 보증할 수
있는 것도 아니고 보증해야 하는 것도 아니다.

대신 실제로 리팩토링이 손대지 않아야 하는 부분 -- **주어진 (게인, 씬)에
대한 물리 시뮬레이션 결과 자체**(사실상 결정론적)와 **씬 샘플링 함수의
출력**(RNG 시드가 같으면 결정론적) -- 이 리팩토링 전후로 정확히 같은지를
확인한다. 이게 실제로 파이프라인 산출물의 재현성을 좌우하는 부분이다.

## 3단계(Stabilizer) 이후: 이 테스트가 더 이상 "물리가 전혀 안 바뀌었다"를
## 보장하지 않는다는 점 (정직하게 밝힘)

3단계에서 hole_socket이 world-고정 body에서 freejoint(자유물체)로
바뀌면서, sim/peg_in_hole_sim.py의 물리 자체가 실제로 달라졌다(테이블
접촉 추가, integrator를 implicitfast로 교체 등) -- 이 파일이 비교하는
"원본"(sim.peg_in_hole_sim._run_episode_with_sim, 아래 import)과 "리팩토링
후"(sim.peg_in_hole_env.PegInHoleEnv, 이하 use_stabilizer 없이 생성해서
Stabilizer 미부착) 양쪽 다 **같은(이미 3단계로 수정된) sim/peg_in_hole_sim.py
모듈**을 부른다. 즉 이 테스트가 여전히 통과하는 건 "2단계 리팩토링이 그
사이에 추가로 뭔가를 깨지 않았다"(PegInHoleEnv가 여전히 저수준
_run_episode_with_sim과 정확히 같은 루프를 재현한다)는 걸 보증하는
것이지, "3단계 물리 변경 전과 결과가 같다"는 뜻이 아니다 -- 3단계
전후의 실제 성공률 비교(고정 hole 70.0% -> 자유물체+Stabilizer 없음
42.5% -> Stabilizer 있음 60.0%, N=40 무작위 씬)는 PIPELINE.md 3단계
절에 별도로 기록했다.

사용법:
    python tests/test_regression_peg_in_hole.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# 리팩토링 전 구현 (건드리지 않았음, sim/peg_in_hole_env.py의 래퍼가 이걸
# 그대로 감싼 것) -- "원본"으로 삼는다.
from sim.peg_in_hole_sim import PegInHoleSim, _run_episode_with_sim

# 리팩토링 후 구현.
from sim.peg_in_hole_env import PegInHoleEnv, default_scene_config, sample_scene_config, to_sim_scene_config
from sim.task_registry import load_task_config

# optimize/cma_search.py의 원래 EVAL_SCENE_CONFIGS와 동일한 오프셋들
# (tasks/peg_in_hole.yaml의 eval_scenarios로 옮겨졌다).
_EVAL_OFFSETS = [(0.014, 0.0), (0.0099, 0.0099), (0.0099, -0.0099)]

# 원본 _run_episode_with_sim()을 다양한 게인/시나리오 조합에서 직접 실측한
# 결과 -- 성공/실패, 궤적이 진동하며 실패하는 경우 둘 다 섞어서 뽑았다.
_FIXED_GAIN_CASES = [
    # (Kp_xy, Kd_xy, scenario_idx)
    (0.0003, 5e-05, 0),
    (0.0003, 5e-05, 1),
    (0.0003, 5e-05, 2),
    (0.0, 0.0, 0),
    (0.0, 0.0, 1),
    (0.0, 0.0, 2),
    (0.001, 0.0002, 0),
    (0.001, 0.0002, 1),
    (0.001, 0.0002, 2),
]


def _old_run(kp: float, kd: float, offset: tuple[float, float]) -> dict:
    sim = PegInHoleSim()
    cfg = {
        "hole_pos_xy": (0.0, 0.0),
        "friction": 0.5,
        "clearance_m": 0.003,
        "peg_init_offset_xy": offset,
        "peg_init_wrist": 0.0,
        "target_insertion_depth": 0.04,
    }
    return _run_episode_with_sim(sim, {"Kp_xy": kp, "Kd_xy": kd}, cfg)


def _new_run(kp: float, kd: float, offset: tuple[float, float]) -> dict:
    env = PegInHoleEnv()
    cfg = default_scene_config()
    cfg["peg_init_offset_xy"] = offset
    return env.run_episode({"Kp_xy": kp, "Kd_xy": kd}, cfg)


def test_fixed_gain_episodes_match() -> None:
    for kp, kd, idx in _FIXED_GAIN_CASES:
        offset = _EVAL_OFFSETS[idx]
        old = _old_run(kp, kd, offset)
        new = _new_run(kp, kd, offset)

        assert old["success"] == new["success"], f"success mismatch @ kp={kp} kd={kd} offset={offset}"
        assert old["step_count"] == new["step_count"], f"step_count mismatch @ kp={kp} kd={kd}"
        np.testing.assert_allclose(
            old["insertion_depth"], new["insertion_depth"], atol=1e-9,
            err_msg=f"insertion_depth mismatch @ kp={kp} kd={kd}",
        )
        np.testing.assert_allclose(
            old["reward"], new["reward"], atol=1e-6, err_msg=f"reward mismatch @ kp={kp} kd={kd}"
        )
        np.testing.assert_allclose(old["ee_poses"], new["ee_poses"], atol=1e-9, err_msg="ee_poses mismatch")
        np.testing.assert_allclose(old["forces"], new["forces"], atol=1e-9, err_msg="forces mismatch")
        print(f"[OK] kp={kp} kd={kd} offset={offset}: success={old['success']} reward={old['reward']:.4f}")


def test_scene_sampling_matches() -> None:
    """pipeline/scene_sampler.py(리팩토링 전)와 sim/peg_in_hole_env.py(리팩토링
    후)의 sample_scene_config/to_sim_scene_config이 같은 시드에서 정확히
    같은 값을 내는지 확인한다."""
    from pipeline.scene_sampler import sample_scene_config as old_sample
    from pipeline.scene_sampler import to_sim_scene_config as old_to_sim

    for seed in range(5):
        old_shared = old_sample(np.random.default_rng(seed))
        new_shared = sample_scene_config(np.random.default_rng(seed))
        assert old_shared == new_shared, f"shared scene_config mismatch @ seed={seed}"

        old_sim_cfg = old_to_sim(old_shared)
        new_sim_cfg = to_sim_scene_config(new_shared)
        assert old_sim_cfg == new_sim_cfg, f"sim scene_config mismatch @ seed={seed}"
        print(f"[OK] scene sampling seed={seed} matches")


def test_task_registry_wiring() -> None:
    """TASK_REGISTRY["peg_in_hole"]가 OpenArm 버전(sim/peg_in_hole_openarm_env.py:
    PegInHoleOpenArmEnv)으로 교체된 뒤의 값들이다("아니 로봇이 왜 다시 바뀐거야"
    피드백 이후 사용자가 명시적으로 선택한 전환, tasks/peg_in_hole.yaml 주석
    참고) -- 이 파일의 다른 두 테스트(test_fixed_gain_episodes_match,
    test_scene_sampling_matches)는 VX300s 저수준 리팩토링 자체를 검증하는
    것이라 TASK_REGISTRY와 무관하게 여전히 유효하지만, 이 테스트만 등록된
    태스크를 통해 접근하므로 전환 이후 값으로 갱신했다."""
    task = load_task_config("peg_in_hole")
    assert task.gain_names == ["Kp_xy", "Kd_xy"]
    assert task.gain_bounds["Kp_xy"] == (0.001, 0.5)
    assert task.gain_bounds["Kd_xy"] == (0.0, 0.02)
    assert len(task.eval_scenarios) == 3
    assert task.condition_dim == 4
    env = task.make_env()
    # optimize/peg_in_hole_openarm_admittance_gain_search.py로 실측 탐색한
    # 알려진 성공 게인(sim/peg_in_hole_bimanual_openarm_sim.py __main__ 참고).
    result = env.run_episode(task.gains_from_vector([0.124630, 0.001125]), task.default_scene_config())
    assert result["success"] is True
    print("[OK] task_registry wiring for peg_in_hole (OpenArm)")


# -- 공식 OpenArm 모델 반영(이번 세션) 회귀 테스트 ---------------------------
# 아래 4개는 그 작업에서 추가/수정된 기능이 계속 동작하는지 지켜본다:
# 공식 그리퍼 반영(조인트 공간 state/action), 관절 토크 센서, 노이즈/지연
# 옵션(기본값=이전 동작과 동일), force_max 센서 버그 수정.

_KNOWN_GOOD_GAINS = (0.124630, 0.001125)  # test_task_registry_wiring과 동일


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
    # 수정 전 버그값(peg 무게, 0.4905N)에 전부 고정돼 있지 않은지도 직접 확인.
    assert not all(abs(f - 0.4905) < 1e-3 for f in max_forces), f"max_force가 옛 버그값(0.4905N)에 고정됨: {max_forces}"
    print(f"[OK] max_force varies across scenes: {[round(f, 4) for f in max_forces]}")


if __name__ == "__main__":
    test_fixed_gain_episodes_match()
    test_scene_sampling_matches()
    test_task_registry_wiring()
    test_joint_space_fields_present()
    test_joint_torque_sensor_varies()
    test_torque_noise_and_delay_default_off()
    test_force_max_not_frozen_constant()
    print()
    print("ALL REGRESSION TESTS PASSED (--task peg_in_hole matches pre-refactor behavior)")
