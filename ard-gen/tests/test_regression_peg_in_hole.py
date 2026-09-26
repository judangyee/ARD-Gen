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
    task = load_task_config("peg_in_hole")
    assert task.gain_names == ["Kp_xy", "Kd_xy"]
    assert task.gain_bounds["Kp_xy"] == (0.00002, 0.003)
    assert task.gain_bounds["Kd_xy"] == (0.0, 0.0005)
    assert len(task.eval_scenarios) == 3
    assert task.condition_dim == 7
    env = task.make_env()
    result = env.run_episode(task.gains_from_vector([0.0003, 5e-05]), task.default_scene_config())
    assert result["success"] is True
    print("[OK] task_registry wiring for peg_in_hole")


if __name__ == "__main__":
    test_fixed_gain_episodes_match()
    test_scene_sampling_matches()
    test_task_registry_wiring()
    print()
    print("ALL REGRESSION TESTS PASSED (--task peg_in_hole matches pre-refactor behavior)")
