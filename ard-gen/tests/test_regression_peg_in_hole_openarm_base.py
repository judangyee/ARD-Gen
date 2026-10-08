"""sim/peg_in_hole_bimanual_openarm_sim.py를 sim/openarm_bimanual_base.py:
OpenArmBimanualBase 상속으로 리팩토링한 뒤, 리팩토링 전과 물리/제어 결과가
바이트 단위로 같은지 확인하는 회귀 테스트.

tests/test_regression_peg_in_hole.py(VX300s -> 태스크 무관 파이프라인 리팩토링)
와 같은 원칙 -- 결정론적인 "주어진 (게인, 씬)에 대한 물리 시뮬레이션 결과"를
비교한다. 다만 이번엔 "리팩토링 전 모듈을 그대로 import"가 불가능하다(이
파일 자체를 수정하는 리팩토링이라 git에 "전" 버전이 남아있지 않음) -- 대신
tests/fixtures/peg_in_hole_openarm_pre_refactor.npz에 리팩토링 전 run_episode()
출력(ee_poses/forces/torques/actions/reward/success/step_count/insertion_depth,
성공 3케이스 + 실패/진동 2케이스)을 미리 저장해두고 그걸 "원본"으로 쓴다.

사용법:
    python tests/test_regression_peg_in_hole_openarm_base.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import _default_scene_config, run_episode

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "peg_in_hole_openarm_pre_refactor.npz")

# tests/fixtures/peg_in_hole_openarm_pre_refactor.npz를 만들 때 쓴 것과 정확히
# 같은 (게인, 오프셋) 조합/순서 -- 성공 3케이스(알려진 좋은 게인) + 실패/진동
# 2케이스(CMA-ES x0, 게인 0).
_CASES = [
    (0.124630, 0.001125, (0.014, 0.0)),
    (0.124630, 0.001125, (0.0099, 0.0099)),
    (0.124630, 0.001125, (0.0099, -0.0099)),
    (0.01, 0.0001, (0.014, 0.0)),
    (0.0, 0.0, (0.014, 0.0)),
]


def test_refactor_matches_pre_refactor_fixture() -> None:
    fixture = np.load(_FIXTURE_PATH)
    for i, (kp, kd, offset) in enumerate(_CASES):
        cfg = _default_scene_config()
        cfg["peg_init_offset_xy"] = offset
        result = run_episode({"Kp_xy": kp, "Kd_xy": kd}, cfg)
        prefix = f"case{i}_"

        assert bool(fixture[prefix + "success"]) == result["success"], f"success mismatch @ case{i}"
        assert int(fixture[prefix + "step_count"]) == result["step_count"], f"step_count mismatch @ case{i}"
        np.testing.assert_allclose(
            fixture[prefix + "insertion_depth"], result["insertion_depth"], atol=1e-9,
            err_msg=f"insertion_depth mismatch @ case{i}",
        )
        np.testing.assert_allclose(
            fixture[prefix + "reward"], result["reward"], atol=1e-6, err_msg=f"reward mismatch @ case{i}"
        )
        np.testing.assert_allclose(
            fixture[prefix + "ee_poses"], result["ee_poses"], atol=1e-9, err_msg=f"ee_poses mismatch @ case{i}"
        )
        np.testing.assert_allclose(
            fixture[prefix + "forces"], result["forces"], atol=1e-9, err_msg=f"forces mismatch @ case{i}"
        )
        np.testing.assert_allclose(
            fixture[prefix + "torques"], result["torques"], atol=1e-9, err_msg=f"torques mismatch @ case{i}"
        )
        np.testing.assert_allclose(
            fixture[prefix + "actions"], result["actions"], atol=1e-9, err_msg=f"actions mismatch @ case{i}"
        )
        print(
            f"[OK] case{i} kp={kp} kd={kd} offset={offset}: success={result['success']} "
            f"steps={result['step_count']} reward={result['reward']:.4f}"
        )


if __name__ == "__main__":
    test_refactor_matches_pre_refactor_fixture()
    print()
    print("ALL REGRESSION TESTS PASSED (openarm_bimanual_base refactor matches pre-refactor behavior)")
