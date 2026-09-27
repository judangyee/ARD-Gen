"""tacker(타카) 태스크 검증, OpenArm 양팔 버전(sim/tacker_openarm_env.py).
일반 Stabilizer(가상 EE) 버전은 peg_in_hole과 같은 이유로 레거시가 됐고
TASK_REGISTRY에서도 빠졌다 -- 이 파일은 그 버전을 대체한다.

이 태스크는 "이전 버전과 동일한지"를 확인할 회귀 대상이 없다(신규 태스크
설계). 대신 이 태스크를 추가한 목적 자체 -- 발사 반동을 왼팔(실제 7-DOF
팔, 강화 위치 게인)이 흡수하는지가 성공/실패를 가른다 -- 가 실제로
성립하는지를 검증한다(PIPELINE.md의 tacker 절 참고, N=150 전체 통계는
거기 기록돼 있다).

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

사용법:
    python tests/test_tacker_task.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.task_registry import load_task_config
from sim.tacker_openarm_env import TackerOpenArmEnv

_SEED_GAINS = {"Kp_approach": 0.2}
_N = 40


def test_task_registry_wiring() -> None:
    task = load_task_config("tacker")
    assert task.gain_names == ["Kp_approach"]
    assert task.gain_bounds["Kp_approach"] == (0.02, 3.0)
    assert len(task.eval_scenarios) == 3
    assert task.env_class is TackerOpenArmEnv
    env = task.make_env()
    result = env.run_episode(_SEED_GAINS, task.default_scene_config())
    assert result["success"] is True
    assert result["fired"] is True
    print("[OK] task_registry wiring for tacker (OpenArm)")


def test_left_arm_reinforced_gains_matter() -> None:
    """이 태스크를 만든 이유 자체에 대한 검증: 같은 씬 시퀀스, 같은 게인으로
    왼팔이 강화 위치 게인(kp=2273.98 등)일 때 vs OpenArm vendor 기본 게인
    (kp=230 등)으로 되돌렸을 때 성공률/변위가 실제로 크게 갈려야 한다."""
    task = load_task_config("tacker")

    def run(env) -> tuple[int, np.ndarray]:
        rng = np.random.default_rng(7)
        succ = 0
        disps = []
        for _ in range(_N):
            cfg = task.sample_scene_config(rng)
            result = env.run_episode(_SEED_GAINS, cfg)
            succ += int(result["success"])
            if result["fired"]:
                disps.append(result["displacement"])
        return succ, np.array(disps)

    succ_reinforced, disp_reinforced = run(TackerOpenArmEnv(left_arm_reinforced=True))
    succ_weak, disp_weak = run(TackerOpenArmEnv(left_arm_reinforced=False))

    print(
        f"[tacker] 왼팔 강화 게인: {succ_reinforced}/{_N} 성공, 중앙값 변위={np.median(disp_reinforced) * 1000:.3f}mm | "
        f"왼팔 vendor 게인: {succ_weak}/{_N} 성공, 중앙값 변위={np.median(disp_weak) * 1000:.3f}mm"
    )
    assert succ_weak < succ_reinforced, "왼팔 게인 강화 유무로 성공률 차이가 없음 -- recoil_strength 재설정 필요"
    assert np.median(disp_weak) > np.median(disp_reinforced), "왼팔 게인 강화 유무로 변위 차이가 없음"
    print("[OK] 왼팔 강화 게인이 발사 반동을 흡수한다는 게 실측으로 확인됨")


if __name__ == "__main__":
    test_task_registry_wiring()
    test_left_arm_reinforced_gains_matter()
    print()
    print("ALL TACKER (OPENARM) TESTS PASSED")
