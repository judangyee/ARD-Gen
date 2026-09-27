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
    assert result["retracted"] is True
    print("[OK] task_registry wiring for tacker (OpenArm)")


def test_retract_phase_completes_without_new_failures() -> None:
    """후퇴(retract) 단계를 추가한 뒤에도 새로운 실패 모드가 안 생기는지 --
    기본 설정(release_grip_after="none", 왼팔 강화 게인 끝까지 유지)에서는
    발사 반동만 성공/실패를 가르고, 후퇴 자체는 항상 완료되며 후퇴 중
    workpiece가 눈에 띄게 더 밀리지 않아야 한다(PIPELINE.md의 retract 절
    N=350 실측: 필터링 성공률이 retract 추가 전후로 동일했다, 여기서는
    더 작은 N으로 빠르게 같은 것을 확인)."""
    task = load_task_config("tacker")
    env = task.make_env()
    rng = np.random.default_rng(11)
    retract_bumps = []
    for _ in range(_N):
        cfg = task.sample_scene_config(rng)
        result = env.run_episode(_SEED_GAINS, cfg)
        assert result["retracted"] is True, "기본 설정(그립 유지)에서는 후퇴가 항상 완료돼야 함"
        retract_bumps.append(result["retract_bump"])
    # 그립을 안 놓으면 후퇴 중 밀림이 발사 반동 자체보다 훨씬 작아야 한다
    # (실측: 중앙값 0.1mm대) -- 이게 커지면 후퇴 로직 자체에 회귀가 생긴 것.
    assert np.median(retract_bumps) < 0.001, "후퇴 중 workpiece가 예상보다 많이 밀림 -- 회귀 의심"
    print(f"[OK] retract 단계 정상 동작 확인 (중앙값 retract_bump={np.median(retract_bumps) * 1000:.4f}mm)")


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


def test_releasing_grip_early_increases_retract_bump() -> None:
    """"왼팔 강화 게인을 언제 놔도 되는가" 실험(PIPELINE.md의 retract 절
    N=150 실측 참고): 정착 직후(retract 시작 전)에 놓으면(release_grip_after
    ="fire_settle") 그 다음 후퇴 구간에서 workpiece가 "none"(끝까지 유지)
    보다 뚜렷하게 더 밀려야 한다 -- 안 그러면 이 실험 결론(끝까지 유지하는
    게 제일 안전하다) 자체가 재현이 안 되는 것."""
    task = load_task_config("tacker")

    def run(env) -> np.ndarray:
        rng = np.random.default_rng(5)
        bumps = []
        for _ in range(_N):
            cfg = task.sample_scene_config(rng)
            result = env.run_episode(_SEED_GAINS, cfg)
            bumps.append(result["retract_bump"])
        return np.array(bumps)

    bump_none = run(TackerOpenArmEnv(release_grip_after="none"))
    bump_early = run(TackerOpenArmEnv(release_grip_after="fire_settle"))

    print(
        f"[tacker] retract_bump 중앙값 -- 그립 유지: {np.median(bump_none) * 1000:.4f}mm | "
        f"정착 직후 해제: {np.median(bump_early) * 1000:.4f}mm"
    )
    assert np.median(bump_early) > np.median(bump_none) * 5, "정착 직후 그립을 놔도 후퇴 중 밀림이 안 늘어남 -- 회귀 의심"
    print("[OK] 그립을 일찍 놓으면 후퇴 중 밀림이 뚜렷하게 늘어난다는 게 실측으로 확인됨")


if __name__ == "__main__":
    test_task_registry_wiring()
    test_retract_phase_completes_without_new_failures()
    test_left_arm_reinforced_gains_matter()
    test_releasing_grip_early_increases_retract_bump()
    print()
    print("ALL TACKER (OPENARM) TESTS PASSED")
