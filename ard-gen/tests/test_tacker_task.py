"""tacker(타카) 태스크 검증 -- 이 태스크는 다른 태스크들과 달리 "이전 버전과
동일한지"를 확인할 대상이 없다(신규 태스크). 대신 이 태스크를 추가한 목적
자체 -- 발사 반동을 Stabilizer가 흡수하는지가 성공/실패를 가른다 -- 가
실제로 성립하는지를 검증한다(PIPELINE.md의 tacker 절 참고, N=150 전체
통계는 거기 기록돼 있다 -- 여기는 더 적은 N으로 빠르게 회귀를 잡는 용도).

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

_SEED_GAINS = {"Kp_approach": 0.5}
_N = 40


def test_task_registry_wiring() -> None:
    task = load_task_config("tacker")
    assert task.gain_names == ["Kp_approach"]
    assert task.gain_bounds["Kp_approach"] == (0.02, 3.0)
    assert len(task.eval_scenarios) == 3
    env = task.make_env()
    result = env.run_episode(_SEED_GAINS, task.default_scene_config())
    assert result["success"] is True
    assert result["fired"] is True
    print("[OK] task_registry wiring for tacker")


def test_stabilizer_absorbs_recoil() -> None:
    """이 태스크를 만든 이유 자체에 대한 검증: 같은 씬 시퀀스, 같은 게인으로
    Stabilizer가 있을 때 vs 없을 때 성공률/변위가 실제로 크게 갈려야 한다.
    안 갈리면(예: 둘 다 비슷하게 성공/실패) 반동 강도(recoil_strength) 설정을
    다시 봐야 한다는 신호다."""
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

    succ_with, disp_with = run(task.make_env())
    succ_without, disp_without = run(task.make_env(use_stabilizer=False))

    print(
        f"[tacker] Stabilizer 있음: {succ_with}/{_N} 성공, 중앙값 변위={np.median(disp_with) * 1000:.3f}mm | "
        f"Stabilizer 없음: {succ_without}/{_N} 성공, 중앙값 변위={np.median(disp_without) * 1000:.3f}mm"
    )
    # Stabilizer가 있으면 사실상 항상 성공해야 한다(정지 상태 대비 발사
    # 순간의 순간적 부하 정도는 kp=20000 weld가 가볍게 버텨야 함).
    assert succ_with >= _N * 0.9, "Stabilizer가 있는데도 성공률이 낮음 -- 회귀 의심"
    # Stabilizer 없이는 눈에 띄게 나빠져야 한다(그래야 이 태스크가 Stabilizer의
    # 존재 가치를 검증하는 의미가 있다).
    assert succ_without < succ_with, "Stabilizer 유무로 성공률 차이가 없음 -- 반동 강도 재설정 필요"
    assert np.median(disp_without) > np.median(disp_with) * 10, "Stabilizer 유무로 변위 차이가 크지 않음"
    print("[OK] Stabilizer가 발사 반동을 흡수한다는 게 실측으로 확인됨")


if __name__ == "__main__":
    test_task_registry_wiring()
    test_stabilizer_absorbs_recoil()
    print()
    print("ALL TACKER TESTS PASSED")
