"""RoboTwin 2.0 이식 Part 3-4(validate_new_task.py) 테스트.

실제로 0->1->2A 전체를 (짧게) 돌려서 크래시 없이 성공률을 계산해
내는지 확인한다 -- subprocess로 cma_search.py/bootstrap.py를 실제로
실행하므로 다른 단위 테스트보다 느리다(수 초~수십 초).

사용법:
    python tests/test_validate_new_task.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validate_new_task import validate


def test_validate_demo_block_runs_end_to_end() -> None:
    """demo_block(torque_reactive, scaffold_task.py로 생성)에 대해
    0->1->2A를 짧게 돌려서 성공률이 [0,1] 범위의 유효한 float로
    나오는지 확인한다 -- 크래시 없이 전체 파이프라인이 연결됐다는
    증거(성공률 자체의 적정성은 validate_new_task.py가 직접 경고로
    보고하므로 여기서는 재검증하지 않는다)."""
    success_rate = validate("demo_block", max_generations=5, popsize=4, n_trials=10, seed=0)
    assert 0.0 <= success_rate <= 1.0
    print(f"[OK] demo_block: 0->1->2A 전체 파이프라인이 끝까지 실행됨(성공률={success_rate:.1%})")


if __name__ == "__main__":
    test_validate_demo_block_runs_end_to_end()
    print()
    print("ALL TESTS PASSED (validate_new_task)")
