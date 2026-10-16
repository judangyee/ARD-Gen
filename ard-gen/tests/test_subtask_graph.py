"""peg_in_hole의 서브태스크 그래프(tasks/peg_in_hole.graph.yaml) + 진행도/
완료 조건/실패 유형 함수(sim/peg_in_hole_bimanual_openarm_sim.py의
align_progress/insert_progress/align_subtask_finished/insert_subtask_finished/
classify_failure) 검증 -- ARD-VLA 폐루프 작업 3단계, PIPELINE.md "폐루프
데이터" 절 참고.

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

사용법:
    python tests/test_subtask_graph.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from sim.peg_in_hole_bimanual_openarm_sim import (
    MAX_STEPS,
    _MAX_SANE_DEPTH_M,
    BimanualPegInHoleOpenArmSim,
    align_progress,
    align_subtask_finished,
    classify_failure,
    controller_step,
    insert_progress,
    insert_subtask_finished,
)

_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "peg_in_hole.graph.yaml")
_TACKER_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "tacker.graph.yaml")
_GAINS = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}


def test_graph_yaml_schema() -> None:
    """decision_logic.load_task_graph()가 요구하는 키({next, recoveries,
    max_retries})가 다 있는지(ARD-VLA 저장소 없이도 확인 가능 -- 그냥
    스키마 확인) + ARD-Gen 전용 completion 필드도 있는지."""
    with open(_GRAPH_PATH) as f:
        graph = yaml.safe_load(f)
    assert set(graph.keys()) == {"align", "insert"}
    assert graph["align"]["next"] == "insert"
    assert graph["insert"]["next"] is None
    for node in graph.values():
        assert "recoveries" in node and len(node["recoveries"]) > 0
        assert "max_retries" in node
        assert "completion" in node
    with open(_TACKER_GRAPH_PATH) as f:
        tacker_graph = yaml.safe_load(f)
    assert list(tacker_graph.keys()) == ["approach", "settle", "retract", "final_settle"]
    print("[OK] peg_in_hole.graph.yaml / tacker.graph.yaml schema sane")


def test_align_then_insert_on_real_episode() -> None:
    """알려진 좋은 게인으로 실제 에피소드를 돌리면서 align_subtask_finished/
    insert_subtask_finished를 매 스텝 추적한다 -- align이 insert보다 먼저
    끝나야 하고(그래프 순서), insert가 끝나는 스텝이 기존 is_success()
    판정(에피소드 전체 성공)과 정확히 일치해야 한다."""
    sim = BimanualPegInHoleOpenArmSim()
    cfg = {"friction": 0.5, "clearance_m": 0.003, "peg_init_offset_xy": (0.014, 0.0), "target_insertion_depth": 0.055 / 3}
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    prev_dx, prev_dy = 0.0, 0.0
    align_hold = 0
    insert_hold = 0
    align_done_step = None
    insert_done_step = None
    touched_wall_ever = False
    final_dx = final_dy = final_depth = 0.0

    for step in range(1, MAX_STEPS + 1):
        delta, prev_dx, prev_dy = controller_step(sim, _GAINS, prev_dx, prev_dy, outer_half, target_depth)
        sim.step(delta)
        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        depth = raw_depth if xy_ok else 0.0
        final_dx, final_dy, final_depth = dx, dy, depth

        if sim.is_peg_touching_hole_wall():
            touched_wall_ever = True

        if align_done_step is None:
            align_hold = align_hold + 1 if xy_ok else 0
            if align_subtask_finished(dx, dy, outer_half, align_hold):
                align_done_step = step

        insert_hold = insert_hold + 1 if depth >= target_depth else 0
        if insert_subtask_finished(depth, target_depth, insert_hold):
            insert_done_step = step
            break

    assert align_done_step is not None, "align 서브태스크가 끝나지 않았다"
    assert insert_done_step is not None, "insert 서브태스크가 끝나지 않았다(알려진 좋은 게인인데 실패)"
    assert align_done_step < insert_done_step, "align이 insert보다 먼저 끝나야 한다(그래프 순서)"
    assert insert_done_step == 157, f"알려진 성공 스텝 수(157)와 다르다: {insert_done_step}"
    assert touched_wall_ever, "이 시나리오는 실측으로 hole 벽 접촉이 있었다(F1 메커니즘이 실제로 발동하는지 확인)"

    progress = insert_progress(final_depth, target_depth)
    assert progress == 1.0

    label = classify_failure(
        success=True,
        right_weld_active=True,
        right_cumulative_shift_m=0.0,
        left_weld_active=True,
        left_cumulative_shift_m=0.0,
        peg_touching_hole_wall=touched_wall_ever,
        raw_depth=final_depth,
        target_depth=target_depth,
        timed_out=False,
    )
    assert label == "none"
    print(
        f"[OK] align_done@step{align_done_step} < insert_done@step{insert_done_step} "
        f"(== overall success), wall contact observed, classify_failure='none'"
    )


def test_classify_failure_priority() -> None:
    """F2 > F3 > F1 > timeout > none 우선순위가 실제로 지켜지는지(여러
    조건이 동시에 참이어도 가장 앞 유형이 나오는지) 직접 입력으로 확인."""
    # success면 다른 조건과 무관하게 항상 none.
    assert classify_failure(True, False, 0.05, False, 0.05, True, 0.0, 0.018, True) == "none"
    # 오른팔 weld 꺼짐(release) -- F2가 가장 먼저.
    assert classify_failure(False, False, 0.0, True, 0.05, True, 0.0, 0.018, True) == "F2"
    # 오른팔 이동량 초과(force_limited/injected) -- F2.
    assert classify_failure(False, True, 0.01, True, 0.0, False, 0.018, 0.018, False) == "F2"
    # 오른팔은 정상, 왼팔 weld 꺼짐 -- F3.
    assert classify_failure(False, True, 0.0, False, 0.0, True, 0.0, 0.018, True) == "F3"
    # 양팔 다 정상, 깊이 미달 + 벽 접촉 -- F1.
    assert classify_failure(False, True, 0.0, True, 0.0, True, 0.0, 0.018, False) == "F1"
    # 양팔 정상, 깊이 미달, 벽 접촉 없음, timeout -- timeout.
    assert classify_failure(False, True, 0.0, True, 0.0, False, 0.0, 0.018, True) == "timeout"
    print("[OK] classify_failure priority F2 > F3 > F1 > timeout > none")


def test_align_progress_monotonic_on_real_episode() -> None:
    """align_progress가 초기 0 근처에서 시작해 align 완료 시점에는 1.0에
    가까워야 한다(엄격한 단조성은 admittance PD 특성상 보장 못 하지만,
    끝값이 충분히 커야 한다는 약한 확인만 한다)."""
    sim = BimanualPegInHoleOpenArmSim()
    cfg = {"friction": 0.5, "clearance_m": 0.003, "peg_init_offset_xy": (0.014, 0.0), "target_insertion_depth": 0.055 / 3}
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    peg_tip0 = sim.get_peg_tip_pos()
    hole0 = sim.get_hole_center_pos()
    initial_xy_err = float(np.hypot(hole0[0] - peg_tip0[0], hole0[1] - peg_tip0[1]))

    prev_dx, prev_dy = 0.0, 0.0
    progresses = []
    for step in range(1, 30):
        delta, prev_dx, prev_dy = controller_step(sim, _GAINS, prev_dx, prev_dy, outer_half, target_depth)
        sim.step(delta)
        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        progresses.append(align_progress(dx, dy, initial_xy_err))

    assert progresses[-1] > 0.5, f"align_progress가 30스텝 뒤에도 낮다: {progresses[-1]}"
    print(f"[OK] align_progress reaches {progresses[-1]:.3f} after 30 steps (initial_xy_err={initial_xy_err*1000:.2f}mm)")


if __name__ == "__main__":
    test_graph_yaml_schema()
    test_align_then_insert_on_real_episode()
    test_classify_failure_priority()
    test_align_progress_monotonic_on_real_episode()
    print()
    print("ALL SUBTASK GRAPH TESTS PASSED")
