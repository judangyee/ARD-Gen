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
    classify_failure_by_subtask,
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
    스키마 확인) + ARD-Gen 전용 completion 필드도 있는지. `failure_recovery`
    는 후속 작업("실패 유형 재정의")이 추가한 노드 레벨이 아닌 그래프
    최상위 전용 필드(decision_logic.load_task_graph()는 모르는 키라 노드
    순회 루프에서 빼야 함)."""
    with open(_GRAPH_PATH) as f:
        graph = yaml.safe_load(f)
    assert set(graph.keys()) == {"align", "insert", "failure_recovery"}
    assert graph["align"]["next"] == "insert"
    assert graph["insert"]["next"] is None
    for node_id in ("align", "insert"):
        node = graph[node_id]
        assert "recoveries" in node and len(node["recoveries"]) > 0
        assert "max_retries" in node
        assert "completion" in node
    assert set(graph["failure_recovery"].keys()) == {"F1a", "F1b", "F2", "F3", "timeout"}
    for recovery_id in graph["failure_recovery"].values():
        assert recovery_id in ("R1", "R2", "R3", "R4")
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


def _hist(dx=None, dy=None, raw_depth=None, wall_contact=None, n: int = 30) -> dict:
    """합성 시계열 history dict -- classify_failure_by_subtask() 테스트용.
    안 준 키는 0(또는 False)으로 n개 채운다."""
    return {
        "dx": list(dx) if dx is not None else [0.0] * n,
        "dy": list(dy) if dy is not None else [0.0] * n,
        "raw_depth": list(raw_depth) if raw_depth is not None else [0.0] * n,
        "wall_contact": list(wall_contact) if wall_contact is not None else [False] * n,
    }


def test_classify_failure_by_subtask_categories() -> None:
    """후속 작업("실패 유형 재정의") 핵심 산출물 -- 합성 시계열로 F1a/
    F1b/timeout/F2/F3/none 6가지가 각각 의도대로 분류되는지. 임계값은
    pipeline/diagnose_timeout_boundaries.py의 228개 실측(PIPELINE.md
    "실패 유형 재정의" 절)으로 고른 값과 동일(classify_f1_or_timeout
    docstring 참고)."""
    outer_half = 0.02  # 20mm
    target_depth = 0.018  # 18mm

    # success=True -> 항상 none(미끄러짐/걸림 여부와 무관).
    assert (
        classify_failure_by_subtask(
            "insert", _hist(raw_depth=[0.02] * 30), True, True, 0.0, True, 0.0, outer_half, target_depth
        )
        == "none"
    )

    # F2 > F3 > F1a/F1b/timeout 우선순위 -- jam처럼 보이는 history라도
    # weld 상태가 더 먼저 걸린다.
    jam_hist = _hist(raw_depth=[0.003] * 30, wall_contact=[True] * 25 + [False] * 5)
    assert classify_failure_by_subtask(
        "insert", jam_hist, False, False, 0.0, True, 0.0, outer_half, target_depth
    ) == "F2", "오른팔 weld 꺼짐이면 F1b처럼 보여도 F2가 먼저"
    assert classify_failure_by_subtask(
        "insert", jam_hist, False, True, 0.01, True, 0.0, outer_half, target_depth
    ) == "F2", "오른팔 누적이동 초과도 F2"
    assert classify_failure_by_subtask(
        "insert", jam_hist, False, True, 0.0, False, 0.0, outer_half, target_depth
    ) == "F3", "왼팔 weld 꺼짐 -- F3"

    # align, 거의성공(timeout) -- xy오차가 끝에 outer_half 안으로 들어옴.
    align_near = _hist(dx=[0.005] * 30)
    assert classify_failure_by_subtask(
        "align", align_near, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "timeout"

    # align, 진동(F1a) -- xy오차가 부호를 계속 바꾸고 끝에도 outer_half 밖.
    align_osc = _hist(dx=[0.03, -0.03] * 15)
    assert classify_failure_by_subtask(
        "align", align_osc, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "F1a"

    # align, 정렬_미수렴(F1a) -- 계속 같은 큰 오차에 멈춰있음(진동도 거의성공도 아님).
    align_stuck = _hist(dx=[0.03] * 30)
    assert classify_failure_by_subtask(
        "align", align_stuck, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "F1a"

    # insert, 거의성공(timeout) -- 깊이가 목표의 90%+ 도달.
    insert_near = _hist(raw_depth=list(np.linspace(0.0, target_depth * 0.95, 30)))
    assert classify_failure_by_subtask(
        "insert", insert_near, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "timeout"

    # insert, 걸림(F1b) -- 벽접촉비율>0.3, 깊이가 목표의 80% 미달.
    assert classify_failure_by_subtask(
        "insert", jam_hist, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "F1b"

    # insert, 진동_발산(F1a) -- 깊이가 오르내림을 반복(벽접촉 없음, 거의성공도 아님).
    insert_osc = _hist(raw_depth=[0.012, 0.002] * 15, wall_contact=[False] * 30)
    assert classify_failure_by_subtask(
        "insert", insert_osc, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "F1a"

    # insert, 정렬_미수렴(F1a) -- 얕은 깊이에 멈춰있음(벽접촉/진동/거의성공 다 아님).
    insert_stuck = _hist(raw_depth=[0.005] * 30, wall_contact=[False] * 30)
    assert classify_failure_by_subtask(
        "insert", insert_stuck, False, True, 0.0, True, 0.0, outer_half, target_depth
    ) == "F1a"

    print("[OK] classify_failure_by_subtask: none/F2/F3/align-timeout/align-F1a(진동+정체)/insert-timeout/F1b/insert-F1a(진동+정체) 전부 의도대로 분류")


def test_legacy_vs_new_rule_on_same_episodes() -> None:
    """legacy_failure_rule=True/False로 같은 시드의 B 에피소드를 똑같이
    돌려서(물리는 동일해야 함) 경계 판정 분포를 비교한다(사용자 지시
    "legacy vs 새 규칙 분포 비교"). 물리(right_arm.action)는 완전히
    같아야 하고(판정 로직은 물리에 영향 안 줌 -- 복구 선택이 달라지는
    경우만 빼고), 판정 유형 집합만 달라야 한다."""
    import pipeline.generate_closed_loop_episodes as gen

    rng_legacy = np.random.default_rng(2222)
    rng_new = np.random.default_rng(2222)
    legacy_labels: dict[str, int] = {}
    new_labels: dict[str, int] = {}
    n_matching_recovery_ep = 0
    for i in range(30):
        ep_legacy = gen.run_one_episode(rng_legacy, "B", seed=i, legacy_failure_rule=True)
        ep_new = gen.run_one_episode(rng_new, "B", seed=i, legacy_failure_rule=False)
        for b in ep_legacy["subtask_boundaries"]:
            legacy_labels[b["failure_type"]] = legacy_labels.get(b["failure_type"], 0) + 1
        for b in ep_new["subtask_boundaries"]:
            new_labels[b["failure_type"]] = new_labels.get(b["failure_type"], 0) + 1
        # 둘 다 첫 attempt까지는(복구가 아직 안 갈렸으므로) 물리가 똑같아야 한다.
        if len(ep_legacy["subtask_boundaries"]) >= 1 and len(ep_new["subtask_boundaries"]) >= 1:
            b0_legacy, b0_new = ep_legacy["subtask_boundaries"][0], ep_new["subtask_boundaries"][0]
            assert b0_legacy["start_step"] == b0_new["start_step"]
            assert b0_legacy["end_step"] == b0_new["end_step"]
            n_matching_recovery_ep += 1
    assert n_matching_recovery_ep > 0
    assert "F1" in legacy_labels or "timeout" in legacy_labels, f"legacy 라벨 집합이 이상하다: {legacy_labels}"
    assert set(new_labels.keys()) <= {"F1a", "F1b", "F2", "F3", "timeout", "none"}, f"새 라벨 집합이 이상하다: {new_labels}"
    assert "F1" not in new_labels, "새 규칙인데 legacy 라벨 F1이 나왔다"
    print(f"[OK] legacy vs 새 규칙 분포 비교 (N=30): legacy={legacy_labels} / new={new_labels}")


def test_legacy_recovery_always_r2() -> None:
    """legacy_failure_rule=True면 복구가 유형과 무관하게 항상 R2여야
    한다(이전 동작 그대로, graph yaml의 failure_recovery 맵을 안 쓴다)."""
    import pipeline.generate_closed_loop_episodes as gen

    rng = np.random.default_rng(333)
    n_recoveries = 0
    for i in range(20):
        ep = gen.run_one_episode(rng, "B", seed=i, legacy_failure_rule=True)
        for b in ep["subtask_boundaries"]:
            if b["recovery_applied"] is not None:
                n_recoveries += 1
                assert b["recovery_applied"] == "R2"
    assert n_recoveries > 0, "복구가 한 번도 안 일어나서 이 테스트가 아무것도 확인 못 했다"
    print(f"[OK] legacy_failure_rule=True: 복구 {n_recoveries}건 전부 R2")


if __name__ == "__main__":
    test_graph_yaml_schema()
    test_align_then_insert_on_real_episode()
    test_classify_failure_priority()
    test_align_progress_monotonic_on_real_episode()
    test_classify_failure_by_subtask_categories()
    test_legacy_vs_new_rule_on_same_episodes()
    test_legacy_recovery_always_r2()
    print()
    print("ALL SUBTASK GRAPH TESTS PASSED")
