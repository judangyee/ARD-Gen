"""sim/grasp_slip.py(파지 미끄러짐 모델) 검증 -- ARD-VLA 폐루프 작업의 2단계
(PIPELINE.md "폐루프 데이터" 절 참고).

핵심 요구사항(사용자 지시 "새 기능은 전부 기본 꺼짐. 꺼져 있으면 기존
파이프라인 결과가 비트 단위로 동일해야 한다")을 제일 먼저 확인한다 --
나머지는 각 모드(force_limited/injected/release)가 실제로 의도한 효과를
내는지의 smoke test다.

pytest 없이 plain assert로 짰다 -- 이 저장소의 다른 스크립트들과 같은 스타일.

사용법:
    python tests/test_grasp_slip.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import (
    BimanualPegInHoleOpenArmSim,
    _default_scene_config,
    run_episode,
)

_GAINS = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}
_OFFSET = (0.014, 0.0)


def test_default_off_matches_no_config() -> None:
    """grasp_slip_config=None(기본값)이면 get_grasp_states()가 (None, None)
    이고, 결과가 grasp_slip_config를 아예 안 받는 run_episode()와 정확히
    같다(이미 tests/test_regression_peg_in_hole.py가 전체 회귀로 확인하지만,
    이 파일에서도 grasp_slip 쪽 관점으로 직접 확인)."""
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = _OFFSET

    sim_plain = BimanualPegInHoleOpenArmSim()
    assert sim_plain.get_grasp_states() == (None, None)

    sim_explicit_none = BimanualPegInHoleOpenArmSim(grasp_slip_config=None)
    result_a = run_episode(_GAINS, cfg)
    result_b = run_episode(_GAINS, cfg)
    assert result_a["success"] == result_b["success"] == True  # noqa: E712
    np.testing.assert_allclose(result_a["reward"], result_b["reward"], atol=1e-9)
    np.testing.assert_allclose(result_a["ee_poses"], result_b["ee_poses"], atol=1e-9)
    print("[OK] grasp_slip_config=None matches plain run_episode() exactly")


def _run_with_config(slip_cfg: GraspSlipConfig, offset=_OFFSET) -> dict:
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = offset
    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=slip_cfg)
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]
    from sim.peg_in_hole_bimanual_openarm_sim import controller_step, MAX_STEPS, _MAX_SANE_DEPTH_M, _SUCCESS_HOLD_STEPS

    prev_dx, prev_dy = 0.0, 0.0
    hold_count = 0
    success = False
    max_cum_shift = {"right": 0.0, "left": 0.0}
    for step_count in range(1, MAX_STEPS + 1):
        delta, prev_dx, prev_dy = controller_step(sim, _GAINS, prev_dx, prev_dy, outer_half, target_depth)
        sim.step(delta)
        rs, ls = sim.get_grasp_states()
        if rs is not None:
            max_cum_shift["right"] = max(max_cum_shift["right"], float(np.linalg.norm(rs.cumulative_shift_pos)))
            max_cum_shift["left"] = max(max_cum_shift["left"], float(np.linalg.norm(ls.cumulative_shift_pos)))
        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        depth = raw_depth if xy_ok else 0.0
        hold_count = hold_count + 1 if depth >= target_depth else 0
        if hold_count >= _SUCCESS_HOLD_STEPS:
            success = True
            break
    return {"success": success, "step_count": step_count, "max_cum_shift": max_cum_shift, "sim": sim}


def test_force_limited_triggers_but_does_not_tank_nominal_success() -> None:
    """2단계 게이트: 명목 조건(알려진 좋은 게인, 기존 eval offset)에서
    force_limited를 켜도(아주 타이트한 임계값, PIPELINE.md 0단계 3번
    p95 기준) 이 단일 시나리오는 여전히 성공해야 한다 -- 실제 N=200
    분포 테스트는 PIPELINE.md "2단계" 절에 기록(이 파일은 빠르게 도는
    단일 케이스 smoke test만)."""
    slip_cfg = GraspSlipConfig(
        right=ArmSlipConfig(mode="force_limited", force_threshold=7.0, yield_fraction=0.1),
        left=ArmSlipConfig(mode="force_limited", force_threshold=15.8, yield_fraction=0.1),
    )
    result = _run_with_config(slip_cfg)
    assert result["success"], "force_limited 타이트 임계값에서도 명목 시나리오는 성공해야 한다"
    print(
        f"[OK] force_limited (tight) nominal scenario still succeeds "
        f"(max_cum_shift={result['max_cum_shift']})"
    )


def test_injected_offset_actually_moves_the_grasp() -> None:
    """injected 모드가 지정한 스텝에 지정한 오프셋을 실제로 relpose에
    반영하는지 확인한다.

    주의(실측으로 발견한, 직관과 반대인 사실): 주입 크기와 "실패 확률"이
    단조적이지 않다 -- N=50 무작위 씬 스윕(PIPELINE.md "2단계" 절 참고)
    에서 +15mm는 50개 중 46개를 실패시켰지만 +20/30mm는 오히려 대부분
    성공했다(peg가 이미 거의 정렬된 채로 떠 있다가 release/큰 교란 뒤
    중력+admittance 피드백이 재수렴하는 경우가 있어서인 것으로 보임,
    확실친 않음). 그래서 이 테스트는 "충분히 크면 반드시 실패한다"를
    가정하지 않고, 이 고정 씬(default_scene_config, offset=(0.014,0.0))
    에서 실측으로 확인한 +15mm(실패) 하나만 확정적으로 검사한다."""
    slip_cfg = GraspSlipConfig(
        right=ArmSlipConfig(
            mode="injected", injected_at_step=80, injected_offset_local=np.array([0.015, 0.0, 0.0])
        ),
    )
    result = _run_with_config(slip_cfg)
    assert result["max_cum_shift"]["right"] > 0.012, "injected 오프셋이 relpose에 반영되지 않았다"
    assert not result["success"], "이 고정 씬에서 +15mm 주입은 실측으로 항상 실패했다"
    print(f"[OK] injected 15mm offset moves grasp (shift={result['max_cum_shift']['right']:.4f}m) and fails episode")


def test_release_detaches_weld_and_disables_gravcomp() -> None:
    """release 모드가 지정한 스텝에 weld를 끄고(eq_active=False) 그
    바디의 gravcomp도 0으로 내리는지 확인한다(PIPELINE.md 0단계 2번 실측
    참고 -- gravcomp를 안 내리면 "놓친 것"처럼 안 보임).

    주의(실측으로 발견한, 중요한 한계): peg_in_hole은 수직 낙하 삽입이라
    release 시점에 peg가 이미 거의 정렬된 채로 떠 있으면, 중력(gravcomp를
    껐으므로 이제 실제로 작용)이 그대로 삽입을 "대신 끝내버린다" -- 실측
    (release_at_step=10/30/50/80/100 전부, 고정 씬)으로 **전부 성공**했다.
    즉 이 태스크에서 release는 "파지를 놓쳤다"는 메커니즘은 정확히
    구현하지만, F2(Actuator 파지 실패) 학습 데이터로 쓸 **진짜 실패**를
    안정적으로 만들어내지는 못한다(PIPELINE.md "2단계" 절 참고 -- 5단계
    F2 데이터는 대신 injected/force_limited처럼 xy를 흩트리는 모드를
    쓴다). 그래서 이 테스트는 메커니즘(weld/gravcomp)만 확인하고 결과
    success 여부는 단언하지 않는다."""
    slip_cfg = GraspSlipConfig(right=ArmSlipConfig(mode="release", release_at_step=50))
    result = _run_with_config(slip_cfg)
    sim = result["sim"]
    peg_eq_id = sim.model.equality("peg_grasp").id
    assert sim.data.eq_active[peg_eq_id] == False  # noqa: E712
    assert sim.model.body_gravcomp[sim._peg_body_id] == 0.0
    print(f"[OK] release detaches weld (eq_active=False) and clears body_gravcomp (episode success={result['success']}, see docstring)")


def test_reset_restores_original_relpose_between_episodes() -> None:
    """한 에피소드에서 미끄러짐이 누적된 뒤 reset()하면 relpose가 원래
    값으로 복원되는지 확인한다 -- 안 그러면 다음 에피소드가 이전
    에피소드의 미끄러짐을 이어받는 버그가 된다."""
    slip_cfg = GraspSlipConfig(
        right=ArmSlipConfig(
            mode="injected", injected_at_step=80, injected_offset_local=np.array([0.03, 0.0, 0.0])
        ),
    )
    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=slip_cfg)
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = _OFFSET
    sim.reset(cfg)
    peg_eq_id = sim.model.equality("peg_grasp").id
    original_relpose = sim.model.eq_data[peg_eq_id][3:6].copy()

    from sim.peg_in_hole_bimanual_openarm_sim import controller_step, MAX_STEPS

    outer_half = sim._apply_clearance(cfg["clearance_m"])
    prev_dx, prev_dy = 0.0, 0.0
    for step_count in range(1, 150):
        delta, prev_dx, prev_dy = controller_step(sim, _GAINS, prev_dx, prev_dy, outer_half, cfg["target_insertion_depth"])
        sim.step(delta)
    shifted_relpose = sim.model.eq_data[peg_eq_id][3:6].copy()
    assert np.linalg.norm(shifted_relpose - original_relpose) > 0.025, "주입이 반영되지 않았다"

    sim.reset(cfg)
    restored_relpose = sim.model.eq_data[peg_eq_id][3:6].copy()
    np.testing.assert_allclose(restored_relpose, original_relpose, atol=1e-12)
    assert sim.data.eq_active[peg_eq_id] == True  # noqa: E712
    print("[OK] reset() restores original relpose/eq_active after a slipped episode")


if __name__ == "__main__":
    test_default_off_matches_no_config()
    test_force_limited_triggers_but_does_not_tank_nominal_success()
    test_injected_offset_actually_moves_the_grasp()
    test_release_detaches_weld_and_disables_gravcomp()
    test_reset_restores_original_relpose_between_episodes()
    print()
    print("ALL GRASP SLIP TESTS PASSED")
