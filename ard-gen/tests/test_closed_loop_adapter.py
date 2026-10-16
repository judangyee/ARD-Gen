"""closedloop/peg_in_hole_adapter.py 검증 -- ARD-VLA(judangyee/ARD-VLA,
closed-loop-structure) 폐루프 작업 4단계, PIPELINE.md "폐루프 데이터" 절
참고.

두 그룹의 테스트:
1. ARD-VLA 저장소 없이도 도는 어댑터 자체 smoke test(프로토콜 다섯
   메서드 + run_recovery 네 가지가 안 죽고 의도대로 동작하는지).
2. ARD-VLA 저장소 경로가 `ARD_VLA_REPO` 환경변수로 주어지면, 그 저장소의
   실제 `decision_logic.py`/`closed_loop.py`를 import해서 스크립트
   정책(오라클)+이 어댑터로 peg_in_hole 한 에피소드를 끝까지 돌리고,
   주입한 F2(Actuator 파지 실패) 미끄러짐 -> R2 복구 -> 성공 흐름을 1회
   확인한다. 환경변수가 없으면 skip(사용자 지시 그대로).

사용법:
    python tests/test_closed_loop_adapter.py
    ARD_VLA_REPO=/path/to/ard-vla python tests/test_closed_loop_adapter.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

from closedloop.peg_in_hole_adapter import FAILURE_TYPES, PegInHoleClosedLoopEnv, ScriptedOraclePolicy
from sim.grasp_slip import ArmSlipConfig, GraspSlipConfig
from sim.peg_in_hole_bimanual_openarm_sim import BimanualPegInHoleOpenArmSim, _default_scene_config

_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "tasks", "peg_in_hole.graph.yaml")
_GAINS = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}


def _load_graph() -> dict:
    with open(_GRAPH_PATH) as f:
        return yaml.safe_load(f)


def _make_env(grasp_slip_config=None) -> PegInHoleClosedLoopEnv:
    sim = BimanualPegInHoleOpenArmSim(grasp_slip_config=grasp_slip_config)
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (0.014, 0.0)
    return PegInHoleClosedLoopEnv(sim, _load_graph(), cfg, dict(_GAINS), render=False)


def test_protocol_smoke_no_ard_vla() -> None:
    """ARD-VLA 없이 어댑터 혼자서 프로토콜 다섯 메서드를 올바르게 구현하는지."""
    env = _make_env()
    outer_half = env.reset_episode("align")
    assert outer_half > 0.0
    assert env.current_node == "align"

    obs = env.observe()
    assert set(["right_arm_qpos", "right_gripper_qpos", "left_arm_qpos", "left_gripper_qpos", "subtask"]) <= set(obs)
    assert obs["right_arm_qpos"].shape == (7,)
    assert obs["subtask"] == "align"

    policy = ScriptedOraclePolicy(env)
    align_done_step = None
    for step in range(1, 400):
        actions, decisions = policy.predict_action_chunk_with_decisions(obs)
        assert actions.shape == (1, 1, 3), "오라클 청크는 항상 (1,1,3)이어야 한다(모듈 docstring 참고)"
        assert decisions["success_prob"] in (0.0, 1.0)
        assert len(decisions["failure_probs"]) == len(FAILURE_TYPES)
        env.execute(actions[0, 0])
        if env.subtask_finished():
            align_done_step = step
            break
        obs = env.observe()
    assert align_done_step is not None, "align이 안 끝났다"
    assert env._subtask_done_naturally and not env._subtask_timed_out
    print(f"[OK] protocol smoke: align finished naturally at step {align_done_step}")

    env.settle()  # render=False라 no-op이어야 함(예외 없이).
    env.safe_stop()
    assert env._stopped
    env.execute(np.zeros(3))  # safe_stop 이후 execute는 no-op이어야 한다(물리 전진 없음).
    print("[OK] settle()/safe_stop() no-op without render, execute() ignored after stop")


def test_run_recovery_all_four_ids() -> None:
    """R1/R2/R3/R4가 전부 예외 없이 동작하고, 의도한 메커니즘(R2=체크포인트
    복원+재파지, R3=게인 상승+재파지, R4=safe_stop)을 실제로 수행하는지."""
    slip_cfg = GraspSlipConfig(
        right=ArmSlipConfig(mode="injected", injected_at_step=20, injected_offset_local=np.array([0.01, 0.0, 0.0]))
    )
    env = _make_env(slip_cfg)
    env.reset_episode("align")
    for _ in range(25):
        delta, env._prev_dx, env._prev_dy = __import__(
            "sim.peg_in_hole_bimanual_openarm_sim", fromlist=["controller_step"]
        ).controller_step(env.sim, env.gains, env._prev_dx, env._prev_dy, env._outer_half, env._target_depth)
        env.execute(delta)

    peg_eq_id = env.sim.model.equality("peg_grasp").id
    shift_before = float(np.linalg.norm(env.sim.model.eq_data[peg_eq_id][3:6] - env.sim._right_slip._orig_relpose_pos))
    assert shift_before > 0.008, "injected 오프셋이 반영되지 않았다"

    env.run_recovery("R1")  # no-op이어야 함 -- 상태 안 바뀜.
    shift_after_r1 = float(np.linalg.norm(env.sim.model.eq_data[peg_eq_id][3:6] - env.sim._right_slip._orig_relpose_pos))
    assert abs(shift_after_r1 - shift_before) < 1e-12

    gains_before = dict(env.gains)
    env.run_recovery("R3")  # 게인 상승 + 재파지(그립만 복원, injected 플래그는 유지).
    shift_after_r3 = float(np.linalg.norm(env.sim.model.eq_data[peg_eq_id][3:6] - env.sim._right_slip._orig_relpose_pos))
    assert shift_after_r3 < 1e-9, "R3 재파지 후에도 relpose가 복원되지 않았다"
    assert env.gains["Kp_xy"] > gains_before["Kp_xy"], "R3가 게인을 올리지 않았다"
    assert env.sim._right_slip._injected_done, "R3가 injected '이미 발생함' 플래그를 잘못 풀었다(재발동 버그 재발)"

    env.run_recovery("R2")  # 체크포인트(align 시작 자세) 복원 + 재파지.
    assert env._step_in_subtask == 0

    env.run_recovery("R4")
    assert env._stopped
    print("[OK] R1(no-op)/R2(checkpoint restore)/R3(gain boost+regrasp, no re-injection)/R4(safe_stop) all behave as designed")


def test_ard_vla_integration() -> None:
    """ARD_VLA_REPO가 설정돼 있으면 실제 decision_logic.py/closed_loop.py로
    peg_in_hole 한 에피소드를 끝까지 돌려서, 주입한 F2 실패 -> R2 복구 ->
    성공 흐름을 확인한다. 설정 안 돼 있으면 skip(사용자 지시 그대로)."""
    repo = os.environ.get("ARD_VLA_REPO")
    if not repo:
        print("[SKIP] ARD_VLA_REPO not set -- ARD-VLA integration test skipped")
        return

    lerobot_src = os.path.join(repo, "third_party", "lerobot", "src")
    sys.path.insert(0, lerobot_src)
    from lerobot.policies.smolvla.closed_loop import run_closed_loop
    from lerobot.policies.smolvla.decision_logic import DecisionLogic

    graph = _load_graph()
    # F2(index1)/F3(index2) -> R2(복귀 후 재실행+재파지), F1(index0) -> R2도,
    # timeout(index3) -> R4(안전 정지). F2/F3를 R3(재파지만)로 매핑하면
    # 실측으로 재현 가능한 문제가 있다 -- PIPELINE.md "4단계" 절 참고: 이
    # injected 교란(15mm, 스텝 80)은 그립만 고쳐서는(R3) 600스텝 예산 안에
    # 복구가 안 됐다(물리적으로 흐트러진 위치/속도까지 R2가 되돌려야
    # 복구됨). 그래서 이 통합 테스트는 F1/F2/F3를 전부 R2로 매핑한다.
    decision_logic = DecisionLogic(
        task_graph=graph,
        option_vocab=["R1", "R2", "R3", "R4"],
        tau_success=0.5,
        tau_fail=0.5,
        max_recheck=3,
        failure_type_to_recovery={0: "R2", 1: "R2", 2: "R2", 3: "R4"},
    )

    slip_cfg = GraspSlipConfig(
        right=ArmSlipConfig(mode="injected", injected_at_step=80, injected_offset_local=np.array([0.015, 0.0, 0.0]))
    )
    env = _make_env(slip_cfg)
    env.reset_episode("align")
    policy = ScriptedOraclePolicy(env)

    events: list[tuple[str, str | None, str | None]] = []

    def post_decision(decision, decisions, state):
        events.append((decision.kind.value, decision.target, decisions.get("failure_type_label")))
        if decision.kind.value == "advance" and state.current_node:
            env.set_subtask(state.current_node)

    log = run_closed_loop(
        policy,
        env,
        decision_logic,
        goal="insert the peg into the hole",
        start_node="align",
        max_iterations=3000,
        max_subtask_steps=None,  # 어댑터 자신의 subtask_finished()가 노드별 max_steps로 timeout까지 판단한다.
        post_decision=post_decision,
    )

    assert log["termination_reason"] == "task_graph_end", f"에피소드가 성공으로 끝나지 않았다: {log}"
    kinds = [e[0] for e in events]
    assert "recover" in kinds, f"복구가 한 번도 일어나지 않았다: {events}"
    recover_events = [e for e in events if e[0] == "recover"]
    assert any(e[2] == "F2" for e in recover_events), f"F2(Actuator 파지 실패)로 복구가 트리거되지 않았다: {events}"
    print(f"[OK] ARD-VLA integration: injected F2 slip -> recover -> success. events={events}")


if __name__ == "__main__":
    test_protocol_smoke_no_ard_vla()
    test_run_recovery_all_four_ids()
    test_ard_vla_integration()
    print()
    print("ALL CLOSED LOOP ADAPTER TESTS PASSED (ARD-VLA integration skipped if ARD_VLA_REPO unset)")
