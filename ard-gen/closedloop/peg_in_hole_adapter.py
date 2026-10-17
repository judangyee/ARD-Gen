"""ARD-VLA(judangyee/ARD-VLA, closed-loop-structure)의 `ClosedLoopEnv`
duck-typing 프로토콜(`observe`/`execute`/`subtask_finished`/`settle`/
`safe_stop`, 선택적 `run_recovery`)을 peg_in_hole 위에 구현한 어댑터 +
그 러너가 그대로 돌릴 수 있는 스크립트 정책/오라클 판정기 스텁.

ARD-VLA 쪽 코드는 이 저장소에 없다(judangyee/ARD-VLA 저장소를 참고만
했다) -- `tests/test_closed_loop_adapter.py`의 통합 테스트가 그 저장소
경로를 환경변수로 받아 실제 `decision_logic.py`/`closed_loop.py`를
import해서 돌린다(경로 없으면 skip).

## set_subtask()는 ARD-VLA 러너가 모른다 -- post_decision 콜백으로 연결

`run_closed_loop()`는 이 프로토콜 다섯 메서드 + 선택적 `run_recovery`만
안다. 서브태스크가 바뀌었을 때(`DecisionKind.ADVANCE`) 이 환경이 다음
노드의 완료 조건을 보도록 하려면, 호출부가 `run_closed_loop(...,
post_decision=...)` 콜백에서 `decision.kind == ADVANCE`일 때
`env.set_subtask(state.current_node)`를 직접 불러줘야 한다(사용자 지시
"ARD-VLA 쪽에는 나중에 연결한다고 PIPELINE.md에 적는다" -- 이 파일이
그 "나중"의 설계이고, 실제 연결은 `tests/test_closed_loop_adapter.py`의
통합 테스트가 보여준다).

## 오라클의 액션 청크는 항상 길이 1이다

admittance 컨트롤러는 매 스텝 실제 관측(peg_tip/hole_center)에 의존하는
연속 피드백 루프라서(controller_step 참고), 한 번의 관측만으로 여러
스텝 앞의 행동을 미리 정확하게 예측할 수 없다(real SmolVLA 같은 학습된
정책의 "액션 청크"와는 다른 성질). 그래서 `ScriptedOraclePolicy`는 매
`predict_action_chunk_with_decisions()` 호출마다 **정확히 한 스텝짜리
청크**(`actions.shape == (1, 1, 3)`)만 돌려준다 -- 매 물리 스텝마다
새로 forward를 부르는 비용을 감수하지만, 수치적으로 정확하다(이게 "오라클
기준선"이라는 이름에 걸맞다 -- 근사/예측이 아니라 그 순간의 정확한 제어
명령).

## 판정(decisions)은 "방금 끝난 서브태스크"를 본다, 다시 계산하지 않는다

`subtask_finished()`는 호출될 때마다 hold-steps 카운터를 갱신하는 부수
효과가 있다(이미 끝났다고 한 번 판정한 뒤 또 부르면 카운터가 망가진다).
그래서 오라클은 `subtask_finished()`를 직접 부르지 않고, 그 메서드가
True를 반환한 **그 순간** 어댑터에 남겨둔 `_subtask_done_naturally`/
`_subtask_timed_out` 플래그만 읽는다(ARD-VLA 러너의 실제 호출 순서 --
`_execute_chunk()`가 매 스텝 `subtask_finished()`를 부르고, 그 다음
iteration의 `predict_action_chunk_with_decisions()`가 그 결과를 판정에
쓰는 것 -- 과 맞물려서, 정확히 "방금 끝난 서브태스크"의 결과를 본다).
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.peg_in_hole_bimanual_openarm_sim import (
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

# 후속 작업("timeout 진단 -> 실패 유형 규칙 재정의", PIPELINE.md 참고)으로
# 기본값이 5개짜리 새 체계(F1a/F1b로 F1을 복구 방법 기준으로 나누고,
# 판정도 순간 스냅샷이 아니라 구간 시계열 기반)로 바뀌었다 -- 이전 4개
# 체계는 FAILURE_TYPES_LEGACY로 그대로 남겨뒀다(사용자 지시 "기본값
# 변경은 허용하되 이전 규칙을 legacy로 남긴다", `legacy_failure_rule=True`
# 로 선택). ARD-VLA의 judge_num_failure_types와 맞출 최종 인덱스 순서는
# PIPELINE.md "실패 유형 재정의" 절 표에 고정해뒀다.
FAILURE_TYPES_LEGACY = ["F1", "F2", "F3", "timeout"]
FAILURE_TYPES = ["F1a", "F1b", "F2", "F3", "timeout"]

# Judge 데이터 v2에서 근접 카메라 2대를 추가했다(assets/peg_in_hole_bimanual_
# openarm.xml의 hole_socket 바디 자식 close_cam_a/close_cam_b, PIPELINE.md
# "Judge 데이터 v2" 절 참고) -- top_cam/wide_cam은 기존 그대로 유지(비교용).
# 이 리스트 하나로 settle()/render_frames()/observe()가 전부 카메라를
# 돈다 -- 카메라를 더 늘리거나 줄일 때 이 한 곳만 고치면 된다.
JUDGE_CAMERAS = ["top_cam", "wide_cam", "close_cam_a", "close_cam_b"]


class PegInHoleClosedLoopEnv:
    """하나의 peg_in_hole 에피소드를 ARD-VLA 폐루프 러너가 구동할 수 있게
    감싼다. `render=True`면 판정 시점(settle() 호출 시)에 `JUDGE_CAMERAS`
    (top_cam/wide_cam/close_cam_a/close_cam_b, v2에서 근접 카메라 2대 추가)
    프레임을 캡처한다 -- 그때만 `MUJOCO_GL=osmesa`를 건드린다(렌더링을
    안 쓰면 이 모듈은 그 환경변수를 전혀 안 건드린다, PIPELINE.md의
    "렌더링 안 하는 스크립트에서는 MUJOCO_GL=osmesa 설정 금지" 규칙 --
    이 프로세스가 torch Adam을 만들 일이 없는 한(이 파일도 통합 테스트도
    optimizer를 만들지 않는다) osmesa와 같이 둬도 세그폴트 조건(PyTorch
    Adam 생성 시점)에 안 걸린다는 뜻으로 이해하고 적용했다)."""

    def __init__(
        self,
        sim: BimanualPegInHoleOpenArmSim,
        graph: dict[str, dict],
        scene_config: dict[str, Any],
        gains: dict[str, float],
        render: bool = False,
        r3_kp_boost: float = 1.3,
        legacy_failure_rule: bool = False,
    ):
        self.sim = sim
        self.graph = graph
        self.scene_config = scene_config
        self.gains = dict(gains)
        self._render = render
        self._r3_kp_boost = r3_kp_boost
        self._legacy_failure_rule = legacy_failure_rule

        self.current_node: str | None = None
        self._step_in_subtask = 0
        self._hold_count = 0
        self._outer_half = 0.0
        self._target_depth = float(scene_config["target_insertion_depth"])
        self._prev_dx = 0.0
        self._prev_dy = 0.0
        self._initial_xy_err = 0.0
        self._checkpoint: tuple[np.ndarray, np.ndarray, dict[str, float]] | None = None
        self._stopped = False
        self._subtask_done_naturally = False
        self._subtask_timed_out = False
        # 새 실패 유형 규칙(기본, legacy_failure_rule=False)이 쓰는 이번
        # attempt의 구간 시계열 -- set_subtask()마다 리셋, execute()마다
        # 한 스텝씩 쌓인다(sim/peg_in_hole_bimanual_openarm_sim.py의
        # compute_subtask_features()가 쓰는 바로 그 형식). legacy
        # 모드에서도 채우기는 하지만(비용이 작아서 끄지 않음) 안 읽는다.
        self._history: dict[str, list] = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": []}

        self._renderer = None
        if render:
            os.environ.setdefault("MUJOCO_GL", "osmesa")
            self._renderer = mujoco.Renderer(sim.model, height=240, width=320)
        self._frames: dict[str, np.ndarray | None] = {cam: None for cam in JUDGE_CAMERAS}

    # ------------------------------------------------------------------
    def reset_episode(self, start_node: str = "align") -> float:
        """ARD-VLA의 `run_closed_loop()`는 환경을 reset하지 않는다(그건
        호출부 책임) -- 호출부가 에피소드 시작 전에 이걸 불러야 한다."""
        self._outer_half = self.sim.reset(self.scene_config)
        self._prev_dx, self._prev_dy = 0.0, 0.0
        self._stopped = False
        peg_tip = self.sim.get_peg_tip_pos()
        hole_center = self.sim.get_hole_center_pos()
        self._initial_xy_err = float(np.hypot(hole_center[0] - peg_tip[0], hole_center[1] - peg_tip[1]))
        self.set_subtask(start_node)
        return self._outer_half

    def set_subtask(self, node_id: str) -> None:
        """사용자 지시의 확장 메서드 -- 러너가 노드를 바꿀 때 이 환경이
        어떤 완료 조건을 볼지 알려준다(ARD-VLA 쪽에는 연결 안 돼 있다,
        모듈 docstring 참고). R2(시작 자세 복귀)용 체크포인트도 여기서
        찍는다."""
        self.current_node = node_id
        self._step_in_subtask = 0
        self._hold_count = 0
        self._subtask_done_naturally = False
        self._subtask_timed_out = False
        self._history = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": []}
        self._checkpoint = (
            self.sim.data.qpos.copy(),
            self.sim.data.qvel.copy(),
            dict(self.sim._virtual_qpos),
        )

    # ---- ClosedLoopEnv 프로토콜 ---------------------------------------
    def observe(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "right_arm_qpos": self.sim.right_arm_qpos(),
            "right_gripper_qpos": self.sim.right_gripper_qpos(),
            "left_arm_qpos": self.sim.left_arm_qpos(),
            "left_gripper_qpos": self.sim.left_gripper_qpos(),
            "subtask": self.current_node,
        }
        if self._render:
            for cam in JUDGE_CAMERAS:
                obs[cam] = self._frames[cam]
        return obs

    def execute(self, action: Any) -> None:
        if self._stopped:
            return
        self.sim.step(np.asarray(action, dtype=float))
        self._step_in_subtask += 1
        # 새 실패 유형 규칙용 구간 시계열 기록(위 __init__의 self._history
        # docstring 참고) -- 비용이 작아서 legacy 모드에서도 채운다(안
        # 읽히기만 할 뿐).
        dx, dy = self.current_dx_dy()
        hole_center = self.sim.get_hole_center_pos()
        peg_tip = self.sim.get_peg_tip_pos()
        self._history["dx"].append(dx)
        self._history["dy"].append(dy)
        self._history["raw_depth"].append(float(hole_center[2] - peg_tip[2]))
        self._history["wall_contact"].append(bool(self.sim.is_peg_touching_hole_wall()))

    def subtask_finished(self) -> bool:
        """true를 돌려줄 때 그 이유(자연 완료 vs 스텝 예산 초과)를
        `_subtask_done_naturally`/`_subtask_timed_out`에 남긴다 -- 다음
        iteration의 오라클이 다시 계산하지 않고 이 값을 읽는다(모듈
        docstring 참고)."""
        node = self.graph[self.current_node]
        peg_tip = self.sim.get_peg_tip_pos()
        hole_center = self.sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_ok = abs(dx) < self._outer_half and abs(dy) < self._outer_half

        if self.current_node == "align":
            self._hold_count = self._hold_count + 1 if xy_ok else 0
            done = align_subtask_finished(
                dx, dy, self._outer_half, self._hold_count, node["completion"]["hold_steps"]
            )
        elif self.current_node == "insert":
            raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
            depth = raw_depth if xy_ok else 0.0
            self._hold_count = self._hold_count + 1 if depth >= self._target_depth else 0
            done = insert_subtask_finished(
                depth, self._target_depth, self._hold_count, node["completion"]["hold_steps"]
            )
        else:
            raise ValueError(f"unknown node {self.current_node!r}")

        if done:
            self._subtask_done_naturally = True
            self._subtask_timed_out = False
            return True
        if self._step_in_subtask >= node["max_steps"]:
            self._subtask_done_naturally = False
            self._subtask_timed_out = True
            return True
        return False

    def settle(self) -> None:
        """판정 직전 호출 -- 렌더링을 쓰면 여기서 프레임을 새로 캡처한다
        (물리적으로는 추가 스텝 없음, 이 admittance 루프는 별도 "정착"
        동역학이 없어서). `render=False`면 완전히 no-op."""
        if self._render:
            for cam in JUDGE_CAMERAS:
                self._frames[cam] = self._render_camera(cam)

    def render_frames(self) -> dict[str, np.ndarray]:
        """지금 이 순간의 `JUDGE_CAMERAS` 전체(top_cam/wide_cam/close_cam_a/
        close_cam_b)를 즉시 렌더링해서 돌려준다(후속 작업 "작업 1" 방법 A
        -- 생성 중 서브태스크 경계에서 바로 캡처). `settle()`의 캐시
        (`self._frames`)와는 별개로 호출 시점의 실제 장면을 새로 그린다 --
        복구 점프 직후처럼 `settle()`을 또 부르기 애매한 시점에도 쓸 수
        있게. `render=False`로 생성됐으면 `RuntimeError`(렌더러 자체가
        없음)."""
        if not self._render:
            raise RuntimeError("render_frames()는 render=True로 만든 환경에서만 쓸 수 있다")
        return {cam: self._render_camera(cam) for cam in JUDGE_CAMERAS}

    def safe_stop(self) -> None:
        self._stopped = True

    def run_recovery(self, recovery_id: str) -> None:
        """스크립트 복구(학습 없음, controller_step 재사용) -- R1(같은
        서브태스크 재실행)/R2(시작 자세 복귀)/R3(설정 변경+재파지)/
        R4(안전 정지)."""
        if recovery_id == "R1":
            # 물리 상태는 전혀 안 건드린다(현재 자세 그대로) -- 하지만
            # _step_in_subtask/_hold_count는 반드시 리셋해야 한다. 실측으로
            # 발견한 버그(PIPELINE.md "실패 유형 재정의" 절 참고): 이걸 안
            # 하면 subtask_finished()가 "아직도 max_steps를 넘었다"를 다음
            # 스텝에 바로 다시 보고 즉시 또 timeout으로 끝나버려서,
            # "그 자리에서 재시도"가 사실상 전혀 재시도가 안 됐다(실측
            # 복구 성공률 0%였던 원인). ARD-VLA의 실제 DecisionLogic이
            # 자기 쪽에서 별도 스텝 예산을 관리하는 경우는 이 리셋이 있어도
            # 안전하다(이 어댑터의 max_steps 자체가 그 예산보다 먼저 걸릴
            # 일이 없다면 영향 없음).
            self._step_in_subtask = 0
            self._hold_count = 0
            self._history = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": []}
            return
        if recovery_id == "R2":
            qpos, qvel, virtual_qpos = self._checkpoint
            self.sim.data.qpos[:] = qpos
            self.sim.data.qvel[:] = qvel
            self.sim._virtual_qpos = dict(virtual_qpos)
            # 물리 상태(qpos/qvel)만 되돌리면 grasp_slip이 model.eq_data에
            # 만들어둔 relpose 이동(모델 레벨 변화, data 복원으로는 안
            # 지워짐)이 남아있어서 "시작 자세로 복귀"가 불완전해진다 --
            # regrasp()도 같이 불러서 그립도 그 시점 기준으로 되돌린다.
            if self.sim._right_slip is not None:
                self.sim._right_slip.regrasp(self.sim.model, self.sim.data)
            if self.sim._left_slip is not None:
                self.sim._left_slip.regrasp(self.sim.model, self.sim.data)
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self._step_in_subtask = 0
            self._hold_count = 0
            self._history = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": []}
            return
        if recovery_id == "R3":
            self.gains = {k: v * self._r3_kp_boost for k, v in self.gains.items()}
            # regrasp()(reset_episode()가 아니라)를 쓴다 -- sim/grasp_slip.py
            # GraspSlipTracker.regrasp() docstring 참고: reset_episode()는
            # "이미 발생함" 플래그까지 풀어서, injected_at_step을 이미
            # 넘긴 상태에서는 다음 스텝에 같은 교란이 또 발동해 재시도마다
            # 똑같이 실패하는 무한 루프가 됐다(실측으로 발견).
            if self.sim._right_slip is not None:
                self.sim._right_slip.regrasp(self.sim.model, self.sim.data)
            if self.sim._left_slip is not None:
                self.sim._left_slip.regrasp(self.sim.model, self.sim.data)
            self._step_in_subtask = 0
            self._hold_count = 0
            self._history = {"dx": [], "dy": [], "raw_depth": [], "wall_contact": []}
            return
        if recovery_id == "R4":
            self.safe_stop()
            return
        raise ValueError(f"unknown recovery_id {recovery_id!r}")

    # ------------------------------------------------------------------
    def _render_camera(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.sim.data, camera=camera)
        return self._renderer.render().copy()

    def current_dx_dy(self) -> tuple[float, float]:
        peg_tip = self.sim.get_peg_tip_pos()
        hole_center = self.sim.get_hole_center_pos()
        return float(hole_center[0] - peg_tip[0]), float(hole_center[1] - peg_tip[1])

    def current_depth(self) -> float:
        dx, dy = self.current_dx_dy()
        xy_ok = abs(dx) < self._outer_half and abs(dy) < self._outer_half
        hole_center = self.sim.get_hole_center_pos()
        peg_tip = self.sim.get_peg_tip_pos()
        raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
        return raw_depth if xy_ok else 0.0

    def current_raw_depth_unclipped(self) -> float:
        """`current_depth()`와 같은 깊이값이지만 xy 정렬 게이팅이 없다 --
        성공 판정(`current_depth()`)에는 "정렬 안 됐으면 깊이 0"이 맞지만,
        Judge 진행도 라벨에는 틀리다(정렬이 살짝 벗어난 채 깊게 꽂힌 jam
        케이스가 진행도 0으로 보여서는 안 됨 -- v2 작업 "진행도 라벨
        수정" 근거)."""
        hole_center = self.sim.get_hole_center_pos()
        peg_tip = self.sim.get_peg_tip_pos()
        return min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))


class ScriptedOraclePolicy:
    """ARD-VLA 러너가 그대로 돌릴 수 있는 `predict_action_chunk_with_
    decisions(batch, selector_mask=None) -> (actions, decisions)` +
    `cancel_pending_actions()`를 가진 오라클 -- 학습된 모델이 아니라
    controller_step()과 물리값으로 직접 계산한다(모듈 docstring의
    "오라클의 액션 청크는 항상 길이 1이다" / "판정은 방금 끝난
    서브태스크를 본다" 참고). `selector_mask`는 받긴 하지만 안 쓴다
    (이 오라클은 Selector 없이 rule table만으로 동작하는 걸 전제로
    설계됨, decision_logic.DecisionLogic과 같은 전제)."""

    def __init__(self, env: PegInHoleClosedLoopEnv):
        self.env = env

    def predict_action_chunk_with_decisions(
        self, batch: dict[str, Any], selector_mask: Any = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        env = self.env
        delta, env._prev_dx, env._prev_dy = controller_step(
            env.sim, env.gains, env._prev_dx, env._prev_dy, env._outer_half, env._target_depth
        )
        actions = np.asarray(delta, dtype=np.float32).reshape(1, 1, -1)

        dx, dy = env.current_dx_dy()
        if env.current_node == "align":
            progress = align_progress(dx, dy, env._initial_xy_err)
        else:
            progress = insert_progress(env.current_depth(), env._target_depth)

        success_prob = 1.0 if env._subtask_done_naturally else 0.0

        right_state, left_state = env.sim.get_grasp_states()
        right_active = True if right_state is None else right_state.weld_active
        right_shift = 0.0 if right_state is None else float(np.linalg.norm(right_state.cumulative_shift_pos))
        left_active = True if left_state is None else left_state.weld_active
        left_shift = 0.0 if left_state is None else float(np.linalg.norm(left_state.cumulative_shift_pos))

        if env._legacy_failure_rule:
            label = classify_failure(
                success=env._subtask_done_naturally,
                right_weld_active=right_active,
                right_cumulative_shift_m=right_shift,
                left_weld_active=left_active,
                left_cumulative_shift_m=left_shift,
                peg_touching_hole_wall=env.sim.is_peg_touching_hole_wall(),
                raw_depth=env.current_depth(),
                target_depth=env._target_depth,
                timed_out=env._subtask_timed_out,
            )
            active_types = FAILURE_TYPES_LEGACY
        else:
            label = classify_failure_by_subtask(
                node=env.current_node,
                history=env._history,
                success=env._subtask_done_naturally,
                right_weld_active=right_active,
                right_cumulative_shift_m=right_shift,
                left_weld_active=left_active,
                left_cumulative_shift_m=left_shift,
                outer_half=env._outer_half,
                target_depth=env._target_depth,
            )
            active_types = FAILURE_TYPES
        failure_probs = [0.0] * len(active_types)
        if label in active_types:
            failure_probs[active_types.index(label)] = 1.0

        decisions = {
            "success_prob": success_prob,
            "failure_probs": failure_probs,
            "progress": progress,
            "failure_type_label": label,  # 디버그/로그용, decision_logic은 이 키를 안 읽음
        }
        return actions, decisions

    def cancel_pending_actions(self) -> None:
        pass  # 청크 크기가 항상 1이라 쌓인 액션이 없다(모듈 docstring 참고).
