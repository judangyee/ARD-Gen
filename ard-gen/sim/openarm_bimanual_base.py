"""OpenArm 양팔(Actuator=오른팔/Stabilizer=왼팔) 태스크들의 공통 기반 클래스.

sim/peg_in_hole_bimanual_openarm_sim.py(BimanualPegInHoleOpenArmSim)에서
리팩토링으로 추출했다 -- sim/tacker_openarm_env.py(BimanualTackerOpenArmSim)
에도 거의 동일한 코드가 복붙돼 있던 걸 보면(_ARM_JOINTS/_LEFT_ARM_HOME_QPOS/
vendor 액추에이터 무력화/gravcomp 주입/_virtual_*_tip_and_jac/_advance_virtual/
역동역학 step() 전부 구조가 같다), 이 메커니즘 자체가 태스크 무관의 "OpenArm
양팔로 뭔가를 쥐고 상대 물체에 접근" 공통 패턴이라는 뜻이다. 이 파일은 그
공통 부분만 뽑은 것이고, tacker는 이번 리팩토링 범위가 아니라 건드리지
않았다(사용자가 "기존 환경 리팩터링"으로 지정한 건 peg_in_hole뿐).

## 뽑은 것 vs 남긴 것

**뽑은 것** (두 구현에서 바이트 단위로 똑같던 메커니즘):
- vendor <position> 액추에이터(left/right_jointN_ctrl) 무력화
- gravcomp 주입 (openarm_left_*/openarm_right_* 바디)
- 오른팔: pos_right_jN 무력화 + torque_right_jN(역동역학)로 전환
- 왼팔: pos_left_jN(강화 위치 게인)으로 고정, 그리퍼 고정
- mj_jac(point, body) 직접 호출 (peg/hole이 weld로 붙은 자유 바디라
  mj_jacSite가 안 통하는 문제, 이 파일 상단이 아니라 원본 파일들 docstring
  "2. peg/hole이..." 절 참고)
- shadow-state(가상 기구학 상태) resolved-rate IK + nullspace 투영 +
  joint-limit freeze
- 오른팔 역동역학(computed-torque) step()
- 반복 Jacobian IK로 초기 자세 푸는 것(reset() 중에만, 실제 물리 아님)

**안 뽑은 것** (태스크마다 다른 것, 서브클래스에 남김):
- reset()의 전체 오케스트레이션(마찰/clearance/scene_config 적용, hole/socket
  위치 읽기, IK 목표점 계산) -- 태스크마다 물체 geometry가 다르므로.
- force/torque 센서 이름, 성공/리워드 판정, Z 하강 속도 같은 제어 법칙의
  "의미" 부분.
- home qpos/grasp anchor offset/그리퍼 ctrl 값 자체 -- 메커니즘은 같지만
  가치는 태스크(쥐는 물체의 geometry)마다 다시 탐색해야 하는 것들이다.

## "오른팔이 쥔 포인트"의 두 오프셋이 왜 다른가

오른팔이 쥔 물체(peg 등)의 "추적점"(resolved-rate IK가 목표로 삼는 점,
예: peg tip)과 "앵커점"(물체의 free-joint qpos를 ee FK로 동기화할 때 쓰는
물체 원점 오프셋)은 같은 ee 로컬 프레임 기준이지만 다른 벡터다 -- 전자는
물체의 "끝"(hole에 꽂히는 지점), 후자는 "그리퍼가 쥔 지점"(물체 원점)이라
물체 길이만큼 떨어져 있을 수 있다(peg_in_hole_bimanual_openarm_sim.py의
_PEG_LOCAL_OFFSET vs _virtual_peg_tip_and_jac의 "+ R @ [0,0,-0.04]" 추가항
참고). 이 파일은 둘을 `right_anchor_offset_ee`(물체 FK 동기화용)와
`right_track_point_offset_ee`(resolved-rate IK 추적용)로 분리해서 받는다 --
두 값이 같은 태스크라면 그냥 같은 벡터를 두 번 넘기면 된다.
"""
from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

_RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]


class OpenArmBimanualBase:
    """OpenArm 양팔 MJCF 공통 메커니즘. xml_path는 assets/openarm/openarm_bimanual.xml
    을 <include>/<attach>하는 태스크별 모델이어야 한다(오른팔=Actuator가 뭔가를
    쥐고 왼팔=Stabilizer 쪽으로 접근, joint/actuator 이름 컨벤션은 모든 태스크가
    공유 -- openarm_{left,right}_jointN, openarm_{left,right}_finger_jointN,
    {left,right}_finger1_ctrl, pos_{left,right}_jN, torque_right_jN,
    {left,right}_jointN_ctrl(vendor), openarm_{left,right}_ee_base_link)."""

    def __init__(
        self,
        xml_path: str,
        right_home_qpos: dict[str, float],
        left_home_qpos: dict[str, float],
        right_anchor_offset_ee: np.ndarray,
        right_track_point_offset_ee: np.ndarray,
        right_held_free_joint: str,
        nullspace_gain: float = 0.03,
        limit_freeze_margin: float = 0.08,
        jac_damping: float = 1e-4,
        ik_max_iters: int = 200,
        ik_step_scale: float = 0.5,
        torque_omega_n: float = 40.0,
        torque_zeta: float = 1.0,
        n_substeps: int = 5,
    ):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.right_home_qpos = dict(right_home_qpos)
        self.left_home_qpos = dict(left_home_qpos)
        self._right_anchor_offset_ee = np.asarray(right_anchor_offset_ee, dtype=float)
        self._right_track_point_offset_ee = np.asarray(right_track_point_offset_ee, dtype=float)

        self._nullspace_gain = nullspace_gain
        self._limit_freeze_margin = limit_freeze_margin
        self._jac_damping = jac_damping
        self._ik_max_iters = ik_max_iters
        self._ik_step_scale = ik_step_scale
        self._torque_omega_n = torque_omega_n
        self._torque_zeta = torque_zeta
        self.n_substeps = n_substeps

        self._right_ee_body_id = self.model.body("openarm_right_ee_base_link").id
        self._left_ee_body_id = self.model.body("openarm_left_ee_base_link").id

        self._right_arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _RIGHT_ARM_JOINTS}
        self._right_arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _RIGHT_ARM_JOINTS}
        self._left_arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_JOINTS}

        self._right_gripper_actuator_id = self.model.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = self.model.joint("openarm_right_finger_joint1").qposadr[0]
        self._right_finger2_qposadr = self.model.joint("openarm_right_finger_joint2").qposadr[0]

        self._left_gripper_actuator_id = self.model.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = self.model.joint("openarm_left_finger_joint1").qposadr[0]
        self._left_finger2_qposadr = self.model.joint("openarm_left_finger_joint2").qposadr[0]

        self._left_arm_actuator_ids = {
            name: self.model.actuator(f"pos_left_j{i}").id for i, name in enumerate(_LEFT_ARM_JOINTS, start=1)
        }

        # vendor <position> 액추에이터 14개 무력화 (원본 파일들 docstring 참고 --
        # ctrl=0 그대로 남으면 "그 관절을 0으로 끌어당기는" 토크가 되어 home
        # 자세와 계속 싸운다).
        for prefix, joints in (("right", _RIGHT_ARM_JOINTS), ("left", _LEFT_ARM_JOINTS)):
            for i in range(1, 8):
                act_id = self.model.actuator(f"{prefix}_joint{i}_ctrl").id
                self.model.actuator_gainprm[act_id] = 0.0
                self.model.actuator_biasprm[act_id] = 0.0

        # 오른팔: pos_right_jN(관절별 독립 PD)도 무력화하고 torque_right_jN
        # (역동역학)로 구동한다.
        for i in range(1, 8):
            act_id = self.model.actuator(f"pos_right_j{i}").id
            self.model.actuator_gainprm[act_id] = 0.0
            self.model.actuator_biasprm[act_id] = 0.0
        self._torque_actuator_ids = {
            name: self.model.actuator(f"torque_right_j{i}").id for i, name in enumerate(_RIGHT_ARM_JOINTS, start=1)
        }
        self._right_arm_dofs = [self._right_arm_dofadr[name] for name in _RIGHT_ARM_JOINTS]

        # gravcomp -- vendor 파일에는 없다, 모든 팔 바디에 직접 주입한다.
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                self.model.body_gravcomp[i] = 1.0

        self._right_held_qposadr = self.model.joint(right_held_free_joint).qposadr[0]

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # planning은 실제 시뮬레이션 상태(self.data, PD 지연이 낀 상태)가 아니라
        # 별도의 순수 기구학 상태에서 한다 -- 원본 파일 docstring "gravcomp를
        # 넣어도..." 절 참고(오차가 이동 거리에 비례해 누적되는 걸 막기 위함).
        self._shadow_data = mujoco.MjData(self.model)
        self._virtual_qpos: dict[str, float] = dict(self.right_home_qpos)

    # ------------------------------------------------------------------
    # 오른팔 Jacobian / IK
    # ------------------------------------------------------------------
    def _jac_at_point(self, world_point: np.ndarray, data: mujoco.MjData | None = None) -> np.ndarray:
        """world_point가 openarm_right_ee_base_link에 강체로 붙어있다고 가정한
        3xnv 위치 Jacobian (mj_jacSite가 weld-연결 자유 바디에 안 통하는 문제 회피,
        모듈 docstring 참고)."""
        d = data if data is not None else self.data
        mujoco.mj_jac(self.model, d, self._jacp, self._jacr, world_point, self._right_ee_body_id)
        return self._jacp

    def _jac_solve(self, current_point_world: np.ndarray, delta_pos_world: np.ndarray) -> np.ndarray:
        jacp = self._jac_at_point(current_point_world)
        jjt = jacp @ jacp.T + self._jac_damping * np.eye(3)
        return jacp.T @ np.linalg.solve(jjt, delta_pos_world)

    def _sync_right_held_to_fk(self) -> None:
        """오른팔이 쥔 물체(free body)의 qpos를 지금 오른팔 ee pose로부터 FK로
        직접 계산해서 덮어쓴다(mj_forward만으로는 weld로 연결된 자유 바디가 안
        움직이므로). reset()의 반복 IK 중에만 쓴다(실제 접촉 물리가 시작되는
        step()에서는 절대 안 씀)."""
        ee_pos = self.data.xpos[self._right_ee_body_id]
        ee_quat = self.data.xquat[self._right_ee_body_id]
        R = self.data.xmat[self._right_ee_body_id].reshape(3, 3)
        held_pos = ee_pos + R @ self._right_anchor_offset_ee
        qadr = self._right_held_qposadr
        self.data.qpos[qadr : qadr + 3] = held_pos
        self.data.qpos[qadr + 3 : qadr + 7] = ee_quat

    def solve_right_initial_pose(
        self, target_pos_world: np.ndarray, get_current_point: Callable[[], np.ndarray]
    ) -> None:
        """오른팔을 home 자세에서 시작해 반복 Jacobian IK로 target_pos_world에
        도달시킨다(실제 물리 아님, reset()에서만 쓴다). get_current_point()는
        "지금 추적 중인 점"의 실제 월드 좌표를 읽는 콜백(태스크마다 site 이름이
        다르므로 서브클래스가 넘긴다, 예: site_xpos[peg_tip_site_id])."""
        for name, value in self.right_home_qpos.items():
            self.data.qpos[self._right_arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        self._sync_right_held_to_fk()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self._ik_max_iters):
            current = get_current_point()
            err = target_pos_world - current
            if np.linalg.norm(err) < 1e-5:
                break
            dq = self._jac_solve(current, err * self._ik_step_scale)
            for name in _RIGHT_ARM_JOINTS:
                dof = self._right_arm_dofadr[name]
                qadr = self._right_arm_qposadr[name]
                lo, hi = self.model.jnt_range[self.model.joint(name).id]
                self.data.qpos[qadr] = np.clip(self.data.qpos[qadr] + dq[dof], lo, hi)
            mujoco.mj_forward(self.model, self.data)
            self._sync_right_held_to_fk()
            mujoco.mj_forward(self.model, self.data)
        # torque_right_jN에 ctrl을 맞출 필요 없음 -- step()이 매 스텝 새로 계산한다.

    def capture_virtual_qpos_from_data(self) -> None:
        """reset()이 초기 자세를 다 잡은 뒤, step()의 resolved-rate 계획이 이
        시점부터 실제 상태가 아니라 이 값에서 이어가도록 가상 상태를 동기화한다."""
        self._virtual_qpos = {
            name: float(self.data.qpos[self._right_arm_qposadr[name]]) for name in _RIGHT_ARM_JOINTS
        }

    # ------------------------------------------------------------------
    # 왼팔(Stabilizer) 고정 홀드
    # ------------------------------------------------------------------
    def reset_left_arm_fixed_hold(
        self,
        left_gripper_ctrl: float,
        left_held_free_joint: str | None = None,
        left_held_home_qpos: np.ndarray | None = None,
    ) -> None:
        """왼팔을 이 태스크 내내 고정될 home 자세로 두고(강화 위치 게인
        pos_left_jN이 그 자세를 유지), 그리퍼를 닫는다. left_held_free_joint가
        주어지면 그 free body(왼팔이 쥔 물체)의 qpos(7)도 left_held_home_qpos로
        덮어쓴다(왼팔이 안 움직이므로 매 reset마다 고정값을 그대로 복사)."""
        for name, value in self.left_home_qpos.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
            self.data.ctrl[self._left_arm_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = left_gripper_ctrl
        self.data.qpos[self._left_finger2_qposadr] = left_gripper_ctrl
        self.data.ctrl[self._left_gripper_actuator_id] = left_gripper_ctrl
        if left_held_free_joint is not None:
            qadr = self.model.joint(left_held_free_joint).qposadr[0]
            self.data.qpos[qadr : qadr + 7] = left_held_home_qpos

    def get_left_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._left_ee_body_id].copy()

    # ------------------------------------------------------------------
    # 오른팔 그리퍼
    # ------------------------------------------------------------------
    def set_right_gripper(self, ctrl: float) -> None:
        self.data.qpos[self._right_finger_qposadr] = ctrl
        self.data.qpos[self._right_finger2_qposadr] = ctrl
        self.data.ctrl[self._right_gripper_actuator_id] = ctrl

    # ------------------------------------------------------------------
    # 가상 상태(shadow) resolved-rate IK + 역동역학 step
    # ------------------------------------------------------------------
    def _virtual_right_track_point_and_jac(self) -> tuple[np.ndarray, np.ndarray]:
        """self._virtual_qpos(순수 기구학 상태)에서의 추적점 위치와, 거기서
        openarm_right_ee_base_link에 강체로 붙어있다고 가정한 Jacobian.
        self._shadow_data에만 쓰고 self.data(진짜 시뮬레이션 상태)는 절대
        건드리지 않는다."""
        sd = self._shadow_data
        for name, value in self._virtual_qpos.items():
            sd.qpos[self._right_arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        ee_pos = sd.xpos[self._right_ee_body_id]
        R = sd.xmat[self._right_ee_body_id].reshape(3, 3)
        point = ee_pos + R @ self._right_track_point_offset_ee
        mujoco.mj_jac(self.model, sd, self._jacp, self._jacr, point, self._right_ee_body_id)
        return point, self._jacp

    def _advance_virtual(self, delta_pos_world: np.ndarray) -> None:
        """가상 상태를 delta_pos_world만큼 resolved-rate로 전진시킨다. 7-DOF라
        3개(xyz) 목표를 만족하고도 널스페이스가 4차원 남는데, 감쇠 없이 두면
        관절이 표류하므로 home 자세로 되돌아가려는 2차 목표를 널스페이스에
        투영해서 더한다. 관절 한계 근처에서는(사후 clip 대신) 그 방향으로 더
        미는 열을 Jacobian에서 아예 0으로 만들고 다시 풀어서, 사후 조정이
        만드는 xy 누출을 구조적으로 없앤다(원본 파일 docstring 상세 참고)."""
        _, jacp = self._virtual_right_track_point_and_jac()

        def _solve(J: np.ndarray) -> np.ndarray:
            jjt = J @ J.T + self._jac_damping * np.eye(3)
            return J.T @ np.linalg.inv(jjt)

        jacp_pinv0 = _solve(jacp)
        dq_task0 = jacp_pinv0 @ delta_pos_world
        jacp_frozen = jacp.copy()
        frozen_dofs = []
        for name in _RIGHT_ARM_JOINTS:
            dof = self._right_arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            frac = (self._virtual_qpos[name] - lo) / (hi - lo)
            if (frac < self._limit_freeze_margin and dq_task0[dof] < 0.0) or (
                frac > 1.0 - self._limit_freeze_margin and dq_task0[dof] > 0.0
            ):
                jacp_frozen[:, dof] = 0.0
                frozen_dofs.append(dof)

        jacp_pinv = _solve(jacp_frozen)
        dq_task = jacp_pinv @ delta_pos_world

        nv = self.model.nv
        null_proj = np.eye(nv) - jacp_pinv @ jacp_frozen
        dq_null = np.zeros(nv)
        for name in _RIGHT_ARM_JOINTS:
            dof = self._right_arm_dofadr[name]
            dq_null[dof] = self._nullspace_gain * (self.right_home_qpos[name] - self._virtual_qpos[name])
        dq = dq_task + null_proj @ dq_null
        for dof in frozen_dofs:
            dq[dof] = 0.0

        for name in _RIGHT_ARM_JOINTS:
            dof = self._right_arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            self._virtual_qpos[name] = float(np.clip(self._virtual_qpos[name] + dq[dof], lo, hi))

    def step(self, delta_pos_world: np.ndarray) -> None:
        """가상 상태를 전진시키고, 그 결과(q_des)로 오른팔에 역동역학
        (computed-torque) 토크를 넣어 n_substeps만큼 물리를 진행한다.

            qacc_cmd_i = omega_n^2*(q_des_i - q_i) - 2*zeta*omega_n*qvel_i
            tau = M(q)_rr @ qacc_cmd + (qfrc_bias_r - qfrc_passive_r)

        qfrc_bias - qfrc_passive는 Coriolis/원심력 항만 남긴다(중력은 이미
        body_gravcomp가 qfrc_passive로 상쇄하므로, qfrc_bias를 그대로 더하면
        중력을 두 번 상쇄하게 된다 -- 원본 파일 docstring 참고)."""
        self._advance_virtual(delta_pos_world)
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        M_rr = M[np.ix_(self._right_arm_dofs, self._right_arm_dofs)]
        qacc_cmd = np.empty(len(_RIGHT_ARM_JOINTS))
        bias = np.empty(len(_RIGHT_ARM_JOINTS))
        for i, name in enumerate(_RIGHT_ARM_JOINTS):
            dof = self._right_arm_dofadr[name]
            qpos = self.data.qpos[self._right_arm_qposadr[name]]
            qvel = self.data.qvel[dof]
            q_des = self._virtual_qpos[name]
            qacc_cmd[i] = self._torque_omega_n**2 * (q_des - qpos) - 2.0 * self._torque_zeta * self._torque_omega_n * qvel
            bias[i] = self.data.qfrc_bias[dof] - self.data.qfrc_passive[dof]
        tau = M_rr @ qacc_cmd + bias
        for i, name in enumerate(_RIGHT_ARM_JOINTS):
            self.data.ctrl[self._torque_actuator_ids[name]] = tau[i]
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # ------------------------------------------------------------------
    # 관절 observation (Q1 결정: 액션 스페이스는 delta-position 유지, "16 DoF"는
    # observation으로만 노출 -- 7+1 per arm)
    # ------------------------------------------------------------------
    def right_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self._right_arm_qposadr[n]] for n in _RIGHT_ARM_JOINTS])

    def left_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self._left_arm_qposadr[n]] for n in _LEFT_ARM_JOINTS])

    def right_gripper_qpos(self) -> float:
        return float(self.data.qpos[self._right_finger_qposadr])

    def left_gripper_qpos(self) -> float:
        return float(self.data.qpos[self._left_finger_qposadr])

    def joint_observation(self) -> np.ndarray:
        """16-DoF 관절 observation: [오른팔 7, 오른팔 그리퍼 1, 왼팔 7, 왼팔
        그리퍼 1]. 액션 스페이스가 아니라 observation 전용(이 프로젝트의 CMA-ES
        게인 탐색은 delta-position 액션 + 내부 resolved-rate IK에 의존하므로,
        액션 자체를 절대 관절각으로 바꾸지 않는다)."""
        return np.concatenate(
            [self.right_arm_qpos(), [self.right_gripper_qpos()], self.left_arm_qpos(), [self.left_gripper_qpos()]]
        ).astype(np.float32)
