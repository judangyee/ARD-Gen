"""pin_assembly 태스크의 저수준 MuJoCo 래퍼. assets/pin_assembly_openarm.xml
+ 왼팔(Actuator, 실제로 들고 끼운다)/오른팔(정지, 공식 RoCo 휴지 자세)
위치-전용 제어. 모듈 상단 docstring은 assets/pin_assembly_openarm.xml에
이미 자세히 적어뒀다(home qpos 탐색 과정, 운동학적 carry 설계 이유 등)
-- 여기서는 그 결정들을 코드로 구현만 한다.

## 왼팔 IK: 위치-전용 resolved-rate + 널스페이스(tacker_openarm_env.py와
## 동일한 패턴, 재사용)

tacker_openarm_env.py의 `_advance_virtual()`(댐핑 최소제곱 pinv + 관절
한계 근처 freeze + 널스페이스로 LEFT_HOME_QPOS 쪽으로 당김)을 그대로
왼팔에 적용했다 -- 이 태스크는 접촉 힘 피드백이 전혀 필요 없는 자유공간
pick-place라서, peg_in_hole/tacker가 쓰는 "매 스텝 shadow IK 재계산 +
역동역학 torque" 전체를 가져올 필요가 없다. 대신 **웨이포인트 하나당
키네마틱 IK를 한 번(수렴할 때까지) 풀어서 목표 관절각을 구하고, 그
값을 그대로 pos_left_j* ctrl로 "점프"시킨 뒤 물리(mj_step, 포지션
액추에이터 PD)가 그 쪽으로 수렴하게 둔다** -- 이 방식이 사용자 지시
3-2("처음 10프레임은 유지, 11프레임째 큰 점프, 이후 감속 수렴")가
요구하는 모양과 정확히 같다(포지션 PD의 스텝 응답 자체가 바로 그
모양이다).

## 운동학적 들기(carry): weld 대신

들고 있는 동안(carry 상태) 매 서브스텝 해당 물체의 freejoint qpos를
"그립 시작 시점에 FK로 한 번 계산한 손 기준 로컬 오프셋"으로 덮어쓰고
qvel을 0으로 고정한다 -- XML에 weld anchor를 미리 맞춰둘 필요가 없고
(assets/pin_assembly_openarm.xml 모듈 docstring 참고), 여러 물체를
순차로 들고 놓아야 하는 이 태스크에 자연스럽게 맞는다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "pin_assembly_openarm.xml")

N_SUBSTEPS = 5
DT = N_SUBSTEPS * 0.002  # 0.01s = 10Hz, LeRobot 캡처 레이트와 1:1 (모듈 docstring 참고)

_LEFT_ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
_RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# 왼팔: 실측으로 2차 보정한 값. 1차 시도(목표점 (0.22,0.05,0.50), 로컬 -z=
# world -z 정확히 일치)는 수치적으로는 맞았지만 *그 qpos에서 왼팔 손목
# 링크(link5/link6)가 carrier_plate와 9~34mm 침투*하는 걸 나중에
# 실측으로 발견했다(원인: 캐리어 표면 자체가 "손끝이 닿아야 하는" 높이
# (0.50)에 있는데, 그 높이에서 "정확히 수직"으로 손끝을 대려면 손목이
# 그 표면 바로 옆/안쪽 공간을 지나가야 하는 기구학이었음 -- 캐리어
# 반지름을 아무리 줄여도 손목이 지나가는 자리 자체가 바뀌지 않아서
# 안 고쳐짐, 실측 확인). 침투한 접촉 제약력이 수백 N급으로 튀어
# 포지션 서보를 완전히 무력화시켰다(hold 테스트: ctrl=qpos인데도 20
# 프레임 만에 0.4rad+ 드리프트). 해결: 작업 공간 전체를 8cm 위로 올려서
# (플랫폼 표면 z=0.50->0.60, assets/pin_assembly_openarm.xml 참고)
# 손끝 목표를 (0.22,0.05,0.68)(표면에서 8cm 위, 호버 높이)로 바꾸고
# 다시 탐색 -- 이번엔 정확히 수직은 아니지만(dot=0.874, ry 쪽으로
# 약간 기욺, 공식 데이터 자체도 ry가 -1.26~1.53까지 가므로 "항상
# 정확히 수직"일 필요가 없다는 걸 뒷받침) 손목이 플랫폼과 안 겹치고
# 관절 한계 여유도 있는 해를 찾았다(여유 최소 4.9%).
LEFT_HOME_QPOS: dict[str, float] = {
    "openarm_left_joint1": -2.1712541575692903,
    "openarm_left_joint2": -0.8169997524825003,
    "openarm_left_joint3": 1.3536569377199628,
    "openarm_left_joint4": 2.089582417940605,
    "openarm_left_joint5": -0.5559873737854346,
    "openarm_left_joint6": -0.618838789878911,
    "openarm_left_joint7": -1.4153272663353196,
}

# 오른팔: peg_in_hole_bimanual_openarm.xml의 오른팔 home을 그대로 재사용
# (이미 로컬 -z=world -z 실측 확인됨, dot=0.999999 -- 이 태스크에서 오른팔은
# 장면과 상호작용하지 않으므로 재탐색할 이유가 없다, 이 파일 상단 docstring
# 참고). 에피소드 내내 고정(reset()에서 ctrl을 한 번 설정하고 다시 안 건드림).
RIGHT_HOME_QPOS: dict[str, float] = {
    "openarm_right_joint1": -0.4932859043730308,
    "openarm_right_joint2": 2.7728121142883237,
    "openarm_right_joint3": 0.7779694449047385,
    "openarm_right_joint4": 2.068501962471789,
    "openarm_right_joint5": 0.5568966487876724,
    "openarm_right_joint6": -0.5030469209189363,
    "openarm_right_joint7": 0.973709272141154,
}

LEFT_GRIPPER_OPEN_CTRL = 0.55
LEFT_GRIPPER_GRASP_CTRL = 0.110  # peg_in_hole의 2cm 핀치와 동일(핸들이 같은 단면, 이 파일 docstring 참고)
# 오른팔 그리퍼: 고정값이지만 GripperMapping(기본 target_range 0.06~0.30)으로
# 변환했을 때 공식 데이터의 오른팔 그리퍼 고정값(약 0.2256)에 가깝도록 역산한
# raw 값 -- 오른팔 joint range[-0.7854,0](open=0,closed=-0.7854), ratio=
# (0.2256-0.06)/(0.30-0.06)=0.69333, raw=closed+ratio*(open-closed)
# = -0.7854 + 0.69333*0.7854 = -0.2408.
RIGHT_GRIPPER_REST_CTRL = -0.2408

LEFT_FINGERTIP_LOCAL_OFFSET = np.array([-0.01772, -0.00214, -0.19274])  # peg_in_hole hole_grasp relpose 위치부 재사용(근사 기준점, 이 파일 docstring 참고)

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 300
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.03
_LIMIT_FREEZE_MARGIN = 0.05
_IK_POS_TOL = 1e-4

DISK_BODIES = ("planet_1", "planet_2", "planet_3", "sun", "ring")


class PinAssemblyOpenArmSim:
    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self._left_qposadr = {n: m.joint(n).qposadr[0] for n in _LEFT_ARM_JOINTS}
        self._left_dofadr = {n: m.joint(n).dofadr[0] for n in _LEFT_ARM_JOINTS}
        self._left_ranges = {n: tuple(m.jnt_range[m.joint(n).id]) for n in _LEFT_ARM_JOINTS}
        self._left_actuator_ids = {n: m.actuator(f"pos_left_j{i}").id for i, n in enumerate(_LEFT_ARM_JOINTS, start=1)}

        self._right_qposadr = {n: m.joint(n).qposadr[0] for n in _RIGHT_ARM_JOINTS}
        self._right_dofadr = {n: m.joint(n).dofadr[0] for n in _RIGHT_ARM_JOINTS}
        self._right_actuator_ids = {n: m.actuator(f"pos_right_j{i}").id for i, n in enumerate(_RIGHT_ARM_JOINTS, start=1)}

        self._left_ee_body_id = m.body("openarm_left_ee_base_link").id
        self._right_ee_body_id = m.body("openarm_right_ee_base_link").id

        self._left_gripper_actuator_id = m.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = m.joint("openarm_left_finger_joint1").qposadr[0]
        self._left_finger2_qposadr = m.joint("openarm_left_finger_joint2").qposadr[0]
        self._right_gripper_actuator_id = m.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = m.joint("openarm_right_finger_joint1").qposadr[0]
        self._right_finger2_qposadr = m.joint("openarm_right_finger_joint2").qposadr[0]

        # vendor(assets/openarm/openarm_bimanual.xml) <position> 액추에이터
        # 16개 무력화 -- peg_in_hole/tacker와 동일한 이유(모듈 상단 docstring
        # 전체에 걸쳐 재사용한다고 밝힌 패턴).
        for prefix, joints in (("left", _LEFT_ARM_JOINTS), ("right", _RIGHT_ARM_JOINTS)):
            for i in range(1, 8):
                act_id = m.actuator(f"{prefix}_joint{i}_ctrl").id
                m.actuator_gainprm[act_id] = 0.0
                m.actuator_biasprm[act_id] = 0.0

        for i in range(m.nbody):
            name = m.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                m.body_gravcomp[i] = 1.0

        self._disk_qposadr = {b: m.joint(f"{b}_free").qposadr[0] for b in DISK_BODIES}
        self._disk_dofadr = {b: m.joint(f"{b}_free").dofadr[0] for b in DISK_BODIES}
        self._disk_body_id = {b: m.body(b).id for b in DISK_BODIES}

        self._site_id = {
            name: m.site(name).id
            for name in (
                "pin_center_site",
                "pin_peripheral_1_site",
                "pin_peripheral_2_site",
                "pin_peripheral_3_site",
                "stage_slot_0_site",
                "stage_slot_1_site",
                "stage_slot_2_site",
                "stage_slot_3_site",
                "stage_slot_4_site",
            )
        }

        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._shadow_data = mujoco.MjData(m)

        self._virtual_qpos: dict[str, float] = dict(LEFT_HOME_QPOS)
        self._carrying: str | None = None
        self._carry_offset_pos = np.zeros(3)
        self._carry_offset_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 생성 직후에도 site_world_pos() 등이 유효한 값을 돌려주도록 한 번
        # forward를 돌려둔다 -- 안 그러면 MjData는 xpos/site_xpos가 전부
        # 0으로 남아있어서(reset() 호출 전) carrier/staging 사이트 위치를
        # 미리 읽어 disk_poses를 구성하려는 호출자가 전부 원점(펜데스탈
        # 바로 옆)을 받게 되는 버그가 있었다(실측으로 발견 -- 모든 디스크가
        # 겹쳐서 openarm_body_link0_collision과 1~3.5cm 침투하는 증상으로
        # 나타남). carrier_plate/staging_platform은 조인트가 없는 고정
        # 바디라 qpos와 무관하게 항상 같은 위치이므로, 이 forward는 디스크
        # 자유조인트 qpos(아직 reset() 전이라 의미 없는 기본값)에는 영향을
        # 안 받는다.
        mujoco.mj_forward(self.model, self.data)

    # -- 사이트/조회 ---------------------------------------------------------
    def site_world_pos(self, site_name: str) -> np.ndarray:
        return self.data.site_xpos[self._site_id[site_name]].copy()

    def disk_world_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._disk_body_id[body]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    def left_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self._left_ee_body_id].copy(), self.data.xquat[self._left_ee_body_id].copy()

    def right_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self._right_ee_body_id].copy(), self.data.xquat[self._right_ee_body_id].copy()

    def left_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self._left_qposadr[n]] for n in _LEFT_ARM_JOINTS])

    def left_arm_qvel(self) -> np.ndarray:
        return np.array([self.data.qvel[self._left_dofadr[n]] for n in _LEFT_ARM_JOINTS])

    def right_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self._right_qposadr[n]] for n in _RIGHT_ARM_JOINTS])

    def right_arm_qvel(self) -> np.ndarray:
        return np.zeros(7)  # 오른팔은 이 태스크 내내 고정(절대 안 움직임이 설계 의도)

    def left_gripper_raw(self) -> float:
        return float(self.data.qpos[self._left_finger_qposadr])

    def right_gripper_raw(self) -> float:
        return float(self.data.qpos[self._right_finger_qposadr])

    # -- IK: 위치-전용 resolved-rate + 널스페이스(tacker_openarm_env.py와 동일 패턴) --
    def _fingertip_world(self, qpos: dict[str, float]) -> np.ndarray:
        sd = self._shadow_data
        for name, value in qpos.items():
            sd.qpos[self._left_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        ee_pos = sd.xpos[self._left_ee_body_id]
        R = sd.xmat[self._left_ee_body_id].reshape(3, 3)
        return ee_pos + R @ LEFT_FINGERTIP_LOCAL_OFFSET

    def _fingertip_jac(self, qpos: dict[str, float]) -> np.ndarray:
        sd = self._shadow_data
        for name, value in qpos.items():
            sd.qpos[self._left_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        tip = self._fingertip_world(qpos)
        mujoco.mj_jac(self.model, sd, self._jacp, self._jacr, tip, self._left_ee_body_id)
        return self._jacp

    def solve_ik_left(
        self, target_pos: np.ndarray, start_qpos: dict[str, float] | None = None, max_iters: int = _IK_MAX_ITERS
    ) -> tuple[dict[str, float], bool, float]:
        """fingertip(LEFT_FINGERTIP_LOCAL_OFFSET)이 target_pos(world)에
        도달하도록 왼팔 7관절을 푼다 -- 널스페이스로 LEFT_HOME_QPOS 쪽을
        당기고, 관절 한계 5% 이내에서는 그 방향 자유도를 freeze한다
        (tacker_openarm_env.py `_advance_virtual()`과 동일 구조, 재사용).
        반환: (qpos dict, 수렴 여부, 최종 위치오차 norm) -- 사용자 지시
        3의 "IK 도달성 사전 검사"가 이 반환값을 그대로 쓴다."""
        qpos = dict(start_qpos) if start_qpos is not None else dict(LEFT_HOME_QPOS)
        err_norm = float("inf")
        for _ in range(max_iters):
            tip = self._fingertip_world(qpos)
            err = target_pos - tip
            err_norm = float(np.linalg.norm(err))
            if err_norm < _IK_POS_TOL:
                return qpos, True, err_norm

            jacp = self._fingertip_jac(qpos)

            def _solve(J: np.ndarray) -> np.ndarray:
                jjt = J @ J.T + _JAC_DAMPING * np.eye(3)
                return J.T @ np.linalg.inv(jjt)

            dofs = [self._left_dofadr[n] for n in _LEFT_ARM_JOINTS]
            jacp_arm = jacp[:, dofs]
            jacp_pinv0 = _solve(jacp_arm)
            dq_task0 = jacp_pinv0 @ (err * _IK_STEP_SCALE)

            jacp_frozen = jacp_arm.copy()
            frozen = []
            for i, name in enumerate(_LEFT_ARM_JOINTS):
                lo, hi = self._left_ranges[name]
                frac = (qpos[name] - lo) / (hi - lo)
                if (frac < _LIMIT_FREEZE_MARGIN and dq_task0[i] < 0.0) or (
                    frac > 1.0 - _LIMIT_FREEZE_MARGIN and dq_task0[i] > 0.0
                ):
                    jacp_frozen[:, i] = 0.0
                    frozen.append(i)

            jacp_pinv = _solve(jacp_frozen)
            dq_task = jacp_pinv @ (err * _IK_STEP_SCALE)

            null_proj = np.eye(7) - jacp_pinv @ jacp_frozen
            dq_null = np.array([_NULLSPACE_GAIN * (LEFT_HOME_QPOS[n] - qpos[n]) for n in _LEFT_ARM_JOINTS])
            dq = dq_task + null_proj @ dq_null
            for i in frozen:
                dq[i] = 0.0

            for i, name in enumerate(_LEFT_ARM_JOINTS):
                lo, hi = self._left_ranges[name]
                qpos[name] = float(np.clip(qpos[name] + dq[i], lo, hi))

        return qpos, False, err_norm

    # -- reset ---------------------------------------------------------------
    def reset(self, disk_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """disk_poses: {body_name: (pos(3,), quat_wxyz(4,))} -- DISK_BODIES
        전부에 대해 있어야 한다(시나리오별 초기 상태, sim/pin_assembly_openarm_env.py
        참고)."""
        mujoco.mj_resetData(self.model, self.data)

        for name, value in RIGHT_HOME_QPOS.items():
            self.data.qpos[self._right_qposadr[name]] = value
            self.data.ctrl[self._right_actuator_ids[name]] = value
        self.data.qpos[self._right_finger_qposadr] = RIGHT_GRIPPER_REST_CTRL
        self.data.qpos[self._right_finger2_qposadr] = RIGHT_GRIPPER_REST_CTRL
        self.data.ctrl[self._right_gripper_actuator_id] = RIGHT_GRIPPER_REST_CTRL

        for name, value in LEFT_HOME_QPOS.items():
            self.data.qpos[self._left_qposadr[name]] = value
            self.data.ctrl[self._left_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = LEFT_GRIPPER_OPEN_CTRL
        self.data.qpos[self._left_finger2_qposadr] = LEFT_GRIPPER_OPEN_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = LEFT_GRIPPER_OPEN_CTRL

        for body, (pos, quat) in disk_poses.items():
            qadr = self._disk_qposadr[body]
            self.data.qpos[qadr : qadr + 3] = pos
            self.data.qpos[qadr + 3 : qadr + 7] = quat
            dadr = self._disk_dofadr[body]
            self.data.qvel[dadr : dadr + 6] = 0.0

        self._virtual_qpos = dict(LEFT_HOME_QPOS)
        self._carrying = None
        mujoco.mj_forward(self.model, self.data)

    # -- 그립(운동학적 들기) ---------------------------------------------------
    def start_carry(self, body: str) -> None:
        """현재 실제 FK(왼손 ee pose vs 물체 pose)로 상대 오프셋을 한 번
        계산해서 기록한다 -- 이후 매 스텝 이 오프셋을 그대로 유지하며
        따라간다(모듈 상단 docstring "운동학적 들기" 참고, anchor를 XML에
        미리 맞출 필요가 없다)."""
        ee_pos, ee_quat = self.left_ee_pose()
        obj_pos, obj_quat = self.disk_world_pose(body)
        ee_mat = np.zeros(9)
        mujoco.mju_quat2Mat(ee_mat, ee_quat)
        ee_mat = ee_mat.reshape(3, 3)
        self._carry_offset_pos = ee_mat.T @ (obj_pos - ee_pos)
        ee_quat_neg = np.array([ee_quat[0], -ee_quat[1], -ee_quat[2], -ee_quat[3]])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, ee_quat_neg, obj_quat)
        self._carry_offset_quat = rel
        self._carrying = body

    def stop_carry(self) -> None:
        self._carrying = None

    def _apply_carry(self) -> None:
        if self._carrying is None:
            return
        ee_pos, ee_quat = self.left_ee_pose()
        ee_mat = np.zeros(9)
        mujoco.mju_quat2Mat(ee_mat, ee_quat)
        ee_mat = ee_mat.reshape(3, 3)
        new_pos = ee_pos + ee_mat @ self._carry_offset_pos
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, ee_quat, self._carry_offset_quat)
        qadr = self._disk_qposadr[self._carrying]
        self.data.qpos[qadr : qadr + 3] = new_pos
        self.data.qpos[qadr + 3 : qadr + 7] = new_quat
        dadr = self._disk_dofadr[self._carrying]
        self.data.qvel[dadr : dadr + 6] = 0.0

    # -- 스텝 ------------------------------------------------------------------
    def hold_left(self) -> None:
        """현재 가상 목표(`_virtual_qpos`)에 왼팔을 운동학적으로 고정한다
        (advance_toward()와 같은 "qpos/qvel 직접 덮어쓰기" 원리, 이동
        없이 그대로 유지만). 그립 여닫기처럼 팔이 안 움직여야 하는
        구간에서 쓴다 -- 실측으로 발견: 이 구간을 일반 mj_step(포지션
        서보 동역학)에만 맡기면, LEFT_HOME_QPOS 자체가 중력을 "정확히"
        상쇄하는 평형점이 아니어서 처음 10프레임 동안 EE가 최대 30mm
        가까이 드리프트했다(validate_against_roco.py의 "초반 유지" 검사
        0/75 실패로 발견). advance_toward()를 쓰는 다른 모든 구간은
        이미 이 운동학적 적용을 쓰므로 드리프트가 없었다 -- hold만
        빠져 있었다."""
        for name, value in self._virtual_qpos.items():
            self.data.qpos[self._left_qposadr[name]] = value
            self.data.qvel[self._left_dofadr[name]] = 0.0
            self.data.ctrl[self._left_actuator_ids[name]] = value
        mujoco.mj_forward(self.model, self.data)

    def get_left_virtual_qpos(self) -> dict[str, float]:
        return dict(self._virtual_qpos)

    def set_left_target_qpos(self, qpos: dict[str, float]) -> None:
        self._virtual_qpos = dict(qpos)
        for name, value in qpos.items():
            self.data.ctrl[self._left_actuator_ids[name]] = value

    def advance_toward(self, target_fingertip_pos: np.ndarray, max_step_m: float = 0.012) -> float:
        """목표 손끝 위치로 한 프레임만큼(resolved-rate Jacobian, 한 번)
        전진한다 -- 사용자 지시 3-2(처음 10프레임 유지 후 11프레임째 큰
        점프, 이후 감속 수렴)를 만족하려고 설계했다: 멀 때는 매 프레임
        최대 max_step_m만큼만 움직이고(등속), 가까워지면 Jacobian이 주는
        step이 그보다 작아져서 자연히 감속한다.

        **"가상(shadow) qpos를 독립적으로 전진시키고 그 결과를 ctrl에
        점프시키는" 1차 설계를 실측으로 폐기한 이유**: 이 자세(관절
        구성)에서 포지션 서보가 과도하게 감쇄돼 있다는 걸 나중에 발견했다
        (kp=2273.98/kv=1568.18는 peg_in_hole이 검증한 값이지만, 그건 그
        작업의 관절 구성에서의 유효 관성 기준이라 -- 관성은 관절 구성마다
        다르다 -- 이 작업의 home 근처 구성에서는 감쇄비가 ζ≈33 수준으로
        계산돼 전혀 다르다). 가상 목표가 "목표를 향해 알아서 전진"하고
        ctrl이 그 가상 목표를 그대로 따라가면, **실제 qpos는 ctrl보다
        한참 뒤처진 채로 그 차이가 누적**된다(실측: 15프레임(0.15s) 만에
        관절 하나가 0.8rad 넘게 뒤처짐, 수렴에 1초 이상 걸림 -- 에피소드
        전체가 끝나기 전에 다음 웨이포인트로 넘어가 버려서 손끝이 거의
        안 움직이는 것처럼 보이는 버그로 나타났었다).

        고친 방법: 가상 목표를 독립적으로 전진시키지 않고, **매 프레임
        실제(real) qpos/실제 Jacobian에서 다시 계산**해서 "실제 위치에서
        max_step_m만큼만 더 간 지점"을 그 프레임의 ctrl로 쓴다 -- ctrl이
        실제 qpos보다 앞서가는 양이 항상 max_step_m급 이하로 유지되므로,
        서보가 얼마나 느리게 감쇄돼 있든 ctrl이 실제를 "따라오게"
        시키는 게 아니라 실제를 "조금씩 끌고 가는" 구조가 된다(당근과
        막대기 패턴) -- 그 결과 실제 손끝 속도 ≈ max_step_m/프레임으로
        거의 일정해진다(실측 확인, 이 파일의 테스트 참고).

        반환: 남은 거리(m) -- 호출자가 수렴 판정에 쓴다."""
        real_qpos = {n: float(self.data.qpos[self._left_qposadr[n]]) for n in _LEFT_ARM_JOINTS}
        tip = self._fingertip_world(real_qpos)
        err = target_fingertip_pos - tip
        dist = float(np.linalg.norm(err))
        step = err if dist <= max_step_m else err * (max_step_m / dist)

        jacp = self._fingertip_jac(real_qpos)

        def _solve(J: np.ndarray) -> np.ndarray:
            jjt = J @ J.T + _JAC_DAMPING * np.eye(3)
            return J.T @ np.linalg.inv(jjt)

        dofs = [self._left_dofadr[n] for n in _LEFT_ARM_JOINTS]
        jacp_arm = jacp[:, dofs]
        jacp_pinv0 = _solve(jacp_arm)
        dq_task0 = jacp_pinv0 @ step

        jacp_frozen = jacp_arm.copy()
        frozen = []
        for i, name in enumerate(_LEFT_ARM_JOINTS):
            lo, hi = self._left_ranges[name]
            frac = (real_qpos[name] - lo) / (hi - lo)
            if (frac < _LIMIT_FREEZE_MARGIN and dq_task0[i] < 0.0) or (
                frac > 1.0 - _LIMIT_FREEZE_MARGIN and dq_task0[i] > 0.0
            ):
                jacp_frozen[:, i] = 0.0
                frozen.append(i)

        jacp_pinv = _solve(jacp_frozen)
        dq_task = jacp_pinv @ step

        null_proj = np.eye(7) - jacp_pinv @ jacp_frozen
        dq_null = np.array([_NULLSPACE_GAIN * (LEFT_HOME_QPOS[n] - real_qpos[n]) for n in _LEFT_ARM_JOINTS])
        dq = dq_task + null_proj @ dq_null
        for i in frozen:
            dq[i] = 0.0

        new_qpos = {}
        for i, name in enumerate(_LEFT_ARM_JOINTS):
            lo, hi = self._left_ranges[name]
            new_qpos[name] = float(np.clip(real_qpos[name] + dq[i], lo, hi))
            self.data.ctrl[self._left_actuator_ids[name]] = new_qpos[name]
            # 운동학적 적용(이 메서드 docstring "2차: 포지션 서보 자체를
            # 포기" 참고) -- qpos/qvel을 직접 덮어써서 ctrl(포지션 서보)이
            # 따라올 시간을 기다리지 않는다. 실측으로 발견: 이 home 근처
            # 구성에서 포지션 서보 과감쇄를 kv 재조정(assets/
            # pin_assembly_openarm.xml 참고)으로 어느 정도 고쳤지만, 여러
            # pick-place를 연쇄했을 때 여전히 수렴이 불안정해서(에피소드
            # 하나가 1600프레임을 넘기고도 수렴 못 하거나, 자세가 정해진
            # 목표와 무관하게 크게 벗어나는 경우가 실측으로 나옴) 데이터
            # 생성의 신뢰성을 포기할 수 없었다. 대신 ctrl도 항상 같은
            # 값으로 맞춰두므로(위) mj_step의 포지션 서보 오차가 거의
            # 0이라 추가로 흔들리지 않는다 -- 사실상 "서보가 쫓아올 필요가
            # 없을 만큼 매번 정확히 거기 가져다 놓는" 것과 같다.
            self.data.qpos[self._left_qposadr[name]] = new_qpos[name]
            self.data.qvel[self._left_dofadr[name]] = 0.0
        self._virtual_qpos = new_qpos
        mujoco.mj_forward(self.model, self.data)
        return dist

    def set_left_gripper(self, ctrl: float) -> None:
        self.data.ctrl[self._left_gripper_actuator_id] = ctrl

    def _pin_right_arm(self) -> None:
        """오른팔을 RIGHT_HOME_QPOS에 운동학적으로 고정한다 -- 공식
        RoCo 데이터의 "오른팔은 사실상 정지" 관례(y<=40mm, z<=30mm)를
        만족하려면 일반 포지션 서보 동역학에 맡기는 것만으로는 부족했다
        (실측: validate_against_roco.py의 오른팔 정지 검사에서 y 56mm,
        z 57mm로 한도를 넘김 -- LEFT_HOME_QPOS처럼 RIGHT_HOME_QPOS도
        그 자체가 "중력을 정확히 상쇄하는" 평형점이 아니라서 매 스텝
        약간씩 처지는 게 누적됨). 오른팔은 이 태스크 내내 어차피 전혀
        안 움직이므로, 왼팔의 hold_left()/advance_toward()와 같은
        원리를 상시 적용해도 손실이 없다."""
        for name, value in RIGHT_HOME_QPOS.items():
            self.data.qpos[self._right_qposadr[name]] = value
            self.data.qvel[self._right_dofadr[name]] = 0.0
            self.data.ctrl[self._right_actuator_ids[name]] = value

    def step_frame(self) -> None:
        """출력 1프레임(10fps) 분량 -- N_SUBSTEPS*dt=0.01s. carry 중이면
        매 서브스텝 _apply_carry()로 다시 덮어쓴다(물리 스텝 사이에
        조인트 PD가 수렴하는 동안 매번 다시 맞춰줘야 함). 오른팔도 매
        서브스텝 _pin_right_arm()으로 고정한다(위 docstring 참고)."""
        for _ in range(N_SUBSTEPS):
            self._apply_carry()
            self._pin_right_arm()
            mujoco.mj_step(self.model, self.data)
        self._apply_carry()
        self._pin_right_arm()
        mujoco.mj_forward(self.model, self.data)
