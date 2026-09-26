"""ARD-Gen 나사 조이기(screw driving) 태스크, OpenArm 양팔 버전 시뮬레이터.

peg_in_hole_bimanual_openarm_sim.py의 오른팔 computed-torque 위치 유지
컨트롤(resolved-rate + Jacobian-column-freezing + 널스페이스)과
screw_driving_sim.py의 가상 나사산 커플링 + 토크 리미터 회전 제어를
합쳤다. 자산/설계 배경은 assets/screw_driving_bimanual_openarm.xml 상단
docstring 참고.

## joint7을 xyz 유지 과제에서 완전히(항상) 제외하는 이유

screw_driving_sim.py는 wrist_rotate를 z-추종 Jacobian 과제(_ARM_FOLLOW_JOINTS)
에서 아예 뺐다 -- 회전축과 삽입축이 같은 축이라 뒤섞이면 안 되기 때문
(회전은 토크 리미터가, tip-head 오프셋 유지는 xyz 추종이 각자 독립적으로
맡아야 함). 여기서도 똑같이: joint7의 Jacobian 열을 (한계 근처에서만이
아니라) 항상 0으로 고정해서 xyz 추종 dq가 joint7에 전혀 배분되지 않게
하고, joint7의 _virtual_qpos는 turn/rewind 사이클이 직접 갱신한다.

## bolt 위치를 매 reset()마다 block 위치에 맞춰 procedural로 재배치하는 이유

screw_driving.xml(단일팔)은 block이 world에 고정돼 있어서 bolt(자체
slide/hinge 축이 world "0 0 -1" 고정)와 항상 정렬됐다. 여기서는 block도
hole_socket처럼 왼팔이 쥐는 자유 바디다 -- 다만 block의 세계 위치는
왼팔 home 자세가 고정이라 매번 똑같다(hole_socket과 동일한 이유로
"home" 값을 그대로 재사용, 아래 _BLOCK_HOME_QPOS 참고). bolt는 block의
자식이 아니라 독립된 바디라서, block 바로 위(고정 오프셋
_BOLT_ABOVE_BLOCK_M)에 있어야 slide/hinge 축이 block 벽과 계속 맞는다.
reset()에서 block을 배치한 뒤 실제 xpos를 읽어 `model.body_pos[bolt]`를
거기 맞춰 다시 쓴다 -- mj_forward 전에 model 파라미터(geom_pos/size를
런타임에 덮어쓰는 것과 같은 패턴)를 바꾸는 절차적 배치다.

## TURN_LOW/TURN_HIGH을 openarm_right_joint7 range에 맞게 좁힌 이유

VX300s wrist_rotate는 range ±3.14158(반 바퀴)라 TURN_LOW/HIGH를 ±2.8로
잡을 여유가 있었다. openarm_right_joint7은 range ±1.5708(±90도)뿐이라
비슷한 비율의 여유(약 10%)를 적용해 ±1.4로 좁혔다 -- 사이클당 회전량이
절반으로 줄어드는 대신 사이클 수가 늘어난다(총 필요 회전량 =
target_depth/PITCH_PER_RAD은 그대로이므로) -- MAX_CONTROL_STEPS를
screw_driving_sim.py보다 넉넉히 키운 이유이기도 하다.

## driver_tip 위치를 mj_jacSite가 아니라 mj_jac(point, body)로 구하는 이유

peg_in_hole_bimanual_openarm_sim.py, screw_driving.xml 둘 다 이미 겪은
버그와 동일: driver는 weld로만 연결된 자유 바디라 mj_jacSite로 Jacobian을
구하면 팔 관절에 대해 거의 0이 나온다. openarm_right_ee_base_link에
강체로 붙어있다고 가정한 world 좌표에 대해 직접 mj_jac을 호출해서 우회한다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "screw_driving_bimanual_openarm.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

TARGET_DEPTH = 0.034  # bolt_slide 최대 범위(완전 삽입), screw_driving.xml과 동일
PITCH_PER_RAD = 0.002 / (2 * np.pi)  # 나사산 피치 2mm/rev, screw_driving.xml과 동일

# joint7 range(±1.5708)에 맞춰 좁힌 사이클 범위 (위 docstring 참고).
TURN_LOW = -1.4
TURN_HIGH = 1.4

NOMINAL_RATE = 2.5  # rad/s -- screw_driving_sim.py에서 실측 검증된 값 재사용.

# 사이클당 회전량이 VX300s 버전의 절반(2.8 vs 5.6 rad)이라 필요한 사이클
# 수가 대략 2배 -- 넉넉하게 잡았다(실측 후 조정 가능).
MAX_CONTROL_STEPS = 30000

_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_JOINT7_NAME = "openarm_right_joint7"
_XYZ_JOINTS = _ARM_JOINTS[:6]  # joint7 제외 (위 docstring 참고)

_LEFT_ARM_HOME_QPOS = {
    "openarm_left_joint1": -0.33119160341930914,
    "openarm_left_joint2": -0.6927695819191171,
    "openarm_left_joint3": 1.0264533366778317,
    "openarm_left_joint4": 1.8918419565784326,
    "openarm_left_joint5": -0.9724271650596741,
    "openarm_left_joint6": -0.11753742523671289,
    "openarm_left_joint7": -0.306424166993611,
}
# peg_in_hole_bimanual_openarm_sim.py의 _HOME_QPOS와 동일 -- FK 확인 결과
# (screw_driving_bimanual_openarm.xml 도입 검증 스크립트) driver_tip과
# bolt_head가 이 자세에서 이미 xy 0.1mm 이내로 정렬돼서 재탐색 불필요.
_HOME_QPOS = {
    "openarm_right_joint1": -0.4932859043730308,
    "openarm_right_joint2": 2.7728121142883237,
    "openarm_right_joint3": 0.7779694449047385,
    "openarm_right_joint4": 2.068501962471789,
    "openarm_right_joint5": 0.5568966487876724,
    "openarm_right_joint6": -0.5030469209189363,
    "openarm_right_joint7": 0.973709272141154,
}

_RIGHT_GRIPPER_GRASP_CTRL = -0.030  # peg와 동일 굵기 가정 -- 위 docstring/asset 참고.
_LEFT_GRIPPER_GRASP_CTRL = 0.110  # hole_socket과 동일 각도 재사용(placeholder).

# driver_grasp weld의 relpose와 동일한 값 (assets/screw_driving_bimanual_openarm.xml 참고).
_DRIVER_LOCAL_ANCHOR = np.array([-0.02222, 0.0, -0.21214])
_DRIVER_RELPOSE_QUAT = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)  # y축 90도


def _rot_90y(v: np.ndarray) -> np.ndarray:
    """로컬 +x를 로컬 -z로 보내는 90도 회전(quat (1,0,1,0)과 동일 회전).
    driver_tip_site(driver 로컬 +x=0.044)를 ee_base_link 로컬 오프셋으로
    바꿀 때 쓴다 -- 실제 Jacobian/FK 계산은 world quat으로 하므로, 이건
    상수 오프셋(_DRIVER_TIP_OFFSET_IN_EE)을 한 번 미리 구하는 데만 쓰인다."""
    x, y, z = v
    return np.array([z, y, -x])


_DRIVER_TIP_OFFSET_IN_EE = _DRIVER_LOCAL_ANCHOR + _rot_90y(np.array([0.044, 0.0, 0.0]))

# block_grasp weld의 relpose와 동일 -- hole_grasp를 그대로 재사용한 값이라
# hole_socket과 완전히 같은 이유로 이 왼팔 home 자세에서 block이 world
# 기준 무회전이 된다(assets 상단 docstring 참고). 그래서 block의 위치도
# hole_socket처럼 "home" 값 그대로 고정 상수로 쓸 수 있다(왼팔이 안
# 움직이므로 매번 다시 계산할 필요 없음) -- 값은 peg_in_hole_bimanual_openarm.xml
# 의 hole_socket qpos placeholder와 동일(relpose가 같으니 FK 결과도 같음).
_BLOCK_HOME_QPOS = np.array([0.21559240, -0.07038191, 0.48106862, 1.0, 0.0, 0.0, 0.0])

# bolt body 원점을 block body 원점 기준 이 값(m)만큼 위에 procedural로
# 배치한다 -- screw_driving.xml의 block(0.04)-bolt(0.074) 오프셋과 동일
# (0.074-0.04=0.034)한 상대 관계를 그대로 재사용(안전 검증된 seat 간격을
# 깨지 않기 위해 그대로 둠 -- 처음엔 driver_tip과 bolt_head 사이 hover
# 간격을 늘리려고 이 값을 줄여봤는데, bolt shaft 바닥이 seat보다 아래로
# 내려가 t=0부터 벽에 파고드는 걸 계산으로 확인하고 원래 값으로 되돌렸다.
# driver_tip이 bolt_head보다 ~7mm 아래에서 시작하는 것처럼 보이는 정도는
# driver에 충돌 자체가 없어서(assets docstring 참고) 물리적으로 무해하다
# -- 렌더로 확인 후 순수 시각적으로만 다듬을 만한 항목).
_BOLT_ABOVE_BLOCK_M = 0.034

_JAC_DAMPING = 1e-4
_NULLSPACE_GAIN = 0.03
_LIMIT_FREEZE_MARGIN = 0.08
_TRACK_GAIN = 0.8  # screw_driving_sim.py의 _z_track_step "error*0.8"과 동일.

_TORQUE_OMEGA_N = 40.0
_TORQUE_ZETA = 1.0

_NOMINAL_HINGE_FRICTIONLOSS = 0.015


def _default_scene_config() -> dict[str, Any]:
    return {
        "target_depth": TARGET_DEPTH,
        "hinge_friction_scale": 1.0,
    }


class ScrewDrivingBimanualOpenArmSim:
    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._right_ee_body_id = self.model.body("openarm_right_ee_base_link").id
        self._left_ee_body_id = self.model.body("openarm_left_ee_base_link").id
        self._block_body_id = self.model.body("block").id
        self._bolt_body_id = self.model.body("bolt").id

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_JOINTS}
        self._arm_dofs = [self._arm_dofadr[name] for name in _ARM_JOINTS]
        self._joint7_qposadr = self._arm_qposadr[_JOINT7_NAME]

        self._left_arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_HOME_QPOS}
        self._left_arm_actuator_ids = {
            name: self.model.actuator(f"pos_left_j{i}").id for i, name in enumerate(_LEFT_ARM_HOME_QPOS, start=1)
        }
        self._left_gripper_actuator_id = self.model.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = self.model.joint("openarm_left_finger_joint1").qposadr[0]
        self._left_finger2_qposadr = self.model.joint("openarm_left_finger_joint2").qposadr[0]

        self._gripper_actuator_id = self.model.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = self.model.joint("openarm_right_finger_joint1").qposadr[0]
        self._right_finger2_qposadr = self.model.joint("openarm_right_finger_joint2").qposadr[0]

        # vendor 팔 관절 <position> 액추에이터 14개 무력화 (peg_in_hole과 동일 이유).
        for prefix, joints in (("right", _ARM_JOINTS), ("left", list(_LEFT_ARM_HOME_QPOS))):
            for i in range(1, 8):
                act_id = self.model.actuator(f"{prefix}_joint{i}_ctrl").id
                self.model.actuator_gainprm[act_id] = 0.0
                self.model.actuator_biasprm[act_id] = 0.0

        # 오른팔 pos_right_j*(강화 게인 <position>)도 무력화 -- torque_right_j*
        # (computed-torque)로만 구동한다(peg_in_hole과 동일).
        for i in range(1, 8):
            act_id = self.model.actuator(f"pos_right_j{i}").id
            self.model.actuator_gainprm[act_id] = 0.0
            self.model.actuator_biasprm[act_id] = 0.0
        self._torque_actuator_ids = {
            name: self.model.actuator(f"torque_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)
        }

        # vendor 모델에는 gravcomp가 없다 -- peg_in_hole과 동일한 이유로 직접 켠다.
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                self.model.body_gravcomp[i] = 1.0

        self._block_qposadr = self.model.joint("block_free").qposadr[0]
        self._driver_qposadr = self.model.joint("driver_free").qposadr[0]

        self._driver_tip_site_id = self.model.site("driver_tip_site").id
        self._bolt_head_site_id = self.model.site("bolt_head_site").id

        self._bolt_hinge_qposadr = self.model.joint("bolt_hinge").qposadr[0]
        self._bolt_hinge_dofadr = self.model.joint("bolt_hinge").dofadr[0]
        self._bolt_slide_qposadr = self.model.joint("bolt_slide").qposadr[0]
        self._bolt_hinge_drive_id = self.model.actuator("bolt_hinge_drive").id
        self._bolt_slide_drive_id = self.model.actuator("bolt_slide_drive").id
        self._torque_adr = self.model.sensor("bolt_drive_torque").adr[0]

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # peg_in_hole과 동일한 이유(파일 상단 docstring 참고)로, resolved-rate
        # 계획은 실제 시뮬레이션 상태가 아니라 별도의 순수 기구학 shadow 상태에서 한다.
        self._shadow_data = mujoco.MjData(self.model)
        self._virtual_qpos: dict[str, float] = dict(_HOME_QPOS)

        self._target_offset = np.zeros(3)
        self._phase = "turn"
        self._engage_wrist_ref = 0.0
        self._engage_hinge_ref = 0.0
        self._frozen_hinge_ctrl = 0.0

    # ------------------------------------------------------------------
    def _sync_driver_to_arm_fk(self) -> None:
        """driver free body의 qpos를 지금 오른팔 ee pose로부터 FK로 계산해서
        덮어쓴다(mj_forward만으로는 weld로 연결된 자유 바디가 안 움직이는
        패턴, peg_in_hole의 _sync_peg_to_arm_fk와 동일)."""
        ee_pos = self.data.xpos[self._right_ee_body_id]
        ee_quat = self.data.xquat[self._right_ee_body_id].copy()
        R = self.data.xmat[self._right_ee_body_id].reshape(3, 3)
        driver_pos = ee_pos + R @ _DRIVER_LOCAL_ANCHOR
        driver_quat = np.zeros(4)
        mujoco.mju_mulQuat(driver_quat, ee_quat, _DRIVER_RELPOSE_QUAT)
        qadr = self._driver_qposadr
        self.data.qpos[qadr : qadr + 3] = driver_pos
        self.data.qpos[qadr + 3 : qadr + 7] = driver_quat

    def _virtual_driver_tip_and_jac(self) -> tuple[np.ndarray, np.ndarray]:
        """self._virtual_qpos에서의 driver tip 위치와, openarm_right_ee_base_link에
        강체로 붙어있다고 가정한 Jacobian (peg_in_hole의
        _virtual_peg_tip_and_jac와 동일한 이유/기법, 파일 상단 docstring 참고)."""
        sd = self._shadow_data
        for name, value in self._virtual_qpos.items():
            sd.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        ee_pos = sd.xpos[self._right_ee_body_id]
        R = sd.xmat[self._right_ee_body_id].reshape(3, 3)
        driver_tip = ee_pos + R @ _DRIVER_TIP_OFFSET_IN_EE
        mujoco.mj_jac(self.model, sd, self._jacp, self._jacr, driver_tip, self._right_ee_body_id)
        return driver_tip, self._jacp

    def _advance_virtual_xyz(self, delta_pos_world: np.ndarray) -> None:
        """joint7을 제외한 6개 관절로 driver tip을 delta_pos_world만큼
        전진시킨다(resolved-rate, 관절 한계 회피용 Jacobian-column-freezing +
        널스페이스는 peg_in_hole의 _advance_virtual과 동일 기법). joint7의
        Jacobian 열은 (한계와 무관하게) 항상 고정한다 -- 회전은 별도로
        turn/rewind 사이클이 담당하므로(파일 상단 docstring 참고)."""
        _, jacp0 = self._virtual_driver_tip_and_jac()
        joint7_dof = self._arm_dofadr[_JOINT7_NAME]
        jacp = jacp0.copy()
        jacp[:, joint7_dof] = 0.0

        def _solve(J: np.ndarray) -> np.ndarray:
            jjt = J @ J.T + _JAC_DAMPING * np.eye(3)
            return J.T @ np.linalg.inv(jjt)

        jacp_pinv0 = _solve(jacp)
        dq_task0 = jacp_pinv0 @ delta_pos_world
        jacp_frozen = jacp.copy()
        frozen_dofs = [joint7_dof]
        for name in _XYZ_JOINTS:
            dof = self._arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            frac = (self._virtual_qpos[name] - lo) / (hi - lo)
            if (frac < _LIMIT_FREEZE_MARGIN and dq_task0[dof] < 0.0) or (
                frac > 1.0 - _LIMIT_FREEZE_MARGIN and dq_task0[dof] > 0.0
            ):
                jacp_frozen[:, dof] = 0.0
                frozen_dofs.append(dof)

        jacp_pinv = _solve(jacp_frozen)
        dq_task = jacp_pinv @ delta_pos_world

        nv = self.model.nv
        null_proj = np.eye(nv) - jacp_pinv @ jacp_frozen
        dq_null = np.zeros(nv)
        for name in _XYZ_JOINTS:
            dof = self._arm_dofadr[name]
            dq_null[dof] = _NULLSPACE_GAIN * (_HOME_QPOS[name] - self._virtual_qpos[name])
        dq = dq_task + null_proj @ dq_null
        for dof in frozen_dofs:
            dq[dof] = 0.0

        for name in _XYZ_JOINTS:
            dof = self._arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            self._virtual_qpos[name] = float(np.clip(self._virtual_qpos[name] + dq[dof], lo, hi))

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any] | None = None) -> None:
        cfg = _default_scene_config()
        if scene_config:
            cfg.update(scene_config)
        self.model.dof_frictionloss[self._bolt_hinge_dofadr] = (
            _NOMINAL_HINGE_FRICTIONLOSS * cfg["hinge_friction_scale"]
        )

        mujoco.mj_resetData(self.model, self.data)

        # 왼팔(고정) + block: home 값 그대로 (peg_in_hole의 hole_socket과 동일 패턴).
        for name, value in _LEFT_ARM_HOME_QPOS.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
            self.data.ctrl[self._left_arm_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._left_finger2_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = _LEFT_GRIPPER_GRASP_CTRL
        qadr = self._block_qposadr
        self.data.qpos[qadr : qadr + 7] = _BLOCK_HOME_QPOS

        # 오른팔: home qpos에서 시작 -- 이미 이 자세에서 driver_tip이 bolt_head와
        # xy 0.1mm 이내로 정렬되므로 IK 반복이 불필요(파일 상단 docstring 참고).
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        self.data.qpos[self._right_finger_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._right_finger2_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._gripper_actuator_id] = _RIGHT_GRIPPER_GRASP_CTRL

        mujoco.mj_forward(self.model, self.data)
        self._sync_driver_to_arm_fk()
        mujoco.mj_forward(self.model, self.data)

        # bolt를 block의 실제 FK 위치 기준으로 procedural 배치 (파일 상단 docstring 참고).
        block_pos = self.data.xpos[self._block_body_id].copy()
        self.model.body_pos[self._bolt_body_id] = block_pos + np.array([0.0, 0.0, _BOLT_ABOVE_BLOCK_M])
        mujoco.mj_forward(self.model, self.data)

        self._virtual_qpos = dict(_HOME_QPOS)

        self._target_offset = (
            self.data.site_xpos[self._driver_tip_site_id] - self.data.site_xpos[self._bolt_head_site_id]
        ).copy()

        self._phase = "turn"
        self._engage_wrist_ref = float(self.data.qpos[self._joint7_qposadr])
        self._engage_hinge_ref = float(self.data.qpos[self._bolt_hinge_qposadr])
        self._frozen_hinge_ctrl = 0.0
        self.data.ctrl[self._bolt_hinge_drive_id] = 0.0
        self.data.ctrl[self._bolt_slide_drive_id] = 0.0

    def get_torque(self) -> float:
        return float(self.data.sensordata[self._torque_adr])

    def get_insertion_depth(self) -> float:
        return float(self.data.qpos[self._bolt_slide_qposadr])

    def get_driver_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._driver_tip_site_id].copy()

    def get_bolt_head_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._bolt_head_site_id].copy()

    # ------------------------------------------------------------------
    def step(self, gains: dict[str, float]) -> dict[str, Any]:
        """제어 틱 하나를 진행한다. torque_limiter 회전 제어(screw_driving_sim.py
        와 동일 설계, joint7로 이식) + xyz 유지 computed-torque(peg_in_hole과
        동일 설계, joint7 제외) + 나사산 가상 커플링(screw_driving.xml과 동일).

        gains: {"torque_limit": float}
        반환: {"torque", "rate", "phase", "limited"} -- screw_driving_sim.py와 동일.
        """
        torque_limit = float(gains.get("torque_limit", np.inf))
        current_ctrl = self._virtual_qpos[_JOINT7_NAME]
        torque = self.get_torque()
        limited = False

        if self._phase == "turn":
            if abs(torque) >= torque_limit:
                limited = True
                self._phase = "rewind"
                self._frozen_hinge_ctrl = float(self.data.ctrl[self._bolt_hinge_drive_id])
                new_ctrl = current_ctrl
            else:
                new_ctrl = min(current_ctrl + NOMINAL_RATE * DT, TURN_HIGH)
                self._virtual_qpos[_JOINT7_NAME] = new_ctrl
                self.data.ctrl[self._bolt_hinge_drive_id] = (
                    self._engage_hinge_ref
                    + (float(self.data.qpos[self._joint7_qposadr]) - self._engage_wrist_ref)
                )
                if new_ctrl >= TURN_HIGH:
                    self._phase = "rewind"
                    self._frozen_hinge_ctrl = float(self.data.ctrl[self._bolt_hinge_drive_id])
            rate = NOMINAL_RATE if not limited else 0.0
        else:  # rewind: disengaged, 저항 없이 항상 nominal rate로 되감는다
            rate = NOMINAL_RATE
            new_ctrl = max(current_ctrl - rate * DT, TURN_LOW)
            self._virtual_qpos[_JOINT7_NAME] = new_ctrl
            self.data.ctrl[self._bolt_hinge_drive_id] = self._frozen_hinge_ctrl
            if new_ctrl <= TURN_LOW:
                self._phase = "turn"
                self._engage_wrist_ref = float(self.data.qpos[self._joint7_qposadr])
                self._engage_hinge_ref = float(self.data.qpos[self._bolt_hinge_qposadr])

        # xyz 유지 (joint7 제외 6관절) -- 실제 물리 상태 기준 오차를 목표
        # 오프셋으로 되돌리는 P 제어(screw_driving_sim.py의 error*0.8과 동일 게인).
        current_offset = self.get_driver_tip_pos() - self.get_bolt_head_pos()
        error = self._target_offset - current_offset
        self._advance_virtual_xyz(error * _TRACK_GAIN)

        # 나사산 가상 커플링 (screw_driving.xml/screw_driving_sim.py와 동일).
        self.data.ctrl[self._bolt_slide_drive_id] = (
            PITCH_PER_RAD * self.data.qpos[self._bolt_hinge_qposadr]
        )

        # 오른팔 computed-torque (peg_in_hole과 동일 공식, 7관절 전부 -- joint7도
        # 자기 _virtual_qpos 목표(turn/rewind가 갱신)를 그대로 따라간다).
        mujoco.mj_forward(self.model, self.data)
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        M_rr = M[np.ix_(self._arm_dofs, self._arm_dofs)]
        qacc_cmd = np.empty(len(_ARM_JOINTS))
        bias = np.empty(len(_ARM_JOINTS))
        for i, name in enumerate(_ARM_JOINTS):
            dof = self._arm_dofadr[name]
            qpos = self.data.qpos[self._arm_qposadr[name]]
            qvel = self.data.qvel[dof]
            q_des = self._virtual_qpos[name]
            qacc_cmd[i] = _TORQUE_OMEGA_N**2 * (q_des - qpos) - 2.0 * _TORQUE_ZETA * _TORQUE_OMEGA_N * qvel
            bias[i] = self.data.qfrc_bias[dof] - self.data.qfrc_passive[dof]
        tau = M_rr @ qacc_cmd + bias
        for i, name in enumerate(_ARM_JOINTS):
            self.data.ctrl[self._torque_actuator_ids[name]] = tau[i]

        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)

        return {"torque": torque, "rate": rate, "phase": self._phase, "limited": limited}


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = ScrewDrivingBimanualOpenArmSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: ScrewDrivingBimanualOpenArmSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    sim.reset(cfg)
    target_depth = float(cfg["target_depth"])

    depth_profile = []
    torque_profile = []
    phase_profile = []
    max_torque = 0.0
    success = False
    step_count = 0

    for step_count in range(1, MAX_CONTROL_STEPS + 1):
        info = sim.step(gains)
        depth = sim.get_insertion_depth()
        depth_profile.append(depth)
        torque_profile.append(info["torque"])
        phase_profile.append(info["phase"])
        max_torque = max(max_torque, abs(info["torque"]))

        if depth >= target_depth * 0.99:
            success = True
            break

    final_depth = depth_profile[-1] if depth_profile else 0.0
    mean_abs_torque = float(np.mean(np.abs(torque_profile))) if torque_profile else 0.0
    reward = 20.0 * final_depth - 0.001 * step_count
    if success:
        reward += 50.0 - 3.0 * max_torque

    return {
        "depth_profile": np.array(depth_profile, dtype=np.float32),
        "torque_profile": np.array(torque_profile, dtype=np.float32),
        "phase_profile": phase_profile,
        "insertion_depth": float(final_depth),
        "max_torque": float(max_torque),
        "mean_abs_torque": mean_abs_torque,
        "step_count": step_count,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }


if __name__ == "__main__":
    for torque_limit in [np.inf, 1.79, 1.0, 0.6]:
        result = run_episode({"torque_limit": torque_limit})
        label = "no limit" if np.isinf(torque_limit) else f"limit={torque_limit}"
        print(
            f"[screw_driving_bimanual_openarm_sim] {label:12s} success={result['success']} "
            f"depth={result['insertion_depth'] * 1000:.2f}mm "
            f"steps={result['step_count']} max_torque={result['max_torque']:.3f}"
        )
