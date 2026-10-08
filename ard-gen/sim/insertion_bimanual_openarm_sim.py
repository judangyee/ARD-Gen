"""ARD-Gen insertion 태스크(OpenArm 양팔 버전) 시뮬레이션 + admittance controller.

`sim/peg_in_hole_bimanual_openarm_sim.py`("원본")를 거의 그대로 복제한 것이다
-- IK(resolved-rate + damped least squares + nullspace + 관절 한계 동결),
computed-torque(오른팔), gravcomp, 왼팔 고정, 그리퍼 anchor/열어두기 전부
**원본과 완전히 동일**하다(그 파일의 긴 docstring이 설명하는 실측/실패
과정 전부 여기도 그대로 적용됨 -- 반복 설명하지 않고 여기서는 참조만 한다).
바뀐 건 세 가지뿐이다:

1. **socket 모양**: `assets/insertion_bimanual_openarm.xml`은 원본의 단순
   4벽+바닥 대신 gym-aloha(huggingface) InsertionTask의 "통로 중앙에 고정
   pin" 구조를 수직으로 세워 들여왔다(그 asset 파일 상단 docstring 참고).
2. **clearance 기본값**: gym-aloha 쪽이 훨씬 느슨하다(한쪽 틈새 0.0015m ->
   0.008m) -- 지시대로 gym-aloha 값을 기본으로 썼다(`_NOMINAL_CLEARANCE_M`).
3. **보상/성공 판정**: 거리+깊이 기반 연속값 대신, gym-aloha
   InsertionTask.get_reward()의 0~4 단계형 보상을 접촉쌍 기준으로 그대로
   들여왔다(아래 "보상 설계" 절 참고) -- CMA-ES 목적함수로도 이 단계값을
   그대로 쓴다(사용자 확인: sparse해서 수렴이 안 될 수 있다는 걸 알고도
   원본 그대로 포팅하기로 결정, 이번 작업 범위에서는 결과를 있는 그대로
   보고한다).

## 보상 설계: gym-aloha와 다를 수밖에 없는 지점 (정직하게 기록)

gym-aloha의 InsertionTask.get_reward()는 4단계 전부를 **그리퍼-물체 접촉**
으로 판정한다(1=양쪽 그리퍼가 각자 물체에 닿음, 2=그 상태로 테이블 미접촉).
하지만 ARD-Gen의 OpenArm 패턴(원본 peg_in_hole)은 peg/socket을 grasp 손가락과
**<contact><exclude>로 충돌 자체를 꺼둔다**(자기충돌 버그 회피, assets 파일
docstring 참고) -- 즉 "그리퍼가 물체에 닿았다"는 접촉이 이 설계에서는
**구조적으로 영원히 발생하지 않는다**. 대신 peg/socket 둘 다 reset() 시점에
이미 weld로 쥐어진 채 공중에 떠 있다(테이블에 닿을 수조차 없는 높이, 아래
"보상 함수" 참고) -- 즉 gym-aloha의 1~2단계("그리퍼로 쥐고 들어올림")는 이
설계에서는 **매 에피소드 시작부터 이미 항상 참**이다. 그래서 그 두 단계를
그리퍼 접촉으로 재현하려 들면 오히려 항상 거짓으로 멈춰서(접촉 자체가 꺼져
있으므로) 보상이 영원히 0에서 안 움직이는 버그가 된다 -- gym-aloha를 "그대로"
포팅하면서도 이 설계 차이 때문에 바닥(1~2단계)은 상수로 두고, 실제로 의미
있게 변하는 3~4단계(peg가 socket에 닿는지, pin까지 닿는지)만 접촉으로
판정한다.

    reward = 2  # 양팔 모두 이미 쥔 채 공중에 있음(weld, 테이블 접촉 불가능) -- 매 에피소드 시작부터 참
    if peg가 socket 벽/바닥에 접촉: reward = 3
    if peg가 pin에 접촉: reward = 4  # gym-aloha의 pin_touched와 동일한 기준

`is_success()`는 reward>=4(pin 접촉)다. 원본처럼 "연속 N스텝 유지"를
요구하지는 않는다 -- 실측해보니 pin(반지름 6mm)이 peg tip(반지름 1cm)보다
얇아서 완전히 정렬돼도 윗면 가장자리에서 접촉이 매 스텝 미세하게 깜빡여
20스텝 연속 유지가 거의 불가능했다(`sim/insertion_openarm_env.py` 상단
docstring "success: hold-count를 쓰지 않는 이유" 참고). gym-aloha 원본도
애초에 그런 유지 조건이 없어서, "그대로 이식"하려면 빼는 게 맞다 -- pin
접촉은 실제 물리 접촉이라(원본의 깊이 기반 판정과 달리) 한 번만 닿아도
유효한 성공 신호로 본다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "insertion_bimanual_openarm.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s) -- 원본과 동일
DT = N_SUBSTEPS * 0.002

Z_RATE = 0.0006  # m / control step -- 원본과 동일(팔 기구학이 안 바뀌었으므로 재탐색 불필요, 아래 __main__에서 실측 확인)
MAX_STEPS = 1000  # 원본과 동일

_Z_GATE_MIN_FRACTION = 0.15
_Z_GATE_RADIUS_MULT = 2.0

# pin까지 닿으려면 원본의 목표 깊이(hole 길이의 1/3=0.0183m)보다 더 깊이
# 들어가야 한다 -- socket_center_site(로컬 z=0)에서 pin_top_site(로컬
# z=-0.03, assets 파일 참고)까지의 거리. _OVERSHOOT_DEPTH_MULT*TARGET이
# "더 안 내려간다" 컷오프라 1.5*0.03=0.045m로 늘어나지만, socket 바닥
# (로컬 z=-0.055)까지는 여전히 여유가 있다(물리적으로 바닥에 닿기 전에
# 멈춘다). 실제로 이 깊이까지 admittance+adaptive_z_rate가 발산 없이
# 도달하는지는 원본에서 검증되지 않은 부분이라 __main__에서 새로 실측한다.
TARGET_INSERTION_DEPTH = 0.03  # m, socket_center_site -> pin_top_site 거리
_OVERSHOOT_DEPTH_MULT = 1.5


def adaptive_z_rate(dx: float, dy: float, outer_half: float, raw_depth: float, target_depth: float) -> float:
    """원본과 100% 동일한 수식(그 파일 docstring 참고) -- 수직 삽입 축 자체가
    안 바뀌었으므로 게이팅 로직을 바꿀 이유가 없다."""
    if raw_depth >= _OVERSHOOT_DEPTH_MULT * target_depth:
        return 0.0
    xy_err = (dx**2 + dy**2) ** 0.5
    align_quality = max(0.0, 1.0 - xy_err / (outer_half * _Z_GATE_RADIUS_MULT))
    scale = _Z_GATE_MIN_FRACTION + (1.0 - _Z_GATE_MIN_FRACTION) * align_quality
    return Z_RATE * scale


# socket 벽 nominal 치수 -- 원본과 동일(_apply_clearance가 런타임에 재배치).
_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
# gym-aloha 기본값으로 교체(이 파일 상단 docstring "clearance 기본값" 절):
# inner_half = _PEG_HALF_WIDTH + clearance/2 = 0.01 + 0.008 = 0.018m, gym-aloha의
# 소켓 내벽(0.02 중심 - 0.002 반두께 = 0.018)과 peg 반폭(0.01) 차이(0.008m
# 한쪽 틈새)와 동일한 비율. 원본 값(0.003m, 한쪽 틈새 0.0015m)보다 훨씬 느슨.
_NOMINAL_CLEARANCE_M = 0.016

# OpenArm 오른팔 7개 관절 -- 원본과 동일.
_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# home 자세 -- 원본(assets/peg_in_hole_bimanual_openarm.xml)과 완전히 동일한
# 값. socket 내부 구조(벽 위치/pin)를 바꿔도 "왼팔이 어디서 socket을 쥐는지"
# (grasp anchor, 오른팔 ee 로컬 좌표 기준)는 안 바뀌므로 CMA-ES로 다시 풀
# 필요가 없다 -- 아래 __main__에서 실측으로 재확인한다.
_HOME_QPOS = {
    "openarm_right_joint1": -0.4932859043730308,
    "openarm_right_joint2": 2.7728121142883237,
    "openarm_right_joint3": 0.7779694449047385,
    "openarm_right_joint4": 2.068501962471789,
    "openarm_right_joint5": 0.5568966487876724,
    "openarm_right_joint6": -0.5030469209189363,
    "openarm_right_joint7": 0.973709272141154,
}
_LEFT_ARM_HOME_QPOS = {
    "openarm_left_joint1": -0.33119160341930914,
    "openarm_left_joint2": -0.6927695819191171,
    "openarm_left_joint3": 1.0264533366778317,
    "openarm_left_joint4": 1.8918419565784326,
    "openarm_left_joint5": -0.9724271650596741,
    "openarm_left_joint6": -0.11753742523671289,
    "openarm_left_joint7": -0.306424166993611,
}
_RIGHT_GRIPPER_GRASP_CTRL = -0.030  # 원본과 동일
_LEFT_GRIPPER_GRASP_CTRL = 0.110  # 원본과 동일

# peg free body를 오른팔 ee 프레임으로부터 FK로 세팅할 때 쓰는 로컬 오프셋
# -- 원본과 완전히 동일(socket 설계와 무관, 오른팔 grasp anchor 그 자체).
_PEG_LOCAL_OFFSET = np.array([-0.02222, 0, -0.21214])

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 200
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.03
_LIMIT_FREEZE_MARGIN = 0.08

_HOVER_GAP_M = 0.039  # 원본과 동일

_TORQUE_OMEGA_N = 40.0
_TORQUE_ZETA = 1.0

# gym-aloha InsertionTask.get_reward()의 접촉쌍 판정을 이 설계(weld-grasp +
# 그리퍼-물체 충돌 제외)에 맞게 적용한 결과 -- 이 파일 상단 docstring
# "보상 설계" 절 참고. 그리퍼가 쥔 채 공중에 있다는 조건(gym-aloha의
# 1~2단계)은 이 설계에서 매 에피소드 시작부터 항상 참이라 바닥값으로 둔다.
_WELD_GRASP_BASE_STAGE = 2
_PEG_GEOM_NAMES = {"peg_shaft", "peg_tip_ball"}
_SOCKET_WALL_GEOM_NAMES = {"socket_wall_px", "socket_wall_nx", "socket_wall_py", "socket_wall_ny", "socket_floor"}
_PIN_GEOM_NAME = "pin"


def _default_scene_config() -> dict[str, Any]:
    return {
        "friction": 0.5,
        "clearance_m": _NOMINAL_CLEARANCE_M,
        "peg_init_offset_xy": (0.0, 0.0),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def sample_scene_config(rng: np.random.Generator) -> dict[str, Any]:
    """원본과 같은 오프셋 반경(팔 기구학이 안 바뀌어서 그대로 적절할 것으로
    보고 시작, 실측으로 재검토 가능) -- clearance_m 범위만 새 기본값
    (_NOMINAL_CLEARANCE_M=0.016) 주변으로 옮겼다."""
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    radius = rng.uniform(0.009, 0.014)
    return {
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.014, 0.018)),
        "peg_init_offset_xy": (radius * np.cos(angle), radius * np.sin(angle)),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


class BimanualInsertionOpenArmSim:
    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._socket_body_id = self.model.body("socket").id
        self._peg_body_id = self.model.body("peg").id
        self._right_ee_body_id = self.model.body("openarm_right_ee_base_link").id
        self._left_ee_body_id = self.model.body("openarm_left_ee_base_link").id
        self._socket_site_id = self.model.site("socket_center_site").id
        self._pin_top_site_id = self.model.site("pin_top_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]

        self._wall_geom_ids = {
            name: self.model.geom(f"socket_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("socket_floor").id
        self._pin_geom_id = self.model.geom("pin").id

        # 접촉쌍 판정용 geom id 집합 (이 파일 상단 docstring "보상 설계" 참고).
        self._peg_geom_id_set = {self.model.geom(n).id for n in _PEG_GEOM_NAMES}
        self._socket_wall_geom_id_set = {self.model.geom(n).id for n in _SOCKET_WALL_GEOM_NAMES}
        self._pin_geom_id_set = {self.model.geom(_PIN_GEOM_NAME).id}

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_JOINTS}
        self._arm_actuator_ids = {
            name: self.model.actuator(f"pos_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)
        }
        self._gripper_actuator_id = self.model.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = self.model.joint("openarm_right_finger_joint1").qposadr[0]
        self._right_finger2_qposadr = self.model.joint("openarm_right_finger_joint2").qposadr[0]

        self._left_arm_qposadr = {
            name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_HOME_QPOS
        }
        self._left_arm_actuator_ids = {
            name: self.model.actuator(f"pos_left_j{i}").id for i, name in enumerate(_LEFT_ARM_HOME_QPOS, start=1)
        }
        self._left_gripper_actuator_id = self.model.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = self.model.joint("openarm_left_finger_joint1").qposadr[0]
        self._left_finger2_qposadr = self.model.joint("openarm_left_finger_joint2").qposadr[0]

        # vendor <position> 액추에이터 무력화 -- 원본과 동일한 이유
        # (assets/peg_in_hole_bimanual_openarm.xml 상단 docstring "3. 왼팔도..." 참고).
        for prefix, joints in (("right", _ARM_JOINTS), ("left", list(_LEFT_ARM_HOME_QPOS))):
            for i in range(1, 8):
                act_id = self.model.actuator(f"{prefix}_joint{i}_ctrl").id
                self.model.actuator_gainprm[act_id] = 0.0
                self.model.actuator_biasprm[act_id] = 0.0

        for act_id in self._arm_actuator_ids.values():
            self.model.actuator_gainprm[act_id] = 0.0
            self.model.actuator_biasprm[act_id] = 0.0
        self._torque_actuator_ids = {
            name: self.model.actuator(f"torque_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)
        }
        self._arm_dofs = [self._arm_dofadr[name] for name in _ARM_JOINTS]

        # gravcomp -- 원본과 동일한 이유(vendor에 gravcomp가 없음).
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                self.model.body_gravcomp[i] = 1.0

        self._peg_qposadr = self.model.joint("peg_free").qposadr[0]
        self._socket_qposadr = self.model.joint("socket_free").qposadr[0]

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        key_id = self.model.key("home").id
        self._socket_home_qpos = self.model.key_qpos[key_id][
            self._socket_qposadr : self._socket_qposadr + 7
        ].copy()

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        self._shadow_data = mujoco.MjData(self.model)
        self._virtual_qpos: dict[str, float] = dict(_HOME_QPOS)

    # ------------------------------------------------------------------
    def _apply_clearance(self, clearance_m: float) -> float:
        """원본과 100% 동일한 수식 -- socket_wall_*/socket_floor geom id만 이름이 바뀜."""
        inner_half = _PEG_HALF_WIDTH + clearance_m / 2.0
        outer_half = inner_half + _WALL_HALF_THICKNESS * 2.0

        wall_center = inner_half + _WALL_HALF_THICKNESS
        self.model.geom_pos[self._wall_geom_ids["px"]] = [wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["nx"]] = [-wall_center, 0, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["py"]] = [0, wall_center, _WALL_CENTER_Z]
        self.model.geom_pos[self._wall_geom_ids["ny"]] = [0, -wall_center, _WALL_CENTER_Z]

        self.model.geom_size[self._wall_geom_ids["px"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["nx"]] = [_WALL_HALF_THICKNESS, outer_half, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["py"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]
        self.model.geom_size[self._wall_geom_ids["ny"]] = [outer_half, _WALL_HALF_THICKNESS, _WALL_HALF_HEIGHT]

        self.model.geom_size[self._floor_geom_id] = [inner_half, inner_half, _FLOOR_HALF_THICKNESS]
        return outer_half

    def _jac_at_point(self, world_point: np.ndarray) -> np.ndarray:
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, world_point, self._right_ee_body_id)
        return self._jacp

    def _jac_solve(self, delta_pos_world: np.ndarray) -> np.ndarray:
        peg_tip = self.data.site_xpos[self._peg_tip_site_id].copy()
        jacp = self._jac_at_point(peg_tip)
        jjt = jacp @ jacp.T + _JAC_DAMPING * np.eye(3)
        return jacp.T @ np.linalg.solve(jjt, delta_pos_world)

    def _virtual_peg_tip_and_jac(self) -> tuple[np.ndarray, np.ndarray]:
        sd = self._shadow_data
        for name, value in self._virtual_qpos.items():
            sd.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        ee_pos = sd.xpos[self._right_ee_body_id]
        R = sd.xmat[self._right_ee_body_id].reshape(3, 3)
        peg_tip = ee_pos + R @ _PEG_LOCAL_OFFSET + R @ np.array([0, 0, -0.04])
        mujoco.mj_jac(self.model, sd, self._jacp, self._jacr, peg_tip, self._right_ee_body_id)
        return peg_tip, self._jacp

    def _advance_virtual(self, delta_pos_world: np.ndarray) -> None:
        """원본(peg_in_hole_bimanual_openarm_sim.py)의 _advance_virtual()과
        100% 동일한 로직(resolved-rate + nullspace + 관절 한계 동결) --
        그 파일 docstring에 실측 근거가 있다."""
        _, jacp = self._virtual_peg_tip_and_jac()

        def _solve(J: np.ndarray) -> np.ndarray:
            jjt = J @ J.T + _JAC_DAMPING * np.eye(3)
            return J.T @ np.linalg.inv(jjt)

        jacp_pinv0 = _solve(jacp)
        dq_task0 = jacp_pinv0 @ delta_pos_world
        jacp_frozen = jacp.copy()
        frozen_dofs = []
        for name in _ARM_JOINTS:
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
        for name in _ARM_JOINTS:
            dof = self._arm_dofadr[name]
            dq_null[dof] = _NULLSPACE_GAIN * (_HOME_QPOS[name] - self._virtual_qpos[name])
        dq = dq_task + null_proj @ dq_null
        for dof in frozen_dofs:
            dq[dof] = 0.0

        for name in _ARM_JOINTS:
            dof = self._arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            self._virtual_qpos[name] = float(np.clip(self._virtual_qpos[name] + dq[dof], lo, hi))

    def _sync_peg_to_arm_fk(self) -> None:
        ee_pos = self.data.xpos[self._right_ee_body_id]
        ee_quat = self.data.xquat[self._right_ee_body_id]
        R = self.data.xmat[self._right_ee_body_id].reshape(3, 3)
        peg_pos = ee_pos + R @ _PEG_LOCAL_OFFSET
        qadr = self._peg_qposadr
        self.data.qpos[qadr : qadr + 3] = peg_pos
        self.data.qpos[qadr + 3 : qadr + 7] = ee_quat

    def _solve_initial_pose(self, target_pos_world: np.ndarray) -> None:
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        self._sync_peg_to_arm_fk()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(_IK_MAX_ITERS):
            current = self.data.site_xpos[self._peg_tip_site_id]
            err = target_pos_world - current
            if np.linalg.norm(err) < 1e-5:
                break
            dq = self._jac_solve(err * _IK_STEP_SCALE)
            for name in _ARM_JOINTS:
                dof = self._arm_dofadr[name]
                qadr = self._arm_qposadr[name]
                lo, hi = self.model.jnt_range[self.model.joint(name).id]
                self.data.qpos[qadr] = np.clip(self.data.qpos[qadr] + dq[dof], lo, hi)
            mujoco.mj_forward(self.model, self.data)
            self._sync_peg_to_arm_fk()
            mujoco.mj_forward(self.model, self.data)

    def reset(self, scene_config: dict[str, Any]) -> float:
        mujoco.mj_resetData(self.model, self.data)

        for geom_id in self._peg_geom_ids:
            self.model.geom_friction[geom_id][0] = scene_config["friction"]

        outer_half = self._apply_clearance(scene_config["clearance_m"])

        for name, value in _LEFT_ARM_HOME_QPOS.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
            self.data.ctrl[self._left_arm_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._left_finger2_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = _LEFT_GRIPPER_GRASP_CTRL
        qadr = self._socket_qposadr
        self.data.qpos[qadr : qadr + 7] = self._socket_home_qpos
        mujoco.mj_forward(self.model, self.data)

        socket_center = self.data.site_xpos[self._socket_site_id].copy()

        target_pos = np.array(
            [
                socket_center[0] + scene_config["peg_init_offset_xy"][0],
                socket_center[1] + scene_config["peg_init_offset_xy"][1],
                socket_center[2] + _HOVER_GAP_M,
            ]
        )
        self._solve_initial_pose(target_pos)
        self._virtual_qpos = {name: float(self.data.qpos[self._arm_qposadr[name]]) for name in _ARM_JOINTS}

        self.data.qpos[self._right_finger_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._right_finger2_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._gripper_actuator_id] = _RIGHT_GRIPPER_GRASP_CTRL

        mujoco.mj_forward(self.model, self.data)
        return outer_half

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        site_rot = self.data.site_xmat[self._peg_tip_site_id].reshape(3, 3)
        force_local = self.data.sensordata[self._force_slice]
        torque_local = self.data.sensordata[self._torque_slice]
        return site_rot @ force_local, site_rot @ torque_local

    def get_ee_pose(self) -> np.ndarray:
        pos = self.data.site_xpos[self._peg_tip_site_id]
        joint7 = self.data.qpos[self._arm_qposadr["openarm_right_joint7"]]
        return np.array([pos[0], pos[1], pos[2], joint7])

    def get_peg_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._peg_tip_site_id].copy()

    def get_socket_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._socket_site_id].copy()

    def get_pin_top_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._pin_top_site_id].copy()

    def get_left_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._left_ee_body_id].copy()

    def get_stage(self) -> int:
        """gym-aloha InsertionTask.get_reward()를 이 설계(weld-grasp,
        그리퍼-물체 접촉 제외)에 맞게 적용한 접촉쌍 기반 단계 판정 -- 이
        파일 상단 docstring "보상 설계" 절 참고. 2(바닥, 항상 참) ->
        3(peg가 socket 벽/바닥에 접촉) -> 4(peg가 pin에 접촉)."""
        stage = _WELD_GRASP_BASE_STAGE
        touch_socket = False
        touch_pin = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            pair = {g1, g2}
            if pair & self._peg_geom_id_set:
                other = g2 if g1 in self._peg_geom_id_set else g1
                if other in self._socket_wall_geom_id_set:
                    touch_socket = True
                if other in self._pin_geom_id_set:
                    touch_pin = True
        if touch_socket:
            stage = 3
        if touch_pin:
            stage = 4
        return stage

    def step(self, delta_pos_world: np.ndarray) -> None:
        """원본 step()과 100% 동일(관절별 computed-torque, 그 파일 docstring 참고)."""
        self._advance_virtual(delta_pos_world)
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


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    sim = BimanualInsertionOpenArmSim()
    cfg = _default_scene_config()
    outer_half = sim.reset(cfg)
    print(f"[insertion] reset OK, outer_half={outer_half:.4f}")

    # peg_in_hole에서 실측 검증된 게인(Kp_xy=0.124630, Kd_xy=0.001125)을
    # 1차 추정치로 그대로 써본다 -- 팔 기구학/home 자세/grasp anchor가
    # 전부 동일하므로 재사용 가능할 것으로 보고 시작(실측으로 확인).
    kp_xy, kd_xy = 0.124630, 0.001125
    target_depth = cfg["target_insertion_depth"]

    prev_dx, prev_dy = 0.0, 0.0
    max_stage = 2
    for step_count in range(1, MAX_STEPS + 1):
        peg_tip = sim.get_peg_tip_pos()
        socket_center = sim.get_socket_center_pos()
        dx = float(socket_center[0] - peg_tip[0])
        dy = float(socket_center[1] - peg_tip[1])
        d_dx = (dx - prev_dx) / DT
        d_dy = (dy - prev_dy) / DT
        prev_dx, prev_dy = dx, dy
        delta_xy = np.array([kp_xy * dx + kd_xy * d_dx, kp_xy * dy + kd_xy * d_dy])

        raw_depth = max(0.0, float(socket_center[2] - peg_tip[2]))
        z_rate = adaptive_z_rate(dx, dy, outer_half, raw_depth, target_depth)
        delta = np.array([delta_xy[0], delta_xy[1], -z_rate])
        sim.step(delta)

        stage = sim.get_stage()
        max_stage = max(max_stage, stage)
        if step_count % 100 == 0 or stage >= 4:
            print(
                f"step={step_count:4d} dx={dx * 1000:+6.2f}mm dy={dy * 1000:+6.2f}mm "
                f"raw_depth={raw_depth * 1000:5.1f}mm stage={stage} max_stage={max_stage}"
            )
        if stage >= 4:
            print(f"[insertion] pin 접촉(stage=4) 도달, step={step_count}")
            break
    else:
        print(f"[insertion] {MAX_STEPS}스텝 종료, 도달한 최고 단계={max_stage} (4=성공)")
