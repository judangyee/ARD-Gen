"""insertion 태스크(OpenArm 양팔 버전) -- gym-aloha(huggingface)의
AlohaInsertionTask 포팅. sim/peg_in_hole_bimanual_openarm_sim.py와 같은
sim/openarm_bimanual_base.py:OpenArmBimanualBase를 쓰지만, 삽입 방향이
수직 낙하가 아니라 **수평 슬라이드**다(오른팔이 peg를, 왼팔이 socket을
쥐고 서로 다가가며 꽂는다 -- Aloha 원본 구조, assets/insertion_bimanual_openarm.xml
docstring 참고, 사용자 승인된 설계).

## 왜 peg_in_hole의 control loop을 거의 그대로 재사용할 수 있었는가

peg_in_hole의 admittance 루프는 "2개의 피드백 정렬 축(xy) + 1개의 일정/가변
속도 삽입 축(z)"이라는 구조인데, 그 세 축이 전부 **world 축에 고정**돼
있다고 가정했다(hole_socket이 world 기준 무회전이라 local z=world z였기
때문). 이 태스크는 socket을 90도 돌려 쥐어서(위 asset 파일 docstring 참고)
local z축이 world +y 근처를 향하게 만들었을 뿐, "2 피드백 + 1 삽입축"이라는
제어 구조 자체는 동일하다 -- 그래서 world 축 대신 **socket의 실제 local
축(x_hat/y_hat/z_hat, reset() 시점에 한 번 측정해서 에피소드 내내 고정)**에
벡터를 투영하도록만 일반화하면(아래 _run_episode_with_sim), adaptive_z_rate()
를 포함한 나머지 수식은 바이트 단위로 peg_in_hole과 동일하게 쓸 수 있었다.

## 성공 판정: Aloha 원본의 "순간 접촉"이지만, 3-4단계만 포팅했다

Aloha 원본 reward(0~4)는 1~2단계가 "그리퍼가 물체를 접촉 중"(진짜 손끝
접촉 physics)에 기반하는데, 이 저장소의 양팔 OpenArm 태스크들(peg_in_hole
포함)은 전부 weld로 "쥔 상태"를 고정하고 그리퍼-물체 접촉 자체를 꺼둔다
(assets 파일의 <contact><exclude> 참고, 손가락이 실제로 뭔가를 쥐는 접촉
물리를 구현하지 않음 -- peg_in_hole_openarm_env.py 모듈 docstring의
"VX300s 버전과의 근본적 차이" 절과 같은 이유). 그래서 1~2단계는 이
아키텍처에 적용할 수 없어 포팅하지 않았고, 3단계(peg가 socket 벽에 접촉)와
4단계(pin 접촉 = 성공)만 남겼다 -- 4단계는 Aloha 원본과 동일하게 **hold
없이 한 번이라도 접촉하면 즉시 성공**이다(사용자가 명시적으로 "Aloha 원본
방식"을 선택, peg_in_hole의 _SUCCESS_HOLD_STEPS 컨벤션과 다른 점).

## 현재 상태: 성공하는 게인을 아직 못 찾았다 (정직하게 기록, README
## "검증됨 vs 미검증" 컨벤션)

reset()은 잘 수렴한다(IK 반복이 목표에 도달, 충돌 0개, 관절 한계도 15%
여유 안쪽 -- 아래 smoke test 참고) -- 문제는 **step() 루프가 삽입 방향으로
전혀 전진하지 못하고 xy로 발산한다는 것**이다. 실측(admittance 게인
kp=0.124630/kd=0.001125, peg_in_hole의 알려진 좋은 값 그대로 시도):
1000스텝 동안 socket 벽에는 닿지만(peg_touching_socket=True) pin까지는
못 간다.

진단 과정(peg_in_hole과 같은 "실측하고 원인 하나씩 격리" 방식):
1. admittance(xy 피드백)를 완전히 꺼도(kp=kd=0, z_rate만으로 순수 삽입축
   push) 똑같이 발산한다(1000스텝에 dx=84mm, dy=-82mm -- socket 채널
   반폭 outer_half=19.5mm의 4배 넘게 벗어남) -- **게인 문제가 아니다.**
2. hover_gap을 39mm/25mm/15mm/8mm로 줄여서 이동 거리 자체를 줄여봐도
   발산 패턴이 그대로다(오히려 더 심해짐, hover_gap=8mm에서 dx=179mm/
   dy=125mm) -- 이동 거리를 줄이는 것도 해법이 아니다.
3. reset() 직후 peg tip에서의 Jacobian 조건수는 peg_in_hole의 reset()
   직후와 거의 동일하다(둘 다 ~3.38, 특이점 문제 아님) -- **정적
   조건수는 문제가 아니다.**
4. 순수 z_rate push만 1000스텝 돌리면 오른팔 joint1이 하한 근처(frac
   0.083, _LIMIT_FREEZE_MARGIN=0.08 바로 위), joint2가 상한 근처(frac
   0.922, 1-0.08=0.92 바로 위)까지 밀린다 -- **peg_in_hole이 CMA-ES로
   찾은 이 home 자세는 "수직으로 내려가는" 동작에는 관절 한계 여유가
   충분하지만, "이 삽입축(socket local -z, 대략 world +y) 방향으로
   움직이는" 동작에는 금방 joint1/2를 한계로 밀어붙인다**는 뜻으로
   보인다 -- peg_in_hole 자신의 docstring("관절 한계 회피" 절)이 이미
   경고한 것과 같은 실패 양상이, 이번엔 다른 motion axis에서 재현된
   것.

다음에 이어서 풀 사람에게: peg_in_hole의 home 자세(값 재사용, 위 asset
파일 docstring "4" 참고)를 그대로 쓰는 게 바로 이 문제의 원인일 가능성이
크다 -- `optimize/peg_in_hole_openarm_home_pose_search.py`와 같은
CMA-ES 탐색을, 비용 함수를 "이 삽입축 방향으로 움직일 때의 xy 누출"로
바꿔서 이 태스크 전용으로 다시 돌리는 게 다음 단계로 보인다(사용자 지시
"성공 못 하면 억지로 넘어가지 말고 어느 단계에서 막혔는지 보고" 따라,
여기서 더 밀어붙이지 않고 멈춘다).
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.openarm_bimanual_base import OpenArmBimanualBase
from sim.peg_in_hole_bimanual_openarm_sim import adaptive_z_rate

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "insertion_bimanual_openarm.xml")

N_SUBSTEPS = 5
DT = N_SUBSTEPS * 0.002
MAX_STEPS = 1000

# peg_in_hole의 _HOME_QPOS/_LEFT_ARM_HOME_QPOS와 완전히 동일한 값(이 파일
# docstring, assets 파일 docstring "4. 왜 home 자세/grasp anchor를
# 재탐색하지 않았는가" 참고) -- 두 모듈을 서로 import하지 않고 독립적으로
# 유지하기 위해(한쪽을 고치다 다른 쪽이 깨지는 걸 막으려고) 값을 복사했다.
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
_RIGHT_GRIPPER_GRASP_CTRL = -0.030
_LEFT_GRIPPER_GRASP_CTRL = 0.110
_PEG_LOCAL_OFFSET = np.array([-0.02222, 0, -0.21214])
_PEG_TRACK_POINT_OFFSET = _PEG_LOCAL_OFFSET + np.array([0.0, 0.0, -0.04])

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 200
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.03
_LIMIT_FREEZE_MARGIN = 0.08
_TORQUE_OMEGA_N = 40.0
_TORQUE_ZETA = 1.0

# socket_grasp weld의 relpose (assets/insertion_bimanual_openarm.xml과 동일한
# 값 -- socket의 "쉬는 자세"(_socket_home_qpos, __init__에서 한 번 FK로
# 계산)를 구할 때 다시 쓴다).
_SOCKET_GRASP_RELPOSE_POS = np.array([-0.01772, -0.00214, -0.19274])
_SOCKET_GRASP_RELPOSE_QUAT = np.array([0.11444174, 0.82575526, 0.54645194, -0.08013495])

Z_RATE = 0.0006
_Z_GATE_MIN_FRACTION = 0.15
_Z_GATE_RADIUS_MULT = 2.0
_OVERSHOOT_DEPTH_MULT = 1.5

TARGET_INSERTION_DEPTH = 0.035  # m, socket local +z축 기준 pin 근방 깊이(아래 pin 위치 참고)

_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
_NOMINAL_CLEARANCE_M = 0.003  # ARD-Gen 타이트 컨벤션(2-5mm) -- Aloha 원본(~3.6cm)이 아님, 사용자 승인됨.

_HOVER_GAP_M = 0.039


def default_scene_config() -> dict[str, Any]:
    return {
        "friction": 0.5,
        "clearance_m": _NOMINAL_CLEARANCE_M,
        "peg_init_offset_xy": (0.0, 0.0),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    radius = rng.uniform(0.009, 0.014)
    return {
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": (radius * np.cos(angle), radius * np.sin(angle)),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class BimanualInsertionOpenArmSim(OpenArmBimanualBase):
    def __init__(self, xml_path: str | None = None):
        xml_path = xml_path or _DEFAULT_XML
        super().__init__(
            xml_path=xml_path,
            right_home_qpos=_HOME_QPOS,
            left_home_qpos=_LEFT_ARM_HOME_QPOS,
            right_anchor_offset_ee=_PEG_LOCAL_OFFSET,
            right_track_point_offset_ee=_PEG_TRACK_POINT_OFFSET,
            right_held_free_joint="peg_free",
            nullspace_gain=_NULLSPACE_GAIN,
            limit_freeze_margin=_LIMIT_FREEZE_MARGIN,
            jac_damping=_JAC_DAMPING,
            ik_max_iters=_IK_MAX_ITERS,
            ik_step_scale=_IK_STEP_SCALE,
            torque_omega_n=_TORQUE_OMEGA_N,
            torque_zeta=_TORQUE_ZETA,
            n_substeps=N_SUBSTEPS,
        )

        self._socket_body_id = self.model.body("socket").id
        self._peg_body_id = self.model.body("peg").id
        self._socket_site_id = self.model.site("socket_entry_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]
        self._peg_tip_ball_geom_id = self.model.geom("peg_tip_ball").id
        self._pin_geom_id = self.model.geom("pin").id

        self._wall_geom_ids = {
            name: self.model.geom(f"socket_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("socket_floor").id
        self._socket_wall_and_floor_geom_ids = set(self._wall_geom_ids.values()) | {self._floor_geom_id}

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        self._socket_qposadr = self.model.joint("socket_free").qposadr[0]
        self._socket_home_qpos = self._fk_socket_rest_pose()

        # reset()에서 매 에피소드 한 번 측정해서 고정(socket이 안 움직이므로
        # episode 내내 유효) -- 모듈 docstring "왜 peg_in_hole의 control
        # loop을 거의 그대로 재사용할 수 있었는가" 참고.
        self._socket_x_hat = np.array([1.0, 0.0, 0.0])
        self._socket_y_hat = np.array([0.0, 1.0, 0.0])
        self._socket_z_hat = np.array([0.0, 0.0, 1.0])

    # ------------------------------------------------------------------
    def _fk_socket_rest_pose(self) -> np.ndarray:
        """socket_grasp weld가 왼팔 home 자세에서 만족된 상태의 socket free
        joint qpos(7)을 FK로 직접 계산한다(peg_in_hole이 "home" 키프레임에서
        읽던 것과 같은 역할이지만, 이 파일은 keyframe을 안 쓰므로 직접
        계산한다 -- relpose_quat이 항등원이 아니라서(채널을 90도 돌렸으므로)
        quat도 합성해야 한다, _sync_right_held_to_fk()는 relpose_quat=항등원을
        가정하므로 여기 재사용 못 함)."""
        for name, value in self.left_home_qpos.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self._left_ee_body_id].copy()
        ee_quat = self.data.xquat[self._left_ee_body_id].copy()
        R = self.data.xmat[self._left_ee_body_id].reshape(3, 3)
        socket_pos = ee_pos + R @ _SOCKET_GRASP_RELPOSE_POS
        socket_quat = np.zeros(4)
        mujoco.mju_mulQuat(socket_quat, ee_quat, _SOCKET_GRASP_RELPOSE_QUAT)
        mujoco.mj_resetData(self.model, self.data)
        return np.concatenate([socket_pos, socket_quat])

    def _apply_clearance(self, clearance_m: float) -> float:
        """peg_in_hole._apply_clearance()와 완전히 동일한 공식(ARD-Gen 타이트
        컨벤션, 위 모듈 docstring 참고) -- 벽 geom 이름만 socket_wall_*."""
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

    def reset(self, scene_config: dict[str, Any]) -> float:
        mujoco.mj_resetData(self.model, self.data)

        for geom_id in self._peg_geom_ids:
            self.model.geom_friction[geom_id][0] = scene_config["friction"]

        outer_half = self._apply_clearance(scene_config["clearance_m"])

        self.reset_left_arm_fixed_hold(
            _LEFT_GRIPPER_GRASP_CTRL,
            left_held_free_joint="socket_free",
            left_held_home_qpos=self._socket_home_qpos,
        )
        mujoco.mj_forward(self.model, self.data)

        socket_center = self.data.site_xpos[self._socket_site_id].copy()
        R_socket = self.data.xmat[self._socket_body_id].reshape(3, 3).copy()
        self._socket_x_hat = R_socket[:, 0]
        self._socket_y_hat = R_socket[:, 1]
        self._socket_z_hat = R_socket[:, 2]

        target_pos = (
            socket_center
            + _HOVER_GAP_M * self._socket_z_hat
            + scene_config["peg_init_offset_xy"][0] * self._socket_x_hat
            + scene_config["peg_init_offset_xy"][1] * self._socket_y_hat
        )
        self.solve_right_initial_pose(target_pos, lambda: self.data.site_xpos[self._peg_tip_site_id])
        self.capture_virtual_qpos_from_data()

        self.set_right_gripper(_RIGHT_GRIPPER_GRASP_CTRL)

        mujoco.mj_forward(self.model, self.data)
        return outer_half

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        site_rot = self.data.site_xmat[self._peg_tip_site_id].reshape(3, 3)
        force_local = self.data.sensordata[self._force_slice]
        torque_local = self.data.sensordata[self._torque_slice]
        return site_rot @ force_local, site_rot @ torque_local

    def get_ee_pose(self) -> np.ndarray:
        pos = self.data.site_xpos[self._peg_tip_site_id]
        joint7 = self.data.qpos[self._right_arm_qposadr["openarm_right_joint7"]]
        return np.array([pos[0], pos[1], pos[2], joint7])

    def get_peg_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._peg_tip_site_id].copy()

    def get_socket_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._socket_site_id].copy()

    def get_socket_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """socket의 local (x,y,z)축을 world 프레임으로 표현한 단위벡터 --
        reset()에서 한 번 측정해 고정(socket이 안 움직이므로 episode 내내
        유효, 모듈 docstring 참고)."""
        return self._socket_x_hat, self._socket_y_hat, self._socket_z_hat

    def _geom_pair_touching(self, geom_id_a: int, geom_id_b: int) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {c.geom1, c.geom2} == {geom_id_a, geom_id_b}:
                return True
        return False

    def is_pin_touched(self) -> bool:
        """Aloha 원본의 ("red_peg","pin") 접촉 포팅 -- hold 없이 한 번이라도
        닿으면 True(사용자 승인된 "원본 방식(순간 접촉)")."""
        return self._geom_pair_touching(self._peg_tip_ball_geom_id, self._pin_geom_id)

    def is_peg_touching_socket(self) -> bool:
        """Aloha 3단계(peg_touch_socket) 포팅 -- peg가 socket 벽/바닥 중
        아무거나에 닿았는지(그리퍼-물체 접촉은 이 아키텍처에 없으므로 1~2
        단계는 포팅 안 함, 모듈 docstring 참고)."""
        for geom_id in self._peg_geom_ids:
            for wall_id in self._socket_wall_and_floor_geom_ids:
                if self._geom_pair_touching(geom_id, wall_id):
                    return True
        return False


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = BimanualInsertionOpenArmSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: BimanualInsertionOpenArmSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    """peg_in_hole._run_episode_with_sim()과 같은 구조("2 피드백 축 + 1 삽입
    축" admittance), 다만 world x/y/z 대신 socket의 실제 local x_hat/y_hat/
    z_hat에 투영한다(모듈 docstring 참고) -- adaptive_z_rate()는 스칼라만
    받아서 바이트 단위로 재사용."""
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]
    x_hat, y_hat, z_hat = sim.get_socket_axes()

    kp_xy = float(gains["Kp_xy"])
    kd_xy = float(gains["Kd_xy"])

    ee_poses = [sim.get_ee_pose()]
    actions = []
    forces = []
    torques = []

    prev_dx, prev_dy = 0.0, 0.0
    max_force_mag = 0.0
    raw_depth = 0.0
    pin_touched = False
    peg_touching_socket = False
    step_count = 0

    for step_count in range(1, MAX_STEPS + 1):
        force, torque = sim.get_force_torque()
        max_force_mag = max(max_force_mag, float(np.linalg.norm(force)))

        peg_tip = sim.get_peg_tip_pos()
        socket_center = sim.get_socket_center_pos()
        err = socket_center - peg_tip
        cur_dx = float(np.dot(err, x_hat))
        cur_dy = float(np.dot(err, y_hat))
        d_dx = (cur_dx - prev_dx) / DT
        d_dy = (cur_dy - prev_dy) / DT
        prev_dx, prev_dy = cur_dx, cur_dy
        delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

        cur_raw_depth = max(0.0, float(np.dot(err, z_hat)))
        z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
        delta = delta_xy[0] * x_hat + delta_xy[1] * y_hat - z_rate * z_hat

        sim.step(delta)

        actions.append(delta.copy())
        forces.append(force)
        torques.append(torque)
        ee_poses.append(sim.get_ee_pose())

        peg_tip = sim.get_peg_tip_pos()
        socket_center = sim.get_socket_center_pos()
        raw_depth = max(0.0, float(np.dot(socket_center - peg_tip, z_hat)))

        if sim.is_peg_touching_socket():
            peg_touching_socket = True
        if sim.is_pin_touched():
            pin_touched = True
            break

    final_peg_tip = sim.get_peg_tip_pos()
    final_socket_center = sim.get_socket_center_pos()
    dist = float(np.linalg.norm(final_socket_center - final_peg_tip))

    success = pin_touched
    reward = (
        -2.0 * dist
        + 20.0 * raw_depth
        - 0.001 * max(0.0, max_force_mag - 5.0)
        - 0.01 * step_count
    )
    if peg_touching_socket:
        reward += 5.0
    if success:
        reward += 50.0

    return {
        "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
        "ee_poses": np.stack(ee_poses).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "force_profile": np.stack(forces).astype(np.float32),
        "torque_profile": np.stack(torques).astype(np.float32),
        "forces": np.stack(forces).astype(np.float32),
        "torques": np.stack(torques).astype(np.float32),
        "insertion_depth": float(raw_depth),
        "final_distance": dist,
        "max_force": max_force_mag,
        "step_count": step_count,
        "peg_touching_socket": peg_touching_socket,
        "pin_touched": pin_touched,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }


class InsertionOpenArmEnv(BaseTaskEnv):
    def __init__(
        self,
        xml_path: str | None = None,
        stabilizer_config: dict[str, Any] | None = None,
        use_stabilizer: bool = True,
    ):
        self._sim = BimanualInsertionOpenArmSim(xml_path=xml_path)
        self._outer_half = 0.0
        self._target_depth = 0.0

    def reset(self, scene_config: dict[str, Any]) -> float:
        self._outer_half = self._sim.reset(scene_config)
        self._target_depth = scene_config["target_insertion_depth"]
        return self._outer_half

    def step(self, action: np.ndarray) -> None:
        self._sim.step(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        reward = (
            -2.0 * episode_result["final_distance"]
            + 20.0 * episode_result["insertion_depth"]
            - 0.001 * max(0.0, episode_result["max_force"] - 5.0)
            - 0.01 * episode_result["step_count"]
        )
        if episode_result.get("peg_touching_socket"):
            reward += 5.0
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return bool(episode_result.get("pin_touched"))

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        sim = self._sim
        outer_half = self.reset(scene_config)
        target_depth = scene_config["target_insertion_depth"]
        x_hat, y_hat, z_hat = sim.get_socket_axes()

        kp_xy = float(gains["Kp_xy"])
        kd_xy = float(gains["Kd_xy"])

        left_arm_traj = [sim.get_left_ee_pos()]
        ee_poses = [sim.get_ee_pose()]
        actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []

        prev_dx, prev_dy = 0.0, 0.0
        max_force_mag = 0.0
        raw_depth = 0.0
        pin_touched = False
        peg_touching_socket = False
        step_count = 0

        for step_count in range(1, MAX_STEPS + 1):
            force, torque = sim.get_force_torque()
            max_force_mag = max(max_force_mag, float(np.linalg.norm(force)))

            peg_tip = sim.get_peg_tip_pos()
            socket_center = sim.get_socket_center_pos()
            err = socket_center - peg_tip
            cur_dx = float(np.dot(err, x_hat))
            cur_dy = float(np.dot(err, y_hat))
            d_dx = (cur_dx - prev_dx) / DT
            d_dy = (cur_dy - prev_dy) / DT
            prev_dx, prev_dy = cur_dx, cur_dy
            delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

            cur_raw_depth = max(0.0, float(np.dot(err, z_hat)))
            z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
            delta = delta_xy[0] * x_hat + delta_xy[1] * y_hat - z_rate * z_hat

            self.step(delta)
            left_arm_traj.append(sim.get_left_ee_pos())

            actions.append(delta.copy())
            forces.append(force)
            torques.append(torque)
            ee_poses.append(sim.get_ee_pose())

            peg_tip = sim.get_peg_tip_pos()
            socket_center = sim.get_socket_center_pos()
            raw_depth = max(0.0, float(np.dot(socket_center - peg_tip, z_hat)))

            if sim.is_peg_touching_socket():
                peg_touching_socket = True
            if sim.is_pin_touched():
                pin_touched = True
                break

        final_peg_tip = sim.get_peg_tip_pos()
        final_socket_center = sim.get_socket_center_pos()
        dist = float(np.linalg.norm(final_socket_center - final_peg_tip))

        episode_result = {
            "insertion_depth": float(raw_depth),
            "final_distance": dist,
            "max_force": max_force_mag,
            "step_count": step_count,
            "peg_touching_socket": peg_touching_socket,
            "pin_touched": pin_touched,
            "success": pin_touched,
        }
        reward = self.compute_reward(episode_result)

        return {
            "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
            "ee_poses": np.stack(ee_poses).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "force_profile": np.stack(forces).astype(np.float32),
            "torque_profile": np.stack(torques).astype(np.float32),
            "forces": np.stack(forces).astype(np.float32),
            "torques": np.stack(torques).astype(np.float32),
            "insertion_depth": float(raw_depth),
            "final_distance": dist,
            "max_force": max_force_mag,
            "step_count": step_count,
            "peg_touching_socket": peg_touching_socket,
            "pin_touched": pin_touched,
            "success": pin_touched,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "left_arm_traj": np.stack(left_arm_traj).astype(np.float32),
        }


if __name__ == "__main__":
    gains = {"Kp_xy": 0.124630, "Kd_xy": 0.001125}
    cfg = default_scene_config()
    result = run_episode(gains, cfg)
    print(
        f"[insertion_openarm] success={result['success']} peg_touching_socket={result['peg_touching_socket']} "
        f"insertion_depth={result['insertion_depth']:.4f}m max_force={result['max_force']:.2f}N "
        f"step_count={result['step_count']} reward={result['reward']:.2f}"
    )
