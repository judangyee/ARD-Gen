"""tacker(타카) 태스크, OpenArm 양팔 버전. "위치 정확도 + 1회성 발사".

일반 Stabilizer(3-슬라이드 가상 EE, sim/stabilizer.py) 버전(sim/tacker_env.py,
assets/tacker.xml)은 peg_in_hole에서 이미 OpenArm으로 교체되면서 레거시가 된
패턴이라 여기서는 다시 쓰지 않는다(사용자 지시) -- 대신 sim/
peg_in_hole_bimanual_openarm_sim.py + sim/peg_in_hole_openarm_env.py의 구조를
그대로 잇는다:

  - **왼팔**: 실제 OpenArm 7-DOF 팔이 workpiece를 직접 쥔다. 팔 기구학/게인
    (pos_left_j* 강화 위치 게인, gravcomp 오버라이드, vendor 액추에이터
    무력화)은 assets/tacker_openarm.xml + 이 파일에서 peg_in_hole_bimanual_openarm_sim.py
    와 완전히 동일한 값을 그대로 재사용한다 -- 새로 설계하지 않았다.
  - **오른팔**: 실제 OpenArm 7-DOF 팔이 tacker_tool을 쥐고 목표(nail_site)로
    접근한다. resolved-rate Jacobian IK(널스페이스 투영 + 관절 한계 회피) +
    computed-torque(매 스텝 mj_fullM으로 그 순간의 관성/코리올리를 상쇄)
    구조도 peg_in_hole_bimanual_openarm_sim.py와 완전히 동일하게 재사용한다
    -- 바뀐 건 **그 위에 얹히는 태스크 제어 루프뿐**이다: peg_in_hole은
    xy admittance PD + 적응형 z-rate로 "삽입 깊이"를 목표했지만, 여기서는
    admittance(접촉힘 피드백) 자체가 필요 없다 -- 접근 중에는 접촉이
    없고, 목표는 "정확한 위치 도달"뿐이라서 순수 위치오차 비례 제어
    (Kp_approach) 하나로 충분하다(일반 Stabilizer 버전 tacker와 같은 설계
    철학, PIPELINE.md 참고).

## grasp anchor/home 자세를 그대로 재사용한 이유 (기하학적으로 무관하기 때문)

peg_in_hole_bimanual_openarm.xml의 hole_grasp/peg_grasp weld relpose와
_HOME_QPOS/_LEFT_ARM_HOME_QPOS는 "왼팔이 어떤 각도로 뭔가를 쥔 채 오른팔이
그 위 6cm(_HOVER_GAP_M)에서 접근한다"는 순수 기하학적 관계만 인코딩한다 --
쥐는 대상의 형상(hole vs 평평한 workpiece)이나 오른팔이 쥔 도구의 용도(peg
삽입 vs 타카 발사)와 무관하게 유효하다. 그래서 assets/tacker_openarm.xml은
그 숫자들을 그대로 물려받았고(물체/도구의 지오metry와 이름만 바꿈), 이
파일도 _TOOL_LOCAL_OFFSET/_HOME_QPOS/_LEFT_ARM_HOME_QPOS 등을
peg_in_hole_bimanual_openarm_sim.py와 동일한 값으로 유지한다.

## 발사 트리거/반동: 일반 Stabilizer 버전 tacker와 동일한 설계

- EE(tacker_tip_site)와 nail_site 거리가 FIRE_TOLERANCE_M(3mm) 이내로
  들어오는 스텝에서 자동 발사(에피소드당 정확히 1회).
- 발사 순간 workpiece의 freejoint 선속도에 velocity kick을 직접 더한다
  (impulse의 이상화) -- 수평 성분이 주, 수직은 작게. 발사 후 SETTLE_STEPS
  만큼 더 진행해 반동이 가라앉은 뒤의 변위로 성공을 판정한다.
- **차이점**: 구버전은 Stabilizer의 weld가 항상 켜져 있거나(grasped) 꺼져
  있는 이진 상태였지만, 여기서는 grasp weld가 애초에 컴파일 시점부터 항상
  활성(active="true", peg_in_hole_bimanual_openarm.xml과 동일)이다 -- 왼팔이
  workpiece를 "쥐는 순간"이 따로 없고 처음부터 쥐고 있다. 그래서
  run_approach_phase() 같은 접근-후-weld 로직 자체가 필요 없다.

## 왼팔 강화 게인 온/오프 비교 (구버전의 "Stabilizer 있음/없음" 비교에 대응)

구버전은 Stabilizer 클래스 자체를 붙이거나(use_stabilizer=True) 안 붙이는
(False) 방식으로 "왼팔이 버텨주는가"를 비교했다. 여기서는 왼팔이 항상
붙어 있으므로(위 참고) 같은 질문을 다르게 번역해야 한다: **왼팔의 강화
위치 게인(pos_left_j*, kp=2273.98/87.90/411.75)을 켤 때 vs 원래 OpenArm
vendor 기본 게인(motor_DM8009/DM4340/DM4310 -- assets/openarm/openarm_bimanual.xml
의 left_joint*_ctrl 액추에이터 값, kp=230/190/30)으로 되돌릴 때**로 비교한다
-- `left_arm_reinforced=False`로 생성하면 pos_left_j*의 gainprm/biasprm을
vendor 값으로 덮어쓴다(PIPELINE.md의 tacker 절에 N=150 비교 결과 기록).
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.base_task_env import BaseTaskEnv

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "tacker_openarm.xml")

N_SUBSTEPS = 5
DT = N_SUBSTEPS * 0.002

MAX_STEPS = 400  # 접근 예산(발사 전까지) -- peg_in_hole_bimanual_openarm_sim.py(MAX_STEPS=1000)보다
# 훨씬 짧다: 이 태스크는 삽입이 아니라 "한 점에 닿기"만 하면 되므로 원래도
# 훨씬 빨리 끝난다(실측: 대부분 20~60스텝, __main__ 결과 참고).
SETTLE_STEPS = 30  # 발사 후 반동이 가라앉는 걸 관찰하는 스텝 수

FIRE_TOLERANCE_M = 0.003  # 이 이내로 들어오면 자동 발사
MAX_APPROACH_STEP_M = 0.01  # 한 스텝에 허용하는 world-space 최대 이동 목표(resolved-rate 입력 클립)
_NEAR_ZONE_MULT = 4.0
_SAFE_NEAR_STEP_M = 2.0 * FIRE_TOLERANCE_M

SUCCESS_DISPLACEMENT_M = 0.006  # 발사 후 이 이내로 workpiece가 밀리면 성공

# 후퇴(retract) 단계 -- 사용자 피드백("실제 타카 작업처럼 접근->압착->발사->
# 후퇴까지 포함해야 한다") 반영. 발사+정착 이후 오른팔이 타점에서 수직으로
# 안전 거리까지 물러나는 동작을 추가한다 -- 접근 때와 같은 Kp_approach
# 비례 제어를 그대로 재사용한다(목표점만 nail_site 대신 "nail_site 위
# RETRACT_SAFE_DISTANCE_M"로 바뀔 뿐, 제어 로직은 동일).
RETRACT_SAFE_DISTANCE_M = 0.05  # workpiece 표면(nail_site)에서 이만큼 위로 물러나면 "후퇴 완료"
RETRACT_TOLERANCE_M = 0.005  # 후퇴 목표점에 이 이내로 들어오면 완료로 인정(발사 tolerance보다 느슨 -- 정확한 위치가 중요하지 않아서)
RETRACT_MAX_STEPS = 100  # 후퇴 예산(접근과 비슷한 거리라 접근보다 넉넉하게)
FINAL_SETTLE_STEPS = 20  # 왼팔 게인을 낮춘 뒤(release_grip_after 참고) 그 영향을 관찰하는 추가 정착 스텝

# recoil_strength 스케일이 일반 Stabilizer 버전 tacker(0.15~0.45)보다 훨씬
# 큰 이유: 그 버전의 왼팔은 순수 3-슬라이드 가상 EE(kp=20000, 관성 0.1kg)라
# 작은 velocity kick에도 즉각 큰 반응이 나왔지만, 여기서는 진짜 7-DOF
# 팔 전체의 분산된 관성/댐핑이 훨씬 크게 개입해서 같은 displacement를
# 만들려면 수십 배 큰 kick이 필요하다(실측 스윕: reinforced 게인 기준
# recoil_strength<=15에서는 변위<4mm로 항상 성공, 25 이상부터 threshold를
# 넘기 시작해 130에서 238mm까지 커진다 -- __main__ 결과 및 PIPELINE.md
# tacker 절 참고). 이 스윕 결과를 보고 sample_scene_config()의 범위(5~35)를
# "reinforced 게인으로도 성공/실패가 실제로 갈리는" 구간으로 잡았다.

_HOVER_GAP_M = 0.039  # peg_in_hole_bimanual_openarm_sim.py와 동일(grasp anchor를 그대로 재사용하므로 값도 동일)

# 오른팔 7개 관절.
_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# peg_in_hole_bimanual_openarm_sim.py의 _HOME_QPOS/_LEFT_ARM_HOME_QPOS와
# 완전히 동일한 값(grasp anchor를 그대로 재사용하므로 home 자세도 재탐색할
# 이유가 없다 -- 모듈 docstring 참고).
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

_RIGHT_GRIPPER_GRASP_CTRL = -0.030  # peg_in_hole_bimanual_openarm_sim.py와 동일
_LEFT_GRIPPER_GRASP_CTRL = 0.110  # 〃

# tacker_tool을 오른팔 ee 프레임에서 직접 FK로 세팅할 때 쓰는 로컬 오프셋 --
# assets/tacker_openarm.xml의 tacker_grasp weld relpose와 동일한 값
# (peg_in_hole_bimanual_openarm.xml의 _PEG_LOCAL_OFFSET을 그대로 재사용).
_TOOL_LOCAL_OFFSET = np.array([-0.02222, 0, -0.21214])

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 200
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.03
_LIMIT_FREEZE_MARGIN = 0.08

_TORQUE_OMEGA_N = 40.0
_TORQUE_ZETA = 1.0

# 왼팔 강화 게인 온/오프 비교용 -- assets/openarm/openarm_bimanual.xml의
# vendor 기본 <position> 액추에이터 값(motor_DM8009/DM4340/DM4310 클래스,
# left_joint{1..7}_ctrl 정의에서 그대로 읽었다). left_arm_reinforced=False일
# 때 pos_left_j*를 이 값으로 되돌린다(모듈 docstring "왼팔 강화 게인
# 온/오프 비교" 절 참고).
_VENDOR_LEFT_GAINS = {
    "openarm_left_joint1": (230.0, 2.7),
    "openarm_left_joint2": (230.0, 2.7),
    "openarm_left_joint3": (190.0, 2.2),
    "openarm_left_joint4": (190.0, 2.2),
    "openarm_left_joint5": (30.0, 1.5),
    "openarm_left_joint6": (30.0, 1.5),
    "openarm_left_joint7": (30.0, 1.5),
}


def default_scene_config() -> dict[str, Any]:
    return {
        "nail_offset_xy": (0.0, 0.0),
        "tool_init_offset_xy": (0.03, -0.03),
        "recoil_strength": 15.0,
        "recoil_angle": 0.0,
        "success_displacement_m": SUCCESS_DISPLACEMENT_M,
    }


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """이미 sim 레벨 스키마를 그대로 뽑는다(peg_in_hole OpenArm 버전과 같은
    이유로 "1단계 공유 씬" 변환 단계가 따로 없음). "workpiece 초기 자세
    무작위화" 요청은 nail_offset_xy(조준점의 workpiece-로컬 오프셋)와
    tool_init_offset_xy(오른팔 시작 오프셋, peg_in_hole의 peg_init_offset_xy와
    동일한 역할)로 구현했다 -- 이 구조에서 workpiece의 world pose 자체는
    왼팔의 고정 자세로 결정되어 독립적으로 움직일 수 없다(모듈 docstring
    참고)."""
    if rng is None:
        rng = np.random.default_rng()

    nail_angle = rng.uniform(-np.pi, np.pi)
    nail_radius = rng.uniform(0.0, 0.015)
    approach_angle = rng.uniform(-np.pi, np.pi)
    approach_radius = rng.uniform(0.01, 0.025)

    return {
        "nail_offset_xy": (float(nail_radius * np.cos(nail_angle)), float(nail_radius * np.sin(nail_angle))),
        "tool_init_offset_xy": (
            float(approach_radius * np.cos(approach_angle)),
            float(approach_radius * np.sin(approach_angle)),
        ),
        "recoil_strength": float(rng.uniform(5.0, 35.0)),
        "recoil_angle": float(rng.uniform(-np.pi, np.pi)),
        "success_displacement_m": SUCCESS_DISPLACEMENT_M,
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class BimanualTackerOpenArmSim:
    def __init__(self, xml_path: str | None = None, left_arm_reinforced: bool = True):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._workpiece_body_id = self.model.body("workpiece").id
        self._tool_body_id = self.model.body("tacker_tool").id
        self._right_ee_body_id = self.model.body("openarm_right_ee_base_link").id
        self._left_ee_body_id = self.model.body("openarm_left_ee_base_link").id
        self._nail_site_id = self.model.site("nail_site").id
        self._tip_site_id = self.model.site("tacker_tip_site").id

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_JOINTS}
        self._gripper_actuator_id = self.model.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = self.model.joint("openarm_right_finger_joint1").qposadr[0]
        self._right_finger2_qposadr = self.model.joint("openarm_right_finger_joint2").qposadr[0]

        self._left_arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_HOME_QPOS}
        self._left_arm_actuator_ids = {
            name: self.model.actuator(f"pos_left_j{i}").id for i, name in enumerate(_LEFT_ARM_HOME_QPOS, start=1)
        }
        self._left_gripper_actuator_id = self.model.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = self.model.joint("openarm_left_finger_joint1").qposadr[0]
        self._left_finger2_qposadr = self.model.joint("openarm_left_finger_joint2").qposadr[0]

        # vendor 액추에이터(양팔 14개) 무력화 -- peg_in_hole_bimanual_openarm_sim.py
        # 와 동일한 이유/방식(모듈 상단 docstring 참고).
        for prefix, joints in (("right", _ARM_JOINTS), ("left", list(_LEFT_ARM_HOME_QPOS))):
            for i in range(1, 8):
                act_id = self.model.actuator(f"{prefix}_joint{i}_ctrl").id
                self.model.actuator_gainprm[act_id] = 0.0
                self.model.actuator_biasprm[act_id] = 0.0

        # 오른팔은 pos_right_j*(관절별 독립 PD)가 아니라 torque_right_j*
        # (역동역학, step()이 매 스텝 계산)로 구동한다 -- pos_right_j*는
        # vendor 액추에이터와 같은 이유로 무력화.
        arm_actuator_ids = {name: self.model.actuator(f"pos_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)}
        for act_id in arm_actuator_ids.values():
            self.model.actuator_gainprm[act_id] = 0.0
            self.model.actuator_biasprm[act_id] = 0.0
        self._torque_actuator_ids = {
            name: self.model.actuator(f"torque_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)
        }
        self._arm_dofs = [self._arm_dofadr[name] for name in _ARM_JOINTS]

        # 왼팔 강화 게인 온/오프 -- 모듈 docstring "왼팔 강화 게인 온/오프
        # 비교" 절 참고. XML은 항상 강화 값(kp=2273.98 등)으로 컴파일되므로,
        # False일 때만 vendor 값으로 되돌린다(True면 아무 것도 안 건드림 --
        # XML 기본값이 이미 재사용된 강화 게인이라서).
        self._left_arm_reinforced_gains = {
            name: (float(self.model.actuator_gainprm[act_id, 0]), float(-self.model.actuator_biasprm[act_id, 2]))
            for name, act_id in self._left_arm_actuator_ids.items()
        }
        if not left_arm_reinforced:
            self.set_left_arm_reinforced(False)

        # gravcomp -- peg_in_hole_bimanual_openarm_sim.py와 동일.
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                self.model.body_gravcomp[i] = 1.0

        self._tool_qposadr = self.model.joint("tacker_tool_free").qposadr[0]
        self._workpiece_qposadr = self.model.joint("workpiece_free").qposadr[0]
        self._workpiece_dofadr = self.model.joint("workpiece_free").dofadr[0]

        key_id = self.model.key("home").id
        self._workpiece_home_qpos = self.model.key_qpos[key_id][
            self._workpiece_qposadr : self._workpiece_qposadr + 7
        ].copy()
        self._nail_site_local_pos = self.model.site_pos[self._nail_site_id].copy()

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # planning은 실제 시뮬레이션 상태가 아니라 별도의 "가상" 순수 기구학
        # 상태에서 한다 -- peg_in_hole_bimanual_openarm_sim.py의
        # self._shadow_data 주석과 동일한 이유(약간 어긋난 실제 위치에서
        # Jacobian을 계산하면 오차가 누적된다).
        self._shadow_data = mujoco.MjData(self.model)
        self._virtual_qpos: dict[str, float] = dict(_HOME_QPOS)

    # ------------------------------------------------------------------
    def _jac_at_point(self, world_point: np.ndarray) -> np.ndarray:
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, world_point, self._right_ee_body_id)
        return self._jacp

    def _jac_solve(self, delta_pos_world: np.ndarray) -> np.ndarray:
        tip = self.data.site_xpos[self._tip_site_id].copy()
        jacp = self._jac_at_point(tip)
        jjt = jacp @ jacp.T + _JAC_DAMPING * np.eye(3)
        return jacp.T @ np.linalg.solve(jjt, delta_pos_world)

    def _virtual_tool_tip_and_jac(self) -> tuple[np.ndarray, np.ndarray]:
        sd = self._shadow_data
        for name, value in self._virtual_qpos.items():
            sd.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, sd)
        ee_pos = sd.xpos[self._right_ee_body_id]
        R = sd.xmat[self._right_ee_body_id].reshape(3, 3)
        tip = ee_pos + R @ _TOOL_LOCAL_OFFSET + R @ np.array([0, 0, -0.04])
        mujoco.mj_jac(self.model, sd, self._jacp, self._jacr, tip, self._right_ee_body_id)
        return tip, self._jacp

    def _advance_virtual(self, delta_pos_world: np.ndarray) -> None:
        """peg_in_hole_bimanual_openarm_sim.py의 _advance_virtual()과 완전히
        동일(널스페이스 투영 + 관절 한계 회피, 그 파일 docstring 참고) --
        코드 그대로 재사용했고 이름/변수만 이 파일의 명명에 맞췄다."""
        _, jacp = self._virtual_tool_tip_and_jac()

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

    def _sync_tool_to_arm_fk(self) -> None:
        ee_pos = self.data.xpos[self._right_ee_body_id]
        ee_quat = self.data.xquat[self._right_ee_body_id]
        R = self.data.xmat[self._right_ee_body_id].reshape(3, 3)
        tool_pos = ee_pos + R @ _TOOL_LOCAL_OFFSET
        qadr = self._tool_qposadr
        self.data.qpos[qadr : qadr + 3] = tool_pos
        self.data.qpos[qadr + 3 : qadr + 7] = ee_quat

    def _solve_initial_pose(self, target_pos_world: np.ndarray) -> None:
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        self._sync_tool_to_arm_fk()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(_IK_MAX_ITERS):
            current = self.data.site_xpos[self._tip_site_id]
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
            self._sync_tool_to_arm_fk()
            mujoco.mj_forward(self.model, self.data)

    def reset(self, scene_config: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)

        # 왼팔(고정) + workpiece: home 키프레임 값 그대로 복사(peg_in_hole_bimanual_openarm_sim.py
        # 의 reset()과 동일한 이유 -- 왼팔이 이 태스크 내내 안 움직임).
        for name, value in _LEFT_ARM_HOME_QPOS.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
            self.data.ctrl[self._left_arm_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._left_finger2_qposadr] = _LEFT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = _LEFT_GRIPPER_GRASP_CTRL
        qadr = self._workpiece_qposadr
        self.data.qpos[qadr : qadr + 7] = self._workpiece_home_qpos

        # nail_site 로컬 xy를 씬마다 무작위화한다(모듈 docstring 참고).
        nail_xy = scene_config["nail_offset_xy"]
        self.model.site_pos[self._nail_site_id] = [
            self._nail_site_local_pos[0] + nail_xy[0],
            self._nail_site_local_pos[1] + nail_xy[1],
            self._nail_site_local_pos[2],
        ]
        mujoco.mj_forward(self.model, self.data)

        nail_world = self.data.site_xpos[self._nail_site_id].copy()
        target_pos = nail_world + np.array(
            [scene_config["tool_init_offset_xy"][0], scene_config["tool_init_offset_xy"][1], _HOVER_GAP_M]
        )
        self._solve_initial_pose(target_pos)
        self._virtual_qpos = {name: float(self.data.qpos[self._arm_qposadr[name]]) for name in _ARM_JOINTS}

        self.data.qpos[self._right_finger_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.qpos[self._right_finger2_qposadr] = _RIGHT_GRIPPER_GRASP_CTRL
        self.data.ctrl[self._gripper_actuator_id] = _RIGHT_GRIPPER_GRASP_CTRL

        mujoco.mj_forward(self.model, self.data)

    def get_tool_tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site_id].copy()

    def get_nail_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._nail_site_id].copy()

    def get_workpiece_pos(self) -> np.ndarray:
        return self.data.xpos[self._workpiece_body_id].copy()

    def get_ee_pose(self) -> np.ndarray:
        return self.get_tool_tip_pos()

    def get_left_ee_pos(self) -> np.ndarray:
        """왼팔(workpiece를 쥔 채 이 태스크 내내 고정) EE 위치. sim/
        peg_in_hole_openarm_env.py의 get_left_ee_pos()와 같은 이유로
        기록한다 -- 실제로는 거의 안 움직이지만, right_arm과 길이가 같은
        left_arm_traj를 만들어 5단계 role classifier 학습용 스키마 일관성을
        유지한다."""
        return self.data.xpos[self._left_ee_body_id].copy()

    def get_force_torque(self) -> tuple[np.ndarray, np.ndarray]:
        """workpiece(freejoint)에 실제로 작용 중인 구속력 합 -- XML <sensor>
        force/torque를 안 쓰는 이유는 assets/tacker_openarm.xml 상단
        docstring 참고(freejoint에서는 구조적으로 항상 0을 반환한다는 걸
        일반 Stabilizer 버전 tacker에서 실측으로 먼저 발견했다)."""
        dofadr = self._workpiece_dofadr
        force = self.data.qfrc_constraint[dofadr : dofadr + 3].copy()
        torque = self.data.qfrc_constraint[dofadr + 3 : dofadr + 6].copy()
        return force, torque

    def set_left_arm_reinforced(self, reinforced: bool) -> None:
        """왼팔 게인을 언제든(에피소드 중간에도) 강화 값 <-> vendor 값으로
        전환한다 -- MuJoCo의 actuator_gainprm/biasprm은 매 mj_step에서 다시
        읽히므로 시뮬레이션 도중 바꿔도 즉시 반영된다. "왼팔 강화 게인을
        언제 놓아도 되는가"(release_grip_after) 실험에 쓴다."""
        for name, act_id in self._left_arm_actuator_ids.items():
            if reinforced:
                kp, kv = self._left_arm_reinforced_gains[name]
            else:
                kp, kv = _VENDOR_LEFT_GAINS[name]
            self.model.actuator_gainprm[act_id, 0] = kp
            self.model.actuator_biasprm[act_id, 1] = -kp
            self.model.actuator_biasprm[act_id, 2] = -kv

    def apply_recoil(self, kick_vector: np.ndarray) -> None:
        dofadr = self._workpiece_dofadr
        self.data.qvel[dofadr : dofadr + 3] += np.asarray(kick_vector, dtype=float)

    def step(self, delta_pos_world: np.ndarray) -> None:
        """peg_in_hole_bimanual_openarm_sim.py의 step()과 완전히 동일한
        역동역학(computed-torque) 구조 -- 그 파일 docstring 참고. 유일한
        차이는 delta_pos_world가 admittance+z-rate가 아니라 이 파일의
        run_episode()가 계산한 단순 비례 접근 항이라는 것뿐이다."""
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


class TackerOpenArmEnv(BaseTaskEnv):
    """xml_path/left_arm_reinforced만 받는다 -- stabilizer_config/use_stabilizer는
    받지 않는다(왼팔이 generic Stabilizer가 아니라 항상 붙어있는 실제 팔이라
    그 인터페이스 자체가 해당 없음, 모듈 docstring 참고). tasks/tacker.yaml에
    `stabilizer` 절이 없으므로 TaskConfig.make_env()가 애초에 그 kwarg를
    안 넘긴다.

    release_grip_after: 왼팔 강화 게인을 언제 vendor 게인으로 낮추는지 --
    "none"(기본값, 에피소드 내내 강화 게인 유지), "fire_settle"(발사+정착
    직후, 후퇴를 시작하기 전에 낮춤), "retract"(후퇴까지 완료한 뒤에 낮춤).
    세 값 다 낮춘 뒤 FINAL_SETTLE_STEPS만큼 더 관찰해서, 그 시점에 게인을
    낮추는 게 실제로 workpiece를 추가로 밀리게 하는지 측정한다(모듈
    docstring "왼팔 강화 게인 온/오프 비교" 절 + PIPELINE.md의 retract
    절 참고). "none"은 비교 기준선이라 관찰은 하되 실제로 게인을 낮추지
    않는다."""

    def __init__(
        self,
        xml_path: str | None = None,
        left_arm_reinforced: bool = True,
        release_grip_after: str = "none",
    ):
        if release_grip_after not in ("none", "fire_settle", "retract"):
            raise ValueError(f"release_grip_after must be none/fire_settle/retract, got {release_grip_after!r}")
        self._sim = BimanualTackerOpenArmSim(xml_path=xml_path, left_arm_reinforced=left_arm_reinforced)
        self._left_arm_reinforced = left_arm_reinforced
        self._release_grip_after = release_grip_after
        self._success_displacement_m = SUCCESS_DISPLACEMENT_M

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> None:
        self._sim.reset(scene_config)
        # release_grip_after가 이전 에피소드에서 게인을 낮춰놨을 수 있으므로
        # 매 에피소드 시작 시 생성자에서 정한 초기 상태로 되돌린다.
        self._sim.set_left_arm_reinforced(self._left_arm_reinforced)
        self._success_displacement_m = float(scene_config.get("success_displacement_m", SUCCESS_DISPLACEMENT_M))

    def step(self, action: np.ndarray) -> None:
        self._sim.step(np.asarray(action, dtype=float))

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """거리 페널티 + 발사 성공 보너스 - (발사 정착/후퇴 중/최종 정착)
        변위 초과 페널티 3종 - 오버슈트 페널티 - 스텝 페널티 + 후퇴 완료
        보너스 + 최종 성공 보너스. 페널티 3종을 따로 두는 이유: 어느 단계
        에서 workpiece가 밀렸는지(발사 반동 자체 vs 후퇴 동작 vs 그립 해제)
        를 리워드 신호에서도 구분할 수 있게 하기 위해서다."""
        success_disp = episode_result.get("success_displacement_m", SUCCESS_DISPLACEMENT_M)
        reward = (
            -2.0 * episode_result["final_distance"]
            + 20.0 * (1.0 if episode_result.get("fired") else 0.0)
            - 500.0 * max(0.0, episode_result.get("displacement", 0.0) - success_disp)
            - 500.0 * max(0.0, episode_result.get("retract_bump", 0.0) - success_disp)
            - 500.0 * max(0.0, episode_result.get("final_bump", 0.0) - success_disp)
            - 20.0 * episode_result.get("overshoot_penalty", 0.0)
            - 0.01 * episode_result["step_count"]
        )
        if episode_result.get("retracted"):
            reward += 10.0
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        """발사 성공 + (발사 정착/후퇴 중/최종 정착) 변위가 전부 허용치
        이내 + 안전 거리까지 후퇴 완료, 전부 만족해야 성공(사용자 요청
        "발사 성공 + 변위 체크 + 안전 거리까지 후퇴 완료" 그대로 반영)."""
        if not episode_result.get("fired") or not episode_result.get("retracted"):
            return False
        success_disp = episode_result.get("success_displacement_m", SUCCESS_DISPLACEMENT_M)
        return bool(
            episode_result.get("displacement", float("inf")) <= success_disp
            and episode_result.get("retract_bump", float("inf")) <= success_disp
            and episode_result.get("final_bump", float("inf")) <= success_disp
        )

    # ------------------------------------------------------------------
    def _record_step(
        self,
        ee_poses: list[np.ndarray],
        left_arm_traj: list[np.ndarray],
        actions: list[np.ndarray],
        forces: list[np.ndarray],
        torques: list[np.ndarray],
        action: np.ndarray,
    ) -> None:
        sim = self._sim
        force_now, torque_now = sim.get_force_torque()
        forces.append(force_now)
        torques.append(torque_now)
        ee_poses.append(sim.get_ee_pose())
        left_arm_traj.append(sim.get_left_ee_pos())
        actions.append(action.astype(np.float32))

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """접근(approach) -> 정착(settle) -> 후퇴(retract) -> 최종 정착
        (final settle) 네 단계로 구성된다(사용자 피드백: "실제 타카 작업처럼
        접근->압착->발사->후퇴까지 포함해야 한다"). "압착"은 이 태스크에서
        발사 트리거 자체(tolerance 진입)에 해당하므로 별도 단계로 안 나눴다
        -- 접근이 끝나는 바로 그 순간이 곧 압착+발사다.

        후퇴는 접근과 똑같은 Kp_approach 비례 제어를 그대로 재사용한다 --
        목표점만 nail_site에서 "nail_site 위 RETRACT_SAFE_DISTANCE_M"로
        바뀔 뿐 delta 계산/클립 로직은 동일하다(재사용 가능한지 검토해보니
        그대로 재사용 가능했다)."""
        sim = self._sim
        self.reset(scene_config)
        kp_approach = float(gains["Kp_approach"])

        ee_poses = [sim.get_ee_pose()]
        left_arm_traj: list[np.ndarray] = [sim.get_left_ee_pos()]
        actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []

        fired = False
        retracted = False
        pre_fire_workpiece_pos: np.ndarray | None = None
        displacement = 0.0
        retract_bump = 0.0
        final_bump = 0.0
        overshoot_penalty = 0.0
        final_dist = float("nan")
        step_count = 0

        # -- 1) 접근(approach): tolerance 진입 = 압착+발사 --------------------
        for step_count in range(1, MAX_STEPS + 1):
            tip = sim.get_tool_tip_pos()
            nail = sim.get_nail_pos()
            err = nail - tip
            dist = float(np.linalg.norm(err))
            final_dist = dist

            delta = kp_approach * err
            step_norm = float(np.linalg.norm(delta))
            if step_norm > MAX_APPROACH_STEP_M:
                delta = delta * (MAX_APPROACH_STEP_M / step_norm)
                step_norm = MAX_APPROACH_STEP_M
            if dist < FIRE_TOLERANCE_M * _NEAR_ZONE_MULT:
                overshoot_penalty = max(overshoot_penalty, step_norm - _SAFE_NEAR_STEP_M)

            self.step(delta)
            self._record_step(ee_poses, left_arm_traj, actions, forces, torques, delta)

            new_dist = float(np.linalg.norm(sim.get_nail_pos() - sim.get_tool_tip_pos()))
            if new_dist <= FIRE_TOLERANCE_M:
                fired = True
                final_dist = new_dist
                break

        if not fired:
            episode_result = {
                "final_distance": final_dist,
                "fired": False,
                "displacement": displacement,
                "retract_bump": retract_bump,
                "retracted": False,
                "final_bump": final_bump,
                "success_displacement_m": self._success_displacement_m,
                "overshoot_penalty": float(max(0.0, overshoot_penalty)),
                "step_count": step_count,
            }
            episode_result["success"] = self.is_success(episode_result)
            return self._finalize_result(
                gains, scene_config, episode_result, ee_poses, left_arm_traj, actions, forces, torques
            )

        # -- 2) 정착(settle): 발사 반동을 kick하고 가라앉힌다 ------------------
        # step()(computed-torque)을 delta=0으로 계속 불러서 그 순간 위치를
        # 계속 붙잡는다(예전엔 raw mj_step만 불러 ctrl이 고정된 채였는데,
        # "발사 직후 도구를 그 자리에 눌러 유지한다"는 실제 동작에 더
        # 맞도록 고쳤다).
        pre_fire_workpiece_pos = sim.get_workpiece_pos().copy()
        recoil_strength = float(scene_config["recoil_strength"])
        recoil_angle = float(scene_config.get("recoil_angle", 0.0))
        kick = recoil_strength * np.array([np.cos(recoil_angle), np.sin(recoil_angle), 0.2])
        sim.apply_recoil(kick)

        zero_delta = np.zeros(3)
        for _ in range(SETTLE_STEPS):
            step_count += 1
            self.step(zero_delta)
            self._record_step(ee_poses, left_arm_traj, actions, forces, torques, zero_delta)

        displacement = float(np.linalg.norm(sim.get_workpiece_pos() - pre_fire_workpiece_pos))

        if self._release_grip_after == "fire_settle":
            sim.set_left_arm_reinforced(False)

        # -- 3) 후퇴(retract): nail_site 위 안전 거리까지 수직으로 물러난다 ----
        retract_start_pos = sim.get_workpiece_pos().copy()
        retract_target = sim.get_nail_pos() + np.array([0.0, 0.0, RETRACT_SAFE_DISTANCE_M])
        for _ in range(RETRACT_MAX_STEPS):
            tip = sim.get_tool_tip_pos()
            err = retract_target - tip
            dist = float(np.linalg.norm(err))
            if dist <= RETRACT_TOLERANCE_M:
                retracted = True
                break

            delta = kp_approach * err
            step_norm = float(np.linalg.norm(delta))
            if step_norm > MAX_APPROACH_STEP_M:
                delta = delta * (MAX_APPROACH_STEP_M / step_norm)

            step_count += 1
            self.step(delta)
            self._record_step(ee_poses, left_arm_traj, actions, forces, torques, delta)

        retract_bump = float(np.linalg.norm(sim.get_workpiece_pos() - retract_start_pos))

        if self._release_grip_after == "retract":
            sim.set_left_arm_reinforced(False)

        # -- 4) 최종 정착(final settle): 그립을 놓은 시점의 영향을 관찰 ---------
        final_settle_start_pos = sim.get_workpiece_pos().copy()
        for _ in range(FINAL_SETTLE_STEPS):
            step_count += 1
            self.step(zero_delta)
            self._record_step(ee_poses, left_arm_traj, actions, forces, torques, zero_delta)
        final_bump = float(np.linalg.norm(sim.get_workpiece_pos() - final_settle_start_pos))

        episode_result = {
            "final_distance": final_dist,
            "fired": fired,
            "displacement": displacement,
            "retract_bump": retract_bump,
            "retracted": retracted,
            "final_bump": final_bump,
            "success_displacement_m": self._success_displacement_m,
            "overshoot_penalty": float(max(0.0, overshoot_penalty)),
            "step_count": step_count,
        }
        episode_result["success"] = self.is_success(episode_result)
        return self._finalize_result(
            gains, scene_config, episode_result, ee_poses, left_arm_traj, actions, forces, torques
        )

    def _finalize_result(
        self,
        gains: dict[str, float],
        scene_config: dict[str, Any],
        episode_result: dict[str, Any],
        ee_poses: list[np.ndarray],
        left_arm_traj: list[np.ndarray],
        actions: list[np.ndarray],
        forces: list[np.ndarray],
        torques: list[np.ndarray],
    ) -> dict[str, Any]:
        reward = self.compute_reward(episode_result)
        return {
            "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
            "ee_poses": np.stack(ee_poses).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 3), dtype=np.float32),
            "forces": np.stack(forces).astype(np.float32) if forces else np.zeros((0, 3), dtype=np.float32),
            "torques": np.stack(torques).astype(np.float32) if torques else np.zeros((0, 3), dtype=np.float32),
            **episode_result,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "left_arm_traj": np.stack(left_arm_traj).astype(np.float32),
        }


if __name__ == "__main__":
    gains = {"Kp_approach": 0.5}
    cfg = default_scene_config()
    env = TackerOpenArmEnv()
    result = env.run_episode(gains, cfg)
    print(
        f"[tacker_openarm_env] success={result['success']} fired={result['fired']} "
        f"displacement={result['displacement']*1000:.4f}mm step_count={result['step_count']} "
        f"reward={result['reward']:.2f}"
    )
