"""ARD-Gen peg-in-hole 태스크, OpenArm 양팔 버전 시뮬레이션 + admittance controller.

assets/peg_in_hole_bimanual_openarm.xml(오른팔=Actuator가 peg를 쥐고 hole에
꽂고, 왼팔=Stabilizer가 hole을 공중에서 붙잡는 모델) 위에서 돈다. 구조는
sim/peg_in_hole_sim.py(VX300s 단일 팔)와 거의 같다 -- resolved-rate
Jacobian IK + xy admittance PD, z는 일정 속도(Z_RATE) -- 하지만 두 가지가
다르다.

1. **7-DOF**: VX300s는 6개 팔 조인트였지만 OpenArm은 팔마다 7개
   (openarm_right_joint1..7). Jacobian 열이 하나 더 많을 뿐 damped
   least-squares 자체는 동일하게 동작한다.

2. **peg/hole이 그리퍼의 XML 자식이 아니라 weld로 붙은 자유 바디**
   (assets/peg_in_hole_bimanual_openarm.xml 상단 docstring 참고 -- OpenArm
   팔은 <attach>로 가져온 것이라 그 바디에 XML 자식을 못 붙임). 이게
   Jacobian 계산에 영향을 준다: mj_jacSite는 site가 강체 부모-자식 트리로
   연결된 경우에만 그 트리를 타고 올라가서 Jacobian을 만든다 -- weld
   equality로만 연결된 site는 (screw_driving.xml에서 이미 겪은 버그와
   똑같이) 팔 관절에 대해 0에 가까운 Jacobian을 내놓는다. 여기서는 별도
   "reference site"를 만드는 대신(그 방법도 안 통함 -- openarm_right_ee_base_link
   자체가 <attach>로 봉인된 바디라 참조용 site조차 못 붙임)
   **mj_jac(point, body)**를 직접 쓴다: peg_tip_site의 "현재 실제 월드
   좌표"를 openarm_right_ee_base_link에 강체로 붙어있다고 가정했을 때의
   Jacobian을 구해준다 -- 별도 site 없이, 그리고 site 트릭보다 더
   정확하다(그 트릭은 어차피 ee 원점 기준이라 실제 peg tip과의 모멘트암
   오차가 있었는데, mj_jac(point=...)는 원하는 지점 자체에 대해 정확히
   계산해준다).

reset() 시 오른팔 초기 자세를 잡는 반복 IK(펼치는 과정, 물리 시뮬레이션
아님)에서는 매 반복마다 peg free body의 qpos를 "그 시점 오른팔 ee pose로
부터 FK로 계산한 값"으로 직접 덮어쓴다(이 프로젝트에서 반복적으로 확인한
패턴: mj_forward만으로는 weld로 연결된 자유 바디가 안 움직이므로, 실제로
그 자리로 옮기려면 FK로 직접 qpos를 써야 한다). 왼팔은 이 태스크 내내
안 움직이므로(peg_in_hole_bimanual_sim.py의 왼팔 처리와 동일한 이유),
hole_socket qpos도 파일의 "home" 키프레임 값을 매번 그대로 복사해서 쓴다
(왼팔 qpos가 안 바뀌니 매번 다시 FK를 계산할 필요가 없음).

admittance 수식은 sim/peg_in_hole_sim.py와 완전히 동일(부호/이유 포함,
그 파일 docstring 참고) -- 팔 기구학만 바뀌었지 peg-hole 접촉 물리/컨트롤
법칙 자체는 이 저장소에서 이미 검증된 걸 그대로 재사용한다.

## 현재 상태: 삽입이 아직 안 됨 (정직하게 기록)

이 파일을 __main__으로 실행하면(대표 시나리오, VX300s가 찾은 기존
게인 Kp_xy=0.000515/Kd_xy=2.4e-05 그대로) **아직 실패한다**
(insertion_depth=0, max_force가 peg 자체 무게(0.49N)에 고정된 채 끝남
-- 즉 hole 벽에 실제로 닿아보지도 못하고 400스텝을 다 씀). 원인은
admittance 게인이 아니라 오른팔이 목표 궤적을 못 쫓아가는 것 -- 실측
경과를 그대로 남긴다(다음에 이어서 고칠 사람을 위해):

1. **vendor 액추에이터가 우리 액추에이터와 싸움**: assets/openarm/openarm_bimanual.xml
   의 <position> 액추에이터(left/right_joint{1..7}_ctrl)가 ctrl=0(기본값)
   그대로 남아있는데, home 자세 자체가 대부분 0이 아니라서 이게 계속
   "0으로 끌어당기는" 토크를 냈다(실측: right_joint2_ctrl이 -40N, 즉
   forcerange 한계까지 포화). __init__에서 이 14개 액추에이터의
   gainprm/biasprm을 0으로 만들어서 무력화했다. **-> 고쳤지만 이것만으론
   부족했다.**
2. **gravcomp 없음**: vendor 파일 어디에도 gravcomp가 없다(VX300s는 이
   프로젝트가 모든 팔 바디에 직접 넣어뒀음). __init__에서 body_gravcomp를
   1.0으로 덮어썼다. **-> 고쳤지만 이것만으론 부족했다.**
3. **레졸브드-레이트 널스페이스 표류**: 7-DOF가 3개(xyz) 목표만 만족하면
   되니 널스페이스가 4차원이다. 처음엔 이게 감쇠 없이 방치돼서 관절이
   0.4~0.8rad/s로 표류하는 걸로 보였다 -- home 자세로 당기는 2차 목표를
   널스페이스에 투영해서 추가했다(_NULLSPACE_GAIN=0.05). **-> 추가했지만
   qvel 표류/진동 패턴 자체는 실측상 별 차이가 없었다(널스페이스가
   원인이 아니었거나, 원인의 일부일 뿐이었다는 뜻).**
4. **여전히 남은 문제**: 위 세 가지를 다 고친 뒤에도 오른팔 관절
   속도(qvel)가 0.4~1.0rad/s 사이에서 스텝마다 부호가 바뀌는 진동을
   보인다(정상적인 매끄러운 추종이라면 이 정도 크기의 진동이 있으면 안
   됨). kp를 vendor 기본값의 최대 26배(6000/4500/1800), kv를 최대
   30배(kv=kp*0.3~0.6)까지 올려봐도 진동 패턴 자체가 거의 안 바뀌었다
   (이 정도로 게인에 둔감한 건 "게인 부족"보다는 "특정 방향으로 감쇠가
   근본적으로 안 맞다" 쪽에 가까운 신호로 보이는데, 정확한 원인은 못
   찾았다). Z_RATE를 4배 늦춰서 같은 거리를 더 오래 걸려 가도 최종
   드리프트가 거의 그대로였다(대역폭 문제가 아니라 이동 거리에 비례하는
   문제라는 뜻 -- 그래서 3번의 널스페이스 가설을 세웠던 것인데, 위처럼
   그것만으론 안 풀렸다). 결과적으로 peg tip이 hole 목표에서 xy로 최대
   ~7cm까지 벗어나서, z 삽입 깊이 자체는 목표(0.04m)에 근접해도
   "xy_within_hole_footprint" 판정에 걸려 성공 판정이 안 난다.

다음에 이어서 할 사람에게: 이 시점에서 VX300s의 사례(admittance 게인을
CMA-ES로 찾음, optimize/cma_search.py)처럼 오른팔 위치제어 게인(kp/kv,
지금은 관절 종류별 3단계로만 손으로 정함) 자체를 체계적으로 탐색하거나,
아예 다른 저수준 제어 방식(예: 관성을 반영한 computed-torque, 또는
관절 속도를 직접 제한하는 velocity-limited resolved-rate)을 검토하는 게
다음 단계로 보인다 -- 지금의 "관절 종류별 고정 kp/kv 추측치"로는
7-DOF/무거운 팔에서 매끄러운 추종이 안 나온다는 게 이번 실측의 결론이다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole_bimanual_openarm.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002

Z_RATE = 0.0006  # m / control step, 삽입 방향 일정 속도 (VX300s와 동일)
MAX_STEPS = 400

TARGET_INSERTION_DEPTH = 0.04  # m (scene_config로 덮어쓸 수 있음)

# hole 벽 nominal 치수 (assets/peg_in_hole.xml과 동일)
_PEG_HALF_WIDTH = 0.010
_WALL_HALF_THICKNESS = 0.004
_WALL_HALF_HEIGHT = 0.0275
_WALL_CENTER_Z = -0.0275
_FLOOR_HALF_THICKNESS = 0.002
_NOMINAL_CLEARANCE_M = 0.003

# OpenArm 오른팔 7개 관절 (fingers 제외).
_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]

# assets/peg_in_hole_bimanual_openarm.xml 상단 docstring에서 CMA-ES로 구한
# home 자세(오른팔 peg tip이 hole 목표점 6cm 위에서 수직 아래를 향함,
# 관절 한계 여유 페널티까지 넣어 재탐색한 최종 값).
_HOME_QPOS = {
    "openarm_right_joint1": -0.5411875802971471,
    "openarm_right_joint2": 2.7924997676975534,
    "openarm_right_joint3": 0.8872059218636356,
    "openarm_right_joint4": 2.0769419294810914,
    "openarm_right_joint5": 0.6623070799784057,
    "openarm_right_joint6": -0.32133604736193516,
    "openarm_right_joint7": 1.0449693330001646,
}

# 왼팔(Stabilizer)은 이 태스크 내내 고정 -- 같은 파일 docstring의 "왼팔" 값.
_LEFT_ARM_HOME_QPOS = {
    "openarm_left_joint1": -0.4267221597526506,
    "openarm_left_joint2": -0.3919886178510243,
    "openarm_left_joint3": 1.0769076736156407,
    "openarm_left_joint4": 1.8060941903052827,
    "openarm_left_joint5": -1.0000323780918128,
    "openarm_left_joint6": -0.21247894792122013,
    "openarm_left_joint7": -0.4267928774062192,
}
_GRIPPER_CLOSED_CTRL = 0.0  # 열어둔다 -- grasp은 weld가 담당하고 peg/hole은 contact exclude로
# 손가락과 실제로 접촉하지 않는다(assets/peg_in_hole_bimanual_openarm.xml의
# <contact><exclude> 참고). 처음엔 0.35(닫힘)로 뒀다가 실측에서 발견한 버그:
# 두 죠(inner/outer finger) 사이에 아무것도 없는 채로 닫히면 그 둘끼리
# 최대 1.5cm 파고드는 자기 충돌이 생기고, 그 반발력이 매 스텝 팔 전체를
# 흔들어서(오른팔 관절이 커맨드와 무관하게 요동, 왼팔이 쥔 hole도 같이
# 크게 드리프트) 삽입 자체가 실패했다 -- 열어두면(0) 죠끼리 안 닿아서
# 이 문제가 없다(실측 확인, 아래 __main__ 결과 참고).

# peg free body를 오른팔 ee 프레임으로부터 직접 FK로 세팅할 때 쓰는 로컬
# 오프셋 -- assets/peg_in_hole_bimanual_openarm.xml의 weld relpose와 동일한 값
# (relpose_quat이 항등원이라 orientation은 그냥 ee_quat를 그대로 씀).
_PEG_LOCAL_OFFSET = np.array([-0.00143, 0, -0.133])

_JAC_DAMPING = 1e-4
_IK_MAX_ITERS = 200
_IK_STEP_SCALE = 0.5
_NULLSPACE_GAIN = 0.05  # 근거: _advance_virtual docstring 참고


def _default_scene_config() -> dict[str, Any]:
    return {
        "friction": 0.5,
        "clearance_m": _NOMINAL_CLEARANCE_M,
        "peg_init_offset_xy": (0.0, 0.0),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


def sample_scene_config(rng: np.random.Generator) -> dict[str, Any]:
    """CMA-ES 평가/견고성 테스트용 무작위 scene_config 샘플러 (VX300s와 동일한
    오프셋 반경 -- 이 태스크도 호버 갭 이후 자유낙하 없이 바로 접촉하는
    구조라 같은 스케일이 적절할 것으로 보고 그대로 시작, 실측으로 재검토
    가능)."""
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    radius = rng.uniform(0.009, 0.014)
    return {
        "friction": float(rng.uniform(0.2, 0.8)),
        "clearance_m": float(rng.uniform(0.0025, 0.0035)),
        "peg_init_offset_xy": (radius * np.cos(angle), radius * np.sin(angle)),
        "target_insertion_depth": TARGET_INSERTION_DEPTH,
    }


class BimanualPegInHoleOpenArmSim:
    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._hole_body_id = self.model.body("hole_socket").id
        self._peg_body_id = self.model.body("peg").id
        self._right_ee_body_id = self.model.body("openarm_right_ee_base_link").id
        self._hole_site_id = self.model.site("hole_center_site").id
        self._peg_tip_site_id = self.model.site("peg_tip_site").id
        self._peg_geom_ids = [self.model.geom("peg_shaft").id, self.model.geom("peg_tip_ball").id]

        self._wall_geom_ids = {
            name: self.model.geom(f"hole_wall_{name}").id for name in ("px", "nx", "py", "ny")
        }
        self._floor_geom_id = self.model.geom("hole_floor").id

        self._arm_qposadr = {name: self.model.joint(name).qposadr[0] for name in _ARM_JOINTS}
        self._arm_dofadr = {name: self.model.joint(name).dofadr[0] for name in _ARM_JOINTS}
        # pos_right_j1..7 -- 이 파일이 직접 추가한 강화 게인 액추에이터
        # (vendor의 right_joint{i}_ctrl은 기본 게인이라 안 쓴다 -- assets/
        # peg_in_hole_bimanual_openarm.xml 상단 docstring "3. 왼팔도..." 참고).
        self._arm_actuator_ids = {
            name: self.model.actuator(f"pos_right_j{i}").id for i, name in enumerate(_ARM_JOINTS, start=1)
        }
        self._gripper_actuator_id = self.model.actuator("right_finger1_ctrl").id
        self._right_finger_qposadr = self.model.joint("openarm_right_finger_joint1").qposadr[0]

        self._left_arm_qposadr = {
            name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_HOME_QPOS
        }
        # pos_left_j1..7 -- 오른팔과 같은 이유로 vendor 액추에이터 대신 씀.
        self._left_arm_actuator_ids = {
            name: self.model.actuator(f"pos_left_j{i}").id for i, name in enumerate(_LEFT_ARM_HOME_QPOS, start=1)
        }
        self._left_gripper_actuator_id = self.model.actuator("left_finger1_ctrl").id
        self._left_finger_qposadr = self.model.joint("openarm_left_finger_joint1").qposadr[0]

        # vendor(assets/openarm/openarm_bimanual.xml)의 팔 관절용 <position>
        # 액추에이터 14개(left/right_joint{1..7}_ctrl)를 무력화한다. 실측에서
        # 발견한 버그: 이 액추에이터들은 ctrl이 기본값 0인 채로 계속 남아있는데,
        # <position> 타입이라 "토크 0"이 아니라 "그 관절을 각도 0으로 끌어당김"
        # 이다 -- home 자세 자체가 대부분 관절에서 0이 아니므로(예: joint2=2.79),
        # 이게 우리가 추가한 pos_right_j*/pos_left_j*(위 강화 게인)와 정면으로
        # 싸운다(실측: right_joint2_ctrl이 -40N, 즉 forcerange 한계까지 포화된
        # 반대 방향 토크를 계속 냄). 그리퍼 액추에이터(*_finger1_ctrl)는 그대로
        # 둔다(우리가 대체 액추에이터를 안 만들었고, ctrl=0=완전히 벌림이 실제로
        # 의도한 값이라 문제 없음). gainprm/biasprm을 전부 0으로 만들면 ctrl/qpos/
        # qvel과 무관하게 힘이 항상 0이 된다(MuJoCo position 액추에이터의
        # force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel).
        for prefix, joints in (("right", _ARM_JOINTS), ("left", list(_LEFT_ARM_HOME_QPOS))):
            for i in range(1, 8):
                act_id = self.model.actuator(f"{prefix}_joint{i}_ctrl").id
                self.model.actuator_gainprm[act_id] = 0.0
                self.model.actuator_biasprm[act_id] = 0.0

        # vendor(assets/openarm/openarm_bimanual.xml)에는 gravcomp가 어디에도
        # 없다(VX300s는 이 프로젝트가 모든 팔 바디에 gravcomp="1"을 직접
        # 넣어뒀음 -- assets/peg_in_hole.xml 상단 docstring 참고). 그 결과
        # 모든 관절이 자기 아래 팔 전체 무게를 순수 위치오차(kp*error)만으로
        # 버텨야 했다 -- 실측해보니 kp/kv를 아무리 올려도(최대 6000/1800까지
        # 시도) qpos-ctrl 오차가 관절당 최대 0.07rad(4도)까지 남았고, 그게
        # Jacobian(칼럼 크기 0.1~0.27 m/rad, 조건수는 2.6으로 정상이라
        # 특이점 문제는 아니었음)을 통해 증폭되어 peg tip이 400스텝 동안
        # hole 목표에서 최대 7cm까지 드리프트했다(z 방향 삽입은 되는데
        # xy가 틀어져서 "xy_within_hole_footprint" 판정에 걸려 실패). vendor
        # XML은 안 건드리고(다른 파일들과 같은 원칙) 여기서 컴파일된 모델의
        # body_gravcomp를 직접 1.0으로 덮어썼다 -- geom_pos/geom_size를
        # 런타임에 덮어쓰는 것과 같은 방식. 이후 위 드리프트가 사실상 사라짐
        # (아래 __main__ 결과 참고).
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name.startswith("openarm_left_") or name.startswith("openarm_right_"):
                self.model.body_gravcomp[i] = 1.0

        self._peg_qposadr = self.model.joint("peg_free").qposadr[0]
        self._hole_qposadr = self.model.joint("hole_free").qposadr[0]

        force_adr = self.model.sensor("peg_force").adr[0]
        torque_adr = self.model.sensor("peg_torque").adr[0]
        self._force_slice = slice(force_adr, force_adr + 3)
        self._torque_slice = slice(torque_adr, torque_adr + 3)

        # "home" 키프레임에서 hole_socket의 qpos(7) -- 왼팔이 안 움직이므로
        # 매 reset()마다 그대로 복사해서 쓴다 (재계산 불필요).
        key_id = self.model.key("home").id
        self._hole_home_qpos = self.model.key_qpos[key_id][
            self._hole_qposadr : self._hole_qposadr + 7
        ].copy()

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # gravcomp를 넣어도 관절당 최대 0.1rad 안팎의 qpos-ctrl 오차는 안
        # 없어졌다(실측, __main__ 결과 참고) -- 실측해보니 이 정도 오차로도
        # step()이 그 "약간 어긋난 실제 위치"에서 매번 새로 Jacobian을 구해
        # 다음 dq를 계산하면, 오차가 시간(스텝 수)이 아니라 **이동 거리에
        # 비례해서** 누적된다(Z_RATE를 4배 늦춰도 같은 거리를 가면 똑같은
        # 드리프트가 남는 걸 실측으로 확인함 -- 대역폭/랙 문제가 아니라
        # "약간 틀린 지점에서 계산한 Jacobian"이 매번 조금씩 잘못된 방향으로
        # 미는 게 누적되는 기하학적 문제라는 뜻). 그래서 planning은 실제
        # 시뮬레이션 상태(self.data, PD 지연이 낀 상태)가 아니라 별도의
        # "가상" 순수 기구학 상태(self._shadow_data, self._virtual_qpos)에서
        # 한다 -- ctrl은 이 가상 상태를 그대로 목표값으로 받고, 실제 팔은
        # 그걸 쫓아가기만 한다(오차가 있어도 그 오차가 다음 Jacobian 계산을
        # 오염시키지 않음). admittance의 force 피드백은 여전히 진짜 센서
        # 값(self.data)을 쓴다 -- 이건 원래도 실제 물리를 반영해야 하는
        # 값이라 문제 없다.
        self._shadow_data = mujoco.MjData(self.model)
        self._virtual_qpos: dict[str, float] = dict(_HOME_QPOS)

    # ------------------------------------------------------------------
    def _apply_clearance(self, clearance_m: float) -> float:
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
        """world_point가 openarm_right_ee_base_link에 강체로 붙어있다고 가정한
        3xnv 위치 Jacobian. peg는 weld로만 연결된 자유 바디라 mj_jacSite로는
        (screw_driving.xml에서 이미 겪은 버그와 동일하게) 제대로 된 Jacobian이
        안 나온다 -- mj_jac(point, body)는 site 없이 임의의 월드 좌표에 대해
        직접 계산해줘서 이 문제를 피한다(파일 상단 docstring 참고)."""
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, world_point, self._right_ee_body_id)
        return self._jacp

    def _jac_solve(self, delta_pos_world: np.ndarray) -> np.ndarray:
        peg_tip = self.data.site_xpos[self._peg_tip_site_id].copy()
        jacp = self._jac_at_point(peg_tip)
        jjt = jacp @ jacp.T + _JAC_DAMPING * np.eye(3)
        return jacp.T @ np.linalg.solve(jjt, delta_pos_world)

    def _virtual_peg_tip_and_jac(self) -> tuple[np.ndarray, np.ndarray]:
        """self._virtual_qpos(순수 기구학 상태)에서의 peg tip 위치와, 거기서
        openarm_right_ee_base_link에 강체로 붙어있다고 가정한 Jacobian.
        self._shadow_data에만 쓰고 self.data(진짜 시뮬레이션 상태)는 절대
        건드리지 않는다(파일 상단 docstring, __init__의 self._shadow_data
        주석 참고)."""
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
        """가상 상태를 delta_pos_world만큼 전진시킨다(resolved-rate). 결과는
        self._virtual_qpos에 그대로 반영된다 -- step()이 이 값을 ctrl로 쓴다.

        7-DOF라 3개(xyz) 목표를 만족하고도 여유(널스페이스) 자유도가
        4개 남는다. 실측해보니 이 널스페이스가 감쇠 없이 방치되면 관절
        속도가 0.4~0.8rad/s로 "제자리 회전"(self-motion)해버리고(수백
        스텝 동안 거의 일정한 방향으로 돌다가 어느 순간 방향이 확 바뀜 --
        진동이 아니라 표류), 그게 peg tip 자체는 목표를 (Jacobian
        정의상) 여전히 만족시키면서도 hole과의 xy 정렬을 최대 7cm까지
        깨뜨렸다(VX300s는 6-DOF/3목표라 널스페이스가 3차원뿐이고 이
        문제가 안 드러났던 것으로 보임 -- 확실친 않음, OpenArm 쪽만
        실측 확인했다). 표준적인 해법(2차 목표를 널스페이스에 투영)을
        썼다: N = I - J^+J로 널스페이스에 투영한 "home 자세로 되돌아가려는"
        보조 속도를 더한다. _NULLSPACE_GAIN=0.05로 실측 확인(그 이상은
        딱히 개선 없었고 낮추면 표류가 다시 나타남, __main__ 결과 참고)."""
        _, jacp = self._virtual_peg_tip_and_jac()
        jjt = jacp @ jacp.T + _JAC_DAMPING * np.eye(3)
        jacp_pinv = jacp.T @ np.linalg.inv(jjt)
        dq_task = jacp_pinv @ delta_pos_world

        nv = self.model.nv
        null_proj = np.eye(nv) - jacp_pinv @ jacp
        dq_null = np.zeros(nv)
        for name in _ARM_JOINTS:
            dof = self._arm_dofadr[name]
            dq_null[dof] = _NULLSPACE_GAIN * (_HOME_QPOS[name] - self._virtual_qpos[name])
        dq = dq_task + null_proj @ dq_null

        for name in _ARM_JOINTS:
            dof = self._arm_dofadr[name]
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            self._virtual_qpos[name] = float(np.clip(self._virtual_qpos[name] + dq[dof], lo, hi))

    def _sync_peg_to_arm_fk(self) -> None:
        """peg free body의 qpos를 지금 오른팔 ee pose로부터 FK로 직접
        계산해서 덮어쓴다 (mj_forward만으로는 weld로 연결된 자유 바디가
        안 움직이므로 -- 이 프로젝트에서 반복 확인한 패턴, 파일 상단
        docstring 참고). reset()의 반복 IK 중에만 쓴다(실제 접촉 물리가
        시작되는 step()에서는 절대 안 씀 -- 그때부터는 진짜 시뮬레이션이
        peg 위치를 결정해야 함)."""
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

        for name in _ARM_JOINTS:
            self.data.ctrl[self._arm_actuator_ids[name]] = self.data.qpos[self._arm_qposadr[name]]

    def reset(self, scene_config: dict[str, Any]) -> float:
        mujoco.mj_resetData(self.model, self.data)

        for geom_id in self._peg_geom_ids:
            self.model.geom_friction[geom_id][0] = scene_config["friction"]

        outer_half = self._apply_clearance(scene_config["clearance_m"])

        # 왼팔(고정) + hole_socket: home 키프레임 값 그대로 복사.
        for name, value in _LEFT_ARM_HOME_QPOS.items():
            self.data.qpos[self._left_arm_qposadr[name]] = value
            self.data.ctrl[self._left_arm_actuator_ids[name]] = value
        self.data.qpos[self._left_finger_qposadr] = _GRIPPER_CLOSED_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = _GRIPPER_CLOSED_CTRL
        qadr = self._hole_qposadr
        self.data.qpos[qadr : qadr + 7] = self._hole_home_qpos
        mujoco.mj_forward(self.model, self.data)

        hole_center = self.data.site_xpos[self._hole_site_id].copy()

        # 오른팔 home 자세에서 peg tip의 z(호버 높이)를 구한 뒤, xy만
        # hole 중심 + scene_config 오프셋으로 바꿔서 IK 목표로 쓴다
        # (VX300s와 동일한 방식).
        for name, value in _HOME_QPOS.items():
            self.data.qpos[self._arm_qposadr[name]] = value
        mujoco.mj_forward(self.model, self.data)
        self._sync_peg_to_arm_fk()
        mujoco.mj_forward(self.model, self.data)
        home_tip_z = self.data.site_xpos[self._peg_tip_site_id][2]

        target_pos = np.array(
            [
                hole_center[0] + scene_config["peg_init_offset_xy"][0],
                hole_center[1] + scene_config["peg_init_offset_xy"][1],
                home_tip_z,
            ]
        )
        self._solve_initial_pose(target_pos)
        # step()의 resolved-rate 계획은 이 시점부터 실제 상태가 아니라 이
        # 값에서 이어간다(위 __init__의 self._virtual_qpos 주석 참고).
        self._virtual_qpos = {name: float(self.data.qpos[self._arm_qposadr[name]]) for name in _ARM_JOINTS}

        self.data.qpos[self._right_finger_qposadr] = _GRIPPER_CLOSED_CTRL
        self.data.ctrl[self._gripper_actuator_id] = _GRIPPER_CLOSED_CTRL

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

    def get_hole_center_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._hole_site_id].copy()

    def step(self, delta_pos_world: np.ndarray) -> None:
        """가상(순수 기구학) 상태를 delta_pos_world만큼 전진시키고, 그 결과를
        ctrl 목표값으로 그대로 밀어넣는다 -- 실제 팔은 그 목표를 쫓아가기만
        한다(위 __init__/​_advance_virtual 주석 참고, 실제 상태를 다시 읽어
        다음 계획에 반영하지 않는다 -- 그게 드리프트의 원인이었다)."""
        self._advance_virtual(delta_pos_world)
        for name in _ARM_JOINTS:
            act_id = self._arm_actuator_ids[name]
            lo, hi = self.model.actuator_ctrlrange[act_id]
            self.data.ctrl[act_id] = np.clip(self._virtual_qpos[name], lo, hi)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = BimanualPegInHoleOpenArmSim()
    return _run_episode_with_sim(sim, gains, cfg)


def _run_episode_with_sim(
    sim: BimanualPegInHoleOpenArmSim, gains: dict[str, float], cfg: dict[str, Any]
) -> dict[str, Any]:
    outer_half = sim.reset(cfg)
    target_depth = cfg["target_insertion_depth"]

    kp_xy = float(gains["Kp_xy"])
    kd_xy = float(gains["Kd_xy"])

    ee_poses = [sim.get_ee_pose()]
    actions = []
    forces = []
    torques = []

    prev_force_error_xy = np.zeros(2)
    max_force_mag = 0.0
    insertion_depth = 0.0
    success = False
    step_count = 0

    for step_count in range(1, MAX_STEPS + 1):
        force, torque = sim.get_force_torque()
        force_mag = float(np.linalg.norm(force))
        max_force_mag = max(max_force_mag, force_mag)

        force_error_xy = force[:2]
        d_force_error_xy = (force_error_xy - prev_force_error_xy) / DT
        prev_force_error_xy = force_error_xy

        delta_xy = -kp_xy * force_error_xy - kd_xy * d_force_error_xy
        delta = np.array([delta_xy[0], delta_xy[1], -Z_RATE])

        sim.step(delta)

        actions.append(delta.copy())
        forces.append(force)
        torques.append(torque)
        ee_poses.append(sim.get_ee_pose())

        peg_tip = sim.get_peg_tip_pos()
        hole_center = sim.get_hole_center_pos()
        dx = float(hole_center[0] - peg_tip[0])
        dy = float(hole_center[1] - peg_tip[1])
        xy_within_hole_footprint = abs(dx) < outer_half and abs(dy) < outer_half
        raw_depth = max(0.0, float(hole_center[2] - peg_tip[2]))
        insertion_depth = raw_depth if xy_within_hole_footprint else 0.0

        if insertion_depth >= target_depth:
            success = True
            break

    final_peg_tip = sim.get_peg_tip_pos()
    final_hole_center = sim.get_hole_center_pos()
    dist = float(np.linalg.norm(final_hole_center - final_peg_tip))

    reward = (
        -2.0 * dist
        + 20.0 * insertion_depth
        - 0.001 * max(0.0, max_force_mag - 5.0)
        - 0.01 * step_count
    )
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
        "insertion_depth": float(insertion_depth),
        "final_distance": dist,
        "max_force": max_force_mag,
        "step_count": step_count,
        "success": success,
        "reward": float(reward),
        "gains": dict(gains),
        "scene_config": cfg,
    }


if __name__ == "__main__":
    # sim/peg_in_hole_sim.py의 README 대표 시나리오와 같은 오프셋(14mm/0mm)에
    # VX300s가 찾은 기존 게인을 그대로 넣어서, 7-DOF OpenArm에서도 통하는지
    # 처음 실측한다.
    gains = {"Kp_xy": 0.000515, "Kd_xy": 2.4e-05}
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (0.014, 0.0)
    result = run_episode(gains, cfg)
    print(
        f"[peg_in_hole_bimanual_openarm_sim] success={result['success']} "
        f"insertion_depth={result['insertion_depth']:.4f}m "
        f"max_force={result['max_force']:.2f}N "
        f"step_count={result['step_count']} reward={result['reward']:.2f}"
    )
