"""ARD-Gen 시뮬이 RoCo 공식 시뮬 데이터와 같은 인터페이스로 데이터를 내도록
하는 기반 모듈 (사용자 지시 "태스크 자체는 이번에 만들지 않는다" -- 이
파일은 특정 태스크의 물리/제어를 전혀 모른다, OpenArm 양팔 베이스 모델
(assets/bimanual_openarm.xml 계열, sim/peg_in_hole_openarm_env.py가 따르는
규칙)이 공통으로 갖는 바디/조인트 이름만 가정한다).

## 이 파일이 다루는 것 / 안 다루는 것

다루는 것: RoCo 14차원 액션/44차원 state의 정확한 스키마(이름·순서),
연속적인(래핑 없는) Euler XYZ 변환, head/left_hand/right_hand 카메라
추가(MjSpec으로, 아래 "손목 카메라" 절 참고), 그리퍼 값 매핑, 왼팔
작업공간 좌표 변환.

안 다루는 것: 특정 태스크의 제어 루프, scene_config, 성공 판정. 이
모듈의 함수들은 모두 "이미 존재하는 MjModel/MjData(또는 그 스냅샷)"를
입력으로 받아 RoCo 스키마로 변환하기만 한다 -- 로봇을 움직이는 건 이
모듈의 책임이 아니다.

## 손목 카메라: `<attach>`로 들여온 바디에도 자식을 붙일 수 있다 (정정)

이전 작업(폐루프 Judge 데이터)에서 "OpenArm 그리퍼 바디는 `<attach>`로
봉인돼 있어서 자식(카메라 등)을 못 붙인다"고 기록했던 적이 있는데, 이건
**raw XML 텍스트 편집** 기준으로만 맞는 말이었다(`<attach>`로 들여온
바디는 이 파일의 XML 텍스트에 안 적혀 있으니 그 바디 이름으로 `<body>`
블록을 열어 자식을 추가할 문법이 없다). 하지만 `mujoco.MjSpec`(프로그램적
API)는 다르다: `spec.body("openarm_right_ee_base_link")`로 attach된
바디를 이름으로 직접 찾아서 `.add_camera(...)`로 자식을 붙일 수 있다 --
compile()하면 정상적으로 동작한다(실측 확인, 바디/조인트/액추에이터 id도
자식 추가 전후로 완전히 동일하게 유지된다 -- 카메라만 새로 추가되므로).
`build_roco_model()`이 이 방식을 쓴다.

`MjSpec.to_xml()`로 되돌린 XML을 그대로 재파싱하는 건 안 된다 -- 이
베이스 모델이 여러 서브파일을 `<attach>`로 합성한 것이라 (1) 병합된
`<default>` 블록에 class 없는 중첩 `<default>`가 생겨서
("empty class name") 파서가 거부하고, (2) 서브파일마다 다른 meshdir가
있어서 재직렬화된 mesh 파일 경로가 더 이상 안 맞는다(실측 확인, 둘 다).
그래서 `build_roco_model()`은 **컴파일된 MjModel을 그대로 돌려준다**
(XML 텍스트 왕복을 거치지 않음) -- 호출자는 기존 코드 패턴처럼
`BimanualPegInHoleOpenArmSim()`으로 기본 모델을 만든 뒤 `sim.model`/
`sim.data`를 이 함수의 결과로 바꿔치면 된다(바디/조인트/액추에이터 id가
그대로 유지되므로 안전 -- `__init__`이 캐싱한 id들이 전부 여전히 맞다).

## 왼팔이 "액티브" 암인 공식 RoCo 관례와의 차이 (정직하게 기록)

공식 RoCo 데이터의 왼팔 EE 위치/회전 범위는 크고, 오른팔은 "사실상
정지"(y 4cm·z 3cm 이내, 그리퍼 고정)라고 명시돼 있다 -- 즉 공식
관례는 **왼팔=액티브, 오른팔=수동 정지**다. 반면 이 저장소의
peg_in_hole(OpenArm)은 정반대로 **오른팔=Actuator(움직임), 왼팔=
Stabilizer(고정)**이다(sim/peg_in_hole_openarm_env.py 모듈 docstring
참고). 이 모듈의 스키마 자체는 "어느 팔이 움직이는지" 가정하지 않고
left/right를 그대로 보고하므로 문제가 없지만, **이 저장소의 기존
태스크(peg_in_hole)로 만든 데이터를 그대로 변환하면 validate_against_roco.py의
"왼팔 액티브/오른팔 정지" 통계 검사를 통과하지 못한다** -- 이건 버그가
아니라 두 관례가 다르다는 사실 자체다(사용자 지시 "태스크 자체는
만들지 않는다"와 일치 -- 액티브 암이 왼쪽인 새 태스크는 이번 범위
밖). tests/test_roco_interface.py는 이 모듈의 스키마/변환 함수만
검증하고, scripts/validate_against_roco.py로 생성한 샘플 데이터를
돌려보는 건 "파이프라인이 끝까지 도는가"를 보여주는 것이지 "공식
통계를 통과한다"는 뜻이 아니다 -- 완료 보고에 그대로 기록한다.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 스키마 상수
# ---------------------------------------------------------------------------

_SIDES = ("left", "right")
_EE_POSE_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")

# 사용자 지시 그대로: [f"{s}_ee_{a}" for s in ("left","right") for a in
# ("x","y","z","qw","qx","qy","qz")] + [f"{s}_jpos_{i}" ...] + [f"{s}_jvel_{i}" ...]
# + ["left_gripper","right_gripper"]. 44차원.
STATE_NAMES: list[str] = (
    [f"{s}_ee_{a}" for s in _SIDES for a in _EE_POSE_FIELDS]
    + [f"{s}_jpos_{i}" for s in _SIDES for i in range(7)]
    + [f"{s}_jvel_{i}" for s in _SIDES for i in range(7)]
    + ["left_gripper", "right_gripper"]
)
assert len(STATE_NAMES) == 44, len(STATE_NAMES)

# 액션: 14차원 EE 절대 목표 [left x,y,z,rx,ry,rz,gripper, right x,y,z,rx,ry,rz,gripper].
_ACTION_POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz", "gripper")
ACTION_NAMES: list[str] = [f"{s}_{a}" for s in _SIDES for a in _ACTION_POSE_FIELDS]
assert len(ACTION_NAMES) == 14, len(ACTION_NAMES)

CAMERA_NAMES: tuple[str, str, str] = ("head", "left_hand", "right_hand")
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 10


def arm_joint_names(side: str) -> list[str]:
    assert side in _SIDES, side
    return [f"openarm_{side}_joint{i}" for i in range(1, 8)]


def ee_body_name(side: str) -> str:
    assert side in _SIDES, side
    return f"openarm_{side}_ee_base_link"


def gripper_joint_name(side: str) -> str:
    assert side in _SIDES, side
    return f"openarm_{side}_finger_joint1"


# ---------------------------------------------------------------------------
# 연속적인(래핑 없는) Euler XYZ(intrinsic, 라디안) 변환
#
# 사용자 지시 "회전은 Euler XYZ(intrinsic, 라디안)이고 시간축으로
# 연속이어야 한다(wrap 금지)"를 만족하려면 두 단계가 필요하다:
#   1) 쿼터니언 -> 단일 프레임 Euler(주 분기, ry는 [-pi/2,pi/2])
#   2) 프레임 시퀀스 전체에 걸쳐 각 축을 독립적으로 unwrap(2*pi 점프 제거)
# 공식 통계의 rz 범위(-2.95~7.00)가 이미 [-pi,pi]를 벗어난다는 것 자체가
# "매 프레임 wrap하지 않고 누적된 연속값"을 기대한다는 증거다 -- 이
# 모듈은 그 관례를 그대로 따른다.
# ---------------------------------------------------------------------------


def quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz 쿼터니언 -> (3,3) 회전행렬. mujoco의 저수준 함수를 그대로
    쓴다(직접 공식을 다시 구현해서 부호/정규화 버그를 또 만들 위험을
    피하려고)."""
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def euler_xyz_from_mat(mat: np.ndarray) -> np.ndarray:
    """R = Rx(rx) @ Ry(ry) @ Rz(rz) (intrinsic X,Y,Z 순서)로 보고 (rx,ry,rz)를
    뽑는다 -- 주 분기(principal branch)만 반환한다(ry는 [-pi/2,pi/2],
    rx/rz는 [-pi,pi]) -- 짐벌락 근처(|R[0,2]|≈1)에서는 다른 두 각이
    유일하게 안 정해지지만, 이 프로젝트가 다루는 손목 자세에서는
    도달하지 않는다고 가정한다(공식 데이터 ry 범위 -1.26~1.53도 ±pi/2=
    ±1.5708 경계에 닿지 않는다 -- 실측치가 이 가정을 뒷받침).

    전개(R = Rx(a)Ry(b)Rz(c)):
        R[0,0]=cb*cc  R[0,1]=-cb*sc  R[0,2]=sb
        R[1,2]=-sa*cb R[2,2]=ca*cb
        R[0,1]=-cb*sc R[0,0]=cb*cc
    따라서 b=atan2(R[0,2], sqrt(R[0,0]^2+R[0,1]^2)),
    a=atan2(-R[1,2],R[2,2]), c=atan2(-R[0,1],R[0,0])."""
    r = np.asarray(mat, dtype=np.float64)
    cb = np.hypot(r[0, 0], r[0, 1])
    ry = np.arctan2(r[0, 2], cb)
    rx = np.arctan2(-r[1, 2], r[2, 2])
    rz = np.arctan2(-r[0, 1], r[0, 0])
    return np.array([rx, ry, rz], dtype=np.float64)


def euler_xyz_to_mat(euler: np.ndarray) -> np.ndarray:
    """euler_xyz_from_mat()의 역변환(테스트/검증용 -- "왕복" 확인에 쓴다)."""
    rx, ry, rz = (float(v) for v in euler)
    ca, sa = np.cos(rx), np.sin(rx)
    cb, sb = np.cos(ry), np.sin(ry)
    cc, sc = np.cos(rz), np.sin(rz)
    rx_mat = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry_mat = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz_mat = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx_mat @ ry_mat @ rz_mat


def quat_wxyz_from_mat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(-1))
    return quat


def unwrap_euler_sequence(euler_seq: np.ndarray) -> np.ndarray:
    """(T,3) 주-분기 Euler 시퀀스를 축마다 독립적으로 `np.unwrap`해서
    프레임간 2*pi 점프를 없앤다 -- 사용자 지시 "연속 프레임 간 π 이상
    점프 0건"을 만족하는 핵심 단계. 축 사이의 짐벌락 분기 전환(한
    프레임에서 세 각이 동시에 (rx+pi, pi-ry, rz+pi)로 튀는 경우)까지는
    못 잡는다 -- 위 euler_xyz_from_mat() docstring의 가정("짐벌락 근처에
    안 감")이 깨지면 이 함수도 한계가 있다(정직하게 기록)."""
    euler_seq = np.asarray(euler_seq, dtype=np.float64)
    if euler_seq.ndim != 2 or euler_seq.shape[1] != 3:
        raise ValueError(f"euler_seq must be (T,3), got {euler_seq.shape}")
    return np.unwrap(euler_seq, axis=0)


def continuous_euler_xyz_from_quats(quats_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) wxyz 쿼터니언 시퀀스 -> (T,3) 연속(unwrap된) Euler XYZ 시퀀스.
    액션 시퀀스를 만들 때 이 함수를 쓴다(단일 프레임만으로는 연속성을
    보장할 기준점이 없으므로, 항상 전체 시퀀스를 한 번에 넘겨야 한다)."""
    quats_wxyz = np.asarray(quats_wxyz, dtype=np.float64)
    if quats_wxyz.ndim != 2 or quats_wxyz.shape[1] != 4:
        raise ValueError(f"quats_wxyz must be (T,4), got {quats_wxyz.shape}")
    raw = np.stack([euler_xyz_from_mat(quat_wxyz_to_mat(q)) for q in quats_wxyz])
    return unwrap_euler_sequence(raw)


# ---------------------------------------------------------------------------
# 그리퍼 값 매핑 -- 사용자 지시 "공식 데이터의 열림 비율 범위(약
# 0.06~0.30)로 매핑하는 설정"
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GripperMapping:
    """raw(관절 각도, 라디안) <-> RoCo "열림 비율" 선형 변환. `closed_value`/
    `open_value`는 raw 공간에서 "완전히 닫힘"/"완전히 열림"에 대응하는 값
    (부호는 팔마다 다를 수 있다 -- default_gripper_mapping()이 모델의
    실제 joint range에서 "0에 가까운 쪽"을 open, 반대쪽을 closed로 자동
    판정한다, 이 저장소 OpenArm 그리퍼가 실제로 그렇게 생겨서다: 양팔 다
    grasp 각도가 0 근처고 0이 아닌 쪽 극단이 완전히 닫힘). `target_range`
    기본값이 사용자 지시 수치(0.06, 0.30) -- 공식 데이터가 관측된
    범위라서, 우리 그리퍼의 "닫힘~열림" 전체를 이 좁은 구간에 선형으로
    눌러 담는다."""

    closed_value: float
    open_value: float
    target_range: tuple[float, float] = (0.06, 0.30)

    def to_roco(self, raw_value: float) -> float:
        span = self.open_value - self.closed_value
        ratio = 0.0 if abs(span) < 1e-12 else (raw_value - self.closed_value) / span
        ratio = float(np.clip(ratio, 0.0, 1.0))
        lo, hi = self.target_range
        return lo + ratio * (hi - lo)

    def from_roco(self, roco_value: float) -> float:
        lo, hi = self.target_range
        span = hi - lo
        ratio = 0.0 if abs(span) < 1e-12 else (roco_value - lo) / span
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return self.closed_value + ratio * (self.open_value - self.closed_value)


def default_gripper_mapping(model: mujoco.MjModel, side: str) -> GripperMapping:
    """joint range에서 0에 가까운 끝을 open, 먼 쪽을 closed로 자동 판정한다
    (위 GripperMapping docstring 참고 -- 하드코딩 대신 모델에서 직접
    읽어서, XML이 바뀌어도 이 함수가 따라간다)."""
    joint = model.joint(gripper_joint_name(side))
    lo, hi = float(joint.range[0]), float(joint.range[1])
    open_value, closed_value = (lo, hi) if abs(lo) < abs(hi) else (hi, lo)
    return GripperMapping(closed_value=closed_value, open_value=open_value)


# ---------------------------------------------------------------------------
# 작업공간 좌표 프레임 변환 -- 사용자 지시 "OpenArm 작업 공간을 이 범위로
# 옮기는 변환을 설정으로 노출"
# ---------------------------------------------------------------------------

# 공식 데이터의 왼팔 EE 위치 범위(사용자 지시 그대로) -- target의 기본값.
ROCO_OFFICIAL_LEFT_EE_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.01, 0.82),
    "y": (-0.14, 0.76),
    "z": (1.07, 1.80),
}


@dataclasses.dataclass(frozen=True)
class WorkspaceTransform:
    """축별(x/y/z) 선형(affine) 변환 -- source 범위를 target 범위로 옮긴다.
    source는 "이 변환을 쓸 태스크의 실제 작업공간 관측 범위"라서 호출자가
    반드시 줘야 한다(기본값을 박아두지 않는다 -- 태스크마다 다르므로, 위
    모듈 docstring "안 다루는 것" 절과 같은 이유). target은 공식 왼팔 EE
    범위가 기본값이지만 다른 신체 부위/축 조합에도 쓸 수 있게 설정으로
    둔다."""

    source: dict[str, tuple[float, float]]
    target: dict[str, tuple[float, float]] = dataclasses.field(
        default_factory=lambda: dict(ROCO_OFFICIAL_LEFT_EE_RANGE)
    )

    def _axis_names(self) -> list[str]:
        return sorted(self.source.keys())

    def apply(self, pos: np.ndarray) -> np.ndarray:
        """pos: (...,3) xyz(또는 source에 있는 축 개수만큼) -- source 범위를
        target 범위로 선형 매핑한다(클리핑 없음 -- source 범위 밖 입력도
        그대로 외삽돼서 돌아온다, round-trip 테스트가 그걸 전제로 한다)."""
        pos = np.asarray(pos, dtype=np.float64)
        axes = self._axis_names()
        if pos.shape[-1] != len(axes):
            raise ValueError(f"pos 마지막 축 길이({pos.shape[-1]})가 source 축 개수({len(axes)})와 다르다")
        out = pos.copy()
        for i, axis in enumerate(axes):
            s_lo, s_hi = self.source[axis]
            t_lo, t_hi = self.target[axis]
            s_span = s_hi - s_lo
            ratio = 0.0 if abs(s_span) < 1e-12 else (pos[..., i] - s_lo) / s_span
            out[..., i] = t_lo + ratio * (t_hi - t_lo)
        return out

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        """apply()의 역변환."""
        inv = WorkspaceTransform(source=self.target, target=self.source)
        return inv.apply(pos)


# ---------------------------------------------------------------------------
# state/action 빌더
# ---------------------------------------------------------------------------


def get_ee_pose(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(pos(3,), quat_wxyz(4,)) -- ee_base_link 바디 자체의 월드 pose(사용자
    태스크가 거기 뭘 쥐고 있든 상관없이 항상 같은 의미를 갖는 기준점)."""
    body_id = model.body(ee_body_name(side)).id
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def get_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    adrs = [model.joint(name).qposadr[0] for name in arm_joint_names(side)]
    return data.qpos[adrs].copy()


def get_joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    adrs = [model.joint(name).dofadr[0] for name in arm_joint_names(side)]
    return data.qvel[adrs].copy()


def get_gripper_raw(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    adr = model.joint(gripper_joint_name(side)).qposadr[0]
    return float(data.qpos[adr])


def build_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_mappings: dict[str, GripperMapping],
) -> np.ndarray:
    """STATE_NAMES 순서 그대로 44차원 state를 만든다. `gripper_mappings`는
    {"left": GripperMapping, "right": GripperMapping} -- default_gripper_mapping()
    으로 만들거나 호출자가 직접 구성."""
    parts: list[np.ndarray] = []
    for side in _SIDES:
        pos, quat = get_ee_pose(model, data, side)
        parts.append(np.concatenate([pos, quat]))
    for side in _SIDES:
        parts.append(get_joint_qpos(model, data, side))
    for side in _SIDES:
        parts.append(get_joint_qvel(model, data, side))
    gripper_vals = np.array(
        [gripper_mappings[side].to_roco(get_gripper_raw(model, data, side)) for side in _SIDES]
    )
    parts.append(gripper_vals)
    state = np.concatenate(parts).astype(np.float32)
    assert state.shape == (44,), state.shape
    return state


def build_action_sequence(
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    left_gripper_roco: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
    right_gripper_roco: np.ndarray,
) -> np.ndarray:
    """시퀀스 전체(각 (T,3)/(T,4)/(T,))를 받아 (T,14) 액션을 만든다 -- 회전
    연속성(unwrap)은 시퀀스 전체 맥락이 있어야 하므로 프레임 하나씩은
    못 만든다(위 continuous_euler_xyz_from_quats() 참고). 호출자는 보통
    "이 스텝의 action = 다음 스텝에서 도달해야 할 EE 목표"(절대 목표
    제어 데이터셋의 표준 관례, 예: RT-1/Bridge류) 식으로 pos/quat
    시퀀스를 한 스텝 밀어서 넘긴다 -- 이 함수는 그 관례를 강제하지
    않고 받은 그대로 변환만 한다."""
    left_euler = continuous_euler_xyz_from_quats(left_quat)
    right_euler = continuous_euler_xyz_from_quats(right_quat)
    left = np.concatenate(
        [np.asarray(left_pos, dtype=np.float64), left_euler, np.asarray(left_gripper_roco, dtype=np.float64)[:, None]],
        axis=1,
    )
    right = np.concatenate(
        [np.asarray(right_pos, dtype=np.float64), right_euler, np.asarray(right_gripper_roco, dtype=np.float64)[:, None]],
        axis=1,
    )
    action = np.concatenate([left, right], axis=1).astype(np.float32)
    assert action.shape[1] == 14, action.shape
    return action


# ---------------------------------------------------------------------------
# 카메라: head/left_hand/right_hand (MjSpec로 추가, 위 모듈 docstring
# "손목 카메라" 절 참고)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None  # wxyz, None이면 identity(부모 바디 방향 그대로)
    fovy: float = 70.0


# 기본 head 카메라: top_cam과 같은 물리적 자리(받침대 꼭대기, 실제 OpenArm
# 하드웨어 사진 기준 -- assets/peg_in_hole_bimanual_openarm.xml의 top_cam
# 주석 참고)에, 작업공간 쪽(앞+아래)을 보도록 고정 각도를 줬다. top_cam과
# 달리 mode="targetbody"를 안 쓴다 -- 이 모듈은 특정 태스크의 타겟 바디를
# 모르는 범용 인터페이스라서(위 모듈 docstring "안 다루는 것" 참고),
# 실제 로봇 헤드 카메라처럼 고정 각도로 아래를 본다. 이 각도/위치는
# 어디까지나 기본값이고, 태스크마다 다른 CameraConfig를 넘기면 된다.
DEFAULT_HEAD_CAMERA = CameraConfig(
    pos=(0.05, 0.0, 0.75),
    # look-at (0.35, 0, 0.1) 기준(world +z up), look_at 공식(z축=카메라-타겟
    # 정규화, x축=world_up x z 정규화, y축=z x x)으로 미리 계산한 값.
    quat=(0.69064452, 0.15169098, -0.15169098, -0.69064452),
    fovy=70.0,
)
DEFAULT_WRIST_CAMERA_OFFSET = (0.0, 0.0, 0.03)  # ee_base_link 로컬 좌표 -- 손가락 뒤로 살짝 물러난 자리
DEFAULT_WRIST_CAMERA_FOVY = 70.0


def build_roco_model(
    base_xml_path: str,
    head_camera: CameraConfig = DEFAULT_HEAD_CAMERA,
    wrist_camera_offset: tuple[float, float, float] = DEFAULT_WRIST_CAMERA_OFFSET,
    wrist_camera_fovy: float = DEFAULT_WRIST_CAMERA_FOVY,
) -> mujoco.MjModel:
    """base_xml_path(OpenArm 양팔 베이스 위에 어떤 태스크든)를 MjSpec으로
    읽어서 head/left_hand/right_hand 카메라가 없으면 추가하고 컴파일한
    MjModel을 돌려준다. 이미 그 이름의 카메라가 있으면 건드리지 않는다
    (idempotent -- 이미 roco 카메라가 박혀있는 XML을 다시 넣어도 안전).
    XML 텍스트로 왕복하지 않는다(위 모듈 docstring 참고) -- 호출자는
    돌아온 MjModel을 기존 Sim 클래스의 `.model`/`.data`에 그대로 끼워
    넣으면 된다(바디/조인트/액추에이터 id는 원본과 동일하게 유지됨,
    카메라만 추가되므로)."""
    spec = mujoco.MjSpec.from_file(base_xml_path)
    existing_cam_names = {cam.name for cam in spec.cameras}

    if "head" not in existing_cam_names:
        world = spec.worldbody
        world.add_camera(name="head", pos=list(head_camera.pos), fovy=head_camera.fovy, quat=list(head_camera.quat))

    for side in _SIDES:
        cam_name = f"{side}_hand"
        if cam_name in existing_cam_names:
            continue
        body = spec.body(ee_body_name(side))
        body.add_camera(name=cam_name, pos=list(wrist_camera_offset), fovy=wrist_camera_fovy)

    return spec.compile()


def render_roco_cameras(
    renderer: mujoco.Renderer, data: mujoco.MjData
) -> dict[str, np.ndarray]:
    """head/left_hand/right_hand 세 장을 한 번에 렌더링한다 -- renderer는
    호출자가 `mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)`
    로 미리 만들어둔 것(매 프레임 새로 만들면 느리다)."""
    frames = {}
    for name in CAMERA_NAMES:
        renderer.update_scene(data, camera=name)
        frames[name] = renderer.render().copy()
    return frames
