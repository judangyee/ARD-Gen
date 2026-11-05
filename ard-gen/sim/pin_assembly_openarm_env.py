"""pin_assembly 태스크 -- RoCo식 "조립 + 오류 초기 상태 복구", OpenArm
양팔. sim/pin_assembly_openarm_sim.py(저수준 IK/운동학적 들기) 위에
시나리오 S1/S2/S3, 오류 상태 생성기, 복구 서브스킬, 프레임별 라벨링을
얹는다. roco_interface.py의 스키마/변환을 그대로 쓴다(사용자 지시 "프롬프트
1에서 만든 sim/roco_interface.py를 그대로 사용").

## 이 태스크의 관례: 왼팔=Actuator, 오른팔=정지 (peg_in_hole/tacker와 반대)

peg_in_hole/tacker는 오른팔이 움직이고 왼팔이 고정이었다 -- 이 태스크는
반대다. roco_interface.py의 스키마 자체는 어느 팔이 움직이는지 가정하지
않으므로(roco_interface.py 모듈 docstring 참고) 그대로 재사용 가능하고,
오히려 이 태스크가 "왼팔 액티브"라는 공식 RoCo 관례와 **일치**하는 첫
태스크다 -- validate_against_roco.py의 왼팔 회전범위/오른팔 정지 검사를
실제로 통과를 노려볼 수 있는 유일한 데이터다(peg_in_hole 데모는 반대
관례라 애초에 그 검사들을 통과할 수 없었다).

## 그립 핸들, 단순 형상 (사용자 지시 1)

assets/pin_assembly_openarm.xml 모듈 docstring 참고 -- 부품은 실제
기어/구멍을 깎지 않은 단순 원판+그립용 작은 핸들이다. "성공"은 접촉
기반 삽입이 아니라 위치/높이 기준(seated 판정, 아래 `is_seated()`)이다.

## 웨이포인트 프로파일: 호버->그립->들기->이동->호버->놓기->후퇴 (사용자 지시 3-2)

각 pick-place는 8단계(호버 위/그립 전/그립/들어올림/이동/호버 위/놓기/
후퇴)로 구성된다. 각 단계는 `_run_to_qpos()`가 `sim.advance_toward()`를
프레임마다 불러서(Jacobian 한 스텝씩, 매 프레임 실제 qpos에 직접 적용
-- sim/pin_assembly_openarm_sim.py 모듈 docstring "2차 설계" 참고) 목표에
거의 수렴할 때까지 진행한다 -- 멀 때는 등속으로 다가가다가 가까워지면
자연히 감속하는 모양이라, 이게 사용자 지시 3-2("처음 10프레임 유지 후
11프레임째 큰 점프, 이후 감속 수렴")가 요구하는 모양을 만든다. 에피소드
시작 시 HOLD_FRAMES(10) 동안은 홈 자세를 그대로 유지한 뒤 첫 웨이포인트로
넘어간다.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np

import sim.roco_interface as roco
from sim.base_task_env import BaseTaskEnv
from sim.pin_assembly_openarm_sim import (
    DISK_BODIES,
    LEFT_GRIPPER_GRASP_CTRL,
    LEFT_GRIPPER_OPEN_CTRL,
    PinAssemblyOpenArmSim,
)

HOVER_DZ = 0.15
PRE_GRASP_DZ = 0.025
GRASP_DZ = 0.012
PLACE_DZ = 0.015

HOLD_FRAMES = 10  # 사용자 지시 3-2: 처음 10프레임 유지

# 단계별 체류 프레임(10fps 캡처 -- sim 모듈 DT=0.01s와 1:1).
_DWELL = {
    "to_hover": 28,
    "descend_pre_grasp": 12,
    "descend_grasp": 8,
    "close_gripper": 6,
    "lift": 14,
    "transit_hover": 28,
    "descend_place": 14,
    "open_gripper": 6,
    "retreat": 12,
}

PIN_SITES = {
    "planet_1": "pin_peripheral_1_site",
    "planet_2": "pin_peripheral_2_site",
    "planet_3": "pin_peripheral_3_site",
    "sun": "pin_center_site",
}
STAGE_SITES = {
    "planet_1": "stage_slot_0_site",
    "planet_2": "stage_slot_1_site",
    "planet_3": "stage_slot_2_site",
    "sun": "stage_slot_3_site",
    "ring": "stage_slot_4_site",
}
DISK_HALF_HEIGHT = {"planet_1": 0.005, "planet_2": 0.005, "planet_3": 0.005, "sun": 0.005, "ring": 0.006}

SEAT_XY_TOL = 0.012
SEAT_Z_TOL = 0.015


@dataclasses.dataclass
class FrameLabel:
    phase: str  # "normal" | "failure" | "recovery"
    failure_type: str | None
    loss_weight: float


@dataclasses.dataclass
class EpisodeRecording:
    """run_episode()가 채우는 프레임별 원시 로그 -- roco 스키마 변환은
    pipeline 쪽(to_lerobot_pin_assembly.py)이 담당한다(roco_interface.py의
    "이 모듈이 안 다루는 것" 철학과 동일하게, 이 Env도 스키마 변환을
    직접 하지 않고 변환에 필요한 원재료만 모은다)."""

    left_ee_pos: list[np.ndarray] = dataclasses.field(default_factory=list)
    left_ee_quat: list[np.ndarray] = dataclasses.field(default_factory=list)
    right_ee_pos: list[np.ndarray] = dataclasses.field(default_factory=list)
    right_ee_quat: list[np.ndarray] = dataclasses.field(default_factory=list)
    left_jpos: list[np.ndarray] = dataclasses.field(default_factory=list)
    left_jvel: list[np.ndarray] = dataclasses.field(default_factory=list)
    right_jpos: list[np.ndarray] = dataclasses.field(default_factory=list)
    right_jvel: list[np.ndarray] = dataclasses.field(default_factory=list)
    left_gripper_raw: list[float] = dataclasses.field(default_factory=list)
    right_gripper_raw: list[float] = dataclasses.field(default_factory=list)
    phase: list[str] = dataclasses.field(default_factory=list)
    failure_type: list[str | None] = dataclasses.field(default_factory=list)
    loss_weight: list[float] = dataclasses.field(default_factory=list)
    images: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
    renderer: Any = None  # mujoco.Renderer | None -- set 전이면 record()가 이미지를 안 모은다(빠른 비-렌더링 실행용)

    def record(self, sim: PinAssemblyOpenArmSim, label: FrameLabel) -> None:
        lp, lq = sim.left_ee_pose()
        rp, rq = sim.right_ee_pose()
        self.left_ee_pos.append(lp)
        self.left_ee_quat.append(lq)
        self.right_ee_pos.append(rp)
        self.right_ee_quat.append(rq)
        self.left_jpos.append(sim.left_arm_qpos())
        self.left_jvel.append(sim.left_arm_qvel())
        self.right_jpos.append(sim.right_arm_qpos())
        self.right_jvel.append(sim.right_arm_qvel())
        self.left_gripper_raw.append(sim.left_gripper_raw())
        self.right_gripper_raw.append(sim.right_gripper_raw())
        self.phase.append(label.phase)
        self.failure_type.append(label.failure_type)
        self.loss_weight.append(label.loss_weight)
        if self.renderer is not None:
            self.images.append(roco.render_roco_cameras(self.renderer, sim.data))

    def __len__(self) -> int:
        return len(self.left_ee_pos)


_CONVERGE_TOL_M = 0.004


def _run_to_qpos(
    sim: PinAssemblyOpenArmSim,
    target_fingertip: np.ndarray,
    n_frames: int,
    rec: EpisodeRecording,
    label: FrameLabel,
    start_qpos: dict[str, float] | None = None,
) -> dict[str, float]:
    """`sim.advance_toward()`를 프레임마다 한 번씩 불러서(가상 목표를
    매 프레임 조금씩 전진, sim 모듈의 `advance_toward()` docstring
    참고) 목표에 거의 수렴할 때까지(`_CONVERGE_TOL_M` 이내) 진행한다 --
    목표가 멀면 등속으로 다가가다가 가까워지면 자연히 감속한다(사용자
    지시 3-2). n_frames는 "최소" 체류 프레임(호버 등 정확도가 중요하지
    않은 단계도 자연스러운 dwell을 주려고)이자 안전 상한의 기준치로
    쓴다 -- 수렴 전에 다음 단계로 넘어가면 그 시점의 위치오차가 그대로
    "그립 오프셋"에 반영돼 이후 전체 들기/놓기에 고정 편향으로 전파되는
    문제를 실측으로 발견했다(1차 설계: 고정 프레임 수만 돌림 -- 수렴
    전에 끝나서 놓는 위치가 몇 cm씩 어긋남). n_frames*3을 안전 상한으로
    둬서 수렴이 이상하게 느린 경우에도 무한루프에 빠지지 않게 한다.
    start_qpos는 호출자가 이어서 쓸 수 있도록 받지만, 실제 "현재
    목표"는 sim 내부 상태(`_virtual_qpos`)이므로 여기서는 그 상태를
    그대로 이어간다(start_qpos는 최초 호출에서만 의미 있음,
    sim.reset()이 LEFT_HOME_QPOS로 이미 초기화해뒀다)."""
    max_frames = max(n_frames, n_frames * 3)
    for i in range(max_frames):
        dist = sim.advance_toward(target_fingertip)
        sim.step_frame()
        rec.record(sim, label)
        if i + 1 >= n_frames and dist < _CONVERGE_TOL_M:
            break
    return sim.get_left_virtual_qpos()


def _hold(sim: PinAssemblyOpenArmSim, n_frames: int, rec: EpisodeRecording, label: FrameLabel) -> None:
    for _ in range(n_frames):
        sim.hold_left()  # sim 모듈의 hold_left() docstring 참고 -- 드리프트 방지
        sim.step_frame()
        rec.record(sim, label)


def pick_and_place(
    sim: PinAssemblyOpenArmSim,
    body: str,
    source_world_pos: np.ndarray,
    target_world_pos: np.ndarray,
    rec: EpisodeRecording,
    phase: str,
    failure_type: str | None = None,
    loss_weight: float = 1.0,
    start_qpos: dict[str, float] | None = None,
) -> dict[str, float]:
    """8단계 pick-place 한 번. source/target_world_pos는 물체가 "놓인
    자리"(원판 중심 높이, site_world_pos()나 disk_world_pose()에서 그대로
    얻은 값)다. 반환값은 마지막 qpos(다음 호출의 start_qpos로 이어주면
    IK가 매번 안전한 기본 시드에서 다시 출발 안 해도 됨 -- 연속적인
    자세 변화를 위해)."""
    label = FrameLabel(phase, failure_type, loss_weight)
    up = np.array([0.0, 0.0, 1.0])

    qpos = _run_to_qpos(sim, source_world_pos + HOVER_DZ * up, _DWELL["to_hover"], rec, label, start_qpos)
    qpos = _run_to_qpos(sim, source_world_pos + PRE_GRASP_DZ * up, _DWELL["descend_pre_grasp"], rec, label, qpos)
    qpos = _run_to_qpos(sim, source_world_pos + GRASP_DZ * up, _DWELL["descend_grasp"], rec, label, qpos)

    sim.set_left_gripper(LEFT_GRIPPER_GRASP_CTRL)
    _hold(sim, _DWELL["close_gripper"], rec, label)
    sim.start_carry(body)

    qpos = _run_to_qpos(sim, source_world_pos + HOVER_DZ * up, _DWELL["lift"], rec, label, qpos)
    qpos = _run_to_qpos(sim, target_world_pos + HOVER_DZ * up, _DWELL["transit_hover"], rec, label, qpos)
    qpos = _run_to_qpos(sim, target_world_pos + PLACE_DZ * up, _DWELL["descend_place"], rec, label, qpos)

    sim.stop_carry()
    # 운동학적 팔 제어(sim 모듈 advance_toward() docstring "2차" 참고)로도
    # 손끝이 목표에 수 mm 이내로 수렴하지만, 그 작은 잔차가 원판 중심
    # 좌표로 그대로 전파되면 seat 판정 허용오차(12mm)를 가끔 넘긴다 --
    # 놓는 순간 원판을 목표 자리에 정확히 스냅한다(완전한 물리 낙하
    # 시뮬레이션 대신 쓴 단순화, docs/pin_assembly.md에 정직하게 기록).
    qadr = sim._disk_qposadr[body]
    sim.data.qpos[qadr : qadr + 3] = [target_world_pos[0], target_world_pos[1], target_world_pos[2]]
    sim.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    dadr = sim._disk_dofadr[body]
    sim.data.qvel[dadr : dadr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    sim.set_left_gripper(LEFT_GRIPPER_OPEN_CTRL)
    _hold(sim, _DWELL["open_gripper"], rec, label)

    qpos = _run_to_qpos(sim, target_world_pos + HOVER_DZ * up, _DWELL["retreat"], rec, label, qpos)
    return qpos


def recover_misplaced_part(
    sim: PinAssemblyOpenArmSim,
    body: str,
    wrong_world_pos: np.ndarray,
    discard_world_pos: np.ndarray,
    rec: EpisodeRecording,
    start_qpos: dict[str, float] | None = None,
) -> dict[str, float]:
    """복구 서브스킬(사용자 지시 3): 잘못 놓인 부품(body, 현재
    wrong_world_pos에 있음 -- 예: sun이 다른 planet 원판 위)을 집어서
    discard_world_pos(해당 부품의 스테이징 슬롯)로 치운다. pick_and_place()
    를 그대로 재사용하지만 **phase="recovery"**로 라벨링한다(사용자
    지시 5 -- 복구 자체는 유용한 지도학습 신호라 loss_weight=1.0 유지,
    "오류로 이어진 직전 구간"(= 이미 잘못 놓인 상태를 가만히 들고
    시작하는 구간)만 phase="failure"/loss_weight 낮춤, run_episode()의
    S3 분기 참고). 분리된 함수로 둔 이유: 사용자 지시가 명시적으로
    "복구 서브스킬은 분리된 함수로 만든다"고 했고, 테스트에서 독립적으로
    호출/검증하기 위함."""
    return pick_and_place(
        sim, body, wrong_world_pos, discard_world_pos, rec, phase="recovery", failure_type=None, loss_weight=1.0,
        start_qpos=start_qpos,
    )


# ---------------------------------------------------------------------------
# 성공 판정
# ---------------------------------------------------------------------------


def is_seated(sim: PinAssemblyOpenArmSim, body: str, target_site: str) -> bool:
    """body(원판)가 target_site(핀) 기준으로 xy/z 허용오차 안에 올바르게
    놓여 있는지 -- 실제 삽입 접촉이 아니라 위치 기준 판정(모듈 docstring
    "그립 핸들, 단순 형상" 참고)."""
    pos, _quat = sim.disk_world_pose(body)
    target = sim.site_world_pos(target_site)
    xy_err = float(np.linalg.norm(pos[:2] - target[:2]))
    z_err = float(abs(pos[2] - target[2] - DISK_HALF_HEIGHT[body]))
    return xy_err <= SEAT_XY_TOL and z_err <= SEAT_Z_TOL


def count_correctly_assembled(sim: PinAssemblyOpenArmSim) -> int:
    """사용자 지시 2: "성공 지표는 올바르게 조립된 부품 수." planet x3 +
    sun은 각자의 핀에 seated인지, ring은 중앙(carrier 전체를 덮는 자리)에
    xy가 있고 바닥(carrier_top)보다 높이 떠 있는지(= 뭔가 위에 안착,
    모듈 상단 docstring "ring" 바디 주석 참고)로 판정한다."""
    n = 0
    for body, site in PIN_SITES.items():
        if is_seated(sim, body, site):
            n += 1
    ring_pos, _ = sim.disk_world_pose("ring")
    carrier_site = sim.site_world_pos("pin_center_site")
    ring_xy_err = float(np.linalg.norm(ring_pos[:2] - carrier_site[:2]))
    ring_rest_z = carrier_site[2] + 2 * DISK_HALF_HEIGHT["sun"] + DISK_HALF_HEIGHT["ring"]
    if ring_xy_err <= SEAT_XY_TOL * 1.5 and abs(ring_pos[2] - ring_rest_z) <= SEAT_Z_TOL:
        n += 1
    return n


# ---------------------------------------------------------------------------
# 초기 상태(시나리오별) + 오류 상태 생성기
# ---------------------------------------------------------------------------


def _resting_pose(sim: PinAssemblyOpenArmSim, body: str, site: str, xy_jitter: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    p = sim.site_world_pos(site).copy()
    if xy_jitter is not None:
        p[:2] += xy_jitter
    p[2] += DISK_HALF_HEIGHT[body]
    return p, np.array([1.0, 0.0, 0.0, 0.0])


def build_initial_disk_poses(sim: PinAssemblyOpenArmSim, scenario: str, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """scenario: "S1"(처음부터) | "S2"(planet 3개 완료) | "S3"(sun이 임의
    planet 위에 잘못 얹힌 오류 상태). S3의 오류 위치는 `generate_error_state()`
    로 물리적 유효성까지 검사한 뒤 쓴다(사용자 지시 4)."""
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    jitter = lambda scale: rng.uniform(-scale, scale, size=2)

    if scenario == "S1":
        for body, site in STAGE_SITES.items():
            poses[body] = _resting_pose(sim, body, site, jitter(0.004))
        return poses

    if scenario in ("S2", "S3"):
        for body in ("planet_1", "planet_2", "planet_3"):
            poses[body] = _resting_pose(sim, body, PIN_SITES[body], jitter(0.002))
        poses["ring"] = _resting_pose(sim, "ring", STAGE_SITES["ring"], jitter(0.004))
        if scenario == "S2":
            poses["sun"] = _resting_pose(sim, "sun", STAGE_SITES["sun"], jitter(0.004))
        else:
            wrong_body, wrong_pos = generate_error_state(sim, poses, rng)
            poses["sun"] = wrong_pos
        return poses

    raise ValueError(f"unknown scenario {scenario!r}")


def generate_error_state(
    sim: PinAssemblyOpenArmSim, planet_poses: dict[str, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator, max_tries: int = 20
) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    """사용자 지시 4: S3의 오류 초기 상태를 무작위(어느 planet 위인지,
    위치 오차)로 만들고, 실제로 물리적으로 "얹혀 있는지"(짧게 물리를
    돌려서 sun이 그 planet 위에 안정적으로 머무는지) 검사해서 아니면
    폐기하고 다시 뽑는다. 반환: (고른 planet 이름, sun의 (pos,quat))."""
    planet_names = ("planet_1", "planet_2", "planet_3")
    for _ in range(max_tries):
        chosen = planet_names[rng.integers(0, 3)]
        planet_pos, _ = planet_poses[chosen]
        offset = rng.uniform(-0.006, 0.006, size=2)
        sun_z = planet_pos[2] + DISK_HALF_HEIGHT[chosen] + DISK_HALF_HEIGHT["sun"] + 0.0015
        sun_pos = np.array([planet_pos[0] + offset[0], planet_pos[1] + offset[1], sun_z])
        sun_quat = np.array([1.0, 0.0, 0.0, 0.0])

        trial_poses = dict(planet_poses)
        trial_poses["sun"] = (sun_pos, sun_quat)
        trial_poses["ring"] = _resting_pose(sim, "ring", STAGE_SITES["ring"])
        sim.reset(trial_poses)
        for _ in range(40):
            sim.step_frame()
        settled_pos, _ = sim.disk_world_pose("sun")
        xy_drift = float(np.linalg.norm(settled_pos[:2] - planet_pos[:2]))
        still_on_top = settled_pos[2] > planet_pos[2] + DISK_HALF_HEIGHT[chosen] + 0.003
        if xy_drift <= 0.015 and still_on_top:
            return chosen, (settled_pos, sim.disk_world_pose("sun")[1])
    raise RuntimeError(f"generate_error_state: {max_tries}번 시도해도 물리적으로 유효한 오류 상태를 못 만듦")


# ---------------------------------------------------------------------------
# 언어 지시문 (사용자 지시 5 -- 기존 템플릿 방식, 시나리오별)
# ---------------------------------------------------------------------------

_LANGUAGE_TEMPLATES = {
    "S1": "Pick up the three planet disks and fit them onto the peripheral pins, then fit the sun disk onto the center pin, then place the ring over the whole assembly.",
    "S2": "The three planet disks are already in place. Fit the sun disk onto the center pin, then place the ring over the whole assembly.",
    "S3": "The sun disk was placed on top of a planet disk by mistake. Remove it, set it aside, then fit it onto the center pin and place the ring over the whole assembly.",
}


def language_instruction(scenario: str) -> str:
    return _LANGUAGE_TEMPLATES[scenario]


# ---------------------------------------------------------------------------
# 에피소드 실행
# ---------------------------------------------------------------------------


def run_episode(
    scenario: str,
    rng: np.random.Generator,
    sim: PinAssemblyOpenArmSim | None = None,
    renderer: Any = None,
) -> tuple[EpisodeRecording, dict[str, Any]]:
    """renderer(mujoco.Renderer, roco.build_roco_model()으로 만든 카메라
    포함 모델 기준)를 주면 매 프레임 head/left_hand/right_hand 이미지도
    `EpisodeRecording.images`에 모은다(기본값 None이면 렌더링 없이
    빠르게 궤적만 생성 -- IK 도달성 사전 검사 등에서 씀)."""
    sim = sim or PinAssemblyOpenArmSim()
    disk_poses = build_initial_disk_poses(sim, scenario, rng)
    sim.reset(disk_poses)

    rec = EpisodeRecording(renderer=renderer)
    normal_label = FrameLabel("normal", None, 1.0)
    rec.record(sim, normal_label)  # 프레임 0(초기 자세)

    qpos: dict[str, float] | None = None
    _hold(sim, HOLD_FRAMES - 1, rec, normal_label)  # 사용자 지시 3-2: 처음 10프레임 유지

    # ring은 sun/carrier보다 높이 떠서 안착한다(실제로 sun 위에 올라앉는
    # 형태 -- 물리 낙하 시뮬레이션 대신 스냅으로 구현하므로(위
    # pick_and_place() "원판을 목표 자리에 정확히 스냅" 주석 참고) 이
    # 높이도 명시적으로 계산해야 한다: carrier 표면 + sun 전체 두께 +
    # ring 반두께).
    carrier_top_z = sim.site_world_pos(PIN_SITES["sun"])[2]
    ring_rest_z = carrier_top_z + 2 * DISK_HALF_HEIGHT["sun"] + DISK_HALF_HEIGHT["ring"]

    # 이미 올바르게 놓은 원판을 그 뒤 pick-place(특히 운반 중인 원판과의
    # 충돌)가 건드려서 밀어내는 경우가 실측으로 있었다(carry 중인
    # 원판도 다른 원판과는 정상 충돌하므로, 호버 경로가 이미 놓인 원판
    # 바로 위를 지나가면 부딪힐 수 있음). pick_and_place() 한 번이
    # 끝날 때마다 "지금까지 올바르게 놓은" 원판들을 각자의 목표로 다시
    # 스냅해서 이 간섭을 상쇄한다(완전한 충돌 회피 경로 계획 대신 쓴
    # 단순화, docs/pin_assembly.md에 기록).
    placed: list[tuple[str, np.ndarray]] = []

    def _resnap_placed() -> None:
        for b, p in placed:
            qadr = sim._disk_qposadr[b]
            sim.data.qpos[qadr : qadr + 3] = p
            sim.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            sim.data.qvel[sim._disk_dofadr[b] : sim._disk_dofadr[b] + 6] = 0.0
        if placed:
            mujoco.mj_forward(sim.model, sim.data)

    def _place(body: str, src: np.ndarray, dst: np.ndarray, phase: str, qpos_in: dict[str, float] | None) -> dict[str, float]:
        """pick_and_place() 자체도 끝에서 한 번 스냅하지만(그 함수 docstring
        참고), 그 뒤 open_gripper/retreat 동안 실제 물리가 몇 프레임
        더 도는 사이에 **막 놓은 물체 자신도** 밀릴 수 있다는 걸 실측으로
        발견했다 -- 특히 ring처럼 다른 원판 위(평면 대 평면 접촉이라
        옆으로 미끄러지기 쉬움)에 얹히는 경우. 그래서 이번에 놓은 body를
        먼저 placed에 추가한 뒤 resnap을 불러서 자기 자신도 포함되게
        한다."""
        nonlocal placed
        q = pick_and_place(sim, body, src, dst, rec, phase=phase, start_qpos=qpos_in)
        placed.append((body, dst))
        _resnap_placed()
        return q

    if scenario == "S1":
        order = [("planet_1", STAGE_SITES["planet_1"], PIN_SITES["planet_1"])]
        order += [("planet_2", STAGE_SITES["planet_2"], PIN_SITES["planet_2"])]
        order += [("planet_3", STAGE_SITES["planet_3"], PIN_SITES["planet_3"])]
        order += [("sun", STAGE_SITES["sun"], PIN_SITES["sun"])]
        for body, src_site, dst_site in order:
            src = sim.site_world_pos(src_site) + np.array([0, 0, DISK_HALF_HEIGHT[body]])
            dst = sim.site_world_pos(dst_site) + np.array([0, 0, DISK_HALF_HEIGHT[body]])
            qpos = _place(body, src, dst, "normal", qpos)
        ring_src = sim.site_world_pos(STAGE_SITES["ring"]) + np.array([0, 0, DISK_HALF_HEIGHT["ring"]])
        ring_dst = sim.site_world_pos(PIN_SITES["sun"]) + np.array([0, 0, ring_rest_z - carrier_top_z])
        qpos = _place("ring", ring_src, ring_dst, "normal", qpos)

    elif scenario == "S2":
        # planet 3개는 이미 정답 자리에 있다 -- 뒤이은 pick-place가
        # 건드리지 않도록 "이미 놓인" 목록에 처음부터 포함한다. 목표를
        # "현재 측정값"이 아니라 핀 사이트에서 다시 계산한 값으로 두고
        # 바로 재스냅한다 -- 처음 HOLD_FRAMES 동안의 물리 스텝(접촉
        # 해소)만으로도 서로 가까이 놓인 원판 3개가 몇 cm씩 밀리는 걸
        # 실측으로 발견했다(원판 반지름 대비 핀 간격이 빡빡해서).
        for body in ("planet_1", "planet_2", "planet_3"):
            placed.append((body, sim.site_world_pos(PIN_SITES[body]) + np.array([0, 0, DISK_HALF_HEIGHT[body]])))
        _resnap_placed()
        sun_src = sim.site_world_pos(STAGE_SITES["sun"]) + np.array([0, 0, DISK_HALF_HEIGHT["sun"]])
        sun_dst = sim.site_world_pos(PIN_SITES["sun"]) + np.array([0, 0, DISK_HALF_HEIGHT["sun"]])
        qpos = _place("sun", sun_src, sun_dst, "normal", qpos)
        ring_src = sim.site_world_pos(STAGE_SITES["ring"]) + np.array([0, 0, DISK_HALF_HEIGHT["ring"]])
        ring_dst = sim.site_world_pos(PIN_SITES["sun"]) + np.array([0, 0, ring_rest_z - carrier_top_z])
        qpos = _place("ring", ring_src, ring_dst, "normal", qpos)

    elif scenario == "S3":
        for body in ("planet_1", "planet_2", "planet_3"):
            placed.append((body, sim.site_world_pos(PIN_SITES[body]) + np.array([0, 0, DISK_HALF_HEIGHT[body]])))
        _resnap_placed()
        # "오류로 이어진 직전 구간"(= 오류 상태를 그대로 들고 시작하는
        # 구간) -- 이미 잘못 놓인 상태라 "이 행동이 좋은 예"라고 학습시킬
        # 이유가 없어서 loss_weight를 낮춘다(사용자 지시 5).
        wrong_pos, _ = sim.disk_world_pose("sun")
        discard_pos = sim.site_world_pos(STAGE_SITES["sun"]) + np.array([0, 0, DISK_HALF_HEIGHT["sun"]])
        qpos = recover_misplaced_part(sim, "sun", wrong_pos, discard_pos, rec, start_qpos=qpos)
        _resnap_placed()

        sun_dst = sim.site_world_pos(PIN_SITES["sun"]) + np.array([0, 0, DISK_HALF_HEIGHT["sun"]])
        qpos = _place("sun", discard_pos, sun_dst, "normal", qpos)
        ring_src = sim.site_world_pos(STAGE_SITES["ring"]) + np.array([0, 0, DISK_HALF_HEIGHT["ring"]])
        ring_dst = sim.site_world_pos(PIN_SITES["sun"]) + np.array([0, 0, ring_rest_z - carrier_top_z])
        qpos = _place("ring", ring_src, ring_dst, "normal", qpos)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    # "오류로 이어진 직전 구간" 라벨링: S3의 맨 첫 HOLD_FRAMES(오류 상태를
    # 그대로 들고 시작하는 구간)만 되돌아가서 phase="failure"/loss_weight
    # 낮춤으로 바꾼다(루프 중에는 아직 "오류 상태였다"는 사실 자체가
    # 변하지 않으므로 사후에 일괄 처리하는 게 더 명확하다).
    if scenario == "S3":
        for i in range(HOLD_FRAMES):
            rec.phase[i] = "failure"
            rec.failure_type[i] = "sun_on_planet"
            rec.loss_weight[i] = 0.0

    n_assembled = count_correctly_assembled(sim)
    episode_success = n_assembled == 5
    info = {
        "scenario": scenario,
        "n_assembled": n_assembled,
        "episode_success": episode_success,
        "language": language_instruction(scenario),
        "n_frames": len(rec),
    }
    return rec, info


# ---------------------------------------------------------------------------
# sim/task_registry.py 호환 래퍼 (사용자 지시 범위 밖이지만, 이 저장소의
# "새 태스크는 config 하나 + Env 클래스 하나로 추가된다" 관례를 따르려고
# 최소한으로 추가했다). 이 태스크는 CMA-ES로 탐색할 "게인"이 없는
# 전문가-스크립트 태스크라서(사용자 지시 3 "전문가 동작 스크립트"),
# gains는 자리만 채우는 더미(RNG 시드 하나)다 -- optimize/cma_search.py로
# 이 태스크를 실제로 최적화할 일은 없지만, load_task_config("pin_assembly")
# 가 깨지지 않게만 해둔다.
# ---------------------------------------------------------------------------


def default_scene_config() -> dict[str, Any]:
    return {"scenario": "S1", "seed": 0}


def sample_scene_config(rng: np.random.Generator) -> dict[str, Any]:
    scenario = ("S1", "S2", "S3")[int(rng.integers(0, 3))]
    return {"scenario": scenario, "seed": int(rng.integers(0, 2**31 - 1))}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class PinAssemblyOpenArmEnv(BaseTaskEnv):
    """BaseTaskEnv 호환 래퍼 -- 실제 로직은 run_episode()(이 모듈 함수)에
    있다. gains는 안 쓴다(더미, 위 모듈 절 docstring 참고)."""

    def __init__(self, xml_path: str | None = None):
        self._sim = PinAssemblyOpenArmSim(xml_path=xml_path)
        self._last_result: dict[str, Any] = {}

    def reset(self, scene_config: dict[str, Any]) -> Any:
        rng = np.random.default_rng(scene_config.get("seed", 0))
        disk_poses = build_initial_disk_poses(self._sim, scene_config.get("scenario", "S1"), rng)
        self._sim.reset(disk_poses)

    def step(self, action: Any) -> Any:
        raise NotImplementedError("pin_assembly는 run_episode()로 전체 에피소드를 한 번에 실행한다(전문가 스크립트, 사용자 지시 3)")

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        return float(episode_result.get("n_assembled", 0))

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return bool(episode_result.get("episode_success", False))

    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        rng = np.random.default_rng(scene_config.get("seed", 0))
        rec, info = run_episode(scene_config.get("scenario", "S1"), rng, sim=self._sim)
        info["reward"] = self.compute_reward(info)
        self._last_result = info
        return info
