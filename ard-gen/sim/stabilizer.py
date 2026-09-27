"""Stabilizer(왼팔): peg_in_hole/cap_twist 공용, IK가 필요 없는 가상
엔드이펙터 + weld로 물체를 고정한다.

## 왜 실제 팔(Jacobian IK) 대신 3-슬라이드 조인트 가상 엔드이펙터인가

이전 시도(assets/peg_in_hole_bimanual.xml, sim/peg_in_hole_bimanual_sim.py,
git log 참고)는 실제 VX300s 팔을 왼팔로도 하나 더 붙였는데, 그러면 "hole을
쥔 채로 정확히 원래 위치/자세에 오는" 왼팔 홈 자세를 매번 손으로(그리드
서치로) 다시 풀어야 했다 -- 그 파일 자체가 "hole_pos_xy 무작위화를 아직
지원 못 한다"고 명시할 만큼 범용성이 없었다. 이번 요청은 peg_in_hole과
cap_twist 둘 다에 **같은 코드로** 적용돼야 하는데, cap_twist는 애초에
오른팔조차 실제 팔 기구학을 모델링하지 않는 태스크(손목 회전 = cap
힌지 각도 직접 구동)라 왼팔만 실제 팔로 만드는 것도 일관성이 없다.

그래서 Stabilizer의 팔 기구학 자체를 모델링하지 않는다 -- 이 프로젝트가
이미 grasp 물리(peg-그리퍼 결합, cap_twist의 wrist=힌지 직접 구동)를
모델링하지 않는 것과 같은 선상의 단순화다. 대신 world에 직접 매인
3-슬라이드 조인트 점 하나(assets/*.xml의 stabilizer_ee body)로 표현한다:
직렬 링크가 없으므로 "역기구학"이 항등함수가 되어(world 목표 좌표를 그대로
ctrl에 넣으면 끝) 태스크마다 홈 자세를 다시 풀 필요가 없고, 물체가 어디에
있든(peg_in_hole의 hole_pos_xy 무작위화 포함) 곧장 접근할 수 있다 --
"IK 기반, 파라미터 탐색 불필요"를 문자 그대로 만족한다.

## 접근(approach) -> weld -> 유지(hold)

1. reset(): 물체의 **이 시점**(에피소드 시작, Actuator가 아직 아무것도
   건드리지 않은 시점) world 위치에 태스크별 grasp_offset(world-axis-aligned
   오프셋, tasks/*.yaml의 stabilizer.grasp_offset)을 더해 목표를 정하고,
   3개 위치 액추에이터에 그 목표를 바로 명령한다. 물체는 이 시점 이후로도
   Actuator의 반작용이 시작되기 전까지는 그대로 있으므로, 이 목표가
   approach 내내 유효하다.
2. tick(): 매 물리 스텝 뒤 호출한다. 아직 weld 전이면 EE가 목표에
   충분히 가까워졌는지(REACH_TOL_M) 또는 최대 대기 스텝(APPROACH_MAX_STEPS)을
   넘겼는지 확인하고, 둘 중 하나면 **그 순간의 실제 상대 포즈**를 weld의
   relpose로 굳혀서 활성화한다 -- 액추에이터 추종 오차가 약간 있어도
   상관없이 "도달한 그 자리에서" 딱 붙는다(오차를 미리 완벽히 없앨 필요가
   없다는 뜻).
3. weld 이후에는 물체가 EE에 강체로 고정되므로 별도 hold 로직이 필요 없다
   -- 다만 EE 자신의 위치 액추에이터도 같은 목표를 계속 명령해서, EE
   자체가 반작용에 밀려나지 않도록 이중으로 버틴다(weld의 잔류 컴플라이언스
   + EE 자신의 강성, 둘 다 관여 -- assets/peg_in_hole.xml의 weld 주석,
   그리고 이전 시도에서 실제로 "weld는 버텼는데 팔 관절 자체가 처져서
   밀렸다"는 걸 겪은 교훈 참고).

## eq_data 레이아웃 (실측으로 확인, 문서가 없어서 직접 테스트해서 알아냄)

MuJoCo의 weld equality는 eq_data(11칸)를
`[anchor(3), relpos(3), relquat(4), torquescale(1)]`로 채운다 -- **relpos가
0번이 아니라 3번부터 시작한다** (처음엔 relpos를 0:3에, relquat을 3:7에
쓰는 실수를 했다가, 활성화 즉시 물체가 전혀 엉뚱한 자리로 튕겨나가는 걸
보고 컴파일된 모델의 eq_data 기본값을 직접 찍어봐서 실제 레이아웃을
확인했다). anchor(body1 로컬 프레임 기준)는 0으로 둔다 -- body1(EE)의
원점을 그대로 기준점으로 쓴다는 뜻이라, relpos만 계산하면 된다.

## 방향(orientation)을 액추에이트하지 않는 이유

EE 몸체는 회전 조인트가 없다(항상 컴파일 시 방향 = identity). 두 태스크
모두 물체를 특정 각도로 재배치할 필요가 없어서(peg_in_hole의 hole_pose
theta는 아직 물리에 반영되지 않음, PIPELINE.md 참고) -- weld 시점에
물체의 **현재** 방향을 그대로 굳히면 충분하다.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

_EE_ACTUATOR_NAMES = ("stab_x", "stab_y", "stab_z")
_EE_BODY_NAME = "stabilizer_ee"
_WELD_NAME = "stabilizer_weld"

APPROACH_MAX_STEPS = 60  # 이 스텝 안에 못 도달해도 그 자리에서 weld한다(억지로 붙잡지 않음)
REACH_TOL_M = 0.002  # 이 이내면 "도달"로 보고 조기에 weld한다


class Stabilizer:
    """tasks/*.yaml의 stabilizer.object_body/grasp_offset만 읽으면, 어떤
    태스크의 어떤 물체에도 동일하게 동작한다(peg_in_hole의 hole_socket,
    cap_twist의 bottle 둘 다 이 클래스 하나로 다룬다)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, object_body_name: str, grasp_offset):
        self.model = model
        self.data = data
        self._object_body_id = model.body(object_body_name).id
        self._ee_body_id = model.body(_EE_BODY_NAME).id
        self._actuator_ids = [model.actuator(n).id for n in _EE_ACTUATOR_NAMES]
        self._weld_id = model.equality(_WELD_NAME).id
        self._grasp_offset = np.array(grasp_offset, dtype=float)

        self._target: np.ndarray | None = None
        self._grasped = False
        self._steps = 0

    @property
    def grasped(self) -> bool:
        return self._grasped

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ee_body_id].copy()

    def reset(self) -> None:
        """weld를 끄고, 물체의 현재(에피소드 시작 시점) 위치 + grasp_offset을
        목표로 EE 액추에이터에 즉시 명령한다. 실제 도달/weld 판정은
        tick()에서 매 스텝 확인한다."""
        self.data.eq_active[self._weld_id] = 0
        obj_pos = self.data.xpos[self._object_body_id].copy()
        self._target = obj_pos + self._grasp_offset
        for act_id, val in zip(self._actuator_ids, self._target):
            self.data.ctrl[act_id] = float(val)
        self._grasped = False
        self._steps = 0

    def tick(self) -> None:
        """매 mj_step 이후 한 번씩 호출한다. weld 전이면 도달 여부만 확인하고,
        도달했으면(또는 너무 오래 걸렸으면) 그 순간의 실제 상대 포즈로 weld를
        활성화한다."""
        if self._grasped:
            return
        self._steps += 1
        ee_pos = self.data.xpos[self._ee_body_id]
        err = float(np.linalg.norm(ee_pos - self._target))
        if err <= REACH_TOL_M or self._steps >= APPROACH_MAX_STEPS:
            self._activate_weld()

    def _activate_weld(self) -> None:
        ee_pos = self.data.xpos[self._ee_body_id].copy()
        ee_mat = self.data.xmat[self._ee_body_id].reshape(3, 3).copy()
        obj_pos = self.data.xpos[self._object_body_id].copy()
        obj_quat = self.data.xquat[self._object_body_id].copy()

        # relpos/relquat = 물체의 현재 포즈를 EE(body1) 로컬 프레임으로 표현한 것.
        rel_pos = ee_mat.T @ (obj_pos - ee_pos)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())
        ee_quat_inv = np.zeros(4)
        mujoco.mju_negQuat(ee_quat_inv, ee_quat)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, ee_quat_inv, obj_quat)

        # eq_data 레이아웃(모듈 docstring 참고): anchor(0:3)=0, relpos(3:6), relquat(6:10), torquescale(10).
        self.model.eq_data[self._weld_id, 0:3] = 0.0
        self.model.eq_data[self._weld_id, 3:6] = rel_pos
        self.model.eq_data[self._weld_id, 6:10] = rel_quat
        self.model.eq_data[self._weld_id, 10] = 1.0
        self.data.eq_active[self._weld_id] = 1
        self._grasped = True


def run_approach_phase(
    stabilizer: "Stabilizer | None", model: mujoco.MjModel, data: mujoco.MjData, n_substeps: int
) -> list[np.ndarray]:
    """Stabilizer.reset() 직후, run_episode()의 실제 제어 루프(Actuator)가
    시작되기 전에 호출한다 -- grasp될 때까지(또는 APPROACH_MAX_STEPS까지)
    물리만 진행하고, 그동안 Actuator는 아무것도 안 한다(ctrl을 안 건드려서
    reset()이 남긴 정지 상태 그대로 유지).

    ## 왜 tick()을 run_episode() 루프 "안"에서 부르면 안 되는가 (실측으로 발견)

    처음엔 Actuator 루프 매 스텝마다 stabilizer.tick()을 같이 불렀다 --
    그러면 Stabilizer가 목표에 도달하는 최대 APPROACH_MAX_STEPS(60)
    스텝 동안 물체가 **완전히 무구속** 상태인데 Actuator는 이미 1스텝째부터
    삽입/회전을 시작해버린다(같은 루프의 같은 반복이라 순서 경쟁이 생김).
    peg_in_hole 무작위 40씬 회귀에서 이걸로 측정한 성공률이 42.5%(Stabilizer
    전혀 없음)에서 57.5%로만 오르고 원래(고정 hole) 70%를 회복 못 했다 --
    Stabilizer가 붙잡기도 전에 반작용력이 이미 hole을 밀어낸 것. 이 함수로
    "먼저 확실히 쥔 뒤에" Actuator를 시작하도록 순서를 강제한다."""
    if stabilizer is None:
        return []
    traj: list[np.ndarray] = []
    while not stabilizer.grasped:
        mujoco.mj_step(model, data, nstep=n_substeps)
        stabilizer.tick()
        traj.append(stabilizer.ee_pos)
    return traj


def make_stabilizer(model: mujoco.MjModel, data: mujoco.MjData, task_raw_config: dict[str, Any]) -> "Stabilizer | None":
    """tasks/{name}.yaml의 최상위 `stabilizer` 절(있으면)로부터 Stabilizer를
    만든다. 절이 없으면 None을 반환해서, 호출자가 "Stabilizer 없이" 모드를
    구현할 필요 없이 그냥 이 반환값을 검사하기만 하면 되게 한다."""
    cfg = task_raw_config.get("stabilizer")
    if not cfg:
        return None
    return Stabilizer(model, data, cfg["object_body"], cfg["grasp_offset"])
