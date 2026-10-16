"""weld-grasp 태스크(peg_in_hole 등)에서 "파지 실패/미끄러짐"을 런타임에
흉내 내는 모델. ARD-VLA 폐루프(judangyee/ARD-VLA, closed-loop-structure)의
Judge가 F2(Actuator 파지 실패)/F3(Stabilizer 미끄러짐)를 학습하려면 그런
실패가 실제로 "일어날 수 있는" 시뮬이 필요한데, 지금 peg/hole은 그리퍼에
weld로 고정돼 있어서(assets/peg_in_hole_bimanual_openarm.xml의 <equality>,
그리퍼 접촉 자체는 <contact><exclude>로 꺼둠) 파지 실패가 원천적으로
불가능하다(PIPELINE.md "폐루프 데이터" 절 참고).

## 진짜 접촉 물리가 아니다 (명시적으로 밝힘)

이 모델은 그리퍼 손가락과 물체 사이의 마찰 접촉을 구현하지 않는다 --
weld는 계속 켜진 채로, 그 weld의 relpose(물체가 ee 로컬 프레임에서 "쥐어진"
위치)를 런타임에 조작해서 "미끄러짐"을 흉내 낸다. grasp anchor를 실제
접촉 메시 기준으로 재조정하는 데 6차례가 걸렸던 걸 감안하면(assets 파일
docstring 참고), 진짜 손가락 마찰 파지로 바꾸는 건 이 작업 범위 밖이다
(사용자 지시 "진짜 손가락 마찰 파지로 바꾸지 않는다").

## 메커니즘 (PIPELINE.md 0단계 실측 근거)

- weld 활성/비활성은 `data.eq_active[eq_id]`로 매 스텝 바로 토글된다.
- weld relpose(물체를 "쥔" 위치)는 `model.eq_data[eq_id][3:6]`(위치
  3원소, WELD eq_data 레이아웃은 anchor(3)+relpose_pos(3)+relpose_quat(4)+
  torquescale(1))를 바꾸면 다음 `mj_step`부터 반영되고, 물체가 새 목표를
  향해 점진적으로(텔레포트 아님) 수렴한다(실측: 1cm 튕긴 뒤 20스텝 정도
  걸려 수렴).
- weld 구속력은 `data.efc_force`에서 `efc_type==mjCNSTR_EQUALITY`이고
  `efc_id==eq_id`인 행들의 노름으로 읽는다.
- release(완전히 놓침) 시 `data.eq_active`만 끄면 안 된다 -- peg/hole
  바디 둘 다 XML에 `gravcomp="1"`이 박혀 있어서, weld만 꺼도 중력이
  상쇄된 채로 그냥 공중에 떠버린다(실측 확인, PIPELINE.md 0단계 2번
  참고). "진짜로 놓친 것처럼" 보이려면 release 시 그 바디의
  `model.body_gravcomp`도 같이 0으로 내려야 한다.

## 미끄러짐 방향 (force_limited) -- 단순화했다는 걸 명시

진짜 힘 방향(efc_force를 world 벡터로 복원하려면 그 weld의 제약
Jacobian을 통한 투영이 필요해서, 이 모듈이 가진 "힘의 크기(norm)"만으로는
방향까지 정확히 복원할 수 없다)이 아니라, **"현재 그 물체가 실제로 있는
위치" 쪽으로 relpose 목표를 조금씩 양보(yield)시키는** 방식으로
단순화했다 -- 과부하가 걸리면 "weld가 쥐고 있으려는 목표"와 "물체가 실제
외력에 밀려 있는 위치"의 차이(error) 중 일부(`yield_fraction`)만큼을
매 스텝 relpose에 영구히 반영한다. 정성적으로는 같은 효과(과부하가 걸린
방향으로 그립이 "밀려난다")를 내지만, 진짜 힘 방향 기반 미끄러짐이
필요해지면 `data.efc_force`를 weld의 제약 Jacobian(`data.efc_J`, 해당
행들)으로 역투영해서 world 힘 벡터를 복원하는 식으로 확장할 수 있다(이
파일은 그 확장을 안 했다 -- 범위 밖).

## 회전은 다루지 않는다 (명시적 한계)

relpose의 위치(pos) 성분만 조작한다 -- 회전(quat) 성분은 전부
`reset_episode()`가 원래 값으로 복원한 채로 고정이다. "파지 미끄러짐"을
위치 이동만으로 표현하는 건 단순화다(실제로는 손 안에서 물체가 돌아가는
미끄러짐도 있을 수 있음) -- 회전 미끄러짐을 추가하려면 `_rotate_relpose_quat`
같은 헬퍼를 relpose_quat(인덱스 6:10)에 작은 회전을 합성(quat multiply)
하는 식으로 넣으면 된다(이 파일은 범위상 안 했다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

_VALID_MODES = ("none", "force_limited", "injected", "release")


@dataclass
class ArmSlipConfig:
    """한쪽 팔(weld 하나)의 미끄러짐 설정. 기본값(`mode="none"`)이면 이
    모듈이 전혀 개입하지 않는다 -- 새 기능 기본 꺼짐 규칙."""

    mode: str = "none"
    # force_limited: 이 weld 구속력(efc_force 노름)이 이 값을 넘으면 그
    # 스텝에 미끄러짐을 적용한다. PIPELINE.md 0단계 3번 실측 분포(명목
    # peg_grasp mean=0.71/max=11.86, hole_grasp mean=10.04/max=18.55) 기준
    # 으로 "명목 범위보다 확실히 위"의 값을 호출부가 넘겨야 한다 --
    # 두 weld baseline이 전혀 다르므로 공통 기본값을 두지 않는다(None이면
    # force_limited를 켜도 아무 효과가 없다 -- 명시적으로 값을 받아야만
    # 동작, 실수로 켜져서 의도 없이 영향 주는 걸 막음).
    force_threshold: float | None = None
    # 과부하가 걸린 스텝마다 "목표-실제 위치" 차이 중 이만큼(0~1)을 그
    # 스텝의 relpose에 영구 반영한다(위 모듈 docstring "미끄러짐 방향"
    # 참고). 작을수록 여러 스텝에 걸쳐 서서히 미끄러진다.
    yield_fraction: float = 0.1

    # injected: 지정한 스텝에 지정한 위치 오프셋(ee 로컬 프레임, m)을
    # relpose에 한 번 더한다(교란 이벤트, 결정론적 -- 재현 가능).
    injected_at_step: int | None = None
    injected_offset_local: np.ndarray | None = None

    # release: 지정한 스텝에 weld를 비활성화하고(물체가 그 즉시 "놓임")
    # 그 바디의 body_gravcomp도 0으로 내린다(위 모듈 docstring 참고,
    # 안 그러면 떠있는 채로 안 떨어짐).
    release_at_step: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.mode == "force_limited" and self.force_threshold is None:
            raise ValueError("mode='force_limited'이면 force_threshold를 반드시 지정해야 한다")
        if self.mode == "injected" and (self.injected_at_step is None or self.injected_offset_local is None):
            raise ValueError("mode='injected'이면 injected_at_step/injected_offset_local을 반드시 지정해야 한다")
        if self.mode == "release" and self.release_at_step is None:
            raise ValueError("mode='release'이면 release_at_step을 반드시 지정해야 한다")


@dataclass
class GraspSlipConfig:
    """오른팔(peg_grasp, Actuator 파지 실패=F2)/왼팔(hole_grasp, Stabilizer
    미끄러짐=F3) 둘 다 독립적으로 설정한다 -- 사용자 지시 "둘 다 지원"."""

    right: ArmSlipConfig = field(default_factory=ArmSlipConfig)
    left: ArmSlipConfig = field(default_factory=ArmSlipConfig)

    @property
    def is_noop(self) -> bool:
        """둘 다 기본(mode='none')이면 True -- 이 경우 BimanualPegInHoleOpenArmSim은
        GraspSlipTracker를 아예 만들지 않는다(새 기능 꺼짐=비트 단위 동일
        규칙, __init__ 참고)."""
        return self.right.mode == "none" and self.left.mode == "none"


def weld_force_mag(data: mujoco.MjData, eq_id: int) -> float:
    """eq_id번 EQUALITY 제약의 구속력 노름(PIPELINE.md 0단계 2번 실측 방법
    그대로). 최대 6개 행(위치3+회전3)이 있고, 비활성(`eq_active=False`)
    이면 매칭되는 행이 없어 0.0을 돌려준다."""
    rows = []
    for i in range(data.nefc):
        if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY.value and data.efc_id[i] == eq_id:
            rows.append(data.efc_force[i])
    return float(np.linalg.norm(rows)) if rows else 0.0


@dataclass
class GraspState:
    """GraspSlipTracker.step()이 매 스텝 돌려주는, 그 스텝의 파지 상태
    (5단계 npz 스키마가 그대로 저장하는 필드들과 1:1 대응)."""

    weld_active: bool
    weld_force: float
    slipped_this_step: bool
    cumulative_shift_pos: np.ndarray  # (3,) m, 원래 relpose 위치 대비 누적 이동
    cumulative_shift_rot_deg: float = 0.0  # 항상 0.0 -- 회전은 안 다룸(모듈 docstring 참고)


class GraspSlipTracker:
    """weld 하나(eq_id)의 미끄러짐 상태를 에피소드 동안 관리한다. ee_body_id는
    relpose가 "이 바디 로컬 프레임 기준"이라는 걸 나타낸다(weld의 body1)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        eq_id: int,
        ee_body_id: int,
        held_body_id: int,
        config: ArmSlipConfig,
    ):
        self.eq_id = eq_id
        self.ee_body_id = ee_body_id
        self.held_body_id = held_body_id
        self.config = config
        self._orig_relpose_pos = model.eq_data[eq_id][3:6].copy()
        self._orig_body_gravcomp = float(model.body_gravcomp[held_body_id])
        self._released_done = False
        self._injected_done = False

    def reset_episode(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """매 에피소드 reset()에서 호출 -- relpose/eq_active/gravcomp를
        원래 값으로 복원하고 누적 상태를 초기화한다."""
        model.eq_data[self.eq_id][3:6] = self._orig_relpose_pos
        data.eq_active[self.eq_id] = True
        model.body_gravcomp[self.held_body_id] = self._orig_body_gravcomp
        self._released_done = False
        self._injected_done = False

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, step_idx: int) -> GraspState:
        """sim.step()이 mj_step을 부른 *뒤에* 호출한다(그 스텝에 솔버가 계산한
        구속력을 읽어야 하므로) -- 여기서 결정한 relpose/eq_active 변화는
        PIPELINE.md 0단계 2번 실측대로 *다음* mj_step부터 반영된다."""
        cfg = self.config
        force = weld_force_mag(data, self.eq_id)
        slipped = False

        if cfg.mode == "none":
            pass
        elif cfg.mode == "release":
            if not self._released_done and step_idx >= cfg.release_at_step:
                data.eq_active[self.eq_id] = False
                model.body_gravcomp[self.held_body_id] = 0.0
                self._released_done = True
                slipped = True
        elif cfg.mode == "injected":
            if not self._injected_done and step_idx >= cfg.injected_at_step:
                model.eq_data[self.eq_id][3:6] += cfg.injected_offset_local
                self._injected_done = True
                slipped = True
        elif cfg.mode == "force_limited":
            if force > cfg.force_threshold:
                slipped = self._yield_toward_actual(model, data)

        weld_active = bool(data.eq_active[self.eq_id])
        cumulative_shift = model.eq_data[self.eq_id][3:6] - self._orig_relpose_pos
        return GraspState(
            weld_active=weld_active,
            weld_force=force,
            slipped_this_step=slipped,
            cumulative_shift_pos=cumulative_shift.copy(),
        )

    def _yield_toward_actual(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """모듈 docstring "미끄러짐 방향" 절 그대로: 지금 relpose가 요구하는
        목표 위치(ee 프레임으로부터 FK)와 물체의 실제 위치 차이 중
        yield_fraction만큼을 relpose에 영구 반영한다."""
        ee_pos = data.xpos[self.ee_body_id]
        R = data.xmat[self.ee_body_id].reshape(3, 3)
        relpose_pos = model.eq_data[self.eq_id][3:6]
        target_world = ee_pos + R @ relpose_pos
        actual_world = data.xpos[self.held_body_id]
        error_world = actual_world - target_world
        if np.linalg.norm(error_world) < 1e-9:
            return False
        shift_world = self.config.yield_fraction * error_world
        model.eq_data[self.eq_id][3:6] += R.T @ shift_world
        return True
