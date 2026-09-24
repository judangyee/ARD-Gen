"""ARD-Gen 3단계(Stabilizer) 검증: 양팔 peg-in-hole 시뮬레이션.

assets/peg_in_hole_bimanual.xml(단일 팔 버전 + 왼팔(Stabilizer) + hole을
왼팔이 쥐는 자유 바디로 바꾼 것)을 오른팔은 sim/peg_in_hole_sim.py의
PegInHoleSim을 그대로 상속해서 쓴다 -- 오른팔의 이름/게인/admittance
공식은 전혀 안 건드렸다. 이 파일이 추가하는 건 딱 두 가지:

1. reset() 시 왼팔을 "hole을 쥔 홈 자세"로 세팅한다(픽업 애니메이션 없이
   처음부터 쥐고 있는 상태로 시작 -- assets 파일의 weld가 이미
   active="true"라서 별도로 켤 필요도 없다).
2. hole_socket이 이제 world-body_pos가 아니라 freejoint qpos로 위치가
   정해지므로, PegInHoleSim.reset()의 "hole_pos_xy로 body_pos를 덮어쓰는"
   방식이 그대로는 안 먹힌다. **이 버전에서는 hole_pos_xy 시나리오
   무작위화를 지원하지 않는다**(왼팔이 hole을 쥔 채로 hole만 다른 곳으로
   옮기려면 왼팔의 베이스나 홈 자세도 같이 바꿔야 해서, 그건 이 파일의
   범위를 넘는 다음 작업이다) -- hole_pos_xy가 0이 아니면 명시적으로
   에러를 낸다(조용히 무시하지 않음).

## 지금까지 확인한 것 / 아직 확인 안 된 것

모델이 컴파일되고, 왼팔 홈 자세에서 hole_socket이 정확히 원래(단일 팔
버전과 동일) 위치 (0.10,0,0.0438), 무회전에 오는 것까지는 실측 확인했다
(assets/peg_in_hole_bimanual.xml 상단 docstring 참고). **오른팔의 기존
admittance 게인(Kp_xy≈0.000515, Kd_xy≈2.4e-05, sim/cma_search.py가 찾은
값)이 hole이 완전 고정이 아니라 왼팔 weld를 거친 상태에서도 여전히
peg를 삽입시키는지는 이 파일을 만들면서 처음 실측한다** -- 결과는 이
파일 __main__ 실행 결과 참고.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.peg_in_hole_sim import (
    PegInHoleSim,
    _default_scene_config,
    _run_episode_with_sim,
)

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "peg_in_hole_bimanual.xml")

# assets/peg_in_hole_bimanual.xml 상단 docstring에서 계산한, hole을
# (0.10,0,0.0438) 무회전에 정확히 두는 왼팔 홈 자세.
_LEFT_ARM_HOME_QPOS = {
    "waist_l": -1.5708,
    "shoulder_l": 0.700,
    "elbow_l": 0.912,
    "forearm_roll_l": 0.0,
    "wrist_angle_l": 1.533,
    "wrist_rotate_l": 0.0,
}
_LEFT_GRIPPER_CLOSED_CTRL = 0.024  # 오른팔 _GRIPPER_CLOSED_CTRL과 동일 (grasp 자체는 안 다룸)


class BimanualPegInHoleSim(PegInHoleSim):
    """PegInHoleSim을 상속해서 왼팔(Stabilizer) 초기화만 추가한다."""

    def __init__(self, xml_path: str | None = None):
        super().__init__(xml_path or _DEFAULT_XML)
        self._left_arm_qposadr = {
            name: self.model.joint(name).qposadr[0] for name in _LEFT_ARM_HOME_QPOS
        }
        self._left_arm_actuator_ids = {
            name: self.model.actuator(name).id for name in _LEFT_ARM_HOME_QPOS
        }
        self._left_gripper_actuator_id = self.model.actuator("gripper_l").id
        self._left_finger_l_qposadr = self.model.joint("left_finger_l").qposadr[0]

    def reset(self, scene_config: dict[str, Any]) -> float:
        if tuple(scene_config.get("hole_pos_xy", (0.0, 0.0))) != (0.0, 0.0):
            raise NotImplementedError(
                "BimanualPegInHoleSim은 아직 hole_pos_xy 무작위화를 지원하지 않는다 "
                "(hole이 왼팔에 쥐어진 자유 바디라, 위치를 바꾸려면 왼팔 베이스/홈 "
                "자세도 같이 다시 계산해야 함 -- 다음 작업)."
            )
        outer_half = super().reset(scene_config)

        for name, val in _LEFT_ARM_HOME_QPOS.items():
            self.data.qpos[self._left_arm_qposadr[name]] = val
            self.data.ctrl[self._left_arm_actuator_ids[name]] = val
        self.data.qpos[self._left_finger_l_qposadr] = _LEFT_GRIPPER_CLOSED_CTRL
        self.data.ctrl[self._left_gripper_actuator_id] = _LEFT_GRIPPER_CLOSED_CTRL

        mujoco.mj_forward(self.model, self.data)
        return outer_half


def run_episode(gains: dict[str, float], scene_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_scene_config()
    if scene_config:
        cfg.update(scene_config)
    sim = BimanualPegInHoleSim()
    return _run_episode_with_sim(sim, gains, cfg)


if __name__ == "__main__":
    # sim/peg_in_hole_sim.py의 README 실제 결과와 같은 대표 시나리오
    # (오프셋 14mm/0mm)에, 그 실측으로 찾은 기존 게인을 그대로 넣어서
    # hole이 왼팔에 쥐어진 상태에서도 여전히 성공하는지 확인한다.
    gains = {"Kp_xy": 0.000515, "Kd_xy": 2.4e-05}
    cfg = _default_scene_config()
    cfg["peg_init_offset_xy"] = (0.014, 0.0)
    result = run_episode(gains, cfg)
    print(
        f"[peg_in_hole_bimanual_sim] success={result['success']} "
        f"insertion_depth={result['insertion_depth']:.4f}m "
        f"max_force={result['max_force']:.2f}N "
        f"step_count={result['step_count']} reward={result['reward']:.2f}"
    )
