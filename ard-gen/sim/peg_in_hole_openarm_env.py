"""peg-in-hole 태스크(OpenArm 양팔 버전)를 BaseTaskEnv 인터페이스로 감싼 것.

VX300s 버전(sim/peg_in_hole_env.py:PegInHoleEnv)과 마찬가지로 물리/제어
자체는 새로 만들지 않는다 -- sim/peg_in_hole_bimanual_openarm_sim.py의
BimanualPegInHoleOpenArmSim(검증된 위치-타겟팅 + 역동역학 컨트롤러, 그 모듈
docstring "xy를 접촉힘이 아니라 hole 실제 위치로 직접 타겟팅" 절 참고)을
그대로 재사용하고, 그 모듈의 module-level `_run_episode_with_sim()`과 동일한
루프를 옮기되 reward/success 공식만 compute_reward()/is_success()로 뽑아냈다.

## VX300s 버전과의 근본적 차이: 왼팔이 generic Stabilizer가 아니다

VX300s peg_in_hole/cap_twist는 왼팔이 sim/stabilizer.py의 가상 EE(3-슬라이드
조인트, 회전 없음, IK 불필요)로 구현돼 있어서 tasks/*.yaml의 `stabilizer`
절(object_body/grasp_offset)만 주면 어떤 물체든 잡게 할 수 있다. 하지만
OpenArm 버전은 왼팔이 **진짜 7-DOF 팔**이고, hole_socket을 쥔 채 이 태스크
내내 강화된 위치 게인(pos_left_j*)으로 고정돼 있다 -- 이 고정 자체가
assets/peg_in_hole_bimanual_openarm.xml + BimanualPegInHoleOpenArmSim.reset()
안에 이미 구워져 있어서(hole 목표점 계산도 왼팔의 고정 home 자세 FK 기준),
generic Stabilizer 클래스를 붙일 자리가 없다(붙일 필요도 없다 -- 물체가
이미 물리적으로 고정돼 있음). 그래서 이 Env는 stabilizer_config를 받지
않는다 -- tasks/peg_in_hole.yaml에도 `stabilizer` 절을 두지 않았다.

다만 5단계(언어 라벨링, ARD-VLA role classifier 학습)와 4단계
(pipeline/filter_episodes.py)는 여전히 "left_arm 궤적이 있으면 role=stabilizer로
저장한다"는 계약을 기대하므로, 왼팔이 실제로는 안 움직이더라도 매 스텝
왼팔 EE 위치를 기록해서 right_arm과 길이를 맞춘 left_arm_traj를 만들어
돌려준다(constant 값이지만 스키마 일관성을 위해 필요, pipeline/episode_io.py
모듈 docstring의 "왼팔은 접근 단계가 있고 오른팔은..." 문구는 VX300s 얘기고
여기선 둘 다 동시에 시작하므로 길이가 정확히 right_arm과 같다).

## 게인의 의미가 VX300s와 다르다 (이름은 같지만 단위가 다름)

VX300s의 Kp_xy/Kd_xy는 **접촉힘**(N)에 곱하는 admittance 게인이다. OpenArm
버전은 (sim 모듈 docstring "xy를 접촉힘이 아니라..." 절 참고) 접촉힘이 거의
항상 0이라 그 신호 자체가 없어서, Kp_xy/Kd_xy를 **위치 오차**(m)에 곱하는
게인으로 바꿨다 -- 그래서 tasks/peg_in_hole.yaml의 게인 bounds/x0/sigma0도
VX300s 값(0.00002~0.003 스케일)을 그대로 못 쓰고 완전히 다시 잡았다(이
파일이 아니라 tasks/peg_in_hole.yaml 참고, optimize/
peg_in_hole_openarm_admittance_gain_search.py로 실측 탐색한 결과).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.peg_in_hole_bimanual_openarm_sim import (
    DT,
    MAX_STEPS,
    _MAX_SANE_DEPTH_M,
    _SUCCESS_HOLD_STEPS,
    BimanualPegInHoleOpenArmSim,
    adaptive_z_rate,
)
from sim.peg_in_hole_bimanual_openarm_sim import _default_scene_config as _sim_default_scene_config
from sim.peg_in_hole_bimanual_openarm_sim import sample_scene_config as _sim_sample_scene_config


def default_scene_config() -> dict[str, Any]:
    return _sim_default_scene_config()


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """이미 sim 레벨(reset()이 바로 받는) 스키마를 그대로 뽑는다 -- VX300s
    쪽과 달리 "1단계 공유 씬" 별도 스키마(hole_pose 등)가 없다: OpenArm의
    hole 위치는 왼팔이 고정한 자세로 결정되고 scene_config로 옮길 수 있는
    자유도가 아니기 때문이다(위 모듈 docstring 참고). 그래서
    to_sim_scene_config()도 아래처럼 항등 함수다."""
    if rng is None:
        rng = np.random.default_rng()
    return _sim_sample_scene_config(rng)


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class PegInHoleOpenArmEnv(BaseTaskEnv):
    """xml_path 하나만 받는다 -- stabilizer_config/use_stabilizer는 VX300s
    버전과의 인터페이스 호환을 위해 받되 무시한다(위 모듈 docstring "근본적
    차이" 절 참고, tasks/peg_in_hole.yaml에 `stabilizer` 절이 없으므로
    TaskConfig.make_env()가 애초에 넘기지도 않는다)."""

    def __init__(
        self,
        xml_path: str | None = None,
        stabilizer_config: dict[str, Any] | None = None,
        use_stabilizer: bool = True,
    ):
        self._sim = BimanualPegInHoleOpenArmSim(xml_path=xml_path)
        self._outer_half = 0.0
        self._target_depth = 0.0

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> float:
        self._outer_half = self._sim.reset(scene_config)
        self._target_depth = scene_config["target_insertion_depth"]
        return self._outer_half

    def step(self, action: np.ndarray) -> None:
        self._sim.step(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """sim 모듈의 run_episode() 리워드 공식과 동일(VX300s와도 동일한 공식)."""
        reward = (
            -2.0 * episode_result["final_distance"]
            + 20.0 * episode_result["insertion_depth"]
            - 0.001 * max(0.0, episode_result["max_force"] - 5.0)
            - 0.01 * episode_result["step_count"]
        )
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return episode_result["insertion_depth"] >= self._target_depth

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """sim/peg_in_hole_bimanual_openarm_sim.py의 _run_episode_with_sim()과
        동일한 루프 -- reward/success만 compute_reward()/is_success()를 거치도록
        뽑아냈고, left_arm_traj 기록(위 모듈 docstring 참고)을 추가했다."""
        sim = self._sim
        outer_half = self.reset(scene_config)
        target_depth = scene_config["target_insertion_depth"]

        kp_xy = float(gains["Kp_xy"])
        kd_xy = float(gains["Kd_xy"])

        left_arm_traj = [sim.get_left_ee_pos()]
        ee_poses = [sim.get_ee_pose()]
        actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []

        prev_dx, prev_dy = 0.0, 0.0
        max_force_mag = 0.0
        insertion_depth = 0.0
        success = False
        hold_count = 0
        step_count = 0

        for step_count in range(1, MAX_STEPS + 1):
            force, torque = sim.get_force_torque()
            max_force_mag = max(max_force_mag, float(np.linalg.norm(force)))

            cur_peg_tip = sim.get_peg_tip_pos()
            cur_hole_center = sim.get_hole_center_pos()
            cur_dx = float(cur_hole_center[0] - cur_peg_tip[0])
            cur_dy = float(cur_hole_center[1] - cur_peg_tip[1])
            d_dx = (cur_dx - prev_dx) / DT
            d_dy = (cur_dy - prev_dy) / DT
            prev_dx, prev_dy = cur_dx, cur_dy
            delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

            cur_raw_depth = max(0.0, float(cur_hole_center[2] - cur_peg_tip[2]))
            z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
            delta = np.array([delta_xy[0], delta_xy[1], -z_rate])

            self.step(delta)
            left_arm_traj.append(sim.get_left_ee_pos())

            actions.append(delta.copy())
            forces.append(force)
            torques.append(torque)
            ee_poses.append(sim.get_ee_pose())

            peg_tip = sim.get_peg_tip_pos()
            hole_center = sim.get_hole_center_pos()
            dx = float(hole_center[0] - peg_tip[0])
            dy = float(hole_center[1] - peg_tip[1])
            xy_within_hole_footprint = abs(dx) < outer_half and abs(dy) < outer_half
            raw_depth = min(_MAX_SANE_DEPTH_M, max(0.0, float(hole_center[2] - peg_tip[2])))
            insertion_depth = raw_depth if xy_within_hole_footprint else 0.0

            if self.is_success({"insertion_depth": insertion_depth}):
                hold_count += 1
            else:
                hold_count = 0
            if hold_count >= _SUCCESS_HOLD_STEPS:
                success = True
                break

        final_peg_tip = sim.get_peg_tip_pos()
        final_hole_center = sim.get_hole_center_pos()
        dist = float(np.linalg.norm(final_hole_center - final_peg_tip))

        episode_result = {
            "insertion_depth": float(insertion_depth),
            "final_distance": dist,
            "max_force": max_force_mag,
            "step_count": step_count,
            "success": success,
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
            "insertion_depth": float(insertion_depth),
            "final_distance": dist,
            "max_force": max_force_mag,
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "left_arm_traj": np.stack(left_arm_traj).astype(np.float32),
        }
