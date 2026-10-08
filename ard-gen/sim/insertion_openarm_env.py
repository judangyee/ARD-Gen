"""insertion 태스크(OpenArm 양팔 버전)를 BaseTaskEnv 인터페이스로 감싼 것.

`sim/peg_in_hole_openarm_env.py`("원본")와 구조가 거의 동일하다 -- 물리/제어는
새로 만들지 않고 `sim/insertion_bimanual_openarm_sim.py`
(BimanualInsertionOpenArmSim)를 그대로 재사용하고, 그 모듈의
`__main__`(이 파일을 작성하면서 직접 실측 검증한 스크립트, 그 파일 하단 참고)
과 동일한 루프를 옮기되 reward/success만 compute_reward()/is_success()로
뽑아냈다. 왼팔이 generic Stabilizer가 아니라 진짜 7-DOF 팔이라는 점, 왼팔
qpos는 사실상 안 움직여도 매 스텝 기록해서 right_arm과 길이를 맞춘다는 점도
원본과 동일한 이유로 그대로 적용된다(그 파일 docstring 참고).

## reward: gym-aloha InsertionTask의 0~4 단계를 그대로(접촉쌍 기준)

`sim/insertion_bimanual_openarm_sim.py`의 `get_stage()`가 매 스텝 현재
접촉 상태로 2(바닥)/3(peg-socket 접촉)/4(peg-pin 접촉, 완전 삽입)를
판정한다(그 파일 docstring "보상 설계" 절 참고 -- gym-aloha 원본의 1~2단계는
이 설계(weld-grasp)에서 항상 참이라 바닥값 2로 압축했다). `compute_reward()`
는 에피소드 전체에서 도달한 **최고 단계**를 그대로 스칼라로 반환한다
(gym-aloha 자체 평가 관례 -- episode return = max(step rewards) -- 와
동일). `is_success()`는 그 최고 단계가 4(pin 접촉)인지를 본다. CMA-ES
목적함수로도 이 단계값을 그대로 쓴다(사용자 확인 -- sparse 보상이라 게인
탐색이 어려울 수 있다는 걸 알고도 gym-aloha 설계를 그대로 포팅하기로
결정했다).

## success: hold-count를 쓰지 않는 이유 (원본 peg_in_hole과 다른 지점)

원본은 `_SUCCESS_HOLD_STEPS`(20스텝 연속 유지)로 "성공 판정 버그"(xy가
틀어진 채 뚫고 지나가다 스치는 거짓 성공)를 막았다 -- 이건 **깊이 기반**
판정 특유의 문제다. 여기서는 처음에 똑같이 들여왔다가 실측에서 막혔다:
pin(반지름 6mm)이 peg tip(반지름 1cm)보다 얇아서, 완전히 정렬된 채
내려와도 peg가 pin 윗면 가장자리에서 미세하게(sub-mm) 걸리며 접촉이
매 스텝 깜빡인다(실측: depth가 30mm 근방에서 멈춘 채 stage가 2/3/4를
오가며 20스텝 연속을 못 채움) -- 접촉 자체는 실제로 반복해서 일어나는데
"연속 유지"라는 조건이 이 기하학적 떨림 때문에 거의 항상 실패한다.
gym-aloha 원본도 애초에 hold-count 개념이 없다(매 스텝 독립적으로
판정, dm_control의 episode return도 max(step rewards)일 뿐 "유지"를
요구하지 않는다) -- 그래서 지시대로 "그대로 이식"하려면 hold-count를
빼는 게 맞다. 대신 pin 접촉은 실제 물리 접촉 이벤트라(원본의 "허공에서
depth가 우연히 조건을 만족"하는 것과 다름) 한 번이라도 닿으면 그 자체로
유효한 성공 신호로 본다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.insertion_bimanual_openarm_sim import (
    DT,
    MAX_STEPS,
    BimanualInsertionOpenArmSim,
    adaptive_z_rate,
)
from sim.insertion_bimanual_openarm_sim import _default_scene_config as _sim_default_scene_config
from sim.insertion_bimanual_openarm_sim import sample_scene_config as _sim_sample_scene_config


def default_scene_config() -> dict[str, Any]:
    return _sim_default_scene_config()


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """원본과 동일한 이유로 "1단계 공유 씬" 별도 스키마가 없다(socket 위치는
    scene_config로 옮길 수 있는 자유도가 아님, sim 모듈 docstring 참고)."""
    if rng is None:
        rng = np.random.default_rng()
    return _sim_sample_scene_config(rng)


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class InsertionOpenArmEnv(BaseTaskEnv):
    """xml_path 하나만 받는다 -- stabilizer_config/use_stabilizer는 원본과
    동일한 이유로 받되 무시한다(tasks/insertion.yaml에 `stabilizer` 절이
    없으므로 TaskConfig.make_env()가 애초에 넘기지도 않는다)."""

    def __init__(
        self,
        xml_path: str | None = None,
        stabilizer_config: dict[str, Any] | None = None,
        use_stabilizer: bool = True,
    ):
        self._sim = BimanualInsertionOpenArmSim(xml_path=xml_path)
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
        """에피소드에서 도달한 최고 단계(2/3/4)를 그대로 반환한다 -- 위 모듈
        docstring 참고."""
        return float(episode_result["stage"])

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return episode_result["stage"] >= 4

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """sim/insertion_bimanual_openarm_sim.py의 __main__ 스크립트와 동일한
        루프 -- reward/success만 compute_reward()/is_success()를 거치도록
        뽑아냈고, left_arm_traj 기록(원본과 동일한 이유)을 추가했다."""
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
        max_stage = 2  # 바닥값 -- sim 모듈 docstring "보상 설계" 참고
        step_count = 0

        for step_count in range(1, MAX_STEPS + 1):
            force, torque = sim.get_force_torque()
            max_force_mag = max(max_force_mag, float(np.linalg.norm(force)))

            cur_peg_tip = sim.get_peg_tip_pos()
            cur_socket_center = sim.get_socket_center_pos()
            cur_dx = float(cur_socket_center[0] - cur_peg_tip[0])
            cur_dy = float(cur_socket_center[1] - cur_peg_tip[1])
            d_dx = (cur_dx - prev_dx) / DT
            d_dy = (cur_dy - prev_dy) / DT
            prev_dx, prev_dy = cur_dx, cur_dy
            delta_xy = np.array([kp_xy * cur_dx + kd_xy * d_dx, kp_xy * cur_dy + kd_xy * d_dy])

            cur_raw_depth = max(0.0, float(cur_socket_center[2] - cur_peg_tip[2]))
            z_rate = adaptive_z_rate(cur_dx, cur_dy, outer_half, cur_raw_depth, target_depth)
            delta = np.array([delta_xy[0], delta_xy[1], -z_rate])

            self.step(delta)
            left_arm_traj.append(sim.get_left_ee_pos())

            actions.append(delta.copy())
            forces.append(force)
            torques.append(torque)
            ee_poses.append(sim.get_ee_pose())

            stage = sim.get_stage()
            max_stage = max(max_stage, stage)

            if stage >= 4:
                break

        success = max_stage >= 4

        final_peg_tip = sim.get_peg_tip_pos()
        final_socket_center = sim.get_socket_center_pos()
        dist = float(np.linalg.norm(final_socket_center - final_peg_tip))

        episode_result = {
            "stage": max_stage,
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
            "stage": max_stage,
            "final_distance": dist,
            "max_force": max_force_mag,
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "left_arm_traj": np.stack(left_arm_traj).astype(np.float32),
        }
