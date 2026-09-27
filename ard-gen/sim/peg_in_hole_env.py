"""peg-in-hole 태스크를 BaseTaskEnv 인터페이스로 감싼 것.

물리/IK 자체는 새로 만들지 않는다 -- sim/peg_in_hole_sim.py의 PegInHoleSim
(검증된 admittance controller 물리)을 그대로 재사용하고, run_episode()의
제어 루프도 원본 _run_episode_with_sim()과 동일하되, reward/success
공식만 compute_reward()/is_success()로 뽑아냈다(sim/base_task_env.py
상단 docstring 참고). sim/peg_in_hole_sim.py 자체는 건드리지 않았다 --
bootstrap/*.py 등 이 리팩토링 범위 밖의 다른 코드가 여전히 그 모듈을
직접 import하고 있어서(옛 방식), 거기에 영향을 주지 않기 위해서다.

리팩토링 전/후 --task peg_in_hole 결과가 완전히 같아야 하므로
(tests/test_regression_peg_in_hole.py), 원본 _run_episode_with_sim()에서
숫자/순서를 하나도 안 바꾸고 옮겼다.

이 모듈은 pipeline/scene_sampler.py에 있던 "1단계 공유 씬 설정" 샘플링
(sample_scene_config/to_sim_scene_config, 씬 무작위화 범위 포함)도 그대로
옮겨왔다 -- 원래도 peg-in-hole 전용 코드였으니 태스크 모듈 안에 있는 게
자연스럽다. pipeline/scene_sampler.py는 이제 이 함수들을 감싸는 얇은
호환 래퍼로 남는다.

## 두 가지 scene_config 스키마가 공존하는 이유 (원본 그대로 유지)

- default_scene_config(): sim 레벨(env.reset()/env.run_episode()가 바로
  받는) 스키마. cma_search.py의 eval_scenarios는 이 값 위에 필드를
  덮어써서 만든다(원본 optimize/cma_search.py의 EVAL_SCENE_CONFIGS와 동일).
- sample_scene_config()/to_sim_scene_config(): 1단계(공유 씬, hole_pose에
  theta 포함/peg_init_offset 이름) 스키마 -- bootstrap/diffusion/filter
  단계가 쓴다. to_sim_scene_config()로 변환해야 env.reset()에 넘길 수 있다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.peg_in_hole_sim import DT, MAX_STEPS, Z_RATE, PegInHoleSim
from sim.peg_in_hole_sim import _default_scene_config as _sim_default_scene_config
from sim.peg_in_hole_sim import N_SUBSTEPS
from sim.stabilizer import Stabilizer, run_approach_phase

# 1단계(공유 씬) 무작위화 범위 -- pipeline/scene_sampler.py에서 그대로 옮겨옴.
_BASE_FRICTION = 0.5
_HOLE_XY_RANGE_M = 0.01
_HOLE_THETA_RANGE_RAD = 0.1
_FRICTION_SCALE_RANGE = (0.7, 1.3)
_CLEARANCE_RANGE_M = (0.002, 0.005)
_PEG_OFFSET_RANGE_M = 0.015


def default_scene_config() -> dict[str, Any]:
    return _sim_default_scene_config()


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    hole_dx, hole_dy = rng.uniform(-_HOLE_XY_RANGE_M, _HOLE_XY_RANGE_M, size=2)
    hole_theta = rng.uniform(-_HOLE_THETA_RANGE_RAD, _HOLE_THETA_RANGE_RAD)
    friction = _BASE_FRICTION * rng.uniform(*_FRICTION_SCALE_RANGE)
    clearance_m = rng.uniform(*_CLEARANCE_RANGE_M)
    peg_offset = tuple(rng.uniform(-_PEG_OFFSET_RANGE_M, _PEG_OFFSET_RANGE_M, size=2))

    return {
        "hole_pose": (float(hole_dx), float(hole_dy), float(hole_theta)),
        "friction": float(friction),
        "clearance_m": float(clearance_m),
        "peg_init_offset": (float(peg_offset[0]), float(peg_offset[1])),
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    hole_dx, hole_dy, _theta = shared_cfg["hole_pose"]
    cfg = default_scene_config()
    cfg.update(
        {
            "hole_pos_xy": (hole_dx, hole_dy),
            "friction": shared_cfg["friction"],
            "clearance_m": shared_cfg["clearance_m"],
            "peg_init_offset_xy": shared_cfg["peg_init_offset"],
            "peg_init_wrist": 0.0,
        }
    )
    return cfg


class PegInHoleEnv(BaseTaskEnv):
    """stabilizer_config/use_stabilizer: 3단계(Stabilizer, 왼팔) 지원.
    tasks/peg_in_hole.yaml에 `stabilizer` 절이 있으면 sim/task_registry.py의
    TaskConfig.make_env()가 stabilizer_config를 자동으로 넘겨준다 -- 호출자가
    직접 신경 쓸 필요 없음. use_stabilizer=False로 명시하면(자유물체 전환
    전후 성공률 비교 등) 물체는 여전히 freejoint(자유물체)지만 Stabilizer가
    붙지 않아 순수히 중력+테이블 마찰만으로 버티는 상태를 시험할 수 있다."""

    def __init__(
        self,
        xml_path: str | None = None,
        stabilizer_config: dict[str, Any] | None = None,
        use_stabilizer: bool = True,
    ):
        self._sim = PegInHoleSim(xml_path=xml_path)
        self._outer_half = 0.0
        self._target_depth = 0.0
        self._stabilizer: Stabilizer | None = None
        if use_stabilizer and stabilizer_config is not None:
            self._stabilizer = Stabilizer(
                self._sim.model, self._sim.data,
                stabilizer_config["object_body"], stabilizer_config["grasp_offset"],
            )

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> float:
        self._outer_half = self._sim.reset(scene_config)
        self._target_depth = scene_config["target_insertion_depth"]
        if self._stabilizer is not None:
            self._stabilizer.reset()
        return self._outer_half

    def step(self, action: np.ndarray) -> None:
        self._sim.step(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """원본 _run_episode_with_sim()의 리워드 공식 그대로."""
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
        """원본: insertion_depth >= target_depth (xy footprint 게이팅은
        insertion_depth 자체 계산 시 이미 반영됨, run_episode() 참고)."""
        return episode_result["insertion_depth"] >= self._target_depth

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """sim/peg_in_hole_sim.py의 _run_episode_with_sim()과 동일한 루프
        (tests/test_regression_peg_in_hole.py로 동일성 확인) -- reward/success만
        compute_reward()/is_success()를 거치도록 뽑아냈다."""
        sim = self._sim
        outer_half = self.reset(scene_config)

        kp_xy = float(gains["Kp_xy"])
        kd_xy = float(gains["Kd_xy"])

        # Stabilizer가 있으면, Actuator 루프를 시작하기 전에 먼저 확실히
        # 쥐게 한다(run_approach_phase() docstring 참고 -- 루프 안에서
        # tick()을 부르면 붙잡기 전에 반작용력이 먼저 hole을 밀어내는
        # 경쟁이 생겨서 성공률이 덜 회복됐다, 실측 확인).
        left_arm_traj: list[np.ndarray] = run_approach_phase(
            self._stabilizer, sim.model, sim.data, N_SUBSTEPS
        )

        ee_poses = [sim.get_ee_pose()]
        actions: list[np.ndarray] = []
        forces: list[np.ndarray] = []
        torques: list[np.ndarray] = []

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

            self.step(delta)
            if self._stabilizer is not None:
                left_arm_traj.append(self._stabilizer.ee_pos)

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

            if self.is_success({"insertion_depth": insertion_depth}):
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

        result = {
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
            "stabilizer_grasped": bool(self._stabilizer.grasped) if self._stabilizer is not None else None,
        }
        if self._stabilizer is not None:
            result["left_arm_traj"] = np.stack(left_arm_traj).astype(np.float32)
        return result
