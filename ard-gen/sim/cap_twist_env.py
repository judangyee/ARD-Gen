"""cap_twist: 리팩토링된 태스크 플러그인 구조를 보여주기 위한 더미(가상)
태스크.

물리 시뮬레이션(MuJoCo)이 전혀 없다 -- "병뚜껑을 목표 각도만큼 돌린다"를
순수 숫자 적분으로 흉내낸 1차원 장난감 모델이다. 이 파일의 목적은 실제
태스크를 만드는 게 아니라, **새 태스크가 정말로 이 파일(Env 클래스) +
tasks/cap_twist.yaml(설정) 두 개만으로 sim/task_registry.py의
TASK_REGISTRY에 이름 한 줄만 추가하면 optimize/cma_search.py,
pipeline/bootstrap.py, pipeline/diffusion_gains.py,
pipeline/filter_episodes.py, pipeline/language_labeling.py 전체를 그대로
--task cap_twist로 돌릴 수 있는지** 시연하는 것이다(리팩토링 요청 6번
항목). tests/test_regression_peg_in_hole.py가 "기존 태스크가 안 깨졌다"를
확인한다면, 이 파일+cma_search 등 실행 결과가 "새 태스크를 최소 노력으로
추가할 수 있다"를 확인해준다.

## 모델

cap의 현재 각도(angle, rad)를 target_angle까지 돌린다. 매 스텝
`action`(이번 틱에 걸 각속도 명령)을 걸되, 정지 마찰(friction, rad/step)
보다 작은 명령은 전혀 못 움직인다(정지 마찰 모델) -- 그래서 게인(Kp)이
너무 작으면 아예 못 돈다는 게, 이 더미 모델에서도 "게인 탐색이 필요한"
문제를 만들어준다(실제 태스크들과 같은 성질).

## direction/quantity: 언어 라벨링 어휘 분리(리팩토링 4번 항목) 시연

run_episode() 결과에 "direction"("cw"/"ccw", target_angle 부호)과
"quantity"(반바퀴 단위로 반올림한 회전량)를 담아서, pipeline/filter_episodes.py
가 그대로 episode에 실어 나르고 pipeline/language_labeling.py가
tasks/cap_twist.yaml의 direction_words/quantity_words로 문장을 채운다 --
peg-in-hole은 이 두 필드를 안 쓰므로(direction_words/quantity_words가
빈 dict) 굳이 채우지 않는다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.base_task_env import BaseTaskEnv

MAX_STEPS = 200
_SUCCESS_ERROR_RAD = 0.05  # 목표 각도와 이 이내로 가까워지면 성공


def default_scene_config() -> dict[str, Any]:
    return {"target_angle": np.pi, "friction": 0.01}


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """1단계(공유 씬) 무작위 샘플 -- 다른 태스크의 hole_pose/friction/...와
    같은 역할, 이름만 이 태스크 것으로 바뀐 것뿐이다."""
    if rng is None:
        rng = np.random.default_rng()
    turns = rng.uniform(0.25, 1.5)  # 목표 회전량(바퀴 수)
    sign = rng.choice([-1.0, 1.0])
    return {
        "target_turns": float(sign * turns),
        "friction": float(rng.uniform(0.005, 0.03)),
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_angle": shared_cfg["target_turns"] * 2 * np.pi,
        "friction": shared_cfg["friction"],
    }


class CapTwistEnv(BaseTaskEnv):
    def __init__(self):
        self.angle = 0.0
        self.target_angle = 0.0
        self.friction = 0.0

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> None:
        self.angle = 0.0
        self.target_angle = float(scene_config["target_angle"])
        self.friction = float(scene_config["friction"])

    def step(self, action: float) -> None:
        """action: 이번 틱에 걸고 싶은 각속도 명령(rad/step). 정지 마찰보다
        작으면 전혀 안 움직인다(부호 유지, 크기만 마찰만큼 깎임)."""
        if abs(action) <= self.friction:
            return
        self.angle += action - self.friction * np.sign(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        reward = -10.0 * abs(episode_result["final_error"]) - 0.01 * episode_result["step_count"]
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return abs(episode_result["final_error"]) < _SUCCESS_ERROR_RAD

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        self.reset(scene_config)
        kp = float(gains["Kp"])

        step_count = 0
        success = False
        for step_count in range(1, MAX_STEPS + 1):
            error = self.target_angle - self.angle
            self.step(kp * error)
            if self.is_success({"final_error": self.target_angle - self.angle}):
                success = True
                break

        final_error = self.target_angle - self.angle
        episode_result = {"final_error": final_error, "step_count": step_count, "success": success}
        reward = self.compute_reward(episode_result)

        turns = self.target_angle / (2 * np.pi)
        return {
            "final_error": float(final_error),
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "direction": "cw" if self.target_angle >= 0 else "ccw",
            "quantity": round(abs(turns) * 2) / 2,  # 반바퀴 단위로 반올림
        }
