"""나사 조이기(screw driving) 태스크(OpenArm 양팔 버전)를 BaseTaskEnv
인터페이스로 감싼 것.

peg_in_hole_openarm_env.py/tacker_openarm_env.py와 같은 이유로, 물리/제어
자체는 새로 만들지 않는다 -- sim/screw_driving_bimanual_openarm_sim.py의
ScrewDrivingBimanualOpenArmSim(검증된 토크 리미터 turn/rewind 상태 기계 +
computed-torque 컨트롤러, 그 모듈 docstring 참고)을 그대로 재사용하고, 그
모듈의 __main__ 데모/render_screw_driving_bimanual_openarm.py가 이미 쓰는
`sim.step(gains)`를 매 스텝 반복 호출하는 루프를 그대로 옮기되,
reward/success 공식만 compute_reward()/is_success()로 뽑아냈다 -- sim 모듈의
control-loop 로직(step()/_advance_virtual_combined() 등)은 한 줄도 바꾸지
않았다(사용자 지시).

## 왜 이 태스크를 OpenArm 파이프라인에 연결하는 게 이제서야 이뤄지는가

진단(별도 세션)에서 발견된 대로, optimize/screw_driving_cma_search.py는
레거시 VX300s 단일팔 시뮬레이터(sim/screw_driving_sim.py)만 최적화하고
있었고 OpenArm 양팔 버전(이 파일이 감싸는 대상)은 TASK_REGISTRY에도 없어서
범용 파이프라인(optimize/cma_search.py --task, pipeline/bootstrap.py 등)이
전혀 손대지 못했다. 이 파일 + sim/task_registry.py 등록 + tasks/
screw_driving.yaml이 그 공백을 메운다 -- PIPELINE.md의 "screw_driving
OpenArm 파이프라인 연결" 절 참고.

## 게인: torque_limit 하나뿐인 이유

이 태스크는 peg_in_hole/tacker 같은 admittance(힘/위치 오차 피드백)가
아니라 **토크 리미터**다(sim 모듈 docstring "이 탐색이 실제로 찾아낸 것"
절 참고) -- 저항 토크가 torque_limit을 넘는 순간 그 사이클의 "돌리기"를
멈추고 되감는다. 접근 속도(NOMINAL_RATE), 사이클 범위(TURN_LOW/HIGH) 등은
sim 모듈에 하드코딩된 물리/제어 상수고 게인 탐색 대상이 아니다(이미
레거시 screw_driving_cma_search.py 시절부터 그렇게 설계됐고, 그 설계
자체는 바꾸지 않는다) -- 그래서 tasks/screw_driving.yaml의 gains.names는
["torque_limit"] 하나다(episode_io.py 모듈 docstring이 이미 "나사
조이기의 torque_limit 1개"로 이 설계를 전제하고 있었다).

## left_arm_traj: 왼팔이 고정이라도 기록하는 이유

peg_in_hole/tacker OpenArm 버전과 동일한 이유(이 파일들 docstring 참고) --
왼팔이 generic Stabilizer가 아니라 block을 쥔 채 이 태스크 내내 고정된
진짜 7-DOF 팔이라 object_body/grasp_offset을 지정할 자리가 없다. 다만
4/5단계가 기대하는 "left_arm이 있으면 role=stabilizer로 저장" 계약을
스키마 일관성 차원에서 유지하기 위해 왼팔 EE 위치를 매 스텝 기록한다.

## "forces"가 항상 빈 배열인 이유 (정직한 반영, cap_twist와 동일 선택)

이 태스크에는 선형 힘 센서가 없다 -- 실제로 (별도 세션 진단에서 실측
확인된 대로) driver와 bolt는 contact/weld 둘 다 없이 완전히 분리된
물체라서, 나사 조임 저항 토크가 오른팔에 물리적으로 전달되는 경로 자체가
없다(이 설계 공백은 PIPELINE.md에 별도로 기록, 다음 세션에서 설계 방향을
정할 미해결 2번 이슈). 그래서 억지로 숫자를 채우지 않고 cap_twist의
"forces는 선형 힘 센서가 없어 항상 빈 배열"과 같은 선택을 했다 -- "torque"
(bolt_drive_torque 액추에이터 effort, 스칼라)만 (T,1) 형태로 기록한다.

## quantity/direction: 깊이를 수량어로, 방향어는 없음

나사 조이기는 항상 "조이는" 단일 방향이라 cap_twist의 cw/ccw 같은
direction 개념이 없다(direction_words 비움). 대신 최종 삽입 깊이(mm)를
quantity로 채워서 "몇 mm까지 조였는지"를 언어 라벨에 쓸 수 있게 한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim.base_task_env import BaseTaskEnv
from sim.screw_driving_bimanual_openarm_sim import (
    MAX_CONTROL_STEPS,
    TARGET_DEPTH,
    ScrewDrivingBimanualOpenArmSim,
)
from sim.screw_driving_bimanual_openarm_sim import _default_scene_config as _sim_default_scene_config

# 씬 무작위화 범위 (별도 세션 진단에서 실측된 범위를 그대로 반영).
#
# target_depth: bolt_slide의 물리적 최대치(TARGET_DEPTH=0.034m)가 상한이다
# (joint range가 그 값이라 더 크게 잡을 수 없다) -- 그 아래로만 무작위화해서
# "완전히 끝까지는 아니지만 충분히 조여지면 성공"으로 치는 시나리오(실제
# 볼트/부품 치수 공차를 흉내)를 표현한다.
_TARGET_DEPTH_RANGE = (0.028, TARGET_DEPTH)

# hinge_friction_scale: 레거시 screw_driving_cma_search.py가 이미 "기본
# 저항(1.0)"과 "더 뻑뻑한 나사(5.0)"를 대표 시나리오 2개로 실측 검증해둔
# 값이다(그 파일 상단 docstring 참고) -- 그 범위를 그대로 이어받아
# 무작위화 구간으로 쓴다. 별도 세션 진단(probe_screw_d.py)에서 이 구간을
# 50배까지 흔들어도(0.05~1.5N*m 관측 범위를 포함) 발산 없이 안전하다는 걸
# 실측 확인했다.
_HINGE_FRICTION_SCALE_RANGE = (0.5, 5.0)


def default_scene_config() -> dict[str, Any]:
    return _sim_default_scene_config()


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """이미 sim 레벨(reset()이 바로 받는) 스키마를 그대로 뽑는다 -- peg_in_hole/
    tacker OpenArm 버전과 같은 이유로 별도 "1단계 공유 씬" 스키마가 없다."""
    if rng is None:
        rng = np.random.default_rng()
    target_depth = float(rng.uniform(*_TARGET_DEPTH_RANGE))
    hinge_friction_scale = float(rng.uniform(*_HINGE_FRICTION_SCALE_RANGE))
    return {"target_depth": target_depth, "hinge_friction_scale": hinge_friction_scale}


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(shared_cfg)


class ScrewDrivingOpenArmEnv(BaseTaskEnv):
    """xml_path 하나만 받는다 -- stabilizer_config/use_stabilizer를 받지
    않는 이유는 peg_in_hole_openarm_env.py/tacker_openarm_env.py와 동일
    (위 모듈 docstring "left_arm_traj" 절 참고, tasks/screw_driving.yaml에
    `stabilizer` 절이 없으므로 TaskConfig.make_env()가 애초에 그 kwarg를
    안 넘긴다)."""

    def __init__(self, xml_path: str | None = None):
        self._sim = ScrewDrivingBimanualOpenArmSim(xml_path=xml_path)
        self._target_depth = TARGET_DEPTH

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> None:
        self._sim.reset(scene_config)
        self._target_depth = float(scene_config.get("target_depth", TARGET_DEPTH))

    def step(self, action: dict[str, float]) -> dict[str, Any]:
        """action: 이번 틱에 쓸 게인 dict({"torque_limit": float}) --
        sim.step()이 매 스텝 같은 게인을 받는 설계(토크 리미터는 에피소드
        내내 고정값)라서 peg_in_hole/tacker의 "델타 벡터" action과 다르게
        게인 dict 자체가 action이다(sim 모듈 __main__ 데모와 동일 호출
        방식)."""
        return self._sim.step(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """sim 모듈의 _run_episode_with_sim() 리워드 공식과 완전히 동일
        (이미 레거시 screw_driving_cma_search.py로 실측 검증된 설계,
        손대지 않았다)."""
        reward = 20.0 * episode_result["insertion_depth"] - 0.001 * episode_result["step_count"]
        if episode_result.get("success"):
            reward += 50.0 - 3.0 * episode_result["max_torque"]
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        return episode_result["insertion_depth"] >= self._target_depth * 0.99

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """sim/screw_driving_bimanual_openarm_sim.py의 _run_episode_with_sim()과
        동일한 루프 -- reward/success만 compute_reward()/is_success()를 거치도록
        뽑아냈고, left_arm_traj 기록(위 모듈 docstring 참고)을 추가했다."""
        sim = self._sim
        self.reset(scene_config)

        ee_poses = [sim.get_driver_tip_pos()]
        left_arm_traj = [sim.get_left_ee_pos()]
        actions: list[np.ndarray] = []
        torques: list[np.ndarray] = []

        depth = 0.0
        max_torque = 0.0
        success = False
        step_count = 0

        for step_count in range(1, MAX_CONTROL_STEPS + 1):
            info = sim.step(gains)
            depth = sim.get_insertion_depth()
            max_torque = max(max_torque, abs(info["torque"]))

            ee_poses.append(sim.get_driver_tip_pos())
            left_arm_traj.append(sim.get_left_ee_pos())
            torques.append(np.array([info["torque"]], dtype=np.float32))
            signed_rate = info["rate"] if info["phase"] == "turn" else -info["rate"]
            actions.append(np.array([signed_rate], dtype=np.float32))

            partial_result = {"insertion_depth": depth}
            if self.is_success(partial_result):
                success = True
                break

        mean_abs_torque = float(np.mean(np.abs(torques))) if torques else 0.0

        episode_result = {
            "insertion_depth": float(depth),
            "max_torque": float(max_torque),
            "mean_abs_torque": mean_abs_torque,
            "step_count": step_count,
            "success": success,
        }
        reward = self.compute_reward(episode_result)

        return {
            "trajectory": {"ee_poses": np.stack(ee_poses).astype(np.float32)},
            "ee_poses": np.stack(ee_poses).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0, 1), dtype=np.float32),
            "forces": np.zeros((0, 3), dtype=np.float32),
            "torques": np.stack(torques).astype(np.float32) if torques else np.zeros((0, 1), dtype=np.float32),
            "insertion_depth": float(depth),
            "max_torque": float(max_torque),
            "mean_abs_torque": mean_abs_torque,
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "left_arm_traj": np.stack(left_arm_traj).astype(np.float32),
            "quantity": round(float(depth) * 1000.0, 1),
        }


if __name__ == "__main__":
    for torque_limit in [float("inf"), 1.79, 1.0, 0.6]:
        gains = {"torque_limit": torque_limit}
        cfg = default_scene_config()
        env = ScrewDrivingOpenArmEnv()
        result = env.run_episode(gains, cfg)
        label = "no limit" if np.isinf(torque_limit) else f"limit={torque_limit}"
        print(
            f"[screw_driving_openarm_env] {label:12s} success={result['success']} "
            f"depth={result['insertion_depth'] * 1000:.2f}mm steps={result['step_count']} "
            f"max_torque={result['max_torque']:.3f} reward={result['reward']:.2f}"
        )
