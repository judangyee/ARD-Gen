"""cap_twist: 병뚜껑을 돌려서 여는 실제 MuJoCo 물리 태스크.

sim/peg_in_hole_env.py(+sim/peg_in_hole_sim.py)의 admittance controller
패턴을 그대로 따르되, "위치 오차에 반응하는 힘 보정"(force-PD, Delta_xy =
-Kp_xy*F_error - Kd_xy*dF_error/dt)을 "회전 저항 토크에 반응하는 각속도
보정"으로 바꿨다:

    tau_error = tau_measured - tau_target          (tau_target ~= 0: 저항을
                                                     최소화하면서 돌린다)
    omega_correction = -Kp_tau * tau_error
    omega_new = omega_base + omega_correction
    target_angle += omega_new * dt

즉 peg-in-hole이 "z축은 일정 속도로 내려가고(Z_RATE), xy는 힘에 반응해
보정"했던 것처럼, 여기는 "회전은 기본 각속도(OMEGA_BASE)로 계속 진행하되,
저항 토크가 커지면 그만큼 속도를 늦춘다"(저항이 사라지면 다시 기본
속도로 돌아간다 -- peg-in-hole의 F_desired_xy=0과 대응되는 tau_target=0).

## 이전 버전과의 차이 (더미 -> 실제 물리)

이전 sim/cap_twist_env.py는 MuJoCo 없이 각도를 직접 적분하는 1차원 숫자
모델이었다(리팩토링 시연용, git log 참고). 이번 버전은 assets/cap_twist.xml
(bottle 고정 + cap 힌지 조인트 + jointactuatorfrc 토크 센서)을 실제로
mj_step()으로 시뮬레이션한다. peg_in_hole과 같은 단순화 전제를 쓴다:
오른팔이 cap을 이미 강체로 그립했다고 가정하고 팔 자체(Jacobian, IK)는
모델링하지 않는다 -- 그래서 "손목 회전 각도"가 곧 cap 힌지 각도이고,
action은 peg-in-hole의 3차원 위치 델타와 달리 스칼라 하나(이번 틱 목표
회전각)다.

## 저항을 dof_damping(속도 비례)으로 모델링 (assets/cap_twist.xml 상단 docstring)

처음에는 저항을 hinge의 frictionloss(건마찰, 속도 무관하게 거의 일정한
토크)로 모델링했는데, 실측해보니 그러면 회전 속도를 늦춰도 필요 토크가
거의 안 줄어서 Kp_tau가 사실상 무의미해졌다(자세한 경위는
assets/cap_twist.xml 참고). scene_config의 resistance_torque(N*m, "기본
각속도 OMEGA_BASE로 돌릴 때 필요한 토크"로 정의)를 damping =
resistance_torque / OMEGA_BASE로 환산해 dof_damping에 대입한다 -- 이러면
느리게 돌릴수록 실제로 필요 토크가 줄어들어서, admittance controller의
속도 보정이 진짜 효과가 있다.

## "뚜껑이 완전히 풀림"을 물리적으로 표현하는 방법 (disengage_ratio)

실제 뚜껑은 나사산이 다 풀리면 저항이 갑자기 사라진다. 이걸 표현하려고,
누적 회전각이 |target_rotation| * disengage_ratio(scene_config, 0.8~1.0
사이 무작위)에 도달하면 힌지의 dof_damping을 거의 0으로 낮춘다 -- 이후
토크 센서 값이 실제로 뚝 떨어지므로, is_success()의 "토크가 갑자기 0
근처로 떨어짐" 조건이 가짜 신호가 아니라 실제 물리 이벤트에 대응한다.
disengage_ratio가 1.0보다 작을 수 있어서, "누적 회전각이 목표에 도달"
조건보다 이 조건이 먼저 성립하는 경우도 실제로 생긴다(두 성공 조건이
서로 다른 걸 감지하는 게 의미 있어지는 이유).

## 스턱 감지 + 역회전 재시도 (그립 슬립 방지)

측정 토크가 임계값(TAU_STUCK_THRESHOLD, assets/cap_twist.xml의 액추에이터
forcerange보다 낮게 잡음)을 넘으면 "뚜껑이 걸렸다"고 보고, REVERSE_STEPS
스텝 동안 반대 방향으로 살짝 돌려서 누적된 위치 오차/토크를 풀어준 뒤
다시 정방향 admittance 제어로 복귀한다 -- 실제로 뚜껑이 안 열릴 때 사람이
"살짝 반대로 풀었다가 다시 돌리는" 동작과 같다.
"""
from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np

from sim.base_task_env import BaseTaskEnv

_DEFAULT_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "cap_twist.xml")

N_SUBSTEPS = 5  # mj_step 호출당 substep 수 (timestep=0.002 -> 제어 주기 dt=0.01s)
DT = N_SUBSTEPS * 0.002
MAX_STEPS = 1200

OMEGA_BASE = 3.0  # rad/s, 저항이 없을 때 기본으로 유지하려는 회전 속도
TAU_TARGET = 0.0  # 저항 토크를 최소화하면서 돌린다(peg-in-hole의 F_desired_xy=0과 대응)

TAU_STUCK_THRESHOLD = 0.8  # N*m, 이 이상이면 "뚜껑이 걸렸다"고 보고 역회전+재시도
REVERSE_STEPS = 15  # 역회전을 유지하는 스텝 수
OMEGA_REVERSE = 2.0  # rad/s, 역회전 속도

TORQUE_DROP_THRESHOLD = 0.05  # N*m, disengage 이후 이 아래로 떨어지면 "완전히 풀림"
_LOOSE_DAMPING = 0.01  # disengage 후 남는 잔여 저항(수치 안정성용, 거의 0)

_SAFE_TORQUE = 0.5  # N*m, 이 이상 토크에는 리워드 페널티(과도한 힘 사용)


def default_scene_config() -> dict[str, Any]:
    return {
        "target_rotation": 4 * np.pi,  # 2바퀴
        "resistance_torque": 0.5,
        "disengage_ratio": 0.9,
        "cap_initial_angle": 0.0,
    }


def sample_scene_config(rng: np.random.Generator | None = None) -> dict[str, Any]:
    """1단계(공유 씬) 무작위 샘플 -- 병마다 다른 목표 회전량/뻑뻑함/나사산
    길이 비율을 흉내낸다. resistance_torque 상한(1.0)은 TAU_STUCK_THRESHOLD
    (0.8)보다 위라서, Kp_tau=0(보정 없음)이면 정상 각속도에서 요구 토크가
    걸림 임계값을 넘어 stuck-반복에 빠지는 씬이 실제로 섞여 나온다 --
    Kp_tau가 실제로 필요해지는 지점."""
    if rng is None:
        rng = np.random.default_rng()
    turns = rng.uniform(1.0, 2.5)
    sign = rng.choice([-1.0, 1.0])
    return {
        "target_turns": float(sign * turns),
        "resistance_torque": float(rng.uniform(0.2, 1.0)),
        "disengage_ratio": float(rng.uniform(0.8, 1.0)),
    }


def to_sim_scene_config(shared_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_rotation": shared_cfg["target_turns"] * 2 * np.pi,
        "resistance_torque": shared_cfg["resistance_torque"],
        "disengage_ratio": shared_cfg["disengage_ratio"],
        "cap_initial_angle": 0.0,
    }


class CapTwistSim:
    """MjModel/MjData를 재사용하는 시뮬레이션 래퍼 (peg_in_hole_sim.PegInHoleSim과
    같은 역할 -- run_episode()가 매번 새로 컴파일하지 않도록 재사용한다)."""

    def __init__(self, xml_path: str | None = None):
        self.xml_path = xml_path or _DEFAULT_XML
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self._hinge_qposadr = self.model.joint("cap_hinge").qposadr[0]
        self._hinge_dofadr = self.model.joint("cap_hinge").dofadr[0]
        self._actuator_id = self.model.actuator("cap_drive").id
        self._torque_adr = self.model.sensor("cap_torque").adr[0]

    def reset(self, scene_config: dict[str, Any]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # resistance_torque(N*m) = "OMEGA_BASE로 돌릴 때 필요한 토크"로 정의하고
        # damping(N*m*s/rad)으로 환산한다 (모듈 상단 docstring 참고).
        damping = float(scene_config["resistance_torque"]) / OMEGA_BASE
        self.model.dof_damping[self._hinge_dofadr] = damping

        init_angle = float(scene_config.get("cap_initial_angle", 0.0))
        self.data.qpos[self._hinge_qposadr] = init_angle
        self.data.ctrl[self._actuator_id] = init_angle
        mujoco.mj_forward(self.model, self.data)

    def get_torque(self) -> float:
        return float(self.data.sensordata[self._torque_adr])

    def get_angle(self) -> float:
        return float(self.data.qpos[self._hinge_qposadr])

    def set_damping(self, value: float) -> None:
        self.model.dof_damping[self._hinge_dofadr] = float(value)

    def step(self, target_angle: float) -> None:
        self.data.ctrl[self._actuator_id] = target_angle
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)


class CapTwistEnv(BaseTaskEnv):
    def __init__(self, xml_path: str | None = None):
        self._sim = CapTwistSim(xml_path=xml_path)
        self._target_rotation = 0.0

    # ------------------------------------------------------------------
    def reset(self, scene_config: dict[str, Any]) -> None:
        self._sim.reset(scene_config)
        self._target_rotation = float(scene_config["target_rotation"])

    def step(self, action: float) -> None:
        """action: 이번 틱의 목표 회전각(rad, 절대값 -- 위치 액추에이터
        ctrl에 그대로 들어간다)."""
        self._sim.step(action)

    def compute_reward(self, episode_result: dict[str, Any]) -> float:
        """peg_in_hole의 리워드 구조(거리 페널티 + 진행도 보상 + 과도한
        힘 페널티 + 스텝 페널티 + 성공 보너스)를 그대로 따르되, insertion_depth
        자리에 누적 회전 진행도를, max_force 자리에 max_torque를 쓴다.

        진행도는 accumulated_rotation의 절대값이 아니라 forward_progress
        (목표 방향으로 투영한 회전량, run_episode() 참고)로 잰다 -- Kp_tau가
        너무 커서 되레 반대 방향으로 밀려버린 경우(실측 확인, 아래
        run_episode() docstring 참고) abs()만 보면 "방향은 틀렸지만 많이
        돌았으니 성공"으로 잘못 판정하는 버그가 있었다."""
        target_mag = abs(episode_result["target_rotation"])
        forward_progress = episode_result["forward_progress"]
        progress_fraction = max(0.0, min(forward_progress / target_mag, 1.0)) if target_mag > 0 else 0.0
        shortfall = max(0.0, target_mag - forward_progress)
        reward = (
            -1.0 * shortfall
            + 30.0 * progress_fraction
            - 0.02 * max(0.0, episode_result["max_torque"] - _SAFE_TORQUE)
            - 0.01 * episode_result["step_count"]
        )
        if episode_result.get("success"):
            reward += 50.0
        return float(reward)

    def is_success(self, episode_result: dict[str, Any]) -> bool:
        """목표 방향으로의 누적 회전(forward_progress)이 목표 크기에
        도달했거나, 이미 disengage된 상태에서 측정 토크가 거의 0으로
        떨어졌으면(나사산이 풀려서 이제 저항 없이 헛돈다) 성공."""
        target_mag = abs(episode_result["target_rotation"])
        reached_target = episode_result["forward_progress"] >= target_mag
        torque_dropped = episode_result.get("disengaged", False) and abs(
            episode_result.get("current_torque", 1.0)
        ) < TORQUE_DROP_THRESHOLD
        return bool(reached_target or torque_dropped)

    # ------------------------------------------------------------------
    def run_episode(self, gains: dict[str, float], scene_config: dict[str, Any]) -> dict[str, Any]:
        """## forward_progress: 목표 방향으로 투영한 누적 회전량

        Kp_tau가 너무 크면 tau_error에 대한 되먹임이 부호를 오락가락하며
        (고전적인 비례이득 과다 -> 불안정) 실제로 cap이 목표와 반대 방향으로
        밀려나는 경우가 실측된다(예: target_rotation>0인데 최종 각도가
        음수). accumulated_rotation(부호 있는 절대 각도) 대신
        actual_angle*direction(목표 방향 축에 투영한 값, 반대로 돌면 음수가
        됨)을 진행도/성공 판정에 쓴다 -- 그래야 "많이 돌긴 했는데 반대
        방향"인 경우를 성공으로 잘못 세지 않는다."""
        sim = self._sim
        self.reset(scene_config)
        target_rotation = self._target_rotation
        direction = 1.0 if target_rotation >= 0 else -1.0
        target_mag = abs(target_rotation)
        disengage_angle = target_mag * float(scene_config.get("disengage_ratio", 1.0))

        kp_tau = float(gains["Kp_tau"])

        target_angle = sim.get_angle()
        max_torque = 0.0
        disengaged = False
        reverse_counter = 0
        success = False
        step_count = 0
        forward_progress = 0.0

        for step_count in range(1, MAX_STEPS + 1):
            tau = sim.get_torque()
            max_torque = max(max_torque, abs(tau))

            if reverse_counter > 0:
                omega = -direction * OMEGA_REVERSE
                reverse_counter -= 1
            elif abs(tau) > TAU_STUCK_THRESHOLD:
                reverse_counter = REVERSE_STEPS
                omega = -direction * OMEGA_REVERSE
            else:
                tau_error = tau - TAU_TARGET
                omega_correction = -kp_tau * tau_error
                omega = direction * OMEGA_BASE + omega_correction

            target_angle += omega * DT
            self.step(target_angle)

            actual_angle = sim.get_angle()
            forward_progress = actual_angle * direction
            if not disengaged and forward_progress >= disengage_angle:
                sim.set_damping(_LOOSE_DAMPING)
                disengaged = True

            partial_result = {
                "target_rotation": target_rotation,
                "forward_progress": forward_progress,
                "disengaged": disengaged,
                "current_torque": sim.get_torque(),
            }
            if self.is_success(partial_result):
                success = True
                break

        final_angle = sim.get_angle()
        episode_result = {
            "accumulated_rotation": float(final_angle),
            "target_rotation": float(target_rotation),
            "forward_progress": float(forward_progress),
            "max_torque": float(max_torque),
            "step_count": step_count,
            "success": success,
            "disengaged": disengaged,
            "current_torque": sim.get_torque(),
        }
        reward = self.compute_reward(episode_result)

        turns = target_rotation / (2 * np.pi)
        return {
            "accumulated_rotation": float(final_angle),
            "target_rotation": float(target_rotation),
            "forward_progress": float(forward_progress),
            "max_torque": float(max_torque),
            "step_count": step_count,
            "success": success,
            "reward": reward,
            "gains": dict(gains),
            "scene_config": scene_config,
            "direction": "cw" if target_rotation >= 0 else "ccw",
            "quantity": round(abs(turns) * 2) / 2,  # 반바퀴 단위로 반올림
        }
